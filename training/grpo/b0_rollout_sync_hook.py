# training/grpo/b0_rollout_sync_hook.py
"""B0 spike instrumentation: produce the inputs for the rollout-sync (logprob-parity) gate.

The pure guard logic lives in ``scripts/rollout_sync_guard.py`` (covered by
``tests/test_rollout_sync_guard.py``). Nothing in veRL's default reward dump *produces* its
inputs, so this module attaches at the trainer and captures them live during the first two
optimizer updates of the B0 spike.

Enabled by ``B0_ROLLOUT_SYNC`` (set in ``run_verl_main_ppo.main``, AFTER
``apply_verl_runtime_patch()`` and BEFORE the trainer runs). It monkeypatches the actor-update
method on ``verl.trainer.ppo.ray_trainer.RayPPOTrainer`` so it wraps whatever that method already
is (the persona runtime patch also wraps ``_update_actor`` for the ELBO/SFT path — we preserve it
by wrapping the *current* attribute, not the pristine upstream one).

At the first two training generations and update calls, the wrapper captures:
  1. vLLM prompt logprobs on ONE FIXED DP-sized prompt+continuation batch, scored while the
     rollout replicas are awake before each call to ``sleep_replicas``.
  2. HF actor logprobs on those exact tokens via ``_compute_old_log_prob`` before each optimizer
     update.
  3. Each engine's step0->step1 delta. Comparing deltas cancels a stable HF/vLLM absolute offset
     and directly shows whether the rollout engine received the actor update.
  4. Legacy within-step absolute parity remains diagnostic only; it no longer gates the result.
  5. Optional IPC/weight fingerprint: intentionally OMITTED — vLLM weights are TP-sharded/fused
     with no actor-comparable API (~36GB); the logprob signals (1-2) are the gate. Do not fake it.

!!! CONTROLLER: CONFIRM VERL SYMBOL NAMES BEFORE B0 !!!
This env has NO veRL installed, so the names below cannot be verified locally. They were chosen to
match the pinned-tree persona patch in ``training/grpo/verl_runtime_patch.py`` (which wraps
``RayPPOTrainer._update_actor(self, batch)``) and the brief. Before the B0 spike, re-confirm
against ``$VERL_DIR`` at the pinned SHA (>= c791da0b):
  - the actor-update method name         (``_update_actor``; may be ``update_actor``/``update_policy``)
  - batch keys                           (``rollout_log_probs``, ``old_log_probs``, ``response_mask``)
  - the trainer logprob adapter          (``_compute_old_log_prob`` -> ``old_log_probs``)
All capture is wrapped in try/except so instrumentation NEVER crashes the training step; a failure
is recorded in ``rollout_sync.json`` (``ok: false`` + an ``error`` field) so the gate fails loudly
rather than silently.
"""
from __future__ import annotations

import asyncio
import copy
import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

from scripts.rollout_sync_guard import assert_fixed_sequence_delta_synced, logprob_parity

logger = logging.getLogger(__name__)

# Confirm against the pinned veRL SHA before B0 (see module docstring).
_ACTOR_UPDATE_METHOD = "_update_actor"           # RayPPOTrainer method that runs the optimizer step
# veRL-0.9 batch key aliases; the real key varies by version, so try each in order and use the
# first present in the DataProto's .batch (TensorDict). See _first_present_key().
_ROLLOUT_LOGPROB_KEYS = ["rollout_log_probs", "rollout_log_prob"]  # vLLM generation logprobs (calculate_log_probs=True)
_ACTOR_LOGPROB_KEYS = ["old_log_prob", "old_log_probs"]            # actor's recomputed logprobs for the same tokens
_RESPONSE_MASK_KEY = "response_mask"
_ATTENTION_MASK_KEY = "attention_mask"
_MAX_CAPTURES = 2                                 # update-call 0 and 1 only


def _run_dir() -> str:
    return os.environ.get("RL_RUN_DIR") or os.environ.get("PERSONA_REWARD_DUMP_DIR") or os.getcwd()


def _first_present_key(tensordict: Any, candidates: list[str]) -> str:
    """Return the first candidate key present in a DataProto's .batch (TensorDict).

    Raises KeyError (caught by the wrapper's try/except -> ok:false+error) if none match.
    """
    keys = set(tensordict.keys())
    for k in candidates:
        if k in keys:
            return k
    raise KeyError(f"none of {candidates} present in batch; keys={sorted(keys)}")


def _to_np(tensor: Any) -> np.ndarray:
    """Detach/CPU a torch tensor (or coerce anything array-like) to float64 numpy."""
    detach = getattr(tensor, "detach", None)
    if callable(detach):
        tensor = detach()
    cpu = getattr(tensor, "cpu", None)
    if callable(cpu):
        tensor = cpu()
    return np.asarray(tensor, dtype=np.float64)


def _to_array(value: Any) -> np.ndarray:
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()
    return np.asarray(value)


def _masked_flat(values: Any, mask: Any) -> np.ndarray:
    v = _to_np(values)
    if mask is None:
        return v.reshape(-1)
    m = _to_np(mask).astype(bool)
    if m.shape != v.shape:
        # response_mask may cover only the response span; align on the trailing axis length.
        resp_len = v.shape[-1]
        m = m[..., -resp_len:]
    return v[m]


def _within_step_parity(batch: Any) -> dict:
    """logprob_parity(rollout_lp, actor_lp) over the masked response tokens of this batch."""
    tensordict = batch.batch
    rollout = tensordict[_first_present_key(tensordict, _ROLLOUT_LOGPROB_KEYS)]
    actor = tensordict[_first_present_key(tensordict, _ACTOR_LOGPROB_KEYS)]
    mask = None
    keys = set(tensordict.keys())
    if _RESPONSE_MASK_KEY in keys:
        mask = tensordict[_RESPONSE_MASK_KEY]
    elif _ATTENTION_MASK_KEY in keys:
        # Fall back to the response span of attention_mask (last response_len columns).
        mask = tensordict[_ATTENTION_MASK_KEY]
    rollout_lp = _masked_flat(rollout, mask)
    actor_lp = _masked_flat(actor, mask)
    return logprob_parity(rollout_lp, actor_lp)


def _actor_dp_size(trainer: Any) -> int:
    fsdp_config = trainer.config.actor_rollout_ref.actor.fsdp_config
    dp_size = int(fsdp_config.get("fsdp_size", -1) or -1)
    if dp_size <= 0:
        dp_size = int(trainer.config.trainer.n_gpus_per_node) * int(trainer.config.trainer.nnodes)
    return dp_size


def _extract_fixed_sequence_specs(batch: Any, dp_size: int) -> list[dict[str, Any]]:
    """Extract unpadded token sequences plus the actor-selected response positions."""
    tensordict = batch.batch
    prompts = _to_array(tensordict["prompts"])
    responses = _to_array(tensordict["responses"])
    attention_mask = _to_array(tensordict[_ATTENTION_MASK_KEY]).astype(bool)
    response_mask = _to_array(tensordict[_RESPONSE_MASK_KEY]).astype(bool)
    if len(prompts) < dp_size:
        raise ValueError(f"B0 fixed sequence probe needs {dp_size} rows, got {len(prompts)}")

    prompt_width = prompts.shape[-1]
    response_width = responses.shape[-1]
    specs: list[dict[str, Any]] = []
    for row in range(dp_size):
        prompt_keep = attention_mask[row, :prompt_width]
        response_keep = attention_mask[row, prompt_width : prompt_width + response_width]
        prompt_ids = prompts[row][prompt_keep].astype(np.int64).tolist()
        response_ids = responses[row][response_keep].astype(np.int64).tolist()
        selected_mask = response_mask[row, :response_width][response_keep]
        selected_offsets = np.flatnonzero(selected_mask).astype(np.int64).tolist()
        if not prompt_ids or not response_ids or not selected_offsets:
            raise ValueError(
                f"B0 fixed sequence row {row} is empty: prompt={len(prompt_ids)} "
                f"response={len(response_ids)} selected={len(selected_offsets)}"
            )
        specs.append(
            {
                "sequence_ids": prompt_ids + response_ids,
                "prompt_length": len(prompt_ids),
                "selected_response_offsets": selected_offsets,
            }
        )
    return specs


def _extract_selected_prompt_logprobs(output: Any, spec: dict[str, Any]) -> np.ndarray:
    """Align veRL/vLLM prompt-logprob output to actor-selected response tokens."""
    prompt_ids = _to_array(output.extra_fields["prompt_ids"])
    prompt_logprobs = _to_array(output.extra_fields["prompt_logprobs"]).astype(np.float64)
    if prompt_ids.ndim != 2 or prompt_logprobs.ndim != 2 or prompt_ids.shape[1] != 1:
        raise ValueError(
            f"unexpected prompt-logprob shapes: ids={prompt_ids.shape} logprobs={prompt_logprobs.shape}"
        )

    sequence_ids = spec["sequence_ids"]
    prompt_length = int(spec["prompt_length"])
    values: list[float] = []
    for response_offset in spec["selected_response_offsets"]:
        token_position = prompt_length + int(response_offset)
        logprob_position = token_position - 1
        expected_token = int(sequence_ids[token_position])
        returned_token = int(prompt_ids[logprob_position, 0])
        if returned_token != expected_token:
            raise ValueError(
                f"vLLM prompt-logprob token misalignment at position {token_position}: "
                f"expected {expected_token}, got {returned_token}"
            )
        values.append(float(prompt_logprobs[logprob_position, 0]))
    return np.asarray(values, dtype=np.float64)


async def _score_fixed_sequences_async(client: Any, specs: list[dict[str, Any]], call_index: int) -> np.ndarray:
    requests = [
        client.generate(
            request_id=f"b0-fixed-delta-{call_index}-{row}",
            prompt_ids=spec["sequence_ids"],
            sampling_params={"max_tokens": 1, "temperature": 1.0, "prompt_logprobs": 0},
        )
        for row, spec in enumerate(specs)
    ]
    outputs = await asyncio.gather(*requests)
    return np.concatenate(
        [_extract_selected_prompt_logprobs(output, spec) for output, spec in zip(outputs, specs, strict=True)]
    )


def _install_fixed_sequence_rollout_capture(trainer: Any, state: dict[str, Any]) -> None:
    """Score the fixed canary in vLLM after training generation and before rollout sleep."""
    manager = trainer.async_rollout_manager
    if getattr(manager, "_b0_fixed_sequence_capture_installed", False):
        return
    original_generate = manager.generate_sequences
    expected_rows = int(trainer.config.data.train_batch_size) * int(trainer.config.actor_rollout_ref.rollout.n)
    dp_size = _actor_dp_size(trainer)
    client = trainer.llm_server_manager.get_client()

    def patched_generate(prompts, *args, **kwargs):
        output = original_generate(prompts, *args, **kwargs)
        call_index = state["rollout_calls"]
        if call_index >= _MAX_CAPTURES or len(output) != expected_rows:
            return output
        try:
            if state["fixed_specs"] is None:
                state["fixed_specs"] = _extract_fixed_sequence_specs(output, dp_size)
            rollout_lp = asyncio.run(
                _score_fixed_sequences_async(client, state["fixed_specs"], call_index)
            )
            state["rollout_lp"].append(rollout_lp)
            print(
                f"B0_ROLLOUT_SYNC: fixed vLLM call={call_index} tokens={rollout_lp.size}",
                flush=True,
            )
        except Exception as exc:
            state["error"] = f"rollout_capture@call{call_index}: {type(exc).__name__}: {exc}"
            logger.exception("B0_ROLLOUT_SYNC fixed vLLM capture failed at call %s", call_index)
        finally:
            state["rollout_calls"] = call_index + 1
        return output

    manager.generate_sequences = patched_generate
    manager._b0_fixed_sequence_capture_installed = True


def _capture_fixed_dataproto(trainer: Any, batch: Any) -> Any:
    """Deep-copy one actor-DP-sized slice as the fixed teacher-forced batch.

    veRL dispatch requires the input length to be divisible by actor data parallelism. Reusing the
    first DP-sized slice keeps all keys/meta the actor expects, while deep-copying keeps the exact
    same prompt+continuation tokens across both update calls.
    """
    dp_size = _actor_dp_size(trainer)
    if len(batch) < dp_size:
        raise ValueError(f"B0 fixed batch needs at least actor DP size {dp_size}, got {len(batch)}")
    return copy.deepcopy(batch[0:dp_size])


def _teacher_forced_logprob(
    trainer: Any,
    fixed_dp: Any,
    specs: list[dict[str, Any]],
) -> np.ndarray:
    """Actor logprobs on the fixed sequence via veRL's DataProto compatibility adapter."""
    out, _mfu = trainer._compute_old_log_prob(copy.deepcopy(fixed_dp))
    tensordict = out.batch
    logprobs = _to_np(tensordict[_first_present_key(tensordict, _ACTOR_LOGPROB_KEYS)])
    if logprobs.shape[0] != len(specs):
        raise ValueError(f"actor fixed rows differ from specs: {logprobs.shape[0]} vs {len(specs)}")
    selected = [
        logprobs[row, np.asarray(spec["selected_response_offsets"], dtype=np.int64)]
        for row, spec in enumerate(specs)
    ]
    return np.concatenate(selected)


def write_rollout_sync(run_dir: str, payload: dict) -> str:
    """Dump the B0 gate artifact (rollout_sync.json) into run_dir. Returns the path."""
    out_path = Path(run_dir) / "rollout_sync.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"B0_ROLLOUT_SYNC: wrote {out_path} ok={payload.get('ok')}", flush=True)
    return str(out_path)


def install_b0_rollout_sync_hook() -> bool:
    """Monkeypatch RayPPOTrainer's actor-update method to capture the rollout-sync signals.

    Wraps the CURRENT attribute (preserving the persona ELBO/SFT wrapper) so both hooks compose.
    Idempotent. Returns True if installed, False if veRL is unavailable.
    """
    try:
        from verl.trainer.ppo.ray_trainer import RayPPOTrainer
    except ImportError:
        logger.warning("B0_ROLLOUT_SYNC set but veRL RayPPOTrainer is unavailable; hook not installed")
        return False

    if getattr(RayPPOTrainer, "_b0_rollout_sync_hook_installed", False):
        return True

    original_update = getattr(RayPPOTrainer, _ACTOR_UPDATE_METHOD)
    original_fit = RayPPOTrainer.fit

    state: dict[str, Any] = {
        "calls": 0,
        "rollout_calls": 0,
        "step0": None,
        "step1": None,
        "actor_lp": [],
        "rollout_lp": [],
        "fixed_dp": None,
        "fixed_specs": None,
        "error": None,
    }

    def patched_fit(self, *args, **kwargs):
        _install_fixed_sequence_rollout_capture(self, state)
        return original_fit(self, *args, **kwargs)

    def patched_update(self, batch, *args, **kwargs):
        call_index = state["calls"]
        if call_index < _MAX_CAPTURES:
            # Capture BEFORE the optimizer step so old_log_probs reflect this update's rollout.
            try:
                parity = _within_step_parity(batch)
                state[f"step{call_index}"] = parity
                if state["fixed_dp"] is None:
                    state["fixed_dp"] = _capture_fixed_dataproto(self, batch)
                    actor_specs = _extract_fixed_sequence_specs(state["fixed_dp"], _actor_dp_size(self))
                    if actor_specs != state["fixed_specs"]:
                        raise RuntimeError(
                            "fixed vLLM and actor rows differ; run the B0 diagnostic with trainer.balance_batch=False"
                        )
                actor_lp = _teacher_forced_logprob(self, state["fixed_dp"], state["fixed_specs"])
                state["actor_lp"].append(actor_lp)
                print(
                    f"B0_ROLLOUT_SYNC: call={call_index} parity={parity} "
                    f"fixed_actor_tokens={actor_lp.size}",
                    flush=True,
                )
            except Exception as exc:  # never crash the training step on instrumentation
                state["error"] = f"capture@call{call_index}: {type(exc).__name__}: {exc}"
                logger.exception("B0_ROLLOUT_SYNC capture failed at call %s", call_index)

        result = original_update(self, batch, *args, **kwargs)
        state["calls"] = call_index + 1

        # After both captures, run the guard and dump the artifact exactly once.
        if state["calls"] == _MAX_CAPTURES and not state.get("written"):
            state["written"] = True
            payload: dict[str, Any] = {}
            try:
                if state["error"]:
                    raise RuntimeError(state["error"])
                if (
                    state["step0"] is None
                    or state["step1"] is None
                    or len(state["actor_lp"]) < 2
                    or len(state["rollout_lp"]) < 2
                ):
                    raise RuntimeError(
                        f"incomplete B0 capture: step0={state['step0']} step1={state['step1']} "
                        f"actor_captures={len(state['actor_lp'])} rollout_captures={len(state['rollout_lp'])}"
                    )
                verdict = assert_fixed_sequence_delta_synced(
                    state["actor_lp"][0],
                    state["actor_lp"][1],
                    state["rollout_lp"][0],
                    state["rollout_lp"][1],
                )
                payload = {
                    **verdict,
                    "absolute_parity": {"step0": state["step0"], "step1": state["step1"]},
                }
            except Exception as exc:
                payload = {
                    "ok": False,
                    "error": str(exc),
                    "step0": state.get("step0"),
                    "step1": state.get("step1"),
                    "actor_captures": len(state.get("actor_lp", [])),
                    "rollout_captures": len(state.get("rollout_lp", [])),
                }
                logger.exception("B0_ROLLOUT_SYNC guard/write failed")
            try:
                write_rollout_sync(_run_dir(), payload)
            except Exception:
                logger.exception("B0_ROLLOUT_SYNC failed to write rollout_sync.json")

        return result

    RayPPOTrainer.fit = patched_fit
    setattr(RayPPOTrainer, _ACTOR_UPDATE_METHOD, patched_update)
    RayPPOTrainer._b0_rollout_sync_hook_installed = True
    print(
        f"B0_ROLLOUT_SYNC: installed rollout-sync hook on RayPPOTrainer.{_ACTOR_UPDATE_METHOD}; "
        f"artifact -> {_run_dir()}/rollout_sync.json",
        flush=True,
    )
    return True
