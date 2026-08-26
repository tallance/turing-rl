"""Standalone single-token A/B judge scorer for the eval sweep.

The judge is shown the same pairwise prompt header as the full-schema protocol but is
asked for one letter, and the verdict is read out of the logprobs at that single
position rather than out of a 37-field JSON body.

Why this lives here and not in ``training/grpo/reward.py``: that module is the GENERATOR
REWARD path. This protocol rewards nothing today -- it is an eval-side measurement -- so
folding it into the reward module would couple a measurement to the training loop for no
benefit. Nothing here imports ``training.grpo``; the pieces both arms must agree on
(prompt header, order randomization, source-copy watchlist) live in ``shared/``.

Two things this module owes the rest of the pipeline:

* **The dump contract.** ``scripts/analyze_judge_sweep.py`` reads the per-call JSONL and
  needs ``rating_gt_first``/``rating_gen_first`` (exactly one set), ``human_side``,
  ``generated_is_b`` and ``randomized_order``. Emitting those is what makes a
  single-token cell comparable against a full-schema cell at all. The base key list is
  duplicated from ``reward._REWARD_DUMP_KEYS`` (see ``_BASE_DUMP_KEYS``).
* **Thinking off.** ``max_tokens=1`` with thinking ON decodes the think opener, not a
  verdict, and nothing about that failure is loud. ``enable_thinking`` is therefore
  pinned False here regardless of ``PERSONA_JUDGE_ENABLE_THINKING``.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any

from shared.api_client import (
    build_chat_payload,
    get_judge_call_meta,
    post_chat_choice_async,
)
from shared.judge_prompts import TURING_SINGLE_TOKEN_PROMPT
from shared.judge_utils import (
    _stable_turing_generated_is_b,
    build_source_copy_warning,
    format_source_copy_watchlist,
    sanitize_prompt_text,
)
from shared.single_token_verdict import HardFail, extract_verdict

# vLLM's ceiling for an OpenAI-compatible request. The A/B mass is spread across
# tokenizer variants ("A", " A", "▁A", "a", ...), so a small k can truncate real verdict
# mass and push an honest answer under the MIN_AB_MASS floor.
TOP_LOGPROBS = 20

# 1 == "definitely A", 7 == "definitely B" on the existing Likert scale the sweep
# analyzer already speaks. A single token cannot produce a tie, so 4 never appears.
RATING_FOR_A = 1
RATING_FOR_B = 7


def judge_model() -> str:
    """The judge model id for this cell.

    Deliberately has no default. ``reward.py`` falls back to the 397B anchor, which is
    the right call for training but the wrong one here: a sweep cell that silently
    scored against a different model than its directory name claims is exactly the
    plausible-but-wrong artifact this protocol exists to avoid.
    """
    model = os.environ.get("JUDGE_MODEL", "").strip()
    if not model:
        raise ValueError(
            "JUDGE_MODEL must be set for single-token scoring; refusing to guess the judge"
        )
    return model


def _sampling_from_env() -> dict | None:
    raw = os.environ.get("PERSONA_JUDGE_SAMPLING")
    return json.loads(raw) if raw else None


# Per-event-loop request semaphore, read from the SAME env vars the reward path uses so
# the two arms are bounded identically and a throughput difference cannot be mistaken
# for a protocol difference.
_SEMAPHORES: dict[int, "asyncio.Semaphore"] = {}
_SEMAPHORE_LIMITS: dict[int, int] = {}


def _judge_request_limit() -> int:
    for env_name in ("TURING_JUDGE_MAX_CONCURRENCY", "PERSONA_OPENAI_JUDGE_MAX_CONCURRENCY"):
        raw_value = os.environ.get(env_name)
        if raw_value is None:
            continue
        try:
            return max(1, int(raw_value))
        except ValueError as exc:
            raise ValueError(f"{env_name} must be an integer, got {raw_value!r}") from exc
    return 512


def _get_request_semaphore() -> "asyncio.Semaphore":
    loop_id = id(asyncio.get_running_loop())
    limit = _judge_request_limit()
    semaphore = _SEMAPHORES.get(loop_id)
    if semaphore is None or _SEMAPHORE_LIMITS.get(loop_id) != limit:
        semaphore = asyncio.Semaphore(limit)
        _SEMAPHORES[loop_id] = semaphore
        _SEMAPHORE_LIMITS[loop_id] = limit
    return semaphore


# DEBT: duplicated from ``training.grpo.reward._REWARD_DUMP_KEYS``. It cannot be imported
# without importing the reward/scoring path, and reward.py is deliberately untouched by
# this protocol, so the list is copied and pinned instead:
# tests/test_single_token_judge.py asserts the two stay equal.
_BASE_DUMP_KEYS = (
    "generated_is_b", "human_side", "rating_gt_first", "rating_gen_first",
    "randomized_order", "response", "ground_truth", "context", "user_history",
    "judge_response", "judge_prompt", "judge_raw_content", "judge_reasoning",
    "judge_latency_ms", "judge_finish_reason", "judge_model", "judge_usage",
    "final_reward", "turing_judge_score_raw", "turing_judge_score_clipped",
    "source_copy_penalty", "assistant_like_penalty", "wrong_target_or_role_penalty",
    "unsupported_adversarial_reframing_penalty", "call_id", "user_id", "post_id",
    "target_idx", "persona", "ts", "worker_pid", "split",
)

# Single-token extras. ``hard_fail``/``hard_fail_reason`` and ``sampled_token`` are what
# make an abstention auditable: without them a hard-failed row is indistinguishable from
# a row the judge simply never reached.
_SINGLE_TOKEN_DUMP_KEYS = (
    "judge_prompt_style", "pair_id", "human_is_b", "hard_fail", "hard_fail_reason",
    "letter", "p_a", "ab_mass", "off_ab_mass", "sampled_token", "enable_thinking",
)

_DUMP_KEYS = _BASE_DUMP_KEYS + _SINGLE_TOKEN_DUMP_KEYS


def build_dump_row(**fields: Any) -> dict:
    """One per-call dump row, in the shape ``scripts/analyze_judge_sweep.py`` reads."""
    return {k: fields.get(k) for k in _DUMP_KEYS}


def dump_call(row: dict) -> None:
    """Append one row to the per-worker reward JSONL. Best-effort, never fatal.

    Same gate, directory and filename pattern as the reward-layer dump, so the analyzer's
    ``(mode_dir / "reward").glob("*.jsonl")`` picks these up with no changes.
    """
    try:
        try:
            rate = float(os.environ.get("PERSONA_JUDGE_DUMP_RATE", "0") or "0")
        except (TypeError, ValueError):
            return
        if rate <= 0:
            return
        directory = os.environ.get("PERSONA_REWARD_DUMP_DIR")
        if not directory:
            return
        os.makedirs(directory, exist_ok=True)
        job = os.environ.get("SLURM_JOB_ID", "local")
        path = os.path.join(directory, f"reward-{job}-{os.getpid()}.jsonl")
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, default=str) + "\n")
    except Exception as exc:  # noqa: BLE001 - never let dumping break scoring
        print(f"[single-token-dump] skipped (non-fatal): {type(exc).__name__}: {exc}", flush=True)


def _read_verdict_position(choice: dict) -> dict:
    """Return the single decoded position, or raise if the server returned no logprobs.

    Distinct from a HardFail on purpose. A HardFail is a property of one input and is
    recorded as an abstention; a response with no logprobs at all means the request or
    the server is misconfigured, which would otherwise show up as a 100% hard-fail cell
    that still exits 0.
    """
    logprobs = (choice or {}).get("logprobs")
    content = (logprobs or {}).get("content")
    if not content:
        raise RuntimeError(
            "judge response carries no logprobs content; single-token scoring requires "
            f"logprobs=True and a server that honours it (got choice keys: {sorted((choice or {}).keys())})"
        )
    return content[0]


async def score_single_token_with_info(
    session,
    api_key: str,
    response: str,
    ground_truth: str,
    user_history: str,
    context: str,
    calibration_domain: str = "",
    user_id: Any = "",
    post_id: Any = "",
    target_idx: Any = "",
    randomization_seed_material: str = "",
    pair_id: Any = None,
) -> dict[str, Any]:
    """Score one pair with the single-token protocol and dump the per-call row.

    Signature-compatible with ``training.grpo.reward.score_turing_with_info`` so the
    sweep cell can dispatch between the two arms without a second call site.
    ``calibration_domain`` is accepted for that compatibility and unused: it only feeds
    the full-schema reward shaping, which this protocol does not compute.

    A ``HardFail`` (no usable A/B verdict at the position) is RECORDED, not raised: the
    hard-fail rate is a headline number of this protocol, and raising would both fail the
    whole shard and delete the evidence. Transport errors still propagate.
    """
    del calibration_domain

    response = sanitize_prompt_text(response)
    ground_truth = sanitize_prompt_text(ground_truth)
    user_history = sanitize_prompt_text(user_history)
    context = sanitize_prompt_text(context)

    response_warning = build_source_copy_warning(
        response, user_history=user_history, thread_context=context
    )
    ground_truth_warning = build_source_copy_warning(
        ground_truth, user_history=user_history, thread_context=context
    )

    # Same function and same seed material as the full arm (reward.py sanitizes before
    # hashing too), so a given pair lands in the same A/B slot in both arms and the
    # comparison is not confounded by a different randomization.
    generated_is_b = _stable_turing_generated_is_b(
        response,
        user_id=user_id,
        post_id=post_id,
        target_idx=target_idx,
        seed_material=randomization_seed_material,
    )
    if generated_is_b:
        response_a, response_b = ground_truth, response
        warning_a, warning_b = ground_truth_warning, response_warning
        randomized_order = "gt_first"
    else:
        response_a, response_b = response, ground_truth
        warning_a, warning_b = response_warning, ground_truth_warning
        randomized_order = "gen_first"

    prompt = TURING_SINGLE_TOKEN_PROMPT.format(
        user_history=user_history,
        context=context,
        response_a=response_a,
        response_b=response_b,
        source_copy_watchlist=format_source_copy_watchlist(
            [warning_a, warning_b],
            item_label="Response",
            labels=["Response A", "Response B"],
        ),
    )

    payload = build_chat_payload(
        model=judge_model(),
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=1,
        response_format=None,
        reasoning=False,
        sampling=_sampling_from_env(),
        # Pinned, not read from PERSONA_JUDGE_ENABLE_THINKING: with a one-token budget a
        # thinking-on judge spends that token on the think opener and the verdict never
        # exists. Silent, and it looks like a hard-fail cliff rather than a config error.
        chat_template_kwargs={"enable_thinking": False},
    )
    payload["logprobs"] = True
    payload["top_logprobs"] = TOP_LOGPROBS

    # Not wrapped in a parse-retry loop: a hard fail is a property of the input, and
    # retrying would hide it from the hard-fail rate and bias accuracy toward the
    # shorter inputs, where hard fails concentrate.
    choice = await post_chat_choice_async(
        session, payload, semaphore=_get_request_semaphore(), api_key=api_key
    )
    judge_meta = get_judge_call_meta() or {}

    position = _read_verdict_position(choice)
    sampled_token = position.get("token")
    verdict = None
    hard_fail_reason = None
    try:
        verdict = extract_verdict(
            position.get("top_logprobs") or [],
            # The token the model actually emitted. If it is not an A/B variant then the
            # position is not a verdict position, whatever mass the top-k carries -- a
            # structural check, independent of any threshold.
            sampled_token=sampled_token,
        )
    except HardFail as exc:
        hard_fail_reason = str(exc)

    letter = verdict.letter if verdict else None
    rating = None
    if letter is not None:
        rating = RATING_FOR_A if letter == "A" else RATING_FOR_B
    rating_gt_first = rating if generated_is_b else None
    rating_gen_first = None if generated_is_b else rating
    # The full arm's Likert orientation: high == "the generated answer looks human".
    likert_score = None
    if rating is not None:
        likert_score = float(rating) if generated_is_b else float(8 - rating)

    row = build_dump_row(
        judge_prompt_style="single_token",
        pair_id=pair_id,
        generated_is_b=generated_is_b,
        # human (ground_truth) sits on side A when the generated answer is B.
        human_side="A" if generated_is_b else "B",
        human_is_b=not generated_is_b,
        rating_gt_first=rating_gt_first,
        rating_gen_first=rating_gen_first,
        randomized_order=randomized_order,
        response=response,
        ground_truth=ground_truth,
        context=context,
        user_history=user_history,
        judge_prompt=prompt,
        judge_raw_content=sampled_token,
        judge_latency_ms=judge_meta.get("latency_ms"),
        judge_finish_reason=judge_meta.get("finish_reason"),
        judge_model=os.environ.get("JUDGE_MODEL"),
        judge_usage=judge_meta.get("usage") or {},
        turing_judge_score_raw=likert_score,
        hard_fail=verdict is None,
        hard_fail_reason=hard_fail_reason,
        letter=letter,
        p_a=verdict.p_a if verdict else None,
        ab_mass=verdict.ab_mass if verdict else None,
        off_ab_mass=verdict.off_ab_mass if verdict else None,
        sampled_token=sampled_token,
        enable_thinking=False,
        user_id=user_id,
        post_id=post_id,
        target_idx=target_idx,
        ts=time.time(),
        worker_pid=os.getpid(),
        split="eval",
    )
    dump_call(row)
    return row
