# scripts/rollout_sync_guard.py
"""B0 guard: verify the vLLM rollout policy tracks the actor via per-token LOGPROB parity.

Weight hashing is unusable here (vLLM weights are TP-sharded/fused, no actor-comparable API, ~36GB).
veRL logs rollout logprobs (vLLM) and the actor's recomputed old_log_prob for the same tokens;
if they diverge, vLLM is serving different weights than the actor holds (the pre-#7014 stale base).
"""
from __future__ import annotations

import numpy as np


def logprob_parity(rollout_lp, actor_lp, atol: float = 0.1) -> dict:
    """Max abs diff between rollout and actor per-token logprobs for the same tokens."""
    r = np.asarray(rollout_lp, dtype=np.float64)
    a = np.asarray(actor_lp, dtype=np.float64)
    max_abs_diff = float(np.max(np.abs(r - a))) if r.size else float("inf")
    return {"max_abs_diff": max_abs_diff, "close": max_abs_diff <= atol}


def assert_rollout_synced(step0: dict, step1: dict, tf_lp0, tf_lp1, move_atol: float = 1e-3) -> dict:
    """ok iff rollout≈actor at BOTH steps (synced) AND the actor moved on a FIXED teacher-forced seq.

    step0/step1 : within-step logprob_parity dicts (rollout vs actor on the SAME sampled tokens).
    tf_lp0/tf_lp1: actor teacher-forced logprobs on ONE fixed prompt+continuation at step0 & step1 —
                   identical tokens, so the cross-step delta is meaningful. (Do NOT diff sampled
                   rollout logprobs across steps: different tokens/shapes -> meaningless.)
    """
    a0 = np.asarray(tf_lp0, dtype=np.float64)
    a1 = np.asarray(tf_lp1, dtype=np.float64)
    if a0.shape != a1.shape:
        raise ValueError(f"teacher-forced logprobs must be the same fixed sequence: {a0.shape} vs {a1.shape}")
    synced = bool(step0["close"] and step1["close"])
    policy_moved = bool(a0.size and float(np.max(np.abs(a0 - a1))) > move_atol)
    return {"synced": synced, "policy_moved": policy_moved, "ok": synced and policy_moved}


def assert_fixed_sequence_delta_synced(
    actor_lp0,
    actor_lp1,
    rollout_lp0,
    rollout_lp1,
    *,
    delta_atol: float = 0.1,
    delta_quantile: float = 0.99,
    move_atol: float = 1e-3,
) -> dict:
    """Check live weight sync from before/after changes on one fixed token sequence.

    Absolute HF/vLLM logprobs can carry a stable engine-specific offset. Comparing each engine's
    change on identical tokens cancels that offset. The high-quantile error is the gate; maxima and
    means remain in the artifact for diagnosis without letting one numerical outlier dominate.
    """
    arrays = [
        np.asarray(values, dtype=np.float64)
        for values in (actor_lp0, actor_lp1, rollout_lp0, rollout_lp1)
    ]
    shapes = {values.shape for values in arrays}
    if len(shapes) != 1:
        raise ValueError(
            "actor and rollout logprobs must score the same fixed sequence: "
            f"actor0={arrays[0].shape} actor1={arrays[1].shape} "
            f"rollout0={arrays[2].shape} rollout1={arrays[3].shape}"
        )
    if not arrays[0].size:
        raise ValueError("fixed-sequence logprobs must not be empty")
    if not 0.0 <= delta_quantile <= 1.0:
        raise ValueError(f"delta_quantile must be in [0, 1], got {delta_quantile}")

    actor_delta = arrays[1] - arrays[0]
    rollout_delta = arrays[3] - arrays[2]
    delta_error = np.abs(actor_delta - rollout_delta)
    actor_move_max = float(np.max(np.abs(actor_delta)))
    rollout_move_max = float(np.max(np.abs(rollout_delta)))
    error_p = float(np.quantile(delta_error, delta_quantile))
    synced = bool(error_p <= delta_atol)
    policy_moved = bool(actor_move_max > move_atol)
    rollout_moved = bool(rollout_move_max > move_atol)

    if actor_delta.size > 1 and np.std(actor_delta) > 0 and np.std(rollout_delta) > 0:
        delta_correlation = float(np.corrcoef(actor_delta, rollout_delta)[0, 1])
    else:
        delta_correlation = None

    return {
        "synced": synced,
        "policy_moved": policy_moved,
        "rollout_moved": rollout_moved,
        "ok": bool(synced and policy_moved and rollout_moved),
        "num_tokens": int(actor_delta.size),
        "actor_move_max_abs": actor_move_max,
        "rollout_move_max_abs": rollout_move_max,
        "delta_error_max_abs": float(np.max(delta_error)),
        "delta_error_mean_abs": float(np.mean(delta_error)),
        "delta_error_p99_abs": error_p,
        "delta_quantile": float(delta_quantile),
        "delta_atol": float(delta_atol),
        "delta_correlation": delta_correlation,
    }
