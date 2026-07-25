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
