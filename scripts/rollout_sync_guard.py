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
    actor_move = np.abs(actor_delta)
    rollout_move = np.abs(rollout_delta)
    actor_move_max = float(np.max(actor_move))
    rollout_move_max = float(np.max(rollout_move))
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
        "actor_move_mean_abs": float(np.mean(actor_move)),
        "actor_move_p99_abs": float(np.quantile(actor_move, 0.99)),
        "rollout_move_max_abs": rollout_move_max,
        "rollout_move_mean_abs": float(np.mean(rollout_move)),
        "rollout_move_p99_abs": float(np.quantile(rollout_move, 0.99)),
        "delta_error_max_abs": float(np.max(delta_error)),
        "delta_error_mean_abs": float(np.mean(delta_error)),
        "delta_error_p99_abs": error_p,
        "delta_quantile": float(delta_quantile),
        "delta_atol": float(delta_atol),
        "delta_correlation": delta_correlation,
    }


def logprob_error_stats(left, right) -> dict:
    """Distributional parity for two engines scoring the same fixed tokens."""
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape:
        raise ValueError(f"logprob shapes differ: {left.shape} vs {right.shape}")
    if not left.size:
        raise ValueError("fixed-sequence logprobs must not be empty")
    error = np.abs(left - right)
    correlation = None
    if left.size > 1 and np.std(left) > 0 and np.std(right) > 0:
        correlation = float(np.corrcoef(left, right)[0, 1])
    return {
        "max_abs": float(np.max(error)),
        "mean_abs": float(np.mean(error)),
        "p99_abs": float(np.quantile(error, 0.99)),
        "correlation": correlation,
    }


def assert_calibrated_fixed_sequence_synced(
    actor_lp0,
    actor_lp1,
    rollout_lp0,
    rollout_lp1,
    *,
    rollout_versions0,
    rollout_versions1,
    min_raw_correlation: float = 0.995,
    max_mean_error_increase: float = 0.01,
    max_p99_error_increase: float = 0.05,
    movement_ratio_bounds: tuple[float, float] = (0.5, 2.0),
    min_delta_correlation: float = 0.5,
) -> dict:
    """Gate sync relative to the known-synced step-0 engine baseline.

    Qwen3.5 uses different GDN kernels in HF and vLLM, so exact token-level deltas are not a valid
    unconditional gate. A valid live-sync check instead requires every rollout replica's version
    to advance, cross-engine raw-logprob parity to remain as good as at step 0, and both engines to
    move by comparable magnitudes. Delta correlation is additionally required when the actor move
    is large enough to rise above the measured cross-engine baseline noise.
    """
    strict_delta = assert_fixed_sequence_delta_synced(
        actor_lp0,
        actor_lp1,
        rollout_lp0,
        rollout_lp1,
    )
    baseline = logprob_error_stats(actor_lp0, rollout_lp0)
    final = logprob_error_stats(actor_lp1, rollout_lp1)

    versions0 = list(rollout_versions0)
    versions1 = list(rollout_versions1)
    weight_versions_advanced = bool(
        versions0
        and len(versions0) == len(versions1)
        and all(old is not None and new is not None and int(new) > int(old) for old, new in zip(versions0, versions1))
    )
    raw_parity_preserved = bool(
        baseline["correlation"] is not None
        and final["correlation"] is not None
        and baseline["correlation"] >= min_raw_correlation
        and final["correlation"] >= min_raw_correlation
        and final["mean_abs"] <= baseline["mean_abs"] + max_mean_error_increase
        and final["p99_abs"] <= baseline["p99_abs"] + max_p99_error_increase
    )

    actor_move_mean = strict_delta["actor_move_mean_abs"]
    rollout_move_mean = strict_delta["rollout_move_mean_abs"]
    movement_ratio = float(rollout_move_mean / actor_move_mean) if actor_move_mean > 0 else float("inf")
    movement_consistent = bool(
        strict_delta["policy_moved"]
        and strict_delta["rollout_moved"]
        and movement_ratio_bounds[0] <= movement_ratio <= movement_ratio_bounds[1]
    )

    delta_signal_required = bool(actor_move_mean >= max(0.02, 1.5 * baseline["mean_abs"]))
    delta_correlation = strict_delta["delta_correlation"]
    delta_signal_consistent = bool(
        not delta_signal_required
        or (delta_correlation is not None and delta_correlation >= min_delta_correlation)
    )
    synced = bool(
        weight_versions_advanced
        and raw_parity_preserved
        and movement_consistent
        and delta_signal_consistent
    )
    return {
        "ok": synced,
        "synced": synced,
        "policy_moved": strict_delta["policy_moved"],
        "rollout_moved": strict_delta["rollout_moved"],
        "weight_versions_advanced": weight_versions_advanced,
        "raw_parity_preserved": raw_parity_preserved,
        "movement_consistent": movement_consistent,
        "movement_ratio": movement_ratio,
        "delta_signal_required": delta_signal_required,
        "delta_signal_consistent": delta_signal_consistent,
        "baseline_raw_parity": baseline,
        "final_raw_parity": final,
        "strict_delta": strict_delta,
    }
