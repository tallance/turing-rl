"""veRL reward for judge-only RLVR.

The label is known by construction, so this reward is entirely local: no judge server,
no HTTP, no external model. That is the structural difference from the generator's
reward path, where judge calls dominated wall-clock.

Reward = JUDGE_TASK_WEIGHT * task + JUDGE_FORMAT_WEIGHT * format. Format is a minor,
*additive* term rather than a gate: at eval time every model runs under forced schema
decoding, so format is free there and the published comparison is on accuracy alone.
Format matters only as a training-time scaffold for unconstrained rollouts.
"""

from __future__ import annotations

import os
import sys
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from training.grpo.judge_verdict import JudgeVerdict, parse_judge_verdict

ARM_DIRECTIONAL = "directional"
ARM_GRADED = "graded"
ARMS = (ARM_DIRECTIONAL, ARM_GRADED)

DEFAULT_TASK_WEIGHT = 0.9
DEFAULT_FORMAT_WEIGHT = 0.1

# "unclosed_thinking" is distinct from "none": the model never closed its <think> block, so the
# response cap cut it off mid-reasoning and no answer was produced. Worth its own metric because
# under thinking-ON it is the failure mode that decides whether a size is trainable at all --
# the 2B logged a 93.75% clip ratio at step 0.
RECOVERY_RUNGS = (
    "dimensions",
    "score_gap",
    "rating_field",
    "rating_text",
    "unclosed_thinking",
    "none",
)

_TIE_RATING = 4


def resolve_arm() -> str:
    """Read the reward arm from the environment, defaulting to directional."""
    arm = os.environ.get("JUDGE_REWARD_ARM", ARM_DIRECTIONAL)
    if arm not in ARMS:
        raise ValueError(f"JUDGE_REWARD_ARM must be one of {ARMS}, got {arm!r}")
    return arm


def _weights() -> tuple[float, float]:
    return (
        float(os.environ.get("JUDGE_TASK_WEIGHT", DEFAULT_TASK_WEIGHT)),
        float(os.environ.get("JUDGE_FORMAT_WEIGHT", DEFAULT_FORMAT_WEIGHT)),
    )


def directional_task_reward(rating: int, human_is_b: bool) -> float:
    """1 for the right side, 0 for the wrong side, 0.5 for a tie."""
    if rating == _TIE_RATING:
        return 0.5
    return 1.0 if (rating > _TIE_RATING) == human_is_b else 0.0


def graded_task_reward(rating: int, human_is_b: bool) -> float:
    """1 - (p - y)^2 with p = (rating - 1) / 6, a graded version of the directional arm."""
    p = (rating - 1) / 6.0
    y = 1.0 if human_is_b else 0.0
    return 1.0 - (p - y) ** 2


def task_reward(rating: int, human_is_b: bool, arm: str) -> float:
    if arm == ARM_DIRECTIONAL:
        return directional_task_reward(rating, human_is_b)
    if arm == ARM_GRADED:
        return graded_task_reward(rating, human_is_b)
    raise ValueError(f"unknown reward arm {arm!r}; expected one of {ARMS}")


def _metrics(verdict: JudgeVerdict, human_is_b: bool, arm: str) -> dict[str, float]:
    """Per-sample metrics. Every key is judge_-prefixed so verl_metric_patch finds it."""
    rating = verdict.rating
    task = task_reward(rating, human_is_b, arm) if verdict.recovered else 0.0
    task_weight, format_weight = _weights()
    total = task_weight * task + format_weight * verdict.format_score

    is_tie = bool(rating == _TIE_RATING)
    if not verdict.recovered:
        acc = 0.0
        correct_strict = 0.0
        pred_b = 0.0
        p = 0.5
    else:
        acc = directional_task_reward(rating, human_is_b)
        correct_strict = 1.0 if acc == 1.0 else 0.0
        pred_b = 1.0 if rating > _TIE_RATING else 0.0
        p = (rating - 1) / 6.0

    y = 1.0 if human_is_b else 0.0
    metrics: dict[str, float] = {
        "score": total,
        "total_score": total,
        "judge_total": total,
        "judge_task_reward": task,
        "judge_format_score": verdict.format_score,
        "judge_fmt_json_valid": float(verdict.fmt_json_valid),
        "judge_fmt_all_fields": float(verdict.fmt_all_fields),
        "judge_fmt_arith": float(verdict.fmt_arith),
        "judge_fmt_rating_range": float(verdict.fmt_rating_range),
        # The three that actually drive format_score. Coverage is the one to watch: it is dense,
        # so it moves before all_fields/exact_schema ever flip, and a near-zero exact_schema
        # beside a healthy coverage means the </think> split broke, not that the model is weak.
        "judge_fmt_ordered_coverage": float(verdict.fmt_ordered_coverage),
        "judge_fmt_exact_schema": float(verdict.fmt_exact_schema),
        "judge_fmt_strict_json": float(verdict.fmt_strict_json),
        "judge_acc": acc,
        "judge_correct_strict": correct_strict,
        "judge_tie": 1.0 if (verdict.recovered and is_tie) else 0.0,
        "judge_brier": (p - y) ** 2,
        "judge_conf": 2.0 * abs(p - 0.5),
        "judge_rating": float(rating) if verdict.recovered else 0.0,
        "judge_pred_b": pred_b,
        "judge_human_is_b": y,
        "judge_recovered": float(verdict.recovered),
    }
    for rung in RECOVERY_RUNGS:
        metrics[f"judge_rung_{rung}"] = 1.0 if verdict.recovery_rung == rung else 0.0
    for value in range(1, 8):
        metrics[f"judge_rating_{value}"] = 1.0 if rating == value else 0.0
    return metrics


async def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict | None = None,
    **kwargs: Any,
) -> dict:
    """Score one judge rollout. ``ground_truth`` is the slot holding the human: "A"/"B"."""
    _ = data_source
    extra_info = extra_info or {}
    label = str(ground_truth).strip().upper()
    if label not in ("A", "B"):
        raise ValueError(f"judge ground_truth must be 'A' or 'B', got {ground_truth!r}")
    human_is_b = label == "B"

    arm = kwargs.get("arm") or resolve_arm()
    verdict = parse_judge_verdict(solution_str)
    return _metrics(verdict, human_is_b, arm)
