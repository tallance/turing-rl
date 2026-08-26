"""Unit tests for the judge GRPO reward."""

import asyncio
import json

import pytest

from shared.judge_prompts import TURING_RESPONSE_PROPERTIES
from training.grpo.judge_reward import (
    ARM_DIRECTIONAL,
    ARM_GRADED,
    compute_score,
    directional_task_reward,
    graded_task_reward,
    resolve_arm,
    task_reward,
)


def _verdict_json(rating: int) -> str:
    """A verdict whose primitives derive exactly the requested rating."""
    gap_for = {1: -3.0, 2: -1.5, 3: -0.5, 4: 0.0, 5: 0.5, 6: 1.5, 7: 3.0}
    gap = gap_for[rating]
    data = {}
    for name, schema in TURING_RESPONSE_PROPERTIES.items():
        data[name] = "text" if schema["type"] == "string" else 0.0
    # Put the whole gap on one side; each dimension is capped at 1.0.
    if gap >= 0:
        for i, field in enumerate(
            ("immediate_target_score_b", "human_goal_score_b", "communication_style_score_b")
        ):
            data[field] = max(0.0, min(1.0, gap - i))
    else:
        for i, field in enumerate(
            ("immediate_target_score_a", "human_goal_score_a", "communication_style_score_a")
        ):
            data[field] = max(0.0, min(1.0, -gap - i))
    base_a = sum(data[f] for f in ("immediate_target_score_a", "human_goal_score_a", "communication_style_score_a"))
    base_b = sum(data[f] for f in ("immediate_target_score_b", "human_goal_score_b", "communication_style_score_b"))
    data["base_score_a"] = base_a
    data["base_score_b"] = base_b
    data["penalty_a"] = 0.0
    data["penalty_b"] = 0.0
    data["response_a_score"] = base_a
    data["response_b_score"] = base_b
    data["score_gap"] = base_b - base_a
    data["rating"] = rating
    return json.dumps(data)


def _score(solution: str, ground_truth: str, arm: str = ARM_DIRECTIONAL) -> dict:
    return asyncio.run(
        compute_score(
            "prism_judge", solution, ground_truth, {"row_id": "r", "split": "train"}, arm=arm
        )
    )


def test_directional_rewards_a_correct_confident_call():
    assert directional_task_reward(7, human_is_b=True) == 1.0
    assert directional_task_reward(1, human_is_b=False) == 1.0


def test_directional_punishes_a_wrong_call():
    assert directional_task_reward(7, human_is_b=False) == 0.0
    assert directional_task_reward(1, human_is_b=True) == 0.0


def test_directional_pays_half_for_a_tie():
    assert directional_task_reward(4, human_is_b=True) == 0.5
    assert directional_task_reward(4, human_is_b=False) == 0.5


def test_graded_reward_values_match_the_spec():
    assert graded_task_reward(7, human_is_b=True) == pytest.approx(1.0)
    assert graded_task_reward(6, human_is_b=True) == pytest.approx(0.9722, abs=1e-4)
    assert graded_task_reward(5, human_is_b=True) == pytest.approx(0.8889, abs=1e-4)
    assert graded_task_reward(4, human_is_b=True) == pytest.approx(0.75)
    assert graded_task_reward(1, human_is_b=True) == pytest.approx(0.0)


def test_graded_reward_is_symmetric_under_swapping_the_human_side():
    for rating in range(1, 8):
        mirrored = 8 - rating
        assert graded_task_reward(rating, human_is_b=True) == pytest.approx(
            graded_task_reward(mirrored, human_is_b=False)
        )


def test_task_reward_dispatches_on_arm():
    assert task_reward(4, True, ARM_DIRECTIONAL) == 0.5
    assert task_reward(4, True, ARM_GRADED) == pytest.approx(0.75)


def test_unknown_arm_raises():
    with pytest.raises(ValueError):
        task_reward(4, True, "nonsense")


def test_resolve_arm_defaults_to_directional(monkeypatch):
    monkeypatch.delenv("JUDGE_REWARD_ARM", raising=False)
    assert resolve_arm() == ARM_DIRECTIONAL


def test_resolve_arm_rejects_an_unknown_env_value(monkeypatch):
    monkeypatch.setenv("JUDGE_REWARD_ARM", "nonsense")
    with pytest.raises(ValueError):
        resolve_arm()


def test_compute_score_totals_task_and_format():
    result = _score(_verdict_json(7), "B")
    assert result["judge_task_reward"] == 1.0
    assert result["judge_format_score"] == 1.0
    assert result["score"] == pytest.approx(1.0)
    assert result["judge_acc"] == 1.0
    assert result["judge_rating"] == 7
    assert result["judge_pred_b"] == 1.0
    assert result["judge_human_is_b"] == 1.0


def test_compute_score_handles_a_ground_truth_of_a():
    result = _score(_verdict_json(1), "A")
    assert result["judge_acc"] == 1.0
    assert result["judge_human_is_b"] == 0.0
    assert result["judge_pred_b"] == 0.0


def test_unrecoverable_verdict_scores_zero_but_still_reports():
    result = _score("gibberish", "B")
    assert result["judge_task_reward"] == 0.0
    assert result["judge_format_score"] == 0.0
    assert result["score"] == 0.0
    assert result["judge_recovered"] == 0.0
    assert result["judge_rung_none"] == 1.0


def test_malformed_but_parseable_still_earns_task_reward():
    from training.grpo.judge_verdict import TURING_FIELDS

    data = json.loads(_verdict_json(7))
    del data["reasoning"]
    result = _score(json.dumps(data), "B")

    # The point of the ladder: a schema-imperfect verdict still contains a prediction, so the
    # task reward is untouched. Only the format term is docked.
    assert result["judge_task_reward"] == 1.0
    assert result["judge_fmt_all_fields"] == 0.0
    assert result["judge_fmt_exact_schema"] == 0.0
    # Coverage is dense: dropping the second-to-last field costs the tail, not the whole term.
    expected_coverage = TURING_FIELDS.index("reasoning") / len(TURING_FIELDS)
    assert result["judge_fmt_ordered_coverage"] == pytest.approx(expected_coverage)
    assert result["score"] == pytest.approx(0.9 + 0.1 * result["judge_format_score"])
    # Still comfortably above the compact-JSON shortcut, which earns 0.9 + 0.1*0.1.
    assert result["score"] > 0.91


def test_tie_is_reported_and_excluded_from_strict_accuracy():
    result = _score(_verdict_json(4), "B")
    assert result["judge_tie"] == 1.0
    assert result["judge_acc"] == 0.5
    assert result["judge_correct_strict"] == 0.0


def test_rating_histogram_is_one_hot():
    result = _score(_verdict_json(5), "B")
    assert result["judge_rating_5"] == 1.0
    assert sum(result[f"judge_rating_{i}"] for i in range(1, 8)) == 1.0


def test_graded_arm_changes_the_total():
    result = _score(_verdict_json(5), "B", arm=ARM_GRADED)
    assert result["judge_task_reward"] == pytest.approx(0.8889, abs=1e-4)
