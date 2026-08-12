"""Unit tests for judge-metric discovery and group-health metrics."""

from types import SimpleNamespace

from training.grpo.verl_metric_patch import (
    _collect_reward_metric_names,
    append_judge_group_metrics,
)


def _batch(uids, totals, corrects):
    return SimpleNamespace(
        non_tensor_batch={
            "uid": uids,
            "reward_extra_info": [
                {"judge_total": t, "judge_correct_strict": c}
                for t, c in zip(totals, corrects)
            ],
        }
    )


def test_judge_keys_are_discovered():
    names = _collect_reward_metric_names(
        {"reward_extra_info": {"judge_acc": [1.0], "judge_fmt_arith": [0.0]}}
    )
    assert "judge_acc" in names
    assert "judge_fmt_arith" in names


def test_existing_format_and_length_discovery_still_works():
    names = _collect_reward_metric_names(
        {"reward_extra_info": {"format_score": [1.0], "length_ratio": [1.0]}}
    )
    assert "format_score" in names
    assert "length_ratio" in names


def test_group_metrics_flag_a_degenerate_all_correct_group():
    metrics = {}
    append_judge_group_metrics(metrics, _batch(["g1"] * 4, [1.0] * 4, [1.0] * 4))
    assert metrics["judge_group/all_equal_rate"] == 1.0
    assert metrics["judge_group/all_correct_rate"] == 1.0
    assert metrics["judge_group/all_wrong_rate"] == 0.0


def test_group_metrics_flag_a_degenerate_all_wrong_group():
    metrics = {}
    append_judge_group_metrics(metrics, _batch(["g1"] * 4, [0.1] * 4, [0.0] * 4))
    assert metrics["judge_group/all_equal_rate"] == 1.0
    assert metrics["judge_group/all_wrong_rate"] == 1.0


def test_a_mixed_group_is_not_degenerate():
    metrics = {}
    append_judge_group_metrics(metrics, _batch(["g1"] * 4, [1.0, 0.0, 1.0, 0.0], [1.0, 0.0, 1.0, 0.0]))
    assert metrics["judge_group/all_equal_rate"] == 0.0
    assert metrics["judge_group/all_correct_rate"] == 0.0
    assert metrics["judge_group/all_wrong_rate"] == 0.0


def test_rates_average_over_groups():
    metrics = {}
    append_judge_group_metrics(
        metrics,
        _batch(["g1", "g1", "g2", "g2"], [1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 1.0, 0.0]),
    )
    assert metrics["judge_group/all_equal_rate"] == 0.5
    assert metrics["judge_group/n_groups"] == 2


def test_missing_judge_keys_are_a_no_op():
    metrics = {}
    append_judge_group_metrics(metrics, SimpleNamespace(non_tensor_batch={}))
    assert metrics == {}
