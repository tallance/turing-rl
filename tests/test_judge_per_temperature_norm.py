"""The per-temperature counters, normalised into something readable on a dashboard.

judge_acc_tXX is the SUM half of a sum/count pair -- it has to be, because verl reduces every
judge_* key with a plain np.mean over the batch and each step draws a different mix of
temperatures (n_t07 was observed swinging 0.36-0.67 across steps of J2'). Plotted on its own it
is accuracy x composition, which reads about half the true value and twice as noisy.

The ratio cannot be computed per row (the mean of per-row ratios is not the ratio of means), so
it is computed once here, AFTER the batch means exist.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.grpo.verl_metric_patch import append_per_temperature_accuracy  # noqa: E402


def test_the_ratio_is_the_subset_accuracy():
    """Half the batch at each temperature; T=0.7 right 80% of the time, T=1.0 right 30%."""
    metrics = {
        "reward/judge_acc_t07/mean": 0.5 * 0.8,
        "reward/judge_n_t07/mean": 0.5,
        "reward/judge_acc_t10/mean": 0.5 * 0.3,
        "reward/judge_n_t10/mean": 0.5,
    }

    append_per_temperature_accuracy(metrics)

    assert metrics["reward/judge_acc_t07_norm/mean"] == pytest.approx(0.8)
    assert metrics["reward/judge_acc_t10_norm/mean"] == pytest.approx(0.3)


def test_an_uneven_batch_mix_is_handled():
    """The composition varies step to step, which is exactly what the raw numerator conflates
    with accuracy. 25% of rows at T=0.7, all correct -> accuracy 1.0, not 0.25."""
    metrics = {
        "reward/judge_acc_t07/mean": 0.25,
        "reward/judge_n_t07/mean": 0.25,
        "reward/judge_acc_t10/mean": 0.0,
        "reward/judge_n_t10/mean": 0.75,
    }

    append_per_temperature_accuracy(metrics)

    assert metrics["reward/judge_acc_t07_norm/mean"] == pytest.approx(1.0)
    assert metrics["reward/judge_acc_t10_norm/mean"] == pytest.approx(0.0)


def test_an_absent_bucket_emits_nothing_rather_than_zero():
    """The val split carries no gen_temperature, so both counts are 0. Emitting 0.0 there would
    plot a flat line that looks like a measured accuracy of zero; emitting nothing leaves the
    panel honestly empty."""
    metrics = {
        "reward/judge_acc_t07/mean": 0.0,
        "reward/judge_n_t07/mean": 0.0,
        "reward/judge_acc_t10/mean": 0.0,
        "reward/judge_n_t10/mean": 0.0,
    }

    append_per_temperature_accuracy(metrics)

    assert "reward/judge_acc_t07_norm/mean" not in metrics
    assert "reward/judge_acc_t10_norm/mean" not in metrics


def test_missing_counters_are_a_no_op():
    """Runs before this metric existed, and the generator's reward path, have no such keys."""
    metrics = {"reward/judge_acc/mean": 0.4}

    append_per_temperature_accuracy(metrics)

    assert metrics == {"reward/judge_acc/mean": 0.4}


def test_it_does_not_disturb_the_existing_metrics():
    metrics = {
        "reward/judge_acc/mean": 0.42,
        "reward/judge_acc_t07/mean": 0.2,
        "reward/judge_n_t07/mean": 0.5,
    }

    append_per_temperature_accuracy(metrics)

    assert metrics["reward/judge_acc/mean"] == 0.42
    assert metrics["reward/judge_acc_t07/mean"] == 0.2
    assert metrics["reward/judge_n_t07/mean"] == 0.5
    assert metrics["reward/judge_acc_t07_norm/mean"] == pytest.approx(0.4)
