"""Judge accuracy split by the temperature the fake turn was sampled at.

The mixed-temperature corpus is the whole reason this judge exists: a judge trained at one
generator temperature flipped polarity across the generator's sampling range, so the training
pairs are half T=0.7 and half T=1.0. Without these counters `judge_acc` is a single aggregate
over both halves and the design cannot be evaluated -- and judge GRPO writes only aggregate
metrics (no per-call dumps like the generator), so it is not recoverable after the run.

Shape matters. verl_metric_patch reduces every judge_-prefixed key with a plain np.mean over
the batch (verl_metric_patch.py: `metric_mean = float(np.mean(values))`), so:

  * a key present on only SOME rows would be averaged over the wrong denominator;
  * the readable quantity is a sum/count PAIR -- mean(judge_acc_tXX) / mean(judge_n_tXX).
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.grpo.judge_reward import compute_score  # noqa: E402

THINK = "reasoning</think>"
TEMP_KEYS = ("judge_n_t07", "judge_acc_t07", "judge_n_t10", "judge_acc_t10")


def _score(rating, human_is_b, gen_temperature, monkeypatch):
    monkeypatch.setenv("PERSONA_ENABLE_THINKING", "1")
    monkeypatch.setenv("JUDGE_PROMPT_STYLE", "rating_only")
    monkeypatch.setenv("JUDGE_REWARD_ARM", "directional")
    extra = {} if gen_temperature is None else {"gen_temperature": gen_temperature}
    return asyncio.run(
        compute_score("prism_judge", f'{THINK}{{"rating": {rating}}}',
                      "B" if human_is_b else "A", extra)
    )


def test_a_warm_row_counts_only_in_the_warm_bucket(monkeypatch):
    got = _score(7, True, 0.7, monkeypatch)  # rating 7 with human on B == correct

    assert got["judge_n_t07"] == 1.0
    assert got["judge_acc_t07"] == 1.0
    assert got["judge_n_t10"] == 0.0
    assert got["judge_acc_t10"] == 0.0


def test_a_hot_row_counts_only_in_the_hot_bucket(monkeypatch):
    got = _score(7, True, 1.0, monkeypatch)

    assert got["judge_n_t10"] == 1.0
    assert got["judge_acc_t10"] == 1.0
    assert got["judge_n_t07"] == 0.0
    assert got["judge_acc_t07"] == 0.0


def test_a_wrong_answer_counts_in_the_denominator_but_not_the_numerator(monkeypatch):
    """The count is what the row was sampled at; the accuracy is whether it was right."""
    got = _score(1, True, 0.7, monkeypatch)  # says A, human is B -> wrong

    assert got["judge_n_t07"] == 1.0
    assert got["judge_acc_t07"] == 0.0


def test_every_key_is_present_on_every_row(monkeypatch):
    """np.mean over the batch: a key missing from some rows is averaged over the wrong
    denominator, or breaks the aggregation outright."""
    for temperature in (0.7, 1.0, None):
        got = _score(6, True, temperature, monkeypatch)
        for key in TEMP_KEYS:
            assert key in got, f"{key} missing at gen_temperature={temperature}"


def test_a_single_temperature_corpus_counts_in_neither_bucket(monkeypatch):
    """The val split and every pre-mix corpus carry no gen_temperature. Those rows must not be
    silently attributed to a bucket -- both counts stay 0 so the ratio has an honest
    denominator."""
    got = _score(7, True, None, monkeypatch)

    assert got["judge_n_t07"] == 0.0
    assert got["judge_n_t10"] == 0.0
    assert got["judge_acc_t07"] == 0.0
    assert got["judge_acc_t10"] == 0.0


def test_subset_accuracy_is_recoverable_from_the_batch_means(monkeypatch):
    """The property the counters exist for, end to end.

    Batch: three T=0.7 rows (2 correct) and one T=1.0 row (wrong).
    Expected warm accuracy 2/3, hot accuracy 0/1 -- recovered as mean(acc)/mean(n) exactly the
    way verl_metric_patch will reduce them.
    """
    batch = [
        _score(7, True, 0.7, monkeypatch),   # correct
        _score(7, True, 0.7, monkeypatch),   # correct
        _score(1, True, 0.7, monkeypatch),   # wrong
        _score(1, True, 1.0, monkeypatch),   # wrong
    ]

    def mean(key):
        return sum(row[key] for row in batch) / len(batch)

    assert mean("judge_acc_t07") / mean("judge_n_t07") == pytest.approx(2 / 3)
    assert mean("judge_n_t07") == pytest.approx(3 / 4)
    assert mean("judge_acc_t10") / mean("judge_n_t10") == pytest.approx(0.0)
    assert mean("judge_n_t10") == pytest.approx(1 / 4)
    # The buckets partition the mixed rows: no row is counted twice or dropped.
    assert mean("judge_n_t07") + mean("judge_n_t10") == pytest.approx(1.0)


def test_the_aggregate_accuracy_is_unchanged(monkeypatch):
    """Additive only: judge_acc keeps meaning accuracy over the whole batch."""
    got = _score(7, True, 0.7, monkeypatch)

    assert got["judge_acc"] == 1.0
    assert _score(1, True, 1.0, monkeypatch)["judge_acc"] == 0.0
