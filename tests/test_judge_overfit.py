"""Unit tests for the judge overfit subset builder and the gate check."""

import json

import pandas as pd
import pytest

from scripts.build_judge_overfit import build_judge_overfit
from scripts.judge_overfit_gate import gate_verdict, read_metric_series


def _judge_parquet(tmp_path, n_pairs: int = 10):
    rows = []
    for i in range(n_pairs):
        for order, human_is_b in (("human_a", False), ("human_b", True)):
            rows.append(
                {
                    "data_source": "prism_judge",
                    "prompt": [{"role": "user", "content": f"prompt {i}"}],
                    "reward_model": {"style": "rule", "ground_truth": "B" if human_is_b else "A"},
                    "extra_info": {
                        "row_id": f"p{i}::{order}",
                        "pair_id": f"p{i}",
                        "order": order,
                        "human_is_b": human_is_b,
                    },
                }
            )
    path = tmp_path / "train.parquet"
    pd.DataFrame(rows).to_parquet(path)
    return str(path)


def test_overfit_subset_keeps_both_orders_of_each_pair(tmp_path):
    out = str(tmp_path / "overfit.parquet")
    df = build_judge_overfit(_judge_parquet(tmp_path), out, n_pairs=3)
    assert len(df) == 6
    assert df["extra_info"].map(lambda e: e["pair_id"]).nunique() == 3


def test_overfit_subset_stays_balanced(tmp_path):
    out = str(tmp_path / "overfit.parquet")
    df = build_judge_overfit(_judge_parquet(tmp_path), out, n_pairs=4)
    sides = [e["human_is_b"] for e in df["extra_info"]]
    assert sum(sides) * 2 == len(sides)


def test_overfit_subset_is_written_to_disk(tmp_path):
    out = str(tmp_path / "overfit.parquet")
    build_judge_overfit(_judge_parquet(tmp_path), out, n_pairs=2)
    assert len(pd.read_parquet(out)) == 4


def test_requesting_more_pairs_than_exist_raises(tmp_path):
    with pytest.raises(ValueError):
        build_judge_overfit(_judge_parquet(tmp_path, n_pairs=2), str(tmp_path / "o.parquet"), n_pairs=99)


def test_read_metric_series_pulls_one_key(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_text(
        "\n".join(json.dumps({"step": s, "judge_acc": s / 10}) for s in range(5)) + "\n"
    )
    assert read_metric_series(str(path), "judge_acc") == [0.0, 0.1, 0.2, 0.3, 0.4]


def test_read_metric_series_skips_rows_missing_the_key(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_text(json.dumps({"step": 0}) + "\n" + json.dumps({"judge_acc": 1.0}) + "\n")
    assert read_metric_series(str(path), "judge_acc") == [1.0]


def test_gate_passes_when_the_tail_saturates():
    verdict = gate_verdict([0.5, 0.6, 0.9, 0.98, 1.0], threshold=0.95, tail=2)
    assert verdict["passed"] is True
    assert verdict["tail_mean"] == pytest.approx(0.99)


def test_gate_fails_when_the_tail_stays_low():
    verdict = gate_verdict([0.5, 0.5, 0.52, 0.51], threshold=0.95, tail=2)
    assert verdict["passed"] is False


def test_gate_fails_on_an_empty_series():
    assert gate_verdict([], threshold=0.95, tail=2)["passed"] is False
