"""Unit tests for judge eval analysis."""

import pandas as pd
import pytest

from scripts.analyze_judge_training import order_consistency, summarize_judge_eval


def _rows(records):
    return pd.DataFrame(
        [
            {"model": m, "pair_id": p, "order": o, "rating": r, "human_is_b": h}
            for m, p, o, r, h in records
        ]
    )


def test_a_perfect_judge_scores_one():
    df = _rows([("m", "p1", "human_a", 1, False), ("m", "p1", "human_b", 7, True)])
    summary = summarize_judge_eval(df).set_index("model")
    assert summary.loc["m", "accuracy"] == 1.0
    assert summary.loc["m", "tie_rate"] == 0.0
    assert summary.loc["m", "n"] == 2


def test_an_always_tie_judge_scores_half():
    df = _rows([("m", "p1", "human_a", 4, False), ("m", "p1", "human_b", 4, True)])
    summary = summarize_judge_eval(df).set_index("model")
    assert summary.loc["m", "accuracy"] == 0.5
    assert summary.loc["m", "tie_rate"] == 1.0


def test_a_slot_b_biased_judge_is_caught_by_pred_b_rate():
    df = _rows([("m", "p1", "human_a", 7, False), ("m", "p1", "human_b", 7, True)])
    summary = summarize_judge_eval(df).set_index("model")
    assert summary.loc["m", "accuracy"] == 0.5
    assert summary.loc["m", "pred_b_rate"] == 1.0


def test_brier_is_reported():
    df = _rows([("m", "p1", "human_a", 1, False)])
    summary = summarize_judge_eval(df).set_index("model")
    assert summary.loc["m", "brier"] == pytest.approx(0.0)


def test_models_are_summarised_separately():
    df = _rows(
        [
            ("good", "p1", "human_a", 1, False),
            ("good", "p1", "human_b", 7, True),
            ("bad", "p1", "human_a", 7, False),
            ("bad", "p1", "human_b", 1, True),
        ]
    )
    summary = summarize_judge_eval(df).set_index("model")
    assert summary.loc["good", "accuracy"] == 1.0
    assert summary.loc["bad", "accuracy"] == 0.0


def test_order_consistency_is_one_when_both_orders_agree():
    # Both orders name the human: correct in each presentation.
    df = _rows([("m", "p1", "human_a", 1, False), ("m", "p1", "human_b", 7, True)])
    result = order_consistency(df).set_index("model")
    assert result.loc["m", "order_consistency"] == 1.0
    assert result.loc["m", "n_pairs"] == 1


def test_order_consistency_is_zero_for_a_fixed_slot_answer():
    # Always says B regardless of where the human is: inconsistent across orders.
    df = _rows([("m", "p1", "human_a", 7, False), ("m", "p1", "human_b", 7, True)])
    assert order_consistency(df).set_index("model").loc["m", "order_consistency"] == 0.0


def test_order_consistency_ignores_pairs_missing_an_order():
    df = _rows([("m", "p1", "human_a", 1, False)])
    assert order_consistency(df).set_index("model").loc["m", "n_pairs"] == 0
