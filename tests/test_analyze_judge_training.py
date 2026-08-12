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


def test_unrecovered_verdicts_do_not_crash_the_summary():
    """probe_judge_format writes rating=None for unrecoverable verdicts -> NaN on CSV."""
    df = _rows([("m", "p1", "human_a", 1, False), ("m", "p1", "human_b", 7, True)])
    df.loc[len(df)] = {
        "model": "m", "pair_id": "p2", "order": "human_a",
        "rating": float("nan"), "human_is_b": False,
    }
    summary = summarize_judge_eval(df).set_index("model")
    assert summary.loc["m", "n"] == 2
    assert summary.loc["m", "accuracy"] == 1.0


def test_summary_reports_how_much_of_the_eval_set_was_unrecoverable():
    """Accuracy over a shrunken, non-random subset flatters a low-compliance model."""
    df = _rows([("m", "p1", "human_a", 1, False)])
    for pair in ("p2", "p3", "p4"):
        df.loc[len(df)] = {
            "model": "m", "pair_id": pair, "order": "human_a",
            "rating": float("nan"), "human_is_b": False,
        }
    summary = summarize_judge_eval(df).set_index("model")
    assert summary.loc["m", "n"] == 1
    assert summary.loc["m", "n_total"] == 4
    assert summary.loc["m", "unrecovered_rate"] == pytest.approx(0.75)
    # The flattering part: perfect accuracy on the quarter it managed to parse.
    assert summary.loc["m", "accuracy"] == 1.0


def test_order_consistency_drops_unrecovered_rows():
    df = _rows([("m", "p1", "human_a", 1, False)])
    df.loc[len(df)] = {
        "model": "m", "pair_id": "p1", "order": "human_b",
        "rating": float("nan"), "human_is_b": True,
    }
    # p1's second order is unrecoverable, so the pair is incomplete, not consistent.
    assert order_consistency(df).set_index("model").loc["m", "n_pairs"] == 0


def _multi_model_df():
    """Two models, one of which parsed nothing at all.

    A single-model fixture cannot catch the bug this guards: with one model, a global
    row count and a per-model row count are the same number.
    """
    df = _rows(
        [
            ("compliant", "p1", "human_a", 1, False),
            ("compliant", "p1", "human_b", 7, True),
            ("compliant", "p2", "human_a", 1, False),
        ]
    )
    for pair, order, human_is_b in (
        ("p1", "human_a", False), ("p1", "human_b", True), ("p2", "human_a", False)
    ):
        df.loc[len(df)] = {
            "model": "silent", "pair_id": pair, "order": order,
            "rating": float("nan"), "human_is_b": human_is_b,
        }
    return df


def test_a_zero_compliance_model_still_gets_a_row():
    """The loudest possible result must not be the one that vanishes from the table."""
    summary = summarize_judge_eval(_multi_model_df()).set_index("model")
    assert set(summary.index) == {"compliant", "silent"}
    assert summary.loc["silent", "n"] == 0
    assert summary.loc["silent", "n_total"] == 3
    assert summary.loc["silent", "unrecovered_rate"] == pytest.approx(1.0)
    assert pd.isna(summary.loc["silent", "accuracy"])


def test_n_total_is_per_model_not_the_whole_frame():
    summary = summarize_judge_eval(_multi_model_df()).set_index("model")
    assert summary.loc["compliant", "n_total"] == 3
    assert summary.loc["compliant", "n"] == 3
    assert summary.loc["compliant", "unrecovered_rate"] == pytest.approx(0.0)


def test_order_consistency_returns_named_columns_when_every_model_is_silent():
    """Otherwise main()'s merge on "model" dies with a bare KeyError."""
    df = _multi_model_df()
    df = df[df["model"] == "silent"]
    result = order_consistency(df)
    assert list(result.columns) == ["model", "n_pairs", "order_consistency"]
    assert len(result) == 0
    # And the merge main() performs must survive it.
    merged = summarize_judge_eval(df).merge(result, on="model", how="left")
    assert list(merged["model"]) == ["silent"]


def test_the_regime_column_is_surfaced_when_present():
    df = _multi_model_df()
    df["regime"] = "json_schema"
    summary = summarize_judge_eval(df).set_index("model")
    assert summary.loc["compliant", "regime"] == "json_schema"


def test_a_dump_without_a_regime_column_is_still_accepted():
    summary = summarize_judge_eval(_multi_model_df())
    assert "regime" not in summary.columns
