"""Summarise judge eval runs: accuracy plus the diagnostics that qualify it.

Accuracy alone cannot distinguish a judge that discriminates from one that always answers
the same slot, or one that hedges every call. pred_b_rate catches the first, tie_rate the
second, and order_consistency catches a judge whose verdict flips when the same pair is
presented the other way round.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

TIE_RATING = 4
REQUIRED_COLUMNS = ("model", "pair_id", "order", "rating", "human_is_b")


def _check_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")


def _drop_unrecovered(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with no rating.

    scripts/probe_judge_format.py::dump_row writes `rating=None` when a verdict could not
    be recovered from the completion (see parse_judge_verdict). That becomes NaN on the CSV
    round trip. Those rows carry no signal about which side the judge picked, so they are
    excluded here rather than left to blow up `int(nan)` downstream.
    """
    return df.dropna(subset=["rating"]).copy()


def _row_accuracy(rating: int, human_is_b: bool) -> float:
    if rating == TIE_RATING:
        return 0.5
    return 1.0 if (rating > TIE_RATING) == bool(human_is_b) else 0.0


def summarize_judge_eval(df: pd.DataFrame) -> pd.DataFrame:
    """One row per model: accuracy, ties, slot bias, confidence, Brier, and coverage.

    `unrecovered_rate` is not decoration. Accuracy is computed only over verdicts that
    parsed, and that subset is NOT random — it is the cases the model handled well. A model
    with poor format compliance therefore gets a flattering accuracy over a shrunken
    sample, which is exactly the artifact this table exists to expose. Read accuracy
    together with unrecovered_rate or not at all.
    """
    _check_columns(df)
    n_total = df.groupby("model", sort=True).size()
    work = _drop_unrecovered(df)
    work["_acc"] = [
        _row_accuracy(int(r), bool(h)) for r, h in zip(work["rating"], work["human_is_b"])
    ]
    work["_tie"] = (work["rating"] == TIE_RATING).astype(float)
    work["_pred_b"] = (work["rating"] > TIE_RATING).astype(float)
    p = (work["rating"] - 1) / 6.0
    y = work["human_is_b"].astype(float)
    work["_brier"] = (p - y) ** 2
    work["_conf"] = 2.0 * (p - 0.5).abs()

    grouped = work.groupby("model", sort=True)
    summary = pd.DataFrame(
        {
            "n": grouped.size(),
            "accuracy": grouped["_acc"].mean(),
            "tie_rate": grouped["_tie"].mean(),
            "pred_b_rate": grouped["_pred_b"].mean(),
            "brier": grouped["_brier"].mean(),
            "conf_mean": grouped["_conf"].mean(),
            "rating_mean": grouped["rating"].mean(),
        }
    )
    # Coverage: how much of each model's eval set is actually behind its accuracy.
    summary["n_total"] = n_total.reindex(summary.index).fillna(0).astype(int)
    summary["unrecovered_rate"] = 1.0 - (summary["n"] / summary["n_total"])
    return summary.reset_index()


def order_consistency(df: pd.DataFrame) -> pd.DataFrame:
    """Fraction of pairs whose two presentations name the same side as human.

    Only pairs with both orders present are counted; a pair seen once cannot be
    self-inconsistent. Ties count as disagreement with everything, including another tie,
    because a tie names no side. An unrecovered rating on either side also makes the pair
    incomplete, for the same reason a missing order does: there is no verdict to compare.
    """
    _check_columns(df)
    work = _drop_unrecovered(df)
    rows = []
    for model, model_df in work.groupby("model", sort=True):
        consistent = 0
        total = 0
        for _pair_id, pair_df in model_df.groupby("pair_id", sort=True):
            orders = {str(o): (int(r), bool(h)) for o, r, h in
                      zip(pair_df["order"], pair_df["rating"], pair_df["human_is_b"])}
            if "human_a" not in orders or "human_b" not in orders:
                continue
            total += 1
            calls = []
            for order in ("human_a", "human_b"):
                rating, human_is_b = orders[order]
                if rating == TIE_RATING:
                    calls.append(None)
                else:
                    # Did the judge name the slot that actually holds the human?
                    calls.append((rating > TIE_RATING) == human_is_b)
            if calls[0] is not None and calls[0] == calls[1]:
                consistent += 1
        rows.append(
            {
                "model": model,
                "n_pairs": total,
                "order_consistency": (consistent / total) if total else 0.0,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarise judge eval results")
    parser.add_argument("--eval_csv", required=True, help="long-format CSV of judge verdicts")
    parser.add_argument("--out_csv", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.eval_csv)
    summary = summarize_judge_eval(df).merge(order_consistency(df), on="model", how="left")
    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    summary.to_csv(args.out_csv, index=False)
    print(summary.to_markdown(index=False))
    print(f"\nWrote {args.out_csv}")


if __name__ == "__main__":
    main()
