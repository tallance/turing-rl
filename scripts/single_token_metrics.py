"""Per-cell metrics for single-token judge runs.

`accuracy` keeps the existing definition (1 correct / 0 wrong / 0.5 tie) so these cells
drop straight into the tables from the earlier judge evals. The extra columns exist to
distinguish a judge that is genuinely uncertain from one that is answering the same
letter every time — both score accuracy 0.5.
"""

from __future__ import annotations


def _p_human(row: dict) -> float:
    """Probability the judge assigned to the slot that actually holds the human."""
    return (1.0 - row["p_a"]) if row["human_is_b"] else row["p_a"]


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    scored = [r for r in rows if not r.get("hard_fail")]

    correct = [
        1.0 if (r["letter"] == ("B" if r["human_is_b"] else "A")) else 0.0
        for r in scored
    ]
    accuracy = sum(correct) / len(scored) if scored else 0.0
    a_rate = sum(r["letter"] == "A" for r in scored) / len(scored) if scored else 0.0

    # a_rate alone conflates letter-bias with sample imbalance: an 80/20-imbalanced but
    # perfectly-judged sample lands a_rate at 0.8 with nothing wrong. expected_a_rate is
    # what an unbiased judge would score on THIS sample (it answers "A" exactly when the
    # human sits in slot A), so a_rate_excess isolates the bias from the imbalance.
    expected_a_rate = (
        sum(not r["human_is_b"] for r in scored) / len(scored) if scored else 0.0
    )
    a_rate_excess = a_rate - expected_a_rate

    # A pair is order-consistent when its two presentations name the same underlying
    # response. Exactly one of the two orders is correct for a position-locked judge,
    # so it lands at 0.0 while a consistent judge lands at 1.0.
    by_pair: dict[str, list[dict]] = {}
    for r in scored:
        by_pair.setdefault(r["pair_id"], []).append(r)
    complete = [g for g in by_pair.values() if len(g) == 2]
    if complete:
        consistent = sum(
            1
            for g in complete
            if (g[0]["letter"] == ("B" if g[0]["human_is_b"] else "A"))
            == (g[1]["letter"] == ("B" if g[1]["human_is_b"] else "A"))
        )
        order_consistency = consistent / len(complete)
    else:
        # No complete pairs means order consistency is unmeasured, not zero. Treating
        # a missing measurement as 0.0 would flag a cell as degenerate when the truth
        # is "no data", not "position-locked".
        order_consistency = None

    ph = [_p_human(r) for r in scored]
    brier = sum((1.0 - p) ** 2 for p in ph) / len(ph) if ph else 0.0

    # abs(a_rate_excess) cannot exceed 1 - accuracy: with s = accuracy on human-in-A
    # rows and t = accuracy on human-in-B rows, a_rate_excess = (s - t)/2 and
    # accuracy = (s + t)/2, so the ceiling at a given accuracy is exactly 1 - accuracy.
    # At the switch threshold (0.7351) that ceiling is 0.2648, and at the reference
    # (0.7551) it is 0.2455 -- both below 0.3, so a 0.3 threshold can never fire on a
    # cell that passes the accuracy gate. 0.2 matches the old a_rate-outside-[0.3, 0.7]
    # rule on a balanced sample and sits ~12 sigma clear of the n=880 sampling noise
    # (~0.017, 1 sigma) -- do not raise it back toward 0.3.
    degenerate = abs(a_rate_excess) > 0.2 or (
        order_consistency is not None and order_consistency < 0.3
    )

    return {
        "n": n,
        "scored": len(scored),
        "hard_fail": (n - len(scored)) / n if n else 0.0,
        "tie_rate": 0.0,  # structurally impossible for a single token
        "accuracy": accuracy,
        "a_rate": a_rate,
        "expected_a_rate": expected_a_rate,
        "a_rate_excess": a_rate_excess,
        "order_consistency": order_consistency,
        "brier": brier,
        "auc": _auc(ph),
        "degenerate": degenerate,
    }


def _auc(p_human: list[float]) -> float:
    """P(the human slot outscores the generated slot), over all cross pairs.

    Each row contributes p_human for the human slot and 1 - p_human for the other, so
    this is a Mann-Whitney statistic over those two score sets, ties counted as 0.5.
    O(n^2) over the pos/neg cross product; fine at eval-cell sizes.
    """
    pos, neg = p_human, [1.0 - p for p in p_human]
    if not pos or not neg:
        return 0.0
    wins = sum((a > b) + 0.5 * (a == b) for a in pos for b in neg)
    return wins / (len(pos) * len(neg))
