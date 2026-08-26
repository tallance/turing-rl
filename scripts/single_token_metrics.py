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

    # A pair is order-consistent when its two presentations name the same underlying
    # response. Exactly one of the two orders is correct for a position-locked judge,
    # so it lands at 0.0 while a consistent judge lands at 1.0.
    by_pair: dict[str, list[dict]] = {}
    for r in scored:
        by_pair.setdefault(r["pair_id"], []).append(r)
    complete = [g for g in by_pair.values() if len(g) == 2]
    consistent = sum(
        1
        for g in complete
        if (g[0]["letter"] == ("B" if g[0]["human_is_b"] else "A"))
        == (g[1]["letter"] == ("B" if g[1]["human_is_b"] else "A"))
    )
    order_consistency = consistent / len(complete) if complete else 0.0

    ph = [_p_human(r) for r in scored]
    brier = sum((1.0 - p) ** 2 for p in ph) / len(ph) if ph else 0.0

    return {
        "n": n,
        "scored": len(scored),
        "hard_fail": (n - len(scored)) / n if n else 0.0,
        "tie_rate": 0.0,  # structurally impossible for a single token
        "accuracy": accuracy,
        "a_rate": a_rate,
        "order_consistency": order_consistency,
        "brier": brier,
        "auc": _auc(ph),
        "degenerate": not (0.3 <= a_rate <= 0.7) or order_consistency < 0.3,
    }


def _auc(p_human: list[float]) -> float:
    """P(the human slot outscores the generated slot), over all cross pairs.

    Each row contributes p_human for the human slot and 1 - p_human for the other, so
    this is a Mann-Whitney statistic over those two score sets, ties counted as 0.5.
    """
    pos, neg = p_human, [1.0 - p for p in p_human]
    if not pos or not neg:
        return 0.0
    wins = sum((a > b) + 0.5 * (a == b) for a in pos for b in neg)
    return wins / (len(pos) * len(neg))
