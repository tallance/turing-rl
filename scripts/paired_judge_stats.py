"""Paired statistics for arm-vs-arm judge comparisons.

Two arms score the SAME 880 rows, so comparing their marginal confidence intervals
throws away the pairing and badly understates power. And those 880 rows are 440 pairs
seen in two presentation orders, so a row-level interval understates the width.
"""

from __future__ import annotations

import math
import random


def mcnemar(a_correct: list[int], b_correct: list[int]) -> dict:
    """McNemar with continuity correction over paired per-row correctness."""
    if len(a_correct) != len(b_correct):
        raise ValueError("arms must score the same rows")
    b01 = sum(1 for a, b in zip(a_correct, b_correct) if a == 1 and b == 0)
    b10 = sum(1 for a, b in zip(a_correct, b_correct) if a == 0 and b == 1)
    n = b01 + b10
    if n == 0:
        return {"n_discordant": 0, "b01": 0, "b10": 0, "chi2": 0.0, "p_value": 1.0}
    chi2 = (abs(b01 - b10) - 1) ** 2 / n
    p = math.erfc(math.sqrt(chi2 / 2.0))  # 1 - chi2_1.cdf(x) == erfc(sqrt(x/2))
    return {"n_discordant": n, "b01": b01, "b10": b10, "chi2": chi2, "p_value": p}


def clustered_ci(
    correct: list[float], pair_ids: list, *, seed: int = 0, iters: int = 2000,
    alpha: float = 0.05,
) -> dict:
    """Bootstrap CI resampling whole pair_id clusters, not rows."""
    clusters: dict = {}
    for c, pid in zip(correct, pair_ids):
        clusters.setdefault(pid, []).append(c)
    keys = list(clusters)
    rng = random.Random(seed)

    means = []
    for _ in range(iters):
        drawn = [v for _ in keys for v in clusters[rng.choice(keys)]]
        means.append(sum(drawn) / len(drawn))
    means.sort()
    lo = means[int(alpha / 2 * iters)]
    hi = means[int((1 - alpha / 2) * iters) - 1]
    return {
        "mean": sum(correct) / len(correct),
        "lo": lo, "hi": hi, "width": hi - lo, "n_clusters": len(keys),
    }
