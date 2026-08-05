#!/usr/bin/env python3
"""Compare judge curves across eval splits, with user-clustered CIs and composition adjustment.

WHY NOT JUST DIFF THE summary.csv MEANS
---------------------------------------
Two traps make a raw mean-vs-mean read of these curves misleading:

1. CLUSTERING. The held-out test set is 880 rows from only 128 users (~6.9 each, up to 23).
   Rows from one user are correlated, so the naive SE understates the true one -- design effect
   reached 1.67 at step 32, i.e. an effective n of ~526, not 880. A gap that looks like 2 sigma
   under a naive SE can be 1 sigma under a clustered one. Every SE here clusters by user.

2. COMPOSITION. The splits are not compositionally interchangeable. grpo/val is a per-user TAIL
   (one deep turn per user, skewed toward controversy-guided prompts and longer human turns),
   while test is every row of its users. Longer ground-truth turns score lower regardless of
   model, so part of any val/test gap is the prompt mix, not the generator. This script reports
   the gap both raw and post-stratified onto a shared
   (conversation type x target-idx bucket x ground-truth-length bucket) grid.

Usage:
  python scripts/compare_split_curves.py \
      --arm train=results/2026-08-06-trainset-eval-9b-half \
      --arm val=results/2026-08-06-valset-eval-9b-half \
      --arm test=results/2026-08-03-test-eval-9b-half \
      --split_root data/prism/full_s42_history_sft40_grpo60_test10 \
      --out_csv <...>/split_curves.csv --out_md <...>/split_curves.md
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

KEY_FIELDS = ("user_id", "post_id", "target_idx")
_TRAILING_INT = re.compile(r"(\d+)$")

# Reference parquets to pull composition covariates from. Keys are unique across all of them.
COVARIATE_SOURCES = ("test.parquet", "grpo/train.parquet", "grpo/val.parquet")

TARGET_IDX_BINS = ([0, 1, 2, 3, 10**6], ["1", "2", "3", "4+"])
GT_WORD_BINS = ([-1, 5, 10, 20, 40, 10**6], ["<=5", "6-10", "11-20", "21-40", ">40"])


def step_of(gen_key: str) -> int:
    m = _TRAILING_INT.search(gen_key)
    if not m:
        raise SystemExit(f"FAIL: cannot read a step number off gen key {gen_key!r}")
    return int(m.group(1))


def load_reward_rows(reward_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(glob.glob(str(reward_dir / "*.jsonl"))):
        with open(path) as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def likert(row: dict) -> float | None:
    """Valid 1-7 rating; 0 and parse failures dropped, matching summarize_test_eval.py."""
    value = row.get("turing_judge_score_raw")
    if value is None:
        return None
    value = int(round(float(value)))
    return None if value == 0 else float(value)


def load_covariates(split_root: Path) -> dict[tuple[str, str, str], dict]:
    cov: dict[tuple[str, str, str], dict] = {}
    for rel in COVARIATE_SOURCES:
        path = split_root / rel
        if not path.is_file():
            continue
        df = pd.read_parquet(path, columns=["extra_info", "reward_model"])
        for extra, reward in zip(df["extra_info"], df["reward_model"]):
            key = (str(extra["user_id"]), str(extra["post_id"]), str(extra["target_idx"]))
            cov[key] = {
                "ctype": str(extra.get("target_conversation_type", "")),
                "target_idx": int(extra["target_idx"]),
                "gt_words": len(str(reward.get("ground_truth", "")).split()),
            }
    if not cov:
        raise SystemExit(f"FAIL: no covariate parquets under {split_root}")
    return cov


def collect(arms: dict[str, Path], cell: str, mode: str, cov: dict) -> pd.DataFrame:
    records = []
    for label, root in arms.items():
        reward_dirs = sorted(root.glob(f"raw/*/sweep/{cell}/{mode}/reward"))
        if not reward_dirs:
            raise SystemExit(f"FAIL: no reward dirs under {root}/raw/*/sweep/{cell}/{mode}")
        for reward_dir in reward_dirs:
            gen_key = reward_dir.parents[3].name
            for row in load_reward_rows(reward_dir):
                key = tuple(str(row.get(f, "")) for f in KEY_FIELDS)
                meta = cov.get(key)
                if meta is None:
                    raise SystemExit(
                        f"FAIL: scored pair {key} in {label} is in no reference parquet under "
                        "--split_root; the arms are not describable on a common covariate grid."
                    )
                records.append({
                    "arm": label,
                    "step": step_of(gen_key),
                    "user_id": row["user_id"],
                    "likert": likert(row),
                    **meta,
                })
    df = pd.DataFrame(records)
    df["tb"] = pd.cut(df.target_idx, TARGET_IDX_BINS[0], labels=TARGET_IDX_BINS[1])
    df["gb"] = pd.cut(df.gt_words, GT_WORD_BINS[0], labels=GT_WORD_BINS[1])
    df["cell"] = df.ctype.astype(str) + "|" + df.tb.astype(str) + "|" + df.gb.astype(str)
    return df


def clustered(values, clusters) -> tuple[float, float, int, int]:
    """Mean and cluster-robust SE. Clusters are users: rows from one user are correlated."""
    x = np.asarray(values, dtype=float)
    n = len(x)
    if n == 0:
        return float("nan"), float("nan"), 0, 0
    mean = x.mean()
    sums = pd.Series(x - mean).groupby(np.asarray(clusters)).sum().values
    groups = len(sums)
    if groups < 2:
        return mean, float("nan"), n, groups
    var = (groups / (groups - 1)) * (sums**2).sum() / n**2
    return mean, math.sqrt(var), n, groups


def poststratified_mean(sub: pd.DataFrame, weights: pd.Series) -> float:
    """Mean of `sub` reweighted onto another arm's covariate cell distribution."""
    per_cell = sub.groupby("cell").likert.mean()
    common = [c for c in weights.index if c in per_cell.index]
    if not common:
        return float("nan")
    w = weights[common] / weights[common].sum()
    return float((per_cell[common] * w).sum())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", action="append", required=True, metavar="LABEL=EVAL_ROOT",
                    help="Repeatable. e.g. --arm test=results/2026-08-03-test-eval-9b-half")
    ap.add_argument("--split_root", required=True, help="Dataset root holding the reference parquets")
    ap.add_argument("--cell", default="qwen35-9b")
    ap.add_argument("--mode", default="on")
    ap.add_argument("--baseline", default=None,
                    help="Arm every gap is measured against (default: the last --arm given)")
    ap.add_argument("--out_csv", default=None)
    ap.add_argument("--out_md", default=None)
    a = ap.parse_args()

    arms: dict[str, Path] = {}
    for spec in a.arm:
        if "=" not in spec:
            ap.error(f"--arm must be LABEL=EVAL_ROOT, got {spec!r}")
        label, root = spec.split("=", 1)
        arms[label] = Path(root)
    baseline = a.baseline or list(arms)[-1]
    if baseline not in arms:
        ap.error(f"--baseline {baseline!r} is not one of {list(arms)}")

    cov = load_covariates(Path(a.split_root))
    df = collect(arms, a.cell, a.mode, cov)
    scored = df[df.likert.notna()]

    rows_out = []
    for step in sorted(scored.step.unique()):
        at_step = scored[scored.step == step]
        base = at_step[at_step.arm == baseline]
        base_mean, base_se, _, _ = clustered(base.likert, base.user_id)
        base_weights = base.cell.value_counts(normalize=True)
        for label in arms:
            arm = at_step[at_step.arm == label]
            if arm.empty:
                continue
            mean, se, n, n_users = clustered(arm.likert, arm.user_id)
            win, win_se, _, _ = clustered((arm.likert >= 5).astype(float), arm.user_id)
            _, naive_se_ref, _, _ = clustered(arm.likert, np.arange(len(arm)))  # 1 row per cluster
            rec = {
                "step": step, "arm": label, "n": n, "n_users": n_users,
                "likert_mean": round(mean, 4), "likert_se_clustered": round(se, 4),
                "design_effect": round((se / naive_se_ref) ** 2, 3) if naive_se_ref else None,
                "win_rate_ge5": round(win, 4), "win_rate_se_clustered": round(win_se, 4),
            }
            if label != baseline:
                gap = mean - base_mean
                gap_se = math.sqrt(se**2 + base_se**2)
                adj = poststratified_mean(arm, base_weights) - base_mean
                rec.update({
                    f"gap_vs_{baseline}": round(gap, 4),
                    "gap_se": round(gap_se, 4),
                    "gap_z": round(gap / gap_se, 2) if gap_se else None,
                    "gap_ci_lo": round(gap - 1.96 * gap_se, 4),
                    "gap_ci_hi": round(gap + 1.96 * gap_se, 4),
                    f"gap_poststrat_vs_{baseline}": round(adj, 4),
                })
            rows_out.append(rec)

    out = pd.DataFrame(rows_out)
    cols = list(out.columns)
    header = f"| {' | '.join(cols)} |\n|{'|'.join(['---'] * len(cols))}|"
    body = "\n".join(
        "| " + " | ".join("" if pd.isna(r[c]) else str(r[c]) for c in cols) + " |"
        for _, r in out.iterrows()
    )
    note = (f"# arms: {', '.join(arms)}   baseline: {baseline}   judge: {a.cell}/{a.mode}\n"
            f"# SEs cluster by user. gap_poststrat reweights the arm onto {baseline}'s "
            "(ctype x target-idx x gt-words) mix.")
    md = f"{note}\n\n{header}\n{body}"
    print(md)

    if a.out_csv:
        out.to_csv(a.out_csv, index=False)
        print(f"\nwrote {a.out_csv}")
    if a.out_md:
        Path(a.out_md).write_text(md + "\n")
        print(f"wrote {a.out_md}")


if __name__ == "__main__":
    main()
