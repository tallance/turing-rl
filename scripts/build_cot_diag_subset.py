"""Build a failing+control pair subset for the CoT-failure diagnostic (plan: smooth-strolling-lake).

A "failing" pair = a pair in the frozen 880 set whose current thinking-ON reward dumps
for <cell> never recovered a valid 1-7 rating (parse error; almost always a client
timeout -> empty judge content). We replay these (plus a deterministic control sample)
with a generous timeout so they COMPLETE and their raw <think> CoT gets dumped, letting
us inspect the failure mode (hypothesis: runaway repetition).

Reuses analyze_judge_sweep.per_call_features so "failing" is defined identically to the
analyzer's parse_error. Pair identity = (user_id, post_id, target_idx).

Usage (cluster, env turing-rl-train):
  python scripts/build_cot_diag_subset.py --cell qwen35-397b \
      --out results/2026-07-08-judge-sweep/raw/pairs/diag_qwen35-397b_on.parquet
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO_ROOT))
import pandas as pd
from scripts.analyze_judge_sweep import load_cell_rows, per_call_features


def _key(user_id, post_id, target_idx) -> str:
    return f"{user_id}::{post_id}::{target_idx}"


def failing_keys(raw_root: Path, cell: str, mode: str = "on") -> set[str]:
    """Keys with NO valid rating across any dump row (recovered pairs are excluded)."""
    rows = load_cell_rows(raw_root / "sweep" / cell / mode)
    valid_by_key: dict[str, bool] = {}
    for r in rows:
        k = _key(r.get("user_id"), r.get("post_id"), r.get("target_idx"))
        ok = not per_call_features(r)["parse_error"]
        valid_by_key[k] = valid_by_key.get(k, False) or ok
    return {k for k, ok in valid_by_key.items() if not ok}


def main() -> None:
    ap = argparse.ArgumentParser()
    base = REPO_ROOT / "results" / "2026-07-08-judge-sweep"
    ap.add_argument("--cell", required=True)
    ap.add_argument("--mode", default="on")
    ap.add_argument("--raw_root", type=Path, default=base / "raw")
    ap.add_argument("--pairs", type=Path, default=base / "raw" / "pairs" / "prism_heldout_880.parquet")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n_total", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_parquet(args.pairs)
    df = df.copy()
    df["_key"] = [_key(u, p, t) for u, p, t in zip(df["user_id"], df["post_id"], df["target_idx"])]
    if df["_key"].duplicated().any():
        raise SystemExit("880 pair set has duplicate (user_id,post_id,target_idx) keys")

    fail = failing_keys(args.raw_root, args.cell, args.mode)
    fail_in_set = df["_key"].isin(fail)
    n_fail = int(fail_in_set.sum())
    missing = fail - set(df["_key"])
    if missing:
        print(f"[warn] {len(missing)} failing keys not found in the 880 set (ignored)")

    fail_df = df[fail_in_set]
    # deterministic control from the non-failing remainder, up to n_total
    n_control = max(0, args.n_total - n_fail)
    rest = df[~fail_in_set]
    control_df = rest.sample(n=min(n_control, len(rest)), random_state=args.seed) if n_control else rest.iloc[:0]
    out_df = pd.concat([fail_df, control_df]).drop(columns=["_key"]).reset_index(drop=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(args.out, index=False)
    print(f"cell={args.cell} mode={args.mode}")
    print(f"  failing pairs (replayed) : {n_fail}")
    print(f"  control pairs            : {len(control_df)}")
    print(f"  total rows written       : {len(out_df)} -> {args.out}")
    print(f"  columns                  : {list(out_df.columns)}")


if __name__ == "__main__":
    main()
