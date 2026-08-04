"""Summarize the held-out test-set eval: directional accuracy + judge-Likert per checkpoint.

Combines the two views of the same reward dumps:
  * directional accuracy -- sweep-matched, via scripts/eval_rl_generator.py, so the numbers are
    comparable to the judge/generator sweeps (ties and parse-fails excluded from the denominator)
  * Likert distribution -- mean, win rate (>=5) and %7 over valid 1-7 ratings, matching the
    convention in scripts/plot_likert_hist_train_val.py

Refuses to emit a table unless every cell scored all expected pairs, because a partially scored
cell would put the checkpoints on different subsets and silently break comparability.

Usage:
  python scripts/summarize_test_eval.py --eval_root results/2026-08-03-test-eval-9b-half \
      [--cell qwen35-9b --mode on] [--out_csv summary.csv] [--out_md summary.md]
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval_rl_generator import directional_accuracy  # noqa: E402

KEY_FIELDS = ("user_id", "post_id", "target_idx")
# Display order; anything else found is appended alphabetically.
PREFERRED = ["9b-grpo-step0", "9b-grpo-step8", "9b-grpo-step16", "9b-grpo-step24", "9b-grpo-step32"]


def load_rows(reward_dir: Path) -> list[dict]:
    rows = []
    for f in sorted(glob.glob(str(reward_dir / "*.jsonl"))):
        for line in open(f):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def likerts(rows: list[dict]) -> list[int]:
    out = []
    for r in rows:
        v = r.get("turing_judge_score_raw")
        if v is None or int(round(float(v))) == 0:
            continue
        out.append(int(round(float(v))))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval_root", required=True)
    ap.add_argument("--cell", default="qwen35-9b")
    ap.add_argument("--mode", default="on")
    ap.add_argument("--expect_pairs", type=int, default=880)
    ap.add_argument("--out_csv", default=None)
    ap.add_argument("--out_md", default=None)
    a = ap.parse_args()

    root = Path(a.eval_root)
    found = {p.parents[3].name: p for p in root.glob(f"raw/*/sweep/{a.cell}/{a.mode}/reward")}
    if not found:
        raise SystemExit(f"FAIL: no reward dirs under {root}/raw/*/sweep/{a.cell}/{a.mode}")
    order = [k for k in PREFERRED if k in found] + sorted(set(found) - set(PREFERRED))

    rows_out, problems = [], []
    for key in order:
        rows = load_rows(found[key])
        uniq = {tuple(str(r.get(f, "")) for f in KEY_FIELDS) for r in rows}
        if len(uniq) != a.expect_pairs:
            problems.append(f"{key}: {len(uniq)} unique pairs, expected {a.expect_pairs}")
        acc = directional_accuracy(rows)
        lk = likerts(rows)
        rows_out.append({
            "checkpoint": key,
            "n_unique_pairs": len(uniq),
            "n_likert": len(lk),
            "likert_mean": round(statistics.mean(lk), 4) if lk else None,
            "win_rate_ge5": round(sum(1 for v in lk if v >= 5) / len(lk), 4) if lk else None,
            "pct_7": round(100 * sum(1 for v in lk if v == 7) / len(lk), 2) if lk else None,
            "judge_accuracy": round(acc["accuracy"], 4),
            "gen_win_rate": round(acc["gen_win_rate"], 4),
            "n_nontie": acc["n_nontie"],
            "n_tie": acc["n_tie"],
            "n_parse_error": acc["n_parse_error"],
        })

    if problems:
        print("FAILED: cells are not on the same pair set, so the table would not be comparable:",
              file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        raise SystemExit(1)

    cols = list(rows_out[0])
    header = f"| {' | '.join(cols)} |\n|{'|'.join(['---'] * len(cols))}|"
    body = "\n".join("| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows_out)
    md = f"{header}\n{body}"
    print(md)

    if a.out_csv:
        Path(a.out_csv).write_text(
            ",".join(cols) + "\n" + "\n".join(",".join(str(r[c]) for c in cols) for r in rows_out) + "\n")
        print(f"\nwrote {a.out_csv}", file=sys.stderr)
    if a.out_md:
        Path(a.out_md).write_text(md + "\n")
        print(f"wrote {a.out_md}", file=sys.stderr)


if __name__ == "__main__":
    main()
