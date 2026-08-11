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
import re
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval_rl_generator import directional_accuracy  # noqa: E402

KEY_FIELDS = ("user_id", "post_id", "target_idx")
# Display order; anything else found is appended in numeric step order.
PREFERRED = ["9b-grpo-step0", "9b-grpo-step8", "9b-grpo-step16", "9b-grpo-step24", "9b-grpo-step32"]
_TRAILING_INT = re.compile(r"(\d+)$")


def _step_order(gen_key: str) -> tuple[str, int, str]:
    """Sort generator keys by trailing step number rather than lexically."""
    match = _TRAILING_INT.search(gen_key)
    return (
        gen_key[: match.start()] if match else gen_key,
        int(match.group(1)) if match else -1,
        gen_key,
    )


def declared_split(eval_root: Path) -> str:
    """Read the recorded split-guard verdict for the output table."""
    guard = eval_root / "split_guard.json"
    if not guard.is_file():
        return "UNVERIFIED (no split_guard.json)"
    try:
        record = json.loads(guard.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        return f"UNREADABLE split_guard.json ({exc})"
    return (
        f"{record.get('verdict', '?')} expect={record.get('expect', '?')} "
        f"rows={record.get('eval_rows', '?')} users={record.get('eval_users', '?')} "
        f"parquet={record.get('eval_parquet', '?')}"
    )

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
    ap.add_argument("--max_missing_frac", type=float, default=0.0,
                    help="Unscored fraction tolerated, applied to EACH cell AND to the common "
                         "subset. Defaults to 0: a published table should not silently rest on "
                         "incomplete scoring. Raise it only as an explicit diagnostic.")
    ap.add_argument("--common_pairs", action="store_true", default=True,
                    help="Score every checkpoint on the intersection of scored pairs (default)")
    ap.add_argument("--no_common_pairs", dest="common_pairs", action="store_false",
                    help="Score each checkpoint on all pairs it has (not strictly comparable)")
    ap.add_argument("--out_csv", default=None)
    ap.add_argument("--out_md", default=None)
    a = ap.parse_args()

    root = Path(a.eval_root)
    found = {p.parents[3].name: p for p in root.glob(f"raw/*/sweep/{a.cell}/{a.mode}/reward")}
    if not found:
        raise SystemExit(f"FAIL: no reward dirs under {root}/raw/*/sweep/{a.cell}/{a.mode}")
    order = [k for k in PREFERRED if k in found] + sorted(
        set(found) - set(PREFERRED), key=_step_order
    )
    # State the split this table is scored on. Without it a train-set table is visually identical
    # to a held-out one, which is precisely how an overfit curve gets published as generalisation.
    split_note = f"# split: {declared_split(root)}"
    print(split_note + "\n")

    # Judge timeouts drop ~1-3% of pairs, and each checkpoint drops a DIFFERENT few. Scoring each
    # on whatever it happens to have would compare them over different subsets. Restrict every
    # checkpoint to the pairs they ALL scored so the columns are strictly comparable.
    per_key = {k: load_rows(found[k]) for k in order}
    keysets = {k: {tuple(str(r.get(f, "")) for f in KEY_FIELDS) for r in v} for k, v in per_key.items()}
    common = set.intersection(*keysets.values()) if keysets else set()
    dropped = {k: len(v - common) for k, v in keysets.items()}
    floor = a.expect_pairs * (1 - a.max_missing_frac)

    rows_out, problems = [], []

    # The COMMON subset is what the table is actually scored on, and it shrinks with the UNION of
    # each cell's gaps -- so per-cell checks alone are not enough. Observed here: cells at
    # 861/857/870 (worst 97.4%) intersected to just 831/880 = 94.4%, under a 97% per-cell bar.
    # Gate the intersection too; it gets stricter as checkpoints are added.
    if a.common_pairs and len(common) < floor:
        problems.append(
            f"common subset is {len(common)}/{a.expect_pairs} ({len(common)/a.expect_pairs:.1%}) "
            f"across {len(order)} checkpoints, below the {1 - a.max_missing_frac:.1%} floor "
            f"(union of per-cell gaps = {a.expect_pairs - len(common)}). Re-judge the missing pairs "
            f"(verify_judge_completeness.py --write_missing) rather than lowering the bar."
        )
    if any(dropped.values()):
        print(f"# comparability: scoring all checkpoints on the {len(common)} pairs common to every "
              f"cell (dropped per checkpoint: "
              f"{', '.join(f'{k}={n}' for k, n in dropped.items() if n)})\n")

    for key in order:
        uniq = keysets[key]
        if len(uniq) < floor:
            problems.append(f"{key}: {len(uniq)} unique pairs, expected >= {floor:.0f} "
                            f"(--max_missing_frac={a.max_missing_frac:.1%})")
        rows = [r for r in per_key[key]
                if tuple(str(r.get(f, "")) for f in KEY_FIELDS) in common] if a.common_pairs \
            else per_key[key]
        acc = directional_accuracy(rows)
        lk = likerts(rows)
        rows_out.append({
            "checkpoint": key,
            "n_scored": len(rows),
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
        # The note travels with the artifact, not just the console: the .md is what gets pasted
        # into write-ups. (Left out of the CSV, where a comment line would break parsing.)
        Path(a.out_md).write_text(f"{split_note}\n\n{md}\n")
        print(f"wrote {a.out_md}", file=sys.stderr)


if __name__ == "__main__":
    main()
