"""Assemble one comparison table across the full-schema and single-token judge arms.

Both arms score the same frozen 880-pair set, so their cells are directly comparable --
but they report different natives. The full-schema path emits a 1-7 rating where 4 is a
tie and the response can fail to parse; the single-token path emits one A/B token and can
hard-fail. Comparing the two arms' headline numbers as-published would compare different
denominators, so this harmonises them under one rule fixed by the design spec:

    accuracy_half_tie = (correct + 0.5 * ties) / n_pairs

Parse failures and hard failures stay in the denominator and contribute 0. Single-token
cells have no ties by construction, so for them the rule reduces to plain accuracy with
hard failures counted wrong.

``accuracy_parse_ok`` is carried alongside for continuity with the earlier full-schema
reports: it is accuracy over parsed, non-tie calls only, which is what
``analyze_judge_sweep.py`` publishes.

The per-call semantics of a full-schema row are NOT reimplemented here. They come from
``analyze_judge_sweep.per_call_features`` -- the single source of truth for picked_human,
tie and parse-error -- so this script cannot drift from the full arm's own analyser.

Usage:
  python scripts/build_single_token_comparison.py \
    --sweep-root <single-token run>/raw/sweep \
    --sweep-root <full-schema run>/raw/sweep \
    --out <out>/comparison.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_judge_sweep import per_call_features  # noqa: E402

FIELDS = [
    "cell", "prompt_style", "thinking_mode", "n_pairs",
    "accuracy_half_tie", "accuracy_parse_ok",
    "tie_rate", "failure_rate", "pick_a_rate", "run_root",
]


def read_rows(reward_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(reward_dir.glob("*.jsonl")):
        with path.open() as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def summarize_single_token(rows: list[dict]) -> dict:
    """A/B letter against the true human side. No ties; hard failures count wrong."""
    n = len(rows)
    failures = sum(1 for r in rows if r.get("hard_fail"))
    correct = sum(
        1 for r in rows
        if not r.get("hard_fail") and r.get("letter") == ("B" if r.get("human_is_b") else "A")
    )
    picks_a = sum(1 for r in rows if not r.get("hard_fail") and r.get("letter") == "A")
    scored = n - failures
    return {
        "n_pairs": n,
        "accuracy_half_tie": correct / n if n else None,
        "accuracy_parse_ok": correct / scored if scored else None,
        "tie_rate": 0.0,
        "failure_rate": failures / n if n else None,
        "pick_a_rate": picks_a / scored if scored else None,
    }


def summarize_full_schema(rows: list[dict]) -> dict:
    """1-7 rating via the full arm's own per-call parser. rating==4 is a tie."""
    calls = [per_call_features(r) for r in rows]
    n = len(calls)
    ties = sum(1 for c in calls if c["rating"] == 4)
    parse_errors = sum(1 for c in calls if c["parse_error"])
    correct = sum(1 for c in calls if c["picked_human"] == 1)
    scored = sum(1 for c in calls if c["picked_human"] is not None)
    picks_a = sum(
        1 for c in calls
        if c["rating"] is not None and 1 <= c["rating"] <= 7 and c["rating"] < 4
    )
    return {
        "n_pairs": n,
        "accuracy_half_tie": (correct + 0.5 * ties) / n if n else None,
        "accuracy_parse_ok": correct / scored if scored else None,
        "tie_rate": ties / n if n else None,
        "failure_rate": parse_errors / n if n else None,
        "pick_a_rate": picks_a / scored if scored else None,
    }


def discover(sweep_root: Path) -> list[dict]:
    """One record per reward directory under ``<cell>/<mode>[/<style>]/reward``."""
    records = []
    for reward_dir in sorted(sweep_root.glob("*/*/reward")) + sorted(
        sweep_root.glob("*/*/*/reward")
    ):
        relative = reward_dir.relative_to(sweep_root).parts
        cell, mode = relative[0], relative[1]
        style_from_path = relative[2] if len(relative) == 4 else None

        metadata = {}
        metadata_path = reward_dir.parent / "run_metadata.json"
        if metadata_path.exists():
            try:
                metadata = json.loads(metadata_path.read_text())
            except json.JSONDecodeError:
                pass
        # Path and metadata should agree; metadata wins because it is what the job recorded.
        style = metadata.get("prompt_style") or style_from_path or "full"

        rows = read_rows(reward_dir)
        if not rows:
            print(f"[compare] SKIP {cell}/{mode} ({style}): no rows", file=sys.stderr)
            continue

        summary = (
            summarize_single_token(rows) if style == "single_token"
            else summarize_full_schema(rows)
        )
        records.append({
            "cell": cell,
            "prompt_style": style,
            "thinking_mode": metadata.get("thinking_mode") or mode,
            "run_root": str(sweep_root.parent.parent),
            **summary,
        })
    return records


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--sweep-root", action="append", required=True, type=Path,
                        help="raw/sweep directory of a run; repeatable")
    parser.add_argument("--out", type=Path, required=True, help="destination CSV")
    args = parser.parse_args()

    records: list[dict] = []
    for sweep_root in args.sweep_root:
        if not sweep_root.is_dir():
            print(f"FATAL: not a directory: {sweep_root}", file=sys.stderr)
            return 2
        found = discover(sweep_root)
        print(f"[compare] {sweep_root}: {len(found)} cell(s)", file=sys.stderr)
        records.extend(found)

    if not records:
        print("FATAL: no cells found in any sweep root", file=sys.stderr)
        return 2

    records.sort(key=lambda r: (r["prompt_style"], r["thinking_mode"], -(r["accuracy_half_tie"] or 0)))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key) for key in FIELDS})

    width = max(len(r["cell"]) for r in records)
    print(f"{'cell':<{width}}  {'style':<12} {'think':<5} {'n':>4} "
          f"{'acc½tie':>8} {'accParseOk':>10} {'tie':>6} {'fail':>6} {'pickA':>6}")
    for record in records:
        def fmt(key: str, spec: str = "6.3f") -> str:
            value = record.get(key)
            return format(value, spec) if isinstance(value, float) else "     -"
        print(f"{record['cell']:<{width}}  {record['prompt_style']:<12} "
              f"{record['thinking_mode']:<5} {record['n_pairs']:>4} "
              f"{fmt('accuracy_half_tie', '8.3f')} {fmt('accuracy_parse_ok', '10.3f')} "
              f"{fmt('tie_rate')} {fmt('failure_rate')} {fmt('pick_a_rate')}")
    print(f"\nwrote {len(records)} rows -> {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
