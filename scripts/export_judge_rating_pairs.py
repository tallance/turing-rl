"""Export the per-pair judge ratings from the test eval as one tidy CSV.

Runs on the cluster (the reward dumps are ~2.3G and are not mirrored to the Mac) and
emits one row per (checkpoint step, pair), with a rating column per judge cell:

  step,user_id,post_id,target_idx,r_qwen35_4b,r_qwen35_9b,r_qwen35_27b

The rating is turing_judge_score_raw, the order-normalized 1-7 Likert (higher = the
generated turn looked more human). A parse failure (0/None) is written as an empty
field rather than a number, so downstream code cannot mistake it for a rating.

All judge cells scored byte-identical pair-sets, so a row's rating columns describe
the same generated turn; the script asserts this rather than trusting it.

Usage (cluster):
  python scripts/export_judge_rating_pairs.py \
      --eval_root results/2026-08-03-test-eval-9b-half --out judge_rating_pairs.csv
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import re
import sys
from pathlib import Path

KEY_FIELDS = ("user_id", "post_id", "target_idx")


def step_of(checkpoint: str) -> int:
    m = re.search(r"step(\d+)", checkpoint)
    if not m:
        raise SystemExit(f"FAIL: cannot read a step number out of {checkpoint!r}")
    return int(m.group(1))


def rating(r: dict):
    v = r.get("turing_judge_score_raw")
    if v is None:
        return None
    try:
        n = int(round(float(v)))
    except (TypeError, ValueError):
        return None
    return None if n == 0 else n


def load_cell(root: Path, cell: str, mode: str) -> dict[int, dict[tuple, dict]]:
    found = {p.parents[3].name: p for p in root.glob(f"raw/*/sweep/{cell}/{mode}/reward")}
    if not found:
        raise SystemExit(f"FAIL: no reward dirs under {root}/raw/*/sweep/{cell}/{mode}")
    out: dict[int, dict[tuple, dict]] = {}
    for ckpt, reward_dir in found.items():
        step = step_of(ckpt)
        seen: dict[tuple, dict] = {}
        for f in sorted(glob.glob(str(reward_dir / "*.jsonl"))):
            for line in open(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                k = tuple(str(d.get(x, "")) for x in KEY_FIELDS)
                if k in seen:
                    raise SystemExit(f"FAIL: {cell} {ckpt} scored pair {k} twice")
                seen[k] = {"r": rating(d),
                           "resp": str(d.get("response", "")).replace("\n", " ").strip()}
        out[step] = seen
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval_root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cells", default="qwen35-4b,qwen35-9b,qwen35-27b")
    ap.add_argument("--mode", default="on")
    a = ap.parse_args()

    root = Path(a.eval_root)
    cells = [c.strip() for c in a.cells.split(",") if c.strip()]
    data = {c: load_cell(root, c, a.mode) for c in cells}

    steps = sorted(data[cells[0]])
    for c in cells[1:]:
        if sorted(data[c]) != steps:
            raise SystemExit(f"FAIL: cell {c} covers steps {sorted(data[c])}, expected {steps}")

    # Only pairs every cell scored at every step, so each row is complete and the
    # per-step point clouds are drawn over the identical sample.
    common = set.intersection(*(set(data[c][s]) for c in cells for s in steps))
    print(f"# steps={steps} cells={cells} common pairs per step={len(common)}", file=sys.stderr)

    rows = []
    for s in steps:
        for k in sorted(common):
            resp = data[cells[0]][s][k]["resp"]
            rec = {"step": s, "user_id": k[0], "post_id": k[1], "target_idx": k[2]}
            for c in cells:
                cell = data[c][s][k]
                if cell["resp"] != resp:
                    raise SystemExit(
                        f"FAIL: {c} step {s} pair {k} scored a different generated turn than "
                        f"{cells[0]}; the cells are not the same sample")
                rec[f"r_{c.replace('-', '_')}"] = "" if cell["r"] is None else cell["r"]
            rows.append(rec)

    cols = ["step", "user_id", "post_id", "target_idx"] + \
           [f"r_{c.replace('-', '_')}" for c in cells]
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {a.out}  ({len(rows)} rows = {len(steps)} steps x {len(common)} pairs)",
          file=sys.stderr)


if __name__ == "__main__":
    main()
