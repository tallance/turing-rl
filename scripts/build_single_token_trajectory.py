#!/usr/bin/env python
"""Per-checkpoint single-token summaries for a generator trajectory.

Reads the cells written by ``launch_single_token_trajectory.sh`` and emits one
``summary_<cell>.csv`` per judge, using the column names the full-schema package already
uses (``judge_accuracy``, ``gen_win_rate``) so the two can be read side by side.

Metrics are generator-oriented, because the question is about the generator rather than
the judge:

  gen_win_rate  fraction of votes where the judge picked the GENERATED response
  p_gen_mean    mean probability the judge put on the generated side

``p_gen_mean`` replaces the full arm's ``likert_mean``. Single-token has no 1-7 rating --
``RATING_FOR_A``/``RATING_FOR_B`` are 1 and 7, so a mean likert here would just be
``1 + 6 * gen_win_rate`` wearing a graded-looking scale. The probability is the real
graded signal.

A hard fail is not a vote: it leaves both denominators and is reported as n_hard_fail.

  python scripts/build_single_token_trajectory.py --run-root <root> --out-dir <dir>
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.analyze_single_token_cells import (  # noqa: E402
    _pair_key,
    _row_prompt_style,
    cell_mode_style,
    find_reward_files,
    load_rows,
)

FIELDS = [
    "checkpoint", "step", "n_scored", "n_unique_pairs", "judge_accuracy",
    "gen_win_rate", "p_gen_mean", "a_rate", "n_hard_fail",
]

DEFAULT_STEPS = (0, 12, 24, 36, 48, 60, 72, 84, 96, 108, 120)
DEFAULT_PREFIX = "9b-train10pct-step"
DEFAULT_PAIRS = 440


def step_of(checkpoint: str, *, prefix: str) -> int:
    """The trailing step number, as an int.

    Int, never the string: the grid contains 12, 108 and 120, so a lexicographic sort
    orders them 0, 108, 12, 120, 24 ... and renders a scrambled curve that still looks
    like a plausible trajectory. Same reason the full-schema builder parses the step.
    """
    if not checkpoint.startswith(prefix):
        raise ValueError(f"checkpoint {checkpoint!r} does not start with {prefix!r}")
    tail = checkpoint[len(prefix):]
    if not tail.isdigit():
        raise ValueError(f"checkpoint {checkpoint!r} has no numeric step after {prefix!r}")
    return int(tail)


def summarize(rows: list[dict]) -> dict:
    """Generator-oriented metrics for one (cell, checkpoint).

    ``human_is_b`` fixes the orientation: the generated response is A exactly when the
    human is B.
    """
    n = len(rows)
    votes = [r for r in rows if not bool(r.get("hard_fail"))]
    hard_fail = n - len(votes)

    picked_generated = 0
    picked_a = 0
    p_gen_total = 0.0
    for row in votes:
        for field in ("human_is_b", "letter", "p_a"):
            if field not in row:
                raise ValueError(f"row is missing single-token field {field!r}")
        human_is_b = bool(row["human_is_b"])
        letter = row["letter"]
        generated_side = "A" if human_is_b else "B"
        if letter == generated_side:
            picked_generated += 1
        if letter == "A":
            picked_a += 1
        p_a = float(row["p_a"])
        p_gen_total += p_a if human_is_b else 1.0 - p_a

    scored = len(votes)
    return {
        "n_scored": n,
        "n_unique_pairs": len({_pair_key(r) for r in rows}),
        "judge_accuracy": (scored - picked_generated) / scored if scored else None,
        "gen_win_rate": picked_generated / scored if scored else None,
        "p_gen_mean": p_gen_total / scored if scored else None,
        "a_rate": picked_a / scored if scored else None,
        "n_hard_fail": hard_fail,
    }


def discover(run_root: Path, *, prefix: str) -> dict[str, dict[str, list[dict]]]:
    """``{cell: {checkpoint: rows}}`` for every single-token cell under ``run_root``.

    Cells are grouped by the ``raw/<checkpoint>/sweep`` they sit under. Only single-token
    cells are admitted, and the style is taken from the ROWS (via ``_row_prompt_style``)
    rather than the directory name, so a full-schema cell filed under a single_token path
    is rejected instead of silently averaged in.
    """
    found: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for sweep_root in sorted(run_root.glob(f"raw/{prefix}*/sweep")):
        checkpoint = sweep_root.parent.name
        for jsonl in find_reward_files(sweep_root):
            cell, _mode, path_style = cell_mode_style(jsonl, sweep_root)
            if path_style != "single_token":
                continue
            rows = load_rows(jsonl)
            if not rows:
                continue
            _row_prompt_style(rows, cell=cell, source=jsonl, path_style=path_style)
            found[cell][checkpoint].extend(rows)
    return {cell: dict(by_checkpoint) for cell, by_checkpoint in found.items()}


def _check_pair_source(run_root: Path, checkpoint: str, cell: str, *, prefix: str,
                       pairs_tag: int) -> None:
    """The cell must have scored the parquet belonging to ITS checkpoint.

    A cell that scored a neighbouring step is invisible downstream: the row count, the
    metrics and the curve all look normal, only the x position is a lie.
    """
    meta_path = run_root / "raw" / checkpoint / "sweep" / cell / "off" / "single_token" \
        / "run_metadata.json"
    if not meta_path.is_file():
        raise ValueError(f"missing run_metadata.json for {cell} at {checkpoint}: {meta_path}")
    pair_source = json.loads(meta_path.read_text()).get("pair_source")
    if pair_source is None:
        raise ValueError(f"{meta_path} records no pair_source")
    expected = f"gen_{checkpoint}_{pairs_tag}.parquet"
    if Path(pair_source).name != expected:
        raise ValueError(
            f"{cell} at {checkpoint} scored {Path(pair_source).name!r}, expected "
            f"{expected!r} -- this cell is filed under the wrong checkpoint"
        )


def build(run_root: Path, *, prefix: str = DEFAULT_PREFIX,
          expected_steps: tuple[int, ...] = DEFAULT_STEPS,
          expected_pairs: int = DEFAULT_PAIRS,
          pairs_tag: int = DEFAULT_PAIRS) -> dict[str, list[dict]]:
    """``{cell: [summary row per checkpoint, ordered by step]}``, or raise."""
    discovered = discover(run_root, prefix=prefix)
    if not discovered:
        raise ValueError(f"no single-token cells found under {run_root}/raw/{prefix}*/sweep")

    tables: dict[str, list[dict]] = {}
    for cell, by_checkpoint in sorted(discovered.items()):
        rows_out = []
        for checkpoint, rows in by_checkpoint.items():
            _check_pair_source(run_root, checkpoint, cell, prefix=prefix, pairs_tag=pairs_tag)
            summary = summarize(rows)
            if summary["n_scored"] != expected_pairs or summary["n_unique_pairs"] != expected_pairs:
                raise ValueError(
                    f"{cell} at {checkpoint}: expected {expected_pairs} rows and "
                    f"{expected_pairs} unique pairs, got {summary['n_scored']} and "
                    f"{summary['n_unique_pairs']}"
                )
            rows_out.append({
                "checkpoint": checkpoint,
                "step": step_of(checkpoint, prefix=prefix),
                **summary,
            })

        # A chain that died mid-trajectory leaves the later steps missing, and a short
        # line on the plot is indistinguishable from the run having ended there.
        steps = tuple(sorted(row["step"] for row in rows_out))
        if expected_steps is not None and steps != tuple(sorted(expected_steps)):
            missing = sorted(set(expected_steps) - set(steps))
            extra = sorted(set(steps) - set(expected_steps))
            raise ValueError(
                f"{cell} does not cover the expected checkpoints: missing={missing} "
                f"extra={extra}. A truncated chain must not be summarized as a short curve."
            )

        rows_out.sort(key=lambda row: row["step"])
        tables[cell] = rows_out
    return tables


def write_tables(tables: dict[str, list[dict]], out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for cell, rows in tables.items():
        path = out_dir / f"summary_{cell}.csv"
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        written.append(path)
    return written


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-root", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--prefix", default=DEFAULT_PREFIX)
    ap.add_argument("--pairs", type=int, default=DEFAULT_PAIRS,
                    help="Rows expected per checkpoint (also the pair-set filename tag).")
    ap.add_argument("--steps", default=" ".join(str(s) for s in DEFAULT_STEPS),
                    help="Expected checkpoints. Empty string disables the coverage check.")
    args = ap.parse_args(argv)

    expected_steps = tuple(int(s) for s in args.steps.split()) if args.steps.strip() else None
    try:
        tables = build(args.run_root, prefix=args.prefix, expected_steps=expected_steps,
                       expected_pairs=args.pairs, pairs_tag=args.pairs)
    except ValueError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2

    for path in write_tables(tables, args.out_dir):
        print(f"wrote {path}")
    for cell, rows in tables.items():
        first, last = rows[0], rows[-1]
        print(f"{cell:16s} step{first['step']:>4} win={first['gen_win_rate']:.3f} "
              f"p_gen={first['p_gen_mean']:.3f}  ->  step{last['step']:>4} "
              f"win={last['gen_win_rate']:.3f} p_gen={last['p_gen_mean']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
