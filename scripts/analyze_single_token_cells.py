#!/usr/bin/env python
"""Assemble single-token per-call JSONL into one CSV row per cell.

This is the step between the scorer and the table a human reads. Without it a run
produces JSONL and nothing else: ``scripts/single_token_metrics.summarize`` computes every
column the design promises but had no caller, and ``scripts/analyze_judge_sweep.py``
cannot stand in for one --

* it globs ``<cell>/<mode>/reward/*.jsonl`` one level deep, while the single-token writer
  nests a ``<style>`` segment (``<cell>/<mode>/single_token/reward/``), so it finds zero
  rows; and
* it skips any cell absent from ``configs.judge_sweep_cells.SIZE_MAP``, which excludes
  four of the nine matrix cells INCLUDING the reference cell ``judge-9b-graded-step52``.

So this reads the tree the writer actually produces and applies no cell allowlist.

Usage:
  python scripts/analyze_single_token_cells.py \\
      --sweep_root <EVAL_ROOT>/raw/sweep \\
      --out <EVAL_ROOT>/derived/single_token_cells.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.single_token_metrics import summarize  # noqa: E402

PROMPT_STYLE = "single_token"

# cell / prompt_style / thinking_mode identify the row; the rest is summarize()'s output,
# in the order the design lists them, and must match it exactly (checked in build_table).
_IDENTIFIER_COLUMNS = ("cell", "prompt_style", "thinking_mode")
COLUMNS = [
    "cell",
    "prompt_style",
    "thinking_mode",
    "n",
    "scored",
    "hard_fail",
    "accuracy",
    "a_rate",
    "expected_a_rate",
    "a_rate_excess",
    "order_consistency",
    "tie_rate",
    "brier",
    "auc",
    "degenerate",
]


def find_reward_files(sweep_root: Path) -> list[Path]:
    """Every ``reward/*.jsonl`` under the sweep root, at any depth.

    Recursive by necessity: the style segment makes the depth style-dependent, and a
    fixed-depth glob is what produced zero rows in the sweep analyzer. Same pattern as
    ``scripts/eval_rl_generator._rows_from_dump_dir``.
    """
    return sorted(sweep_root.rglob("reward/*.jsonl"))


def cell_mode_style(jsonl_path: Path, sweep_root: Path) -> tuple[str, str, str | None]:
    """Split ``<cell>/<mode>[/<style>]/reward/<file>.jsonl`` into its parts.

    Raises on any other shape rather than guessing. Passing the run root instead of the
    sweep root would otherwise attribute every cell to the literal directory ``raw`` --
    one plausible-looking table row covering the whole matrix.
    """
    rel = jsonl_path.relative_to(sweep_root).parts
    if len(rel) == 4:
        cell, mode, _reward, _name = rel
        style = None
    elif len(rel) == 5:
        cell, mode, style, _reward, _name = rel
    else:
        raise ValueError(
            f"{jsonl_path} is not <cell>/<mode>[/<style>]/reward/<file>.jsonl relative to "
            f"{sweep_root} (got {len(rel)} path segments: {list(rel)}). Pass the SWEEP "
            "root (…/raw/sweep), not the run root."
        )
    return cell, mode, style


def load_rows(jsonl_path: Path) -> list[dict]:
    rows = []
    for line in jsonl_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return rows


def _pair_key(row: dict) -> str:
    """Stable per-pair key. Same fallback as ``analyze_judge_sweep.per_call_features``.

    The single-token scorer carries the real ``pair_id`` through; the fallback exists for
    a pair set that has none, so that two rows of the same pair still group together.
    """
    pair_id = row.get("pair_id")
    if pair_id is not None:
        return str(pair_id)
    return f'{row.get("user_id")}::{row.get("post_id")}::{row.get("target_idx")}'


def metric_row(row: dict, *, source: Path) -> dict:
    """Map one dumped scorer row onto the keys ``summarize`` reads.

    Explicit rather than passing the dump row straight through: the mapping between
    ``eval.single_token_judge._DUMP_KEYS`` and what the metrics module indexes is the
    join this script exists to make, so it should be visible in one place and fail loudly
    when a field is missing.
    """
    missing = [k for k in ("human_is_b", "letter", "p_a", "hard_fail") if k not in row]
    if missing:
        raise ValueError(f"row in {source} is missing single-token field(s) {missing}")
    return {
        "pair_id": _pair_key(row),
        "human_is_b": bool(row["human_is_b"]),
        "letter": row["letter"],
        "p_a": row["p_a"],
        "hard_fail": bool(row["hard_fail"]),
    }


def _row_prompt_style(rows: list[dict], *, cell: str, source: Path, path_style: str | None) -> str:
    """The style the ROWS report, cross-checked against the directory they sit in.

    A cell directory named ``single_token`` holding full-schema rows is the "wrong
    protocol ran under the right label" failure, and it is invisible in the numbers.
    """
    styles = sorted({str(r.get("judge_prompt_style")) for r in rows})
    if styles != [PROMPT_STYLE]:
        raise ValueError(
            f"{source} holds judge_prompt_style {styles}; this analyzer reads "
            f"{PROMPT_STYLE!r} cells only (cell {cell!r})"
        )
    if path_style is not None and path_style != PROMPT_STYLE:
        raise ValueError(
            f"{source} sits under a {path_style!r} directory but its rows report "
            f"{PROMPT_STYLE!r} (cell {cell!r})"
        )
    return PROMPT_STYLE


def build_table(sweep_root: Path) -> list[dict]:
    """One summary row per (cell, thinking mode) found under ``sweep_root``."""
    grouped: dict[tuple[str, str], list[dict]] = {}
    styles: dict[tuple[str, str], str] = {}
    for jsonl_path in find_reward_files(sweep_root):
        rows = load_rows(jsonl_path)
        if not rows:
            continue
        cell, mode, path_style = cell_mode_style(jsonl_path, sweep_root)
        key = (cell, mode)
        styles[key] = _row_prompt_style(
            rows, cell=cell, source=jsonl_path, path_style=path_style
        )
        grouped.setdefault(key, []).extend(
            metric_row(r, source=jsonl_path) for r in rows
        )

    table = []
    for (cell, mode), rows in sorted(grouped.items()):
        summary = summarize(rows)
        # Loud on drift in either direction: a metric added to summarize() that no column
        # carries would vanish from the table, and a column summarize() stopped emitting
        # would silently become an empty cell in every row.
        if sorted(summary) != sorted(set(COLUMNS) - set(_IDENTIFIER_COLUMNS)):
            raise ValueError(
                f"summarize() keys {sorted(summary)} do not match the CSV metric columns "
                f"{sorted(set(COLUMNS) - set(_IDENTIFIER_COLUMNS))}"
            )
        table.append(
            {
                "cell": cell,
                "prompt_style": styles[(cell, mode)],
                "thinking_mode": mode,
                **summary,
            }
        )
    return table


def write_csv(table: list[dict], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        for row in table:
            writer.writerow({k: row.get(k) for k in COLUMNS})


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep_root", type=Path, required=True,
                        help="…/raw/sweep for the run (NOT the run root)")
    parser.add_argument("--out", type=Path, required=True, help="CSV to write")
    args = parser.parse_args(argv)

    if not args.sweep_root.is_dir():
        raise SystemExit(f"sweep root does not exist: {args.sweep_root}")

    table = build_table(args.sweep_root)
    if not table:
        # The failure this script was written for: a run that produced JSONL somewhere the
        # analyzer never looked reported an empty table instead of an error.
        raise SystemExit(
            f"no reward/*.jsonl rows found under {args.sweep_root}; nothing to summarize"
        )

    write_csv(table, args.out)
    for row in table:
        print(
            f"[single-token] {row['cell']}/{row['thinking_mode']}: n={row['n']} "
            f"scored={row['scored']} acc={row['accuracy']:.3f} "
            f"hard_fail={row['hard_fail']:.3f} degenerate={row['degenerate']}",
            flush=True,
        )
    print(f"[single-token] wrote {len(table)} cell(s) -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
