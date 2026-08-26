"""Merge judge-eval cells into the published comparison table.

Concatenates one or more per-run CSVs into the single table the protocol-switch
decision is read from. Rows written before the single-token prompt style existed
have no ``prompt_style`` column; those default to ``"full"`` without ever
overwriting a row that already declares one (so re-merging a single-token CSV
twice cannot silently relabel it). Refuses to merge if two rows collide on
``(model, kind, thinking_mode, prompt_style)`` rather than silently keeping the
last one -- these keys back a published comparison table, and a duplicate would
corrupt it in a way that looks like a result.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

_KEY_COLUMNS = ["model", "kind", "thinking_mode", "prompt_style"]


def merge_cells(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate ``frames`` into the comparison table.

    Missing ``prompt_style`` values (rows predating the concept) are filled
    with ``"full"``; existing values are left untouched. Raises ``ValueError``
    if any ``(model, kind, thinking_mode, prompt_style)`` key is not unique
    across the merged rows.
    """
    merged = pd.concat(list(frames), ignore_index=True, sort=False)

    if "prompt_style" not in merged.columns:
        merged["prompt_style"] = pd.NA
    merged["prompt_style"] = merged["prompt_style"].fillna("full")

    dupes = merged.duplicated(subset=_KEY_COLUMNS, keep=False)
    if dupes.any():
        keys = merged.loc[dupes, _KEY_COLUMNS].drop_duplicates()
        raise ValueError(
            "duplicate cell keys would overwrite published comparison rows:\n"
            f"{keys.to_string(index=False)}"
        )
    return merged


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv", action="append", required=True, metavar="PATH",
        help="input comparison CSV; repeat to merge more than one",
    )
    parser.add_argument("--out", required=True, help="output path for the merged CSV")
    args = parser.parse_args(argv)

    frames = [pd.read_csv(path) for path in args.csv]
    merged = merge_cells(frames)
    merged.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
