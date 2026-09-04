"""RL-generator eval scorer: sweep-matched directional accuracy.

Reproduces the judge sweep's "did the judge pick the true human turn?" convention
so RL-final eval numbers are directly comparable to the SFT baseline (identical
scoring logic, identical pair set).

Directional accuracy = fraction of valid, non-tie pairs where the judge picked the
TRUE human turn. Generator win-rate = 1 - accuracy on those pairs. Ties (rating==4)
and parse failures (no valid 1-7 rating) are excluded from the denominator.

The authoritative sweep scorer is ``scripts/analyze_judge_sweep.py``
(``per_call_features`` -> ``accuracy``). This module deliberately mirrors that exact
rule so the two never drift (see tests/test_eval_parity.py, which asserts equality
against the real sweep functions):

  * canonical rating = ``rating_gt_first`` if set else ``rating_gen_first``
    (analyze_judge_sweep.py:68-70)
  * rating scale "1=strongly A ... 4=tie ... 7=strongly B": among valid 1-7 non-tie
    ratings, ``rating < 4 -> judge picks side A``, ``rating > 4 -> picks side B``
    (analyze_judge_sweep.py:86-91)
  * ``human_side`` = row value else ``"A" if generated_is_b else "B"``
    (analyze_judge_sweep.py:84). By the reward-path coupling
    (training/grpo/reward.py:718-763) ``rating_gt_first`` is set IFF
    ``generated_is_b`` is True IFF the human/GT turn sits on side A. So when
    ``generated_is_b``/``human_side`` are absent (minimal rows), orientation is
    inferred from which rating field is set — matching that coupling exactly.
  * ``picked_human = int(pick == human_side)``; ties / invalid ratings -> excluded.

Net rule: gt-first pair (human on A) -> correct iff rating < 4; gen-first pair
(human on B) -> correct iff rating > 4.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _canonical_rating(row: dict) -> int | None:
    """Rating for whichever ordering ran (gt-first preferred), coerced to a valid
    1-7 int or None. Mirrors analyze_judge_sweep.per_call_features."""
    rating = row.get("rating_gt_first")
    if rating is None:
        rating = row.get("rating_gen_first")
    if not isinstance(rating, (int, float)) or isinstance(rating, bool):
        return None
    rating = int(rating)
    return rating if 1 <= rating <= 7 else None


def _human_side(row: dict) -> str:
    """Side (A/B) the true human turn sits on.

    Uses ``human_side`` if present, else ``generated_is_b`` (human is A iff the
    generator is B), else falls back to the reward-path coupling: ``rating_gt_first``
    is set IFF the GT/human turn was presented first (side A).
    """
    hs = row.get("human_side")
    if hs in ("A", "B"):
        return hs
    generated_is_b = row.get("generated_is_b")
    if generated_is_b is None:
        # Infer from which rating field is populated (reward.py coupling).
        generated_is_b = row.get("rating_gt_first") is not None
    return "A" if bool(generated_is_b) else "B"


def _picked_human(row: dict) -> int | None:
    """1 if the judge picked the true human turn, 0 if it picked the generated turn,
    None for ties (rating==4) and parse failures (no valid 1-7 rating)."""
    rating = _canonical_rating(row)
    if rating is None or rating == 4:
        return None
    pick = "A" if rating < 4 else "B"
    return int(pick == _human_side(row))


def find_reward_dirs(root, cell: str = "*", mode: str = "*") -> list:
    """Reward dirs under ``root``, for both the flat and prompt-style layouts.

    judge_sweep_cell.sh nests a single_token run one level deeper than a full-schema one:

        raw/<gen_key>/sweep/<cell>/<mode>/reward                 full schema
        raw/<gen_key>/sweep/<cell>/<mode>/<style>/reward         single_token

    Callers used to glob only the first and index the gen_key at a fixed depth, so a
    single-token eval was invisible to them -- reporting "no reward dirs" for a sweep that had
    in fact scored every pair. Matching both, and deriving the gen_key by position relative to
    ``raw`` rather than by depth, keeps them correct under either layout.
    """
    from pathlib import Path

    root = Path(root)
    seen: dict = {}
    for pattern in (f"raw/*/sweep/{cell}/{mode}/reward",
                    f"raw/*/sweep/{cell}/{mode}/*/reward"):
        for p in root.glob(pattern):
            seen[p] = None
    return sorted(seen)


def gen_key_of(reward_dir) -> str:
    """The gen_key owning ``reward_dir``: the path component directly under ``raw``.

    Depth-independent on purpose -- see find_reward_dirs for why a fixed index is wrong.
    """
    from pathlib import Path

    parts = Path(reward_dir).resolve().parts
    try:
        # Last "raw" wins: an absolute results path may contain the word earlier.
        raw_at = len(parts) - 1 - parts[::-1].index("raw")
    except ValueError as exc:
        raise ValueError(f"{reward_dir} is not under a raw/ directory") from exc
    if raw_at + 1 >= len(parts):
        raise ValueError(f"{reward_dir} is not under a raw/ directory")
    return parts[raw_at + 1]


def directional_accuracy(rows: Iterable[dict]) -> dict:
    """Sweep-matched directional accuracy over judge pairs.

    Returns ``{n_total, n_nontie, n_tie, n_parse_error, correct, accuracy,
    gen_win_rate, frac_ties}`` where ``accuracy = correct / n_nontie`` (0.0 when
    ``n_nontie == 0``) and ``gen_win_rate = 1 - accuracy`` on the non-tie pairs.
    Ties (rating==4) and parse failures are excluded from ``n_nontie``.
    """
    rows = list(rows)
    n_total = len(rows)
    correct = 0
    n_nontie = 0
    n_tie = 0
    n_parse_error = 0
    for row in rows:
        rating = _canonical_rating(row)
        if rating is None:
            n_parse_error += 1
            continue
        if rating == 4:
            n_tie += 1
            continue
        n_nontie += 1
        correct += _picked_human(row) or 0
    accuracy = (correct / n_nontie) if n_nontie else 0.0
    return {
        "n_total": n_total,
        "n_nontie": n_nontie,
        "n_tie": n_tie,
        "n_parse_error": n_parse_error,
        "correct": correct,
        "accuracy": accuracy,
        "gen_win_rate": 1.0 - accuracy,
        "frac_ties": (n_tie / n_total) if n_total else 0.0,
    }


# --------------------------------------------------------------------------- #
# Thin CLI (not unit-tested here; validated on the cluster in Task 14).        #
# --------------------------------------------------------------------------- #
def _rows_from_parquet(path: str) -> list[dict]:
    """Load post-scored judge pairs from a parquet with rating_* columns."""
    import pandas as pd  # local import: keep pure scoring free of heavy deps

    df = pd.read_parquet(path)
    return df.to_dict("records")


def _rows_from_dump_dir(dump_dir: str) -> list[dict]:
    """Load reward-layer dump rows from ``<dump_dir>/**/*.jsonl``.

    Mirrors scripts/analyze_judge_sweep.load_cell_rows so the same rating fields
    (rating_gt_first / rating_gen_first / generated_is_b / human_side) are read.
    Recurses so either a cell/mode/reward dir or a raw reward dir works.
    """
    rows: list[dict] = []
    for jl in sorted(Path(dump_dir).rglob("*.jsonl")):
        for line in jl.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Sweep-matched directional accuracy for RL-generator eval."
    )
    parser.add_argument("--pairs", default=None,
                        help="Post-scored judge-pairs parquet (with rating_* columns)")
    parser.add_argument("--dump_dir", default=None,
                        help="Reward-dump dir (reads **/*.jsonl, same fields as the sweep)")
    args = parser.parse_args(argv)

    if not args.pairs and not args.dump_dir:
        parser.error("provide --pairs and/or --dump_dir")

    rows: list[dict] = []
    if args.pairs:
        rows.extend(_rows_from_parquet(args.pairs))
    if args.dump_dir:
        rows.extend(_rows_from_dump_dir(args.dump_dir))

    result = directional_accuracy(rows)
    result["sources"] = {
        "pairs": os.path.abspath(args.pairs) if args.pairs else None,
        "dump_dir": os.path.abspath(args.dump_dir) if args.dump_dir else None,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
