"""Guards for the single-token trajectory summaries.

Every case here fails silently in production: the numbers stay in range and the curve
stays readable, so nothing downstream raises. Ordered roughly by how invisible the bug is.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_single_token_trajectory import (  # noqa: E402
    build,
    discover,
    step_of,
    summarize,
)

PREFIX = "9b-train10pct-step"


def _row(*, human_is_b: bool, letter: str | None, p_a: float | None, pair_id: str,
         hard_fail: bool = False, style: str = "single_token") -> dict:
    return {
        "judge_prompt_style": style,
        "pair_id": pair_id,
        "human_is_b": human_is_b,
        "letter": letter,
        "p_a": p_a,
        "hard_fail": hard_fail,
    }


def _cell(run_root: Path, checkpoint: str, cell: str, rows: list[dict], *,
          style: str = "single_token", pair_source: str | None = None) -> Path:
    mode_dir = run_root / "raw" / checkpoint / "sweep" / cell / "off" / style
    reward = mode_dir / "reward"
    reward.mkdir(parents=True, exist_ok=True)
    reward.joinpath("reward-1.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows)
    )
    mode_dir.joinpath("run_metadata.json").write_text(json.dumps({
        "thinking_mode": "off",
        "prompt_style": style,
        "pair_source": pair_source or f"/somewhere/gen_{checkpoint}_2.parquet",
    }))
    return reward


def _two_good_rows(tag: str) -> list[dict]:
    """One pair the judge got right, one it got wrong."""
    return [
        # human is B, so generated is A; judge says B -> picked the human.
        _row(human_is_b=True, letter="B", p_a=0.2, pair_id=f"{tag}-1"),
        # human is A, so generated is B; judge says B -> picked the generated.
        _row(human_is_b=False, letter="B", p_a=0.1, pair_id=f"{tag}-2"),
    ]


# 1. Ordering -------------------------------------------------------------------------

def test_steps_are_numeric_so_the_curve_is_not_scrambled(tmp_path: Path) -> None:
    # Lexicographically these order 0, 108, 12, 120, 24 -- a plausible-looking but wrong
    # curve. Numerically they order 0, 12, 24, 108, 120.
    steps = (0, 12, 24, 108, 120)
    for step in steps:
        _cell(tmp_path, f"{PREFIX}{step}", "j", _two_good_rows(f"s{step}"))

    tables = build(tmp_path, expected_steps=steps, expected_pairs=2, pairs_tag=2)
    assert [row["step"] for row in tables["j"]] == [0, 12, 24, 108, 120]


def test_step_of_parses_the_trailing_integer() -> None:
    assert step_of(f"{PREFIX}108", prefix=PREFIX) == 108
    assert step_of(f"{PREFIX}0", prefix=PREFIX) == 0
    with pytest.raises(ValueError):
        step_of("something-else", prefix=PREFIX)


# 2. Truncated chain ------------------------------------------------------------------

def test_a_truncated_chain_is_refused_not_summarized_short(tmp_path: Path) -> None:
    # The steps run as an afterok chain, so a death at step 24 leaves 36 and 48 missing.
    # Plotted, that is just a shorter line -- it reads as the run ending, not as a failure.
    for step in (0, 12, 24):
        _cell(tmp_path, f"{PREFIX}{step}", "j", _two_good_rows(f"s{step}"))

    with pytest.raises(ValueError, match="does not cover the expected checkpoints"):
        build(tmp_path, expected_steps=(0, 12, 24, 36, 48), expected_pairs=2, pairs_tag=2)


# 3. Orientation ----------------------------------------------------------------------

def test_p_gen_follows_which_side_the_generated_response_is_on() -> None:
    # Generated is A (human is B): p_gen is p_a itself.
    got = summarize([_row(human_is_b=True, letter="A", p_a=0.9, pair_id="x")])
    assert got["p_gen_mean"] == pytest.approx(0.9)
    assert got["gen_win_rate"] == 1.0

    # Generated is B (human is A): p_gen is the complement.
    got = summarize([_row(human_is_b=False, letter="A", p_a=0.9, pair_id="x")])
    assert got["p_gen_mean"] == pytest.approx(0.1)
    assert got["gen_win_rate"] == 0.0


@pytest.mark.parametrize("human_is_b", [True, False])
@pytest.mark.parametrize("p_a", [0.05, 0.3, 0.7, 0.95])
def test_p_gen_and_win_rate_never_disagree(human_is_b: bool, p_a: float) -> None:
    """The verdict is the argmax of the renormalised A/B mass, so the graded and the hard
    metric must land on the same side of 0.5. A flip in EITHER one breaks this; testing
    them one at a time would not catch it.
    """
    letter = "A" if p_a > 0.5 else "B"
    got = summarize([_row(human_is_b=human_is_b, letter=letter, p_a=p_a, pair_id="x")])
    assert (got["p_gen_mean"] > 0.5) == (got["gen_win_rate"] == 1.0)


def test_accuracy_and_win_rate_are_complements() -> None:
    got = summarize(_two_good_rows("t"))
    assert got["judge_accuracy"] == pytest.approx(0.5)
    assert got["gen_win_rate"] == pytest.approx(0.5)
    assert got["judge_accuracy"] + got["gen_win_rate"] == pytest.approx(1.0)


# 4. Style purity ---------------------------------------------------------------------

def test_a_full_schema_cell_in_the_same_tree_is_not_ingested(tmp_path: Path) -> None:
    checkpoint = f"{PREFIX}0"
    _cell(tmp_path, checkpoint, "st-judge", _two_good_rows("a"))
    # A full-schema cell sits at <cell>/<mode>/reward -- one level shallower.
    full = tmp_path / "raw" / checkpoint / "sweep" / "full-judge" / "on" / "reward"
    full.mkdir(parents=True)
    full.joinpath("reward-1.jsonl").write_text(
        json.dumps({"rating_gt_first": 6, "generated_is_b": True}) + "\n"
    )

    assert sorted(discover(tmp_path, prefix=PREFIX)) == ["st-judge"]


def test_full_schema_rows_filed_under_a_single_token_path_are_refused(tmp_path: Path) -> None:
    # The directory says single_token but the rows say otherwise: the wrong protocol ran
    # under the right label, which is invisible in the metrics.
    rows = [_row(human_is_b=True, letter="B", p_a=0.2, pair_id="a", style="full")]
    _cell(tmp_path, f"{PREFIX}0", "j", rows)

    with pytest.raises(ValueError, match="judge_prompt_style"):
        discover(tmp_path, prefix=PREFIX)


# 5. Hard fails and coverage ----------------------------------------------------------

def test_a_hard_fail_leaves_both_denominators(tmp_path: Path) -> None:
    rows = [
        _row(human_is_b=False, letter="B", p_a=0.1, pair_id="a"),   # generator wins
        _row(human_is_b=True, letter=None, p_a=None, pair_id="b", hard_fail=True),
    ]
    got = summarize(rows)
    assert got["n_scored"] == 2
    assert got["n_hard_fail"] == 1
    # One vote, and the generator won it. A failure is not a loss for the generator.
    assert got["gen_win_rate"] == 1.0
    assert got["p_gen_mean"] == pytest.approx(0.9)


def test_a_rate_excess_subtracts_the_samples_own_imbalance() -> None:
    # Both rows have the human in slot B, so an unbiased judge answers B twice and A never.
    # A judge that does exactly that is NOT biased, even though its a_rate is 0.0.
    rows = [
        _row(human_is_b=True, letter="B", p_a=0.1, pair_id="a"),
        _row(human_is_b=True, letter="B", p_a=0.2, pair_id="b"),
    ]
    got = summarize(rows)
    assert got["a_rate"] == 0.0
    assert got["expected_a_rate"] == 0.0
    assert got["a_rate_excess"] == 0.0

    # A judge answering A on that same sample is maximally position-biased.
    biased = summarize([
        _row(human_is_b=True, letter="A", p_a=0.9, pair_id="a"),
        _row(human_is_b=True, letter="A", p_a=0.8, pair_id="b"),
    ])
    assert biased["a_rate_excess"] == 1.0


def test_every_row_hard_failing_is_refused() -> None:
    # Otherwise the cell divides by zero, or silently reports None metrics as a data point.
    with pytest.raises(ValueError, match="every row hard-failed"):
        summarize([_row(human_is_b=True, letter=None, p_a=None, pair_id="a",
                        hard_fail=True)])


def test_a_short_cell_is_refused(tmp_path: Path) -> None:
    _cell(tmp_path, f"{PREFIX}0", "j", _two_good_rows("a")[:1])
    with pytest.raises(ValueError, match="expected 2 rows"):
        build(tmp_path, expected_steps=(0,), expected_pairs=2, pairs_tag=2)


def test_duplicated_pairs_are_refused(tmp_path: Path) -> None:
    # Two rows, one pair -- a re-scored pair inflates n_scored without adding coverage.
    rows = [
        _row(human_is_b=True, letter="B", p_a=0.2, pair_id="same"),
        _row(human_is_b=True, letter="B", p_a=0.2, pair_id="same"),
    ]
    _cell(tmp_path, f"{PREFIX}0", "j", rows)
    with pytest.raises(ValueError, match="unique pairs"):
        build(tmp_path, expected_steps=(0,), expected_pairs=2, pairs_tag=2)


def test_a_cell_that_scored_another_checkpoints_pairs_is_refused(tmp_path: Path) -> None:
    # Right row count, sane metrics, wrong x position. Nothing else catches this.
    _cell(tmp_path, f"{PREFIX}12", "j", _two_good_rows("a"),
          pair_source=f"/somewhere/gen_{PREFIX}24_2.parquet")
    with pytest.raises(ValueError, match="filed under the wrong checkpoint"):
        build(tmp_path, expected_steps=(12,), expected_pairs=2, pairs_tag=2)


def test_a_cell_without_run_metadata_is_refused(tmp_path: Path) -> None:
    checkpoint = f"{PREFIX}0"
    reward = _cell(tmp_path, checkpoint, "j", _two_good_rows("a"))
    (reward.parent / "run_metadata.json").unlink()
    with pytest.raises(ValueError, match="missing run_metadata.json"):
        build(tmp_path, expected_steps=(0,), expected_pairs=2, pairs_tag=2)


def test_an_empty_run_root_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no single-token cells"):
        build(tmp_path, expected_steps=(0,), expected_pairs=2, pairs_tag=2)
