"""The sweep cell's default pair set must be the one the published chart uses.

judge_sweep_cell.sh used to default PAIRS to results/2026-07-08-judge-sweep's
prism_heldout_880.parquet. Its fake turns come from the Qwen3-8B SFT with the stop-token
masking bug: 36% run past 5x the paired human turn (gen mean 2444 chars against 67), so a
judge scores largely by length -- on the length-matched subset a zero-shot 9B sits at 0.500,
and a length-only rule scores 0.561 against the judge's 0.564.

Four rating_only cells (jobs 23558-23561) were scored against it purely by taking that
default, produced 0.691-0.728, and were discarded. Nothing in the pipeline objected: the
per-cell summariser only sees one sweep root at a time, so its pair_source check cannot fire.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CELL = ROOT / "scripts" / "slurm" / "judge_sweep_cell.sh"

CLEAN = "gen_9b-full5ep-step0_880.parquet"
CONTAMINATED = "prism_heldout_880.parquet"


def _default_pairs_line() -> str:
    for line in CELL.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("PAIRS=${PAIRS:-"):
            return stripped
    raise AssertionError("no PAIRS default found in judge_sweep_cell.sh")


def test_the_default_is_the_clean_pair_set():
    assert CLEAN in _default_pairs_line()


def test_the_contaminated_set_is_not_the_default():
    """Not merely 'a good set is present' -- the specific bad one must be gone from the
    default, since taking it silently is the whole failure mode."""
    assert CONTAMINATED not in _default_pairs_line()


def test_the_hazard_is_explained_where_someone_would_change_it():
    """A future edit back to the July set should have to read why not."""
    text = CELL.read_text()
    assert CONTAMINATED in text, "the rejected set should still be named in the comment"
    for token in ("stop-token", "length"):
        assert token in text.lower()


def test_a_missing_pair_set_still_fails_loudly():
    """The existence guard must survive the default change; a silently-empty run would be
    worse than a wrong one."""
    assert 'ERROR: pair-set not found' in CELL.read_text()
