from pathlib import Path

import pandas as pd
import pytest

from scripts.merge_judge_comparison import main, merge_cells

_REAL_CSV = Path(
    "results/2026-08-19-judge-only-rlvr-thinking-off-eval/judge_eval_880.csv"
)


def _existing():
    return pd.DataFrame([
        {"model": "qwen35-9b", "kind": "zero-shot", "thinking_mode": "on", "accuracy": 0.5182},
        {"model": "qwen35-9b", "kind": "zero-shot", "thinking_mode": "off", "accuracy": 0.4477},
    ])


def _new():
    return pd.DataFrame([
        {"model": "qwen35-9b", "kind": "zero-shot", "thinking_mode": "off",
         "prompt_style": "single_token", "accuracy": 0.61},
    ])


def test_every_row_survives_the_merge():
    out = merge_cells([_existing(), _new()])
    assert len(out) == 3


def test_existing_rows_default_to_the_full_prompt_style():
    out = merge_cells([_existing(), _new()])
    full = out[out["prompt_style"] == "full"]
    assert len(full) == 2


def test_duplicate_cell_keys_are_refused():
    """A duplicate key silently overwrites a published number in the final table."""
    with pytest.raises(ValueError, match="duplicate"):
        merge_cells([_new(), _new()])


def test_an_explicit_prompt_style_is_never_relabelled():
    """Merging a CSV that already carries prompt_style twice must not get
    coerced onto "full" -- that would silently relabel a real cell.
    A fillna-based bug (e.g. filling unconditionally instead of only NaNs)
    would pass test_existing_rows_default_to_the_full_prompt_style but wipe
    this row's real label."""
    out = merge_cells([_new()])
    assert out["prompt_style"].tolist() == ["single_token"]


def test_non_colliding_rows_with_different_keys_are_kept_distinct():
    # Same model/kind, different thinking_mode/prompt_style -- must not be
    # flagged as a duplicate by an overly-broad key (e.g. (model, kind) only).
    other_style = _new().assign(thinking_mode="on")
    out = merge_cells([_new(), other_style])
    assert len(out) == 2


def test_merge_is_compatible_with_the_published_comparison_csv():
    """Guards against the merge silently choking on -- or dropping rows from
    -- the real published table this function is meant to extend."""
    if not _REAL_CSV.exists():
        pytest.skip(f"published comparison CSV not present at {_REAL_CSV}")
    published = pd.read_csv(_REAL_CSV)
    out = merge_cells([published])
    assert len(out) == len(published)
    assert (out["prompt_style"] == "full").all()


def test_main_merges_two_csvs_and_writes_the_output(tmp_path):
    existing_path = tmp_path / "existing.csv"
    new_path = tmp_path / "new.csv"
    out_path = tmp_path / "merged.csv"
    _existing().to_csv(existing_path, index=False)
    _new().to_csv(new_path, index=False)

    main(["--csv", str(existing_path), "--csv", str(new_path), "--out", str(out_path)])

    out = pd.read_csv(out_path)
    assert len(out) == 3
    assert (out[out["prompt_style"] == "full"]["model"] == "qwen35-9b").all()


def test_main_refuses_to_write_on_duplicate_keys(tmp_path):
    new_path = tmp_path / "new.csv"
    out_path = tmp_path / "merged.csv"
    _new().to_csv(new_path, index=False)

    with pytest.raises(ValueError, match="duplicate"):
        main(["--csv", str(new_path), "--csv", str(new_path), "--out", str(out_path)])
    assert not out_path.exists()
