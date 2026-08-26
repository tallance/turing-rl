"""Unit tests for deterministic judge-data slicing."""

import pandas as pd
import pytest

from data.judge.slice import in_slice, select_slice, slice_fraction, slice_key


def _rows(n: int) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "data_source": "prism",
                "prompt": [{"role": "user", "content": f"p{i}"}],
                "reward_model": {"ground_truth": f"gt{i}"},
                "extra_info": {"user_id": f"u{i % 7}", "post_id": f"p{i}", "target_idx": i % 3},
            }
            for i in range(n)
        ]
    )


def test_slice_key_is_the_documented_triple():
    assert slice_key("u1", "p2", 3) == "u1::p2::3"


def test_slice_fraction_is_deterministic():
    assert slice_fraction("u1", "p2", 3) == slice_fraction("u1", "p2", 3)


def test_slice_fraction_is_in_unit_interval():
    for i in range(500):
        u = slice_fraction(f"u{i}", f"p{i}", i)
        assert 0.0 <= u < 1.0


def test_adjacent_slices_are_disjoint():
    for i in range(500):
        args = (f"u{i}", f"p{i}", i)
        first = in_slice(*args, lo=0.0, hi=0.1)
        second = in_slice(*args, lo=0.1, hi=0.2)
        assert not (first and second)


def test_slice_covers_roughly_the_requested_fraction():
    hits = sum(1 for i in range(4174) if in_slice(f"u{i}", f"p{i}", i, lo=0.0, hi=0.1))
    assert 340 <= hits <= 500


def test_select_slice_is_independent_of_row_order():
    df = _rows(300)
    forward = select_slice(df, lo=0.0, hi=0.25)
    shuffled = select_slice(df.iloc[::-1].reset_index(drop=True), lo=0.0, hi=0.25)
    assert [r["post_id"] for r in forward["extra_info"]] == [
        r["post_id"] for r in shuffled["extra_info"]
    ]


def test_limit_truncates_deterministically():
    df = _rows(300)
    full = select_slice(df, lo=0.0, hi=1.0)
    capped = select_slice(df, lo=0.0, hi=1.0, limit=10)
    assert len(capped) == 10
    assert list(capped["extra_info"]) == list(full["extra_info"])[:10]


def test_bad_bounds_raise():
    with pytest.raises(ValueError):
        in_slice("u", "p", 0, lo=0.5, hi=0.5)
    with pytest.raises(ValueError):
        in_slice("u", "p", 0, lo=-0.1, hi=0.5)


def test_slicing_is_idempotent():
    """scripts/slurm/judge_train_gen.sh slices before generation and the builder slices
    again with the same bounds. If that were not a no-op the builder would drop rows it
    already has generations for, or worse, keep a different set."""
    df = _rows(300)
    once = select_slice(df, lo=0.0, hi=0.1, limit=17)
    twice = select_slice(once, lo=0.0, hi=0.1, limit=17)
    assert list(twice["extra_info"]) == list(once["extra_info"])


def test_non_dict_extra_info_raises():
    df = pd.DataFrame([{"extra_info": "nope"}])
    with pytest.raises(TypeError):
        select_slice(df, lo=0.0, hi=1.0)
