"""Unit tests for deterministic judge-data slicing."""

import json
import pathlib
import subprocess
import sys

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


# --- scripts/check_slice_bounds.py -----------------------------------------------------
# The tests above prove the slice arithmetic. These prove the checker that verifies a RUN
# actually used a sliced file -- the failure no unit test can otherwise see, because a
# dropped TRAIN_FILE silently falls back to the full split while every metric looks fine.

CHECKER = str(pathlib.Path(__file__).resolve().parents[1] / "scripts" / "check_slice_bounds.py")


def _run_checker(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, CHECKER, *args], capture_output=True, text=True)


def _write_dump(path: pathlib.Path, rows: list[dict]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    with open(path / "reward-1-1.jsonl", "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _keys_in(lo: float, hi: float, n: int) -> list[tuple[str, str, int]]:
    out = []
    for i in range(100000):
        args = (f"u{i}", f"p{i}", i % 3)
        if lo <= slice_fraction(*args) < hi:
            out.append(args)
            if len(out) == n:
                return out
    raise AssertionError(f"could not find {n} keys in [{lo},{hi})")


def test_checker_accepts_a_parquet_fully_inside_the_slice(tmp_path):
    rows = [{"extra_info": {"user_id": u, "post_id": p, "target_idx": t}}
            for u, p, t in _keys_in(0.1, 0.2, 12)]
    src = tmp_path / "in.parquet"
    pd.DataFrame(rows).to_parquet(src, index=False)
    proc = _run_checker("--rows", str(src), "--lo", "0.1", "--hi", "0.2")
    assert proc.returncode == 0, proc.stderr
    assert "OK: all 12 turns" in proc.stdout


def test_checker_rejects_a_single_out_of_bounds_row(tmp_path):
    rows = [{"extra_info": {"user_id": u, "post_id": p, "target_idx": t}}
            for u, p, t in _keys_in(0.1, 0.2, 5)]
    stray = _keys_in(0.0, 0.1, 1)[0]          # a turn from the JUDGE's slice
    rows.append({"extra_info": {"user_id": stray[0], "post_id": stray[1],
                                "target_idx": stray[2]}})
    src = tmp_path / "mixed.parquet"
    pd.DataFrame(rows).to_parquet(src, index=False)
    proc = _run_checker("--rows", str(src), "--lo", "0.1", "--hi", "0.2")
    assert proc.returncode == 1
    assert "1 turn(s) outside" in proc.stderr


def test_split_filter_is_load_bearing_on_a_reward_dump(tmp_path):
    """The dump mixes sliced train rows with deliberately-unsliced val rows.

    Without --split the val rows fail the bounds check even though the run is correct, so
    the flag is not cosmetic: dropping it would make this check unusable post-run.
    """
    rows = [{"split": "train", "user_id": u, "post_id": p, "target_idx": t}
            for u, p, t in _keys_in(0.1, 0.2, 8)]
    rows += [{"split": "val", "user_id": u, "post_id": p, "target_idx": t}
             for u, p, t in _keys_in(0.0, 0.1, 4)]
    dump = tmp_path / "reward_dump"
    _write_dump(dump, rows)

    ok = _run_checker("--rows", str(dump), "--split", "train", "--lo", "0.1", "--hi", "0.2")
    assert ok.returncode == 0, ok.stderr
    assert "OK: all 8 turns" in ok.stdout

    unfiltered = _run_checker("--rows", str(dump), "--lo", "0.1", "--hi", "0.2")
    assert unfiltered.returncode == 1
    assert "4 turn(s) outside" in unfiltered.stderr


def test_checker_rejects_inverted_bounds(tmp_path):
    src = tmp_path / "x.parquet"
    pd.DataFrame([{"extra_info": {"user_id": "u", "post_id": "p", "target_idx": 0}}]).to_parquet(
        src, index=False)
    proc = _run_checker("--rows", str(src), "--lo", "0.5", "--hi", "0.2")
    assert proc.returncode != 0
    assert "lo < hi" in proc.stderr
