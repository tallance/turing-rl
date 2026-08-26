"""Unit tests for the leakage and pair-set identity gates."""

import hashlib
import json

import pandas as pd
import pytest

from scripts.judge_ce_guards import LeakageError, check_no_user_overlap, check_sha256, main


def test_disjoint_users_pass():
    assert check_no_user_overlap({"u1", "u2"}, {"u3"})["overlap"] == []


def test_any_shared_user_raises():
    with pytest.raises(LeakageError, match="u2"):
        check_no_user_overlap({"u1", "u2"}, {"u2", "u3"})


def test_checksum_mismatch_raises(tmp_path):
    f = tmp_path / "pairs.parquet"
    f.write_bytes(b"not the pair set")
    with pytest.raises(ValueError, match="checksum"):
        check_sha256(f, "0" * 64)


def test_checksum_match_passes(tmp_path):
    f = tmp_path / "pairs.parquet"
    f.write_bytes(b"payload")
    check_sha256(f, hashlib.sha256(b"payload").hexdigest())


def _pairs_df(user_ids):
    """A minimal judge-pairs frame: one row per user_id, extra_info shaped like
    build_judge_train_pairs.build_judge_rows output."""
    return pd.DataFrame(
        {
            "extra_info": [
                {"user_id": uid, "post_id": f"p{i}", "target_idx": 0}
                for i, uid in enumerate(user_ids)
            ]
        }
    )


def test_main_writes_split_guard_and_exits_zero_on_clean_split(tmp_path):
    train_path = tmp_path / "train.parquet"
    eval_path = tmp_path / "eval.parquet"
    _pairs_df(["u1", "u2"]).to_parquet(train_path)
    _pairs_df(["u3", "u4"]).to_parquet(eval_path)
    expected_sha = hashlib.sha256(eval_path.read_bytes()).hexdigest()
    out_path = tmp_path / "split_guard.json"

    rc = main(
        [
            "--train-pairs", str(train_path),
            "--eval-pairs", str(eval_path),
            "--expected-sha256", expected_sha,
            "--out", str(out_path),
        ]
    )

    assert rc == 0
    payload = json.loads(out_path.read_text())
    assert payload["status"] == "pass"
    assert payload["overlap"] == []
    assert payload["eval_sha256"] == expected_sha


def test_main_returns_nonzero_on_leaked_user(tmp_path, capsys):
    train_path = tmp_path / "train.parquet"
    eval_path = tmp_path / "eval.parquet"
    _pairs_df(["u1", "u2"]).to_parquet(train_path)
    _pairs_df(["u2", "u3"]).to_parquet(eval_path)
    expected_sha = hashlib.sha256(eval_path.read_bytes()).hexdigest()
    out_path = tmp_path / "split_guard.json"

    rc = main(
        [
            "--train-pairs", str(train_path),
            "--eval-pairs", str(eval_path),
            "--expected-sha256", expected_sha,
            "--out", str(out_path),
        ]
    )

    assert rc != 0
    assert not out_path.exists()
    assert "u2" in capsys.readouterr().err


def test_main_returns_nonzero_on_wrong_eval_file(tmp_path, capsys):
    train_path = tmp_path / "train.parquet"
    eval_path = tmp_path / "eval.parquet"
    _pairs_df(["u1", "u2"]).to_parquet(train_path)
    _pairs_df(["u3", "u4"]).to_parquet(eval_path)
    out_path = tmp_path / "split_guard.json"

    rc = main(
        [
            "--train-pairs", str(train_path),
            "--eval-pairs", str(eval_path),
            "--expected-sha256", "0" * 64,
            "--out", str(out_path),
        ]
    )

    assert rc != 0
    assert not out_path.exists()
    assert "checksum" in capsys.readouterr().err
