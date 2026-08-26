"""Unit tests for the leakage and pair-set identity gates."""

import hashlib
import json

import pandas as pd
import pytest

from scripts.judge_ce_guards import (
    LeakageError,
    check_no_user_overlap,
    check_sha256,
    load_user_ids,
    main,
)


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


def _flat_pairs_df(user_ids):
    """A minimal judge-pairs frame shaped like build_judge_pairs.py's flat COLUMNS
    list: a top-level user_id column and no extra_info at all."""
    return pd.DataFrame(
        {
            "pair_id": [f"pair{i}" for i in range(len(user_ids))],
            "user_id": list(user_ids),
            "post_id": [f"p{i}" for i in range(len(user_ids))],
            "target_idx": [0] * len(user_ids),
        }
    )


def test_load_user_ids_reads_flat_evaluation_pairs_shape(tmp_path):
    # build_judge_pairs.py output has no extra_info column at all -- selecting
    # columns=["extra_info"] out of this shape raises pyarrow.lib.ArrowInvalid, which
    # is why an in-memory DataFrame (no to_parquet round trip) would not catch this.
    f = tmp_path / "eval.parquet"
    _flat_pairs_df(["u1", "u2"]).to_parquet(f)
    assert load_user_ids(f) == {"u1", "u2"}


def test_load_user_ids_reads_nested_training_pairs_shape(tmp_path):
    f = tmp_path / "train.parquet"
    _pairs_df(["u1", "u2"]).to_parquet(f)
    assert load_user_ids(f) == {"u1", "u2"}


def test_load_user_ids_raises_clear_error_when_neither_shape_present(tmp_path):
    f = tmp_path / "junk.parquet"
    pd.DataFrame({"pair_id": ["p1"], "human": ["x"]}).to_parquet(f)
    with pytest.raises(ValueError, match="extra_info.*user_id"):
        load_user_ids(f)


def test_main_handles_the_real_combination_flat_eval_and_nested_train(tmp_path):
    # This is the actual shape mismatch in production: CE training pairs come from
    # build_judge_train_pairs.py (nested extra_info), evaluation pairs come from
    # build_judge_pairs.py (flat user_id). Before the fix, load_user_ids on the flat
    # eval file raised ArrowInvalid and main() reported a confusing FAIL.
    train_path = tmp_path / "train.parquet"
    eval_path = tmp_path / "eval.parquet"
    _pairs_df(["u1", "u2"]).to_parquet(train_path)
    _flat_pairs_df(["u3", "u4"]).to_parquet(eval_path)
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
    assert payload["n_eval_users"] == 2


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
