"""Unit tests for the judge pair-set builder.

The synthetic fixtures mirror the real pickle structure emitted by
``eval/generate_trained.py`` (dict keyed by ``user_id`` -> ``test_targets`` ->
``generations`` list of parsed dicts whose raw text lives in ``raw_completion``).
A second fixture exercises the defensive fallback keys
(``test_results`` / ``outputs`` / ``text``) that the plan calls for.
"""

import pickle

import pandas as pd

from scripts.build_judge_pairs import build_pairs

EXPECTED_COLUMNS = [
    "pair_id",
    "user_id",
    "post_id",
    "target_idx",
    "user_history",
    "context",
    "persona",
    "human",
    "generated",
]


def _make_real(tmp_path):
    """Fixture matching the real generate_trained.py output structure."""
    infer = {
        "u1": {
            "user_id": "u1",
            "test_targets": [
                {
                    "target_idx": 0,
                    "user_id": "u1",
                    "post_id": "p1",
                    "generations": [
                        {
                            "reasoning": "r",
                            "response": "hi there",
                            "raw_completion": "<reasoning>r</reasoning>[HUMAN]: hi there",
                        }
                    ],
                }
            ],
        }
    }
    (tmp_path / "inf.pkl").write_bytes(pickle.dumps(infer))
    pd.DataFrame(
        [
            {
                "reward_model": {"ground_truth": "hello"},
                "extra_info": {
                    "user_id": "u1",
                    "post_id": "p1",
                    "target_idx": 0,
                    "user_history": "h",
                    "context": "c",
                    "persona": "",
                },
            }
        ]
    ).to_parquet(tmp_path / "test.parquet")
    return str(tmp_path / "inf.pkl"), str(tmp_path / "test.parquet")


def _make_fallback(tmp_path):
    """Fixture using defensive fallback keys: test_results / outputs / str text."""
    infer = {
        "u1": {
            "test_results": [
                {
                    "target_idx": 0,
                    "user_id": "u1",
                    "post_id": "p1",
                    "outputs": ["<reasoning>r</reasoning>[HUMAN]: hi there"],
                }
            ]
        }
    }
    (tmp_path / "inf.pkl").write_bytes(pickle.dumps(infer))
    pd.DataFrame(
        [
            {
                "reward_model": {"ground_truth": "hello"},
                "extra_info": {
                    "user_id": "u1",
                    "post_id": "p1",
                    "target_idx": 0,
                    "user_history": "h",
                    "context": "c",
                    "persona": "",
                },
            }
        ]
    ).to_parquet(tmp_path / "test.parquet")
    return str(tmp_path / "inf.pkl"), str(tmp_path / "test.parquet")


def test_cols_and_strip(tmp_path):
    df, meta = build_pairs(*_make_real(tmp_path))
    assert list(df.columns) == EXPECTED_COLUMNS
    assert df.iloc[0]["human"] == "hello"
    assert df.iloc[0]["generated"] == "hi there"
    assert "<reasoning>" not in df.iloc[0]["generated"]
    assert df.iloc[0]["pair_id"] == "u1::p1::0"


def test_flags_exact_matches(tmp_path):
    _, meta = build_pairs(*_make_real(tmp_path))
    assert "exact_match_count" in meta
    assert "exact_match_frac" in meta
    assert meta["exact_match_count"] == 0
    assert meta["n_pairs"] == 1


def test_fallback_keys(tmp_path):
    df, meta = build_pairs(*_make_fallback(tmp_path))
    assert list(df.columns) == EXPECTED_COLUMNS
    assert df.iloc[0]["generated"] == "hi there"
    assert "<reasoning>" not in df.iloc[0]["generated"]


def test_missing_inference_row_raises(tmp_path):
    infer = {"u1": {"test_targets": []}}
    (tmp_path / "inf.pkl").write_bytes(pickle.dumps(infer))
    pd.DataFrame(
        [
            {
                "reward_model": {"ground_truth": "hello"},
                "extra_info": {
                    "user_id": "u1",
                    "post_id": "p1",
                    "target_idx": 0,
                    "user_history": "h",
                    "context": "c",
                    "persona": "",
                },
            }
        ]
    ).to_parquet(tmp_path / "test.parquet")
    try:
        build_pairs(str(tmp_path / "inf.pkl"), str(tmp_path / "test.parquet"))
        raised = False
    except (AssertionError, KeyError, ValueError):
        raised = True
    assert raised, "expected an error when a heldout row has no matching generation"


def _make_with_raw(tmp_path, raw_completion):
    """Build a single-row fixture with a given raw_completion string."""
    infer = {
        "u1": {
            "test_targets": [
                {
                    "target_idx": 0,
                    "user_id": "u1",
                    "post_id": "p1",
                    "generations": [{"raw_completion": raw_completion}],
                }
            ]
        }
    }
    (tmp_path / "inf.pkl").write_bytes(pickle.dumps(infer))
    pd.DataFrame(
        [
            {
                "reward_model": {"ground_truth": "hello"},
                "extra_info": {
                    "user_id": "u1", "post_id": "p1", "target_idx": 0,
                    "user_history": "h", "context": "c", "persona": "",
                },
            }
        ]
    ).to_parquet(tmp_path / "test.parquet")
    return str(tmp_path / "inf.pkl"), str(tmp_path / "test.parquet")


def test_strips_stray_trailing_reasoning_tag(tmp_path):
    # Real bug: model appends a stray </reasoning> after an otherwise-complete turn,
    # which the primary parse leaves attached. It must be stripped, not asserted-on.
    df, meta = build_pairs(*_make_with_raw(tmp_path, "<reasoning>r</reasoning>[HUMAN]: hi there</reasoning>"))
    assert df.iloc[0]["generated"] == "hi there"
    assert "reasoning" not in df.iloc[0]["generated"]
    assert meta["reasoning_residue_stripped"] == 1


def test_strips_leaked_second_reasoning_block(tmp_path):
    # A second full <reasoning>...</reasoning> block after the response must be removed
    # entirely (tags AND the leaked reasoning text).
    df, meta = build_pairs(
        *_make_with_raw(tmp_path, "<reasoning>r1</reasoning>[HUMAN]: hi there<reasoning>leaked r2</reasoning>")
    )
    assert df.iloc[0]["generated"] == "hi there"
    assert "reasoning" not in df.iloc[0]["generated"]
    assert "leaked" not in df.iloc[0]["generated"]
    assert meta["reasoning_residue_stripped"] == 1


def test_no_residue_stripped_on_clean_rows(tmp_path):
    _, meta = build_pairs(*_make_real(tmp_path))
    assert meta["reasoning_residue_stripped"] == 0


def test_exact_match_counted_not_dropped(tmp_path):
    infer = {
        "u1": {
            "test_targets": [
                {
                    "target_idx": 0,
                    "user_id": "u1",
                    "post_id": "p1",
                    "generations": [{"raw_completion": "[HUMAN]: hello"}],
                }
            ]
        }
    }
    (tmp_path / "inf.pkl").write_bytes(pickle.dumps(infer))
    pd.DataFrame(
        [
            {
                "reward_model": {"ground_truth": "hello"},
                "extra_info": {
                    "user_id": "u1",
                    "post_id": "p1",
                    "target_idx": 0,
                    "user_history": "h",
                    "context": "c",
                    "persona": "",
                },
            }
        ]
    ).to_parquet(tmp_path / "test.parquet")
    df, meta = build_pairs(str(tmp_path / "inf.pkl"), str(tmp_path / "test.parquet"))
    assert len(df) == 1  # not dropped
    assert meta["exact_match_count"] == 1
    assert meta["exact_match_frac"] == 1.0
