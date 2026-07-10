"""Row-level verification of the paper-faithful PRISM split.

Runs against data/prism/full_s42_history_sft40_grpo60_test10/.
Complements tests/test_prism_split_determinism.py (which only checks counts and
byte-determinism). Requires the split parquets + the cached raw PRISM dataset, so
this is a cluster-run suite (it skips locally when the parquets are absent).

CONFIRM on first cluster run: EXPECTED_COUNTS below and the extra_info / raw-turns
field names (see tests/prism_verification_helpers.py). Adjust if the cluster reports
different values.
"""
from __future__ import annotations

import random
from pathlib import Path

import pandas as pd
import pytest

from tests.prism_verification_helpers import extra_info_key, load_raw_prism_replies

REPO_ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = REPO_ROOT / "data" / "prism" / "full_s42_history_sft40_grpo60_test10"

EXPECTED_COUNTS = {
    "sft/train.parquet":  {"rows": 3272, "users": 464},
    "grpo/train.parquet": {"rows": 4174, "users": 696},
    "grpo/val.parquet":   {"rows": 705,  "users": 696},
    "test.parquet":       {"rows": 880,  "users": 128},
}


def _load(rel: str) -> pd.DataFrame:
    p = SPLIT_DIR / rel
    if not p.exists():
        pytest.skip(f"missing {p}; rebuild the split first")
    return pd.read_parquet(p)


def _users(df: pd.DataFrame) -> set[str]:
    return {str(e["user_id"]) for e in df["extra_info"]}


def test_1_files_exist():
    for name in list(EXPECTED_COUNTS) + ["split_metadata.json"]:
        assert (SPLIT_DIR / name).exists(), f"missing {name}"


@pytest.mark.parametrize("rel,exp", list(EXPECTED_COUNTS.items()))
def test_2_row_counts(rel, exp):
    assert len(_load(rel)) == exp["rows"], f"{rel} row count"


@pytest.mark.parametrize("rel,exp", list(EXPECTED_COUNTS.items()))
def test_3_user_counts(rel, exp):
    assert len(_users(_load(rel))) == exp["users"], f"{rel} user count"


def test_4_user_disjointness():
    sft = _users(_load("sft/train.parquet"))
    grpo = _users(_load("grpo/train.parquet"))
    held = _users(_load("test.parquet"))
    assert sft & grpo == set(), f"sft ∩ grpo not empty: {list(sft & grpo)[:5]}"
    assert sft & held == set(), f"sft ∩ heldout not empty: {list(sft & held)[:5]}"
    assert grpo & held == set(), f"grpo ∩ heldout not empty: {list(grpo & held)[:5]}"


@pytest.mark.parametrize("rel", list(EXPECTED_COUNTS))
def test_5_prompt_schema(rel):
    df = _load(rel)
    rng = random.Random(42)
    for i in rng.sample(range(len(df)), min(50, len(df))):
        row = df.iloc[i]
        prompt = list(row["prompt"])
        assert prompt, f"{rel}[{i}] empty prompt"
        for msg in prompt:
            assert "role" in msg and "content" in msg, f"{rel}[{i}] prompt msg missing role/content"
        assert str(row["reward_model"]["ground_truth"]).strip(), f"{rel}[{i}] empty ground_truth"
        for k in ("user_id", "post_id", "target_idx", "user_history", "context"):
            assert k in row["extra_info"], f"{rel}[{i}] extra_info missing {k}"
        assert row["data_source"] == "prism_alignment_user_sim", f"{rel}[{i}] data_source"


def test_6_heldout_gt_matches_raw():
    df = _load("test.parquet")
    raw = load_raw_prism_replies()
    rng = random.Random(42)
    bad: list[str] = []
    for i in rng.sample(range(len(df)), 20):
        row = df.iloc[i]
        try:
            key = extra_info_key(dict(row["extra_info"]))
        except (KeyError, TypeError) as e:
            bad.append(f"row {i}: bad extra_info ({e})")
            continue
        reply = raw.get(key)
        got = str(row["reward_model"]["ground_truth"])
        if reply is None:
            bad.append(f"row {i} key {key} not in raw PRISM")
        elif reply.strip() != got.strip():
            bad.append(f"row {i} key {key}: raw != split (raw[:80]={reply[:80]!r})")
    assert not bad, "heldout ground_truth mismatches:\n" + "\n".join(bad)


def test_7_no_text_leak_heldout_from_sft_targets():
    sft = _load("sft/train.parquet")
    held = _load("test.parquet")
    long_targets = {
        str(rm["ground_truth"]) for rm in sft["reward_model"]
        if len(str(rm["ground_truth"])) >= 60
    }
    rng = random.Random(42)
    leaks: list[str] = []
    for i in rng.sample(range(len(held)), 20):
        gt = str(held.iloc[i]["reward_model"]["ground_truth"])
        if any(t in gt for t in long_targets):
            leaks.append(f"heldout[{i}] contains an SFT target verbatim")
    assert not leaks, "text-leak candidates:\n" + "\n".join(leaks)
