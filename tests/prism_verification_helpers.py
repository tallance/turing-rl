"""Helpers for tests/test_prism_split_verification.py.

Loads the raw HuggingFace PRISM dataset (cached on the cluster at
/home/lancewicki/data/hf_cache/datasets--HannahRoseKirk--prism-alignment) for
split-verification test 6 (heldout ground_truth matches raw PRISM).

Raw PRISM schema (confirmed on cluster 2026-07-10): turns live in
`conversation_history` (a list[dict]); `conversation_turns` is an INT turn-count,
NOT a list. Per-turn keys include `role` ('user' / 'model') and `content`.
`post_id` (extra_info) maps to `conversation_id` (raw). `extra_info` carries both
`user_id` and `raw_user_id`, plus `target_idx`. End-to-end lookup verified against
test.parquet[0] (user96/c529/target_idx=1 → 2nd user turn → matches ground_truth).
"""
from __future__ import annotations

import os
from typing import Any

# Force offline; the raw dataset is already cached on the cluster.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HOME", "/home/lancewicki/data/hf_cache")
os.environ.setdefault("HF_DATASETS_CACHE", "/home/lancewicki/data/hf_cache/datasets")


def load_raw_prism_replies() -> dict[tuple[str, str, int], str]:
    """Return {(user_id, conversation_id, turn_idx): reply} for every [HUMAN] turn.

    turn_idx counts user turns within a conversation, starting at 0.
    """
    from datasets import load_dataset

    ds = load_dataset("HannahRoseKirk/prism-alignment", "conversations", split="train")
    out: dict[tuple[str, str, int], str] = {}
    for row in ds:
        uid = str(row["user_id"])
        cid = str(row["conversation_id"])
        user_turn_idx = 0
        # Turns live in conversation_history (list[dict]); conversation_turns is an int count.
        for turn in row.get("conversation_history") or []:
            if str(turn.get("role") or "").lower() != "user":
                continue
            out[(uid, cid, user_turn_idx)] = str(turn.get("content") or "")
            user_turn_idx += 1
    return out


def extra_info_key(extra_info: dict[str, Any]) -> tuple[str, str, int]:
    """Build the raw-PRISM lookup key from a split row's extra_info.

    Uses `raw_user_id` if present, else `user_id`; `post_id` as the conversation id.
    """
    return (
        str(extra_info.get("raw_user_id", extra_info["user_id"])),
        str(extra_info["post_id"]),
        int(extra_info["target_idx"]),
    )
