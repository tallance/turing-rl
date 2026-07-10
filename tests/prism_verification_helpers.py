"""Helpers for tests/test_prism_split_verification.py.

Loads the raw HuggingFace PRISM dataset (cached on the cluster at
/home/lancewicki/data/hf_cache/datasets--HannahRoseKirk--prism-alignment) for
split-verification test 6 (heldout ground_truth matches raw PRISM).

NOTE (confirm on first cluster run): the raw per-turn field name is assumed to be
`conversation_turns` with role=="user"; and `extra_info` is assumed to carry
`user_id`/`post_id`/`target_idx` (with an optional `raw_user_id`). If the cluster
run KeyErrors, inspect one raw row / one split row and adjust the two functions below.
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
        turn_idx = 0
        for turn in row.get("conversation_turns", []) or []:
            if str(turn.get("role") or "").lower() != "user":
                continue
            out[(uid, cid, turn_idx)] = str(turn.get("content") or "")
            turn_idx += 1
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
