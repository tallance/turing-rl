"""Single source of truth for the judge-sweep cell matrix + TP/replica lookup.

Imported by the launcher (Task 16, ``scripts/launch_judge_sweep.sh``) and the
analyzer (Task 21, ``scripts/analyze_judge_sweep.py``) so the size/TP/replica
mapping lives in exactly one tested place.

A "cell" is one judge model (its own vLLM serving shape); the two thinking
modes (off/on) are expanded by the launcher, not here. The 397B anchor is
appended to every family's list so a family sweep always re-baselines against it.
"""
from __future__ import annotations

ANCHOR = {
    "cell_name": "qwen35-397b",
    "model_id": "Qwen/Qwen3.5-397B-A17B-GPTQ-Int4",
    "tp": 8,
    "replicas": 1,
    "size_b": 17,  # active params (MoE)
    "is_moe": True,
}

# (cell_name, model_id, size_b, is_moe) — size_b is total params for dense,
# active params for MoE.
_FAMILIES = {
    "qwen3": [
        ("qwen3-4b", "Qwen/Qwen3-4B", 4, False),
        ("qwen3-8b", "Qwen/Qwen3-8B", 8, False),
        ("qwen3-14b", "Qwen/Qwen3-14B", 14, False),
        ("qwen3-32b", "Qwen/Qwen3-32B", 32, False),
    ],
    "qwen3.5": [
        ("qwen35-4b", "Qwen/Qwen3.5-4B", 4, False),
        ("qwen35-9b", "Qwen/Qwen3.5-9B", 9, False),
        ("qwen35-27b", "Qwen/Qwen3.5-27B", 27, False),
        ("qwen35-35b-a3b", "Qwen/Qwen3.5-35B-A3B-GPTQ-Int4", 35, True),
    ],
}


def tp_for_size(size_b: int, is_moe: bool) -> tuple[int, int]:
    """Serving shape (tensor_parallel, replicas) for one judge on an 8-GPU node.

    Dense models >=20B need TP=4 (4 GPUs each -> 2 replicas); everything else,
    including MoE-Int4 (which fits on a single GPU despite large total params),
    runs TP=1 -> 8 replicas. ``tp*replicas == 8`` in both cases (full node).

    TP=8/1-replica for dense >=20B: on 40GB A100s the Qwen3-Next 27B hybrid died
    at the KV-cache memory-profiling stage at BOTH TP=2 (CUDA OOM) and TP=4/2-rep
    (a TP worker crashed -> shm_broadcast timeout; the two co-located TP groups
    also strain shared-memory/semaphores). TP=8 single-replica gives ~6.75GB/GPU
    for weights (ample headroom) with no co-located groups -- the same topology
    the 397B anchor serves on successfully.
    """
    if not is_moe and size_b >= 20:
        return (8, 1)
    return (1, 8)


def cell_list(family: str) -> list[dict]:
    """Return the judge cells for ``family`` plus the fixed 397B anchor."""
    out = []
    for name, mid, size_b, is_moe in _FAMILIES[family]:
        tp, rep = tp_for_size(size_b, is_moe)
        out.append(
            {
                "cell_name": name,
                "model_id": mid,
                "tp": tp,
                "replicas": rep,
                "size_b": size_b,
                "is_moe": is_moe,
            }
        )
    out.append(dict(ANCHOR))
    return out


# x-axis sizes for plotting (active params for MoE); anchor plotted at 17B active.
SIZE_MAP = {
    "qwen3-4b": 4,
    "qwen3-8b": 8,
    "qwen3-14b": 14,
    "qwen3-32b": 32,
    "qwen35-4b": 4,
    "qwen35-9b": 9,
    "qwen35-27b": 27,
    "qwen35-35b-a3b": 3,
    "qwen35-397b": 17,
}

ANCHOR_CELL = "qwen35-397b"
