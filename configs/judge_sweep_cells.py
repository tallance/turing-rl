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
    "quantized": True,  # bf16 397B (~800GB) can't fit; Int4 is the forced deviation
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
        # Non-quantized (bf16) so every judge is full-precision; only the 397B
        # anchor is forced to Int4 (bf16 397B won't fit). ~70GB -> whole node.
        ("qwen35-35b-a3b", "Qwen/Qwen3.5-35B-A3B", 35, True),
        # 122B is Int4: bf16 (234GB) needs 2 nodes, and fp8 fails on A100 (Marlin
        # tile constraint at the forced TP=8). Int4 (~61GB) fits at TP=8 and
        # matches the anchor's quantization; the one mid-axis quantized judge.
        ("qwen35-122b", "Qwen/Qwen3.5-122B-A10B-GPTQ-Int4", 122, True),
    ],
}

# Usable weight budget per 40GB A100 after KV cache + CUDA-graph headroom.
_PER_GPU_BUDGET_GB = 30.0


def _is_quantized(model_id: str) -> bool:
    return "Int4" in model_id or "Int8" in model_id or "GPTQ" in model_id or "AWQ" in model_id


def tp_for_size(size_b: int, quantized: bool) -> tuple[int, int]:
    """Serving shape (tensor_parallel, replicas) chosen by MEMORY FOOTPRINT.

    footprint_gb = params * bytes/param (2.0 bf16, 0.5 Int4). If it fits one GPU
    with KV/CUDA-graph headroom (<= ~30GB) -> TP=1, 8 replicas (max throughput);
    otherwise the model spans the whole node -> TP=8, 1 replica. ``tp*replicas==8``.

    Footprint, not param count, is what matters on 40GB A100s: e.g. 35B-A3B is
    ~70GB in bf16 (whole node) but ~18GB in Int4 (one GPU); 27B bf16 is ~54GB
    (whole node). This also avoids the earlier TP=2/4 failures for the 27B hybrid
    (custom-all-reduce error at TP>1, since fixed with NCCL fallback), keeping the
    large-dense cells on the single-group TP=8 topology the 397B anchor proved.
    """
    bytes_per_param = 0.5 if quantized else 2.0
    footprint_gb = size_b * bytes_per_param
    if footprint_gb <= _PER_GPU_BUDGET_GB:
        return (1, 8)
    return (8, 1)


def cell_list(family: str) -> list[dict]:
    """Return the judge cells for ``family`` plus the fixed 397B anchor."""
    out = []
    for name, mid, size_b, is_moe in _FAMILIES[family]:
        quantized = _is_quantized(mid)
        tp, rep = tp_for_size(size_b, quantized)
        out.append(
            {
                "cell_name": name,
                "model_id": mid,
                "tp": tp,
                "replicas": rep,
                "size_b": size_b,
                "is_moe": is_moe,
                "quantized": quantized,
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
    "qwen35-122b": 10,  # A10B: 10B active
    "qwen35-397b": 17,
    # 397B sampling-variant diagnostic cells (plotted next to the 397B baseline).
    "qwen35-397b-t07": 17,     # temperature=0.7 (model card)
    "qwen35-397b-reppen": 17,  # repetition_penalty=1.1 (loop hypothesis)
}

ANCHOR_CELL = "qwen35-397b"
