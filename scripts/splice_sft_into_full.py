"""Splice an SFT LoRA (trained on Qwen3_5ForCausalLM, the text tower) into the FULL
multimodal Qwen3.5-9B checkpoint, so vLLM can serve it.

Why: vLLM 0.18 registers only Qwen3_5ForConditionalGeneration (not Qwen3_5ForCausalLM),
can't LoRA-serve the Gated-DeltaNet adapter, and HF autoregressive decoding via the torch
fallback degenerates with the adapter. This produces a drop-in full checkpoint identical to
the base repo EXCEPT the text-tower tensors are LoRA-merged — so vLLM loads it exactly like
the base (correct DeltaNet kernels) but with the SFT applied.

Method (in-place tensor surgery, preserves index/config/mtp/vision):
  1. Merge adapter into Qwen3_5ForCausalLM -> merged text state_dict (keys `model.*`, `lm_head.weight`).
  2. Copy the base snapshot to OUT (deref symlinks).
  3. Rewrite each safetensors shard, replacing base `model.language_model.<S>` with merged
     `model.<S>` (and lm_head), keeping `model.visual.*` / `mtp.*` / everything else as-is.
"""
from __future__ import annotations

import glob
import json
import os
import shutil

import torch
from peft import PeftModel
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM

BASE_ID = "Qwen/Qwen3.5-9B"
TEXT_PREFIX = "model.language_model."


def main() -> None:
    adapter = os.environ["ADAPTER"]
    out = os.environ["OUT"]
    hfc = os.environ.get("HF_HUB_CACHE", "/home/lancewicki/data/hf_cache")
    snaps = sorted(glob.glob(f"{hfc}/models--Qwen--Qwen3.5-9B/snapshots/*"))
    if not snaps:
        raise SystemExit(f"no cached snapshot under {hfc}")
    snap = snaps[0]
    print(f"base snapshot: {snap}", flush=True)
    if os.path.exists(out):
        raise SystemExit(f"OUT already exists: {out}")

    # 1. merge adapter into the text-tower CausalLM (CPU).
    print("loading base CausalLM + merging adapter ...", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_ID, dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True,
    )
    merged = PeftModel.from_pretrained(base, adapter, is_trainable=False).merge_and_unload()
    tsd = {k: v.to(torch.bfloat16).contiguous() for k, v in merged.state_dict().items()}
    del base, merged
    print(f"merged text tensors: {len(tsd)}", flush=True)

    # 2. copy non-weight files from the snapshot (deref symlinks).
    os.makedirs(out)
    for fn in os.listdir(snap):
        src = os.path.join(snap, fn)
        if not os.path.isfile(src) or fn.endswith(".safetensors"):
            continue
        shutil.copy(os.path.realpath(src), os.path.join(out, fn))

    # 3. rewrite each shard, splicing merged text tensors in place.
    index_path = os.path.join(snap, "model.safetensors.index.json")
    shards = sorted(set(json.load(open(index_path))["weight_map"].values()))
    replaced = kept = 0
    for shard in shards:
        data = load_file(os.path.realpath(os.path.join(snap, shard)))
        new: dict[str, torch.Tensor] = {}
        for k, v in data.items():
            if k.startswith(TEXT_PREFIX):
                src = "model." + k[len(TEXT_PREFIX):]
                if src in tsd and tuple(tsd[src].shape) == tuple(v.shape):
                    new[k] = tsd[src]; replaced += 1; continue
            if k == "lm_head.weight":
                cand = tsd.get("lm_head.weight")
                if cand is None:  # tied embeddings
                    cand = tsd.get("model.embed_tokens.weight")
                if cand is not None and tuple(cand.shape) == tuple(v.shape):
                    new[k] = cand; replaced += 1; continue
            new[k] = v; kept += 1
        save_file(new, os.path.join(out, shard), metadata={"format": "pt"})
        print(f"wrote {shard}: {len(new)} tensors", flush=True)

    print(f"DONE. replaced={replaced} kept={kept} -> {out}", flush=True)
    # sanity: every text tensor we had should have landed somewhere
    n_text = sum(1 for k in json.load(open(index_path))["weight_map"] if k.startswith(TEXT_PREFIX))
    print(f"base text tensors={n_text}, merged-source text tensors={len(tsd)}", flush=True)


if __name__ == "__main__":
    main()
