"""Fold a veRL GRPO LoRA adapter into the SFT backbone, producing a dense servable model.

WHY THIS EXISTS
---------------
veRL GRPO checkpoints are NOT dense policies. ``lora.merge=True`` merges weights only
transiently for rollout weight-sync; what persists under ``global_step_N/actor/`` is an
unmerged PEFT model (``lora_A``/``lora_B`` alongside frozen ``base_layer`` weights).
``verl.model_merger`` then *pops* every ``lora_*`` key into ``<target>/lora_adapter/`` and
saves the stripped remainder -- i.e. the SFT base. Serving that directory directly would
silently evaluate the pre-RL checkpoint with no error, so the GRPO delta must be folded
back in explicitly.

WHY merged_ep3 IS THE CONTAINER
-------------------------------
The merge runs under transformers 5.4 (``turing-rl-rl-qwen35``) but generation runs under
transformers 4.57.6 (``turing-rl-train``). The checkpoint's tokenizer declares
``tokenizer_class: TokenizersBackend`` (5.x-only) and its config says
``transformers_version: 5.4.0`` -- neither loads in the generation env. ``merged_ep3`` uses
the 4.x form and is proven servable there. So we take config/tokenizer/chat-template and
every non-target tensor from ``merged_ep3`` verbatim and update only the LoRA targets.

This is pure safetensors tensor math -- no model instantiation -- so it is immune to the
transformers version split. Soundness relies on the validation gate proving
``hf_base == merged_ep3`` on shared tensors (see scripts/validate_grpo_merge.py).

Usage:
  python scripts/merge_grpo_adapter.py \
    --base   checkpoints/sft/.../merged_ep3 \
    --adapter <EVAL_ROOT>/models/step8/hf_base/lora_adapter \
    --out     <EVAL_ROOT>/models/step8/hf_dense
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

# Everything that is not a weight shard is copied verbatim from the base container
# (config.json, tokenizer*, chat_template.jinja, and the safetensors index).
_SKIP_SUFFIXES = (".safetensors",)

LORA_A = ".lora_A"
LORA_B = ".lora_B"


def _module_path(key: str) -> str:
    """``base_model.model.model.<...>.gate_proj.lora_A.weight`` -> ``model.<...>.gate_proj``."""
    for tag in (LORA_A, LORA_B):
        if tag in key:
            stem = key.split(tag)[0]
            break
    else:
        raise ValueError(f"not a lora key: {key}")
    if stem.startswith("base_model.model."):
        stem = stem[len("base_model.model.") :]
    return stem


def load_adapter(adapter_dir: Path) -> tuple[dict[str, tuple[torch.Tensor, torch.Tensor]], float]:
    """Return ``{base_weight_key: (A, B)}`` and the LoRA scaling ``alpha / r``."""
    cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
    r = int(cfg["r"])
    alpha = int(cfg["lora_alpha"])
    if r <= 0:
        raise ValueError(f"invalid LoRA rank r={r}")
    if alpha == 0:
        raise ValueError(
            "adapter_config.json has lora_alpha=0 -- verl.model_merger falls back to 0 when "
            "lora_train_meta.json is missing/invalid, which would make the delta vanish."
        )
    scaling = alpha / r

    a_by_mod: dict[str, torch.Tensor] = {}
    b_by_mod: dict[str, torch.Tensor] = {}
    with safe_open(str(adapter_dir / "adapter_model.safetensors"), framework="pt") as f:
        for key in f.keys():
            mod = _module_path(key)
            if LORA_A in key:
                a_by_mod[mod] = f.get_tensor(key)
            elif LORA_B in key:
                b_by_mod[mod] = f.get_tensor(key)
            else:
                raise ValueError(f"unexpected key in adapter: {key}")

    only_a = set(a_by_mod) - set(b_by_mod)
    only_b = set(b_by_mod) - set(a_by_mod)
    if only_a or only_b:
        raise ValueError(f"unpaired LoRA tensors: A-only={sorted(only_a)} B-only={sorted(only_b)}")

    pairs = {f"{mod}.weight": (a_by_mod[mod], b_by_mod[mod]) for mod in a_by_mod}
    return pairs, scaling


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="Container model dir (merged_ep3): config/tokenizer/weights")
    ap.add_argument("--adapter", required=True, help="verl.model_merger output lora_adapter/ dir")
    ap.add_argument("--out", required=True, help="Destination dense model dir")
    ap.add_argument("--expect_targets", type=int, default=128,
                    help="Expected LoRA target count (Qwen3.5-9B: 32*mlp3 + 8*attn4 = 128)")
    a = ap.parse_args()

    base_dir, adapter_dir, out_dir = Path(a.base), Path(a.adapter), Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs, scaling = load_adapter(adapter_dir)
    print(f"adapter: {len(pairs)} LoRA targets, scaling=alpha/r={scaling}")
    if len(pairs) != a.expect_targets:
        raise SystemExit(f"FAIL: expected {a.expect_targets} LoRA targets, found {len(pairs)}")

    shards = sorted(base_dir.glob("*.safetensors"))
    if not shards:
        raise SystemExit(f"FAIL: no safetensors under {base_dir}")

    applied: dict[str, dict] = {}
    for shard in shards:
        tensors: dict[str, torch.Tensor] = {}
        with safe_open(str(shard), framework="pt") as f:
            metadata = f.metadata() or {}
            for key in f.keys():
                w = f.get_tensor(key)
                if key in pairs:
                    lora_a, lora_b = pairs[key]
                    # float32 accumulate, then cast back -- avoids bf16 rounding in the matmul.
                    delta = (lora_b.float() @ lora_a.float()) * scaling
                    if delta.shape != w.shape:
                        raise SystemExit(
                            f"FAIL: shape mismatch for {key}: base={tuple(w.shape)} delta={tuple(delta.shape)}"
                        )
                    merged = (w.float() + delta).to(w.dtype)
                    applied[key] = {
                        "shard": shard.name,
                        "shape": list(w.shape),
                        "delta_absmax": float(delta.abs().max()),
                        "delta_frobenius": float(delta.norm()),
                    }
                    tensors[key] = merged
                else:
                    tensors[key] = w
        save_file(tensors, str(out_dir / shard.name), metadata=metadata or {"format": "pt"})
        print(f"wrote {shard.name} ({len(tensors)} tensors, {sum(1 for k in tensors if k in applied)} merged)")

    missing = sorted(set(pairs) - set(applied))
    if missing:
        raise SystemExit(f"FAIL: {len(missing)} adapter targets had no matching base tensor: {missing[:5]}")

    zero = sorted(k for k, v in applied.items() if v["delta_absmax"] == 0.0)
    if zero:
        raise SystemExit(f"FAIL: {len(zero)} targets got an all-zero delta (adapter is untrained?): {zero[:5]}")

    # Copy config / tokenizer / chat template / index verbatim from the container.
    copied = []
    for src in sorted(base_dir.iterdir()):
        if not src.is_file() or src.name.endswith(_SKIP_SUFFIXES):
            continue
        shutil.copy2(src, out_dir / src.name)
        copied.append(src.name)
    print(f"copied {len(copied)} container files: {', '.join(copied)}")

    report = {
        "base": str(base_dir),
        "adapter": str(adapter_dir),
        "out": str(out_dir),
        "scaling": scaling,
        "n_targets": len(applied),
        "targets": applied,
        "copied_files": copied,
    }
    (out_dir / "grpo_merge_report.json").write_text(json.dumps(report, indent=2))
    print(f"OK: merged {len(applied)} targets -> {out_dir}")


if __name__ == "__main__":
    main()
