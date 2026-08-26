#!/usr/bin/env python3
"""Merge the PRISM SFT LoRA into Qwen3-8B for use as a GRPO backbone.

veRL's colocated LoRA reference policy is the actor with its active LoRA
disabled. Starting GRPO from a pre-trained SFT LoRA therefore makes the
reference the unadapted base model. This utility creates a standalone SFT
checkpoint so a new RL LoRA can be trained on top and disabled to recover the
actual SFT reference policy.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_MODEL = "Qwen/Qwen3-8B"
DEFAULT_ADAPTER_DIR = REPO_ROOT / "checkpoints/sft/qwen3_8b_prism_full_s42_bf16_fsdp_nopack/final"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "checkpoints/sft/qwen3_8b_prism_full_s42_bf16_fsdp_nopack/merged"
# Stop-token-supervised (proper) SFT checkpoint from the 2026-07-21 trajectory run (job 10715),
# ep3. Distinct from the buggy DEFAULT_* above (whose completion mask excluded <|im_end|>).
PROPER_ADAPTER_DIR_8B = (
    REPO_ROOT / "checkpoints/sft/qwen3_8b_prism_full_s42_bf16_fsdp_nopack_epochsave/checkpoint-78"
)
PROPER_MERGED_DIR_8B = (
    REPO_ROOT / "checkpoints/sft/qwen3_8b_prism_full_s42_bf16_fsdp_nopack_epochsave/merged_ep3"
)
MERGE_METADATA_NAME = "sft_merge_metadata.json"


def load_and_validate_adapter_config(adapter_dir: Path, base_model: str) -> dict[str, Any]:
    """Load the PEFT config and ensure it belongs to the requested base model."""
    adapter_dir = Path(adapter_dir)
    config_path = adapter_dir / "adapter_config.json"
    weights_path = adapter_dir / "adapter_model.safetensors"
    if not config_path.is_file():
        raise FileNotFoundError(f"missing adapter config: {config_path}")
    if not weights_path.is_file():
        raise FileNotFoundError(f"missing adapter weights: {weights_path}")

    config = json.loads(config_path.read_text())
    if str(config.get("peft_type", "")).upper() != "LORA":
        raise ValueError(f"expected a LoRA adapter, got peft_type={config.get('peft_type')!r}")
    adapter_base = str(config.get("base_model_name_or_path", ""))
    if adapter_base != base_model:
        raise ValueError(
            f"base model mismatch: adapter expects {adapter_base!r}, requested {base_model!r}"
        )
    return config


def _weight_files(model_dir: Path) -> list[Path]:
    patterns = ("model*.safetensors", "pytorch_model*.bin")
    return sorted(path for pattern in patterns for path in model_dir.glob(pattern) if path.is_file())


def validate_merged_artifact(model_dir: Path) -> dict[str, Any]:
    """Validate the files needed by Transformers, veRL, and provenance checks."""
    model_dir = Path(model_dir)
    required = ("config.json", "tokenizer_config.json", MERGE_METADATA_NAME)
    missing = [name for name in required if not (model_dir / name).is_file()]
    if missing:
        raise ValueError(f"merged artifact is missing required files: {', '.join(missing)}")
    weights = _weight_files(model_dir)
    if not weights or any(path.stat().st_size == 0 for path in weights):
        raise ValueError("merged artifact has no non-empty model weights")

    metadata = json.loads((model_dir / MERGE_METADATA_NAME).read_text())
    if metadata.get("artifact_type") != "merged_sft_backbone":
        raise ValueError("merged artifact metadata has an unexpected artifact_type")
    return metadata


def _load_merge_runtime():
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    return torch, AutoModelForCausalLM, AutoTokenizer, PeftModel


def _build_metadata(
    *,
    base_model: str,
    adapter_dir: Path,
    adapter_config: dict[str, Any],
    dtype_name: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact_type": "merged_sft_backbone",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_model": base_model,
        "source_adapter_path": str(adapter_dir.resolve()),
        "dtype": dtype_name,
        "safe_merge": True,
        "source_adapter": {
            "peft_type": adapter_config.get("peft_type"),
            "r": adapter_config.get("r"),
            "lora_alpha": adapter_config.get("lora_alpha"),
            "lora_dropout": adapter_config.get("lora_dropout"),
            "target_modules": adapter_config.get("target_modules"),
        },
    }


def merge_sft_adapter(
    *,
    base_model: str,
    adapter_dir: Path,
    output_dir: Path,
    dtype_name: str = "bfloat16",
    max_shard_size: str = "5GB",
) -> Path:
    """Safely merge an SFT LoRA and atomically publish the validated artifact."""
    adapter_dir = Path(adapter_dir)
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")

    adapter_config = load_and_validate_adapter_config(adapter_dir, base_model)
    torch, model_loader, tokenizer_loader, peft_loader = _load_merge_runtime()
    dtype_by_name = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if dtype_name not in dtype_by_name:
        raise ValueError(f"unsupported dtype {dtype_name!r}; choose one of {sorted(dtype_by_name)}")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=str(output_dir.parent))
    )
    try:
        base = model_loader.from_pretrained(
            base_model,
            torch_dtype=dtype_by_name[dtype_name],
            low_cpu_mem_usage=True,
        )
        sft_model = peft_loader.from_pretrained(base, str(adapter_dir), is_trainable=False)
        merged_model = sft_model.merge_and_unload(safe_merge=True, progressbar=True)
        merged_model.save_pretrained(
            temporary_dir,
            safe_serialization=True,
            max_shard_size=max_shard_size,
        )

        tokenizer = tokenizer_loader.from_pretrained(str(adapter_dir), trust_remote_code=False)
        tokenizer.save_pretrained(temporary_dir)
        metadata = _build_metadata(
            base_model=base_model,
            adapter_dir=adapter_dir,
            adapter_config=adapter_config,
            dtype_name=dtype_name,
        )
        (temporary_dir / MERGE_METADATA_NAME).write_text(json.dumps(metadata, indent=2) + "\n")
        validate_merged_artifact(temporary_dir)
        temporary_dir.rename(output_dir)
    finally:
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir)
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--adapter-dir", type=Path, default=DEFAULT_ADAPTER_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--max-shard-size", default="5GB")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = merge_sft_adapter(
        base_model=args.base_model,
        adapter_dir=args.adapter_dir,
        output_dir=args.output_dir,
        dtype_name=args.dtype,
        max_shard_size=args.max_shard_size,
    )
    metadata = validate_merged_artifact(output_dir)
    print(f"wrote merged SFT backbone: {output_dir}")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
