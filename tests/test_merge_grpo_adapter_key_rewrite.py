"""Key-prefix rewrite in merge_grpo_adapter.load_adapter.

An adapter trained against the text-only view of a multimodal checkpoint records
``model.layers.<...>``; the multimodal container stores ``model.language_model.layers.<...>``.
Without the rewrite the merge matches nothing and every LoRA target is silently absent from
the base -- which the caller only catches as a hard failure, never as a correct merge.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.merge_grpo_adapter import load_adapter  # noqa: E402

TEXT_ONLY = "model.layers.0.mlp.down_proj"
MULTIMODAL = "model.language_model.layers.0.mlp.down_proj"


@pytest.fixture()
def adapter_dir(tmp_path: Path) -> Path:
    d = tmp_path / "adapter"
    d.mkdir()
    (d / "adapter_config.json").write_text(json.dumps({"r": 64, "lora_alpha": 128}))
    save_file(
        {
            f"base_model.model.{TEXT_ONLY}.lora_A.weight": torch.zeros(64, 8),
            f"base_model.model.{TEXT_ONLY}.lora_B.weight": torch.zeros(8, 64),
        },
        str(d / "adapter_model.safetensors"),
    )
    return d


def test_without_rewrite_keys_stay_text_only(adapter_dir: Path) -> None:
    pairs, scaling = load_adapter(adapter_dir)
    assert scaling == 2.0
    assert set(pairs) == {f"{TEXT_ONLY}.weight"}


def test_rewrite_maps_onto_the_multimodal_container_key(adapter_dir: Path) -> None:
    pairs, _ = load_adapter(
        adapter_dir, ("model.layers.", "model.language_model.layers.")
    )
    assert set(pairs) == {f"{MULTIMODAL}.weight"}


def test_rewrite_that_does_not_apply_is_an_error(adapter_dir: Path) -> None:
    # A typo in OLD must fail loudly rather than pass keys through unrewritten.
    with pytest.raises(ValueError, match="does not apply"):
        load_adapter(adapter_dir, ("model.decoder.", "model.language_model.layers."))
