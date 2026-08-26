import json
from pathlib import Path

import pytest

import scripts.merge_sft_adapter as merge_script


def _write_adapter(adapter_dir: Path, *, base_model: str = "Qwen/Qwen3-8B") -> None:
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model_name_or_path": base_model,
                "peft_type": "LORA",
                "r": 64,
                "lora_alpha": 128,
                "target_modules": ["q_proj", "v_proj"],
            }
        )
    )
    (adapter_dir / "adapter_model.safetensors").write_bytes(b"adapter")
    (adapter_dir / "tokenizer_config.json").write_text("{}")


def test_validate_adapter_rejects_wrong_base(tmp_path):
    adapter_dir = tmp_path / "adapter"
    _write_adapter(adapter_dir, base_model="other/model")

    with pytest.raises(ValueError, match="base model mismatch"):
        merge_script.load_and_validate_adapter_config(adapter_dir, "Qwen/Qwen3-8B")


def test_validate_merged_artifact_requires_weights(tmp_path):
    output_dir = tmp_path / "merged"
    output_dir.mkdir()
    for name in ("config.json", "tokenizer_config.json", merge_script.MERGE_METADATA_NAME):
        (output_dir / name).write_text("{}")

    with pytest.raises(ValueError, match="model weights"):
        merge_script.validate_merged_artifact(output_dir)


def test_merge_writes_valid_atomic_artifact(tmp_path, monkeypatch):
    adapter_dir = tmp_path / "adapter"
    output_dir = tmp_path / "merged"
    _write_adapter(adapter_dir)
    calls = {}

    class FakeTorch:
        bfloat16 = "bf16"
        float16 = "fp16"
        float32 = "fp32"

    class FakeBaseLoader:
        @staticmethod
        def from_pretrained(model_name, **kwargs):
            calls["base"] = (model_name, kwargs)
            return object()

    class FakeMergedModel:
        def save_pretrained(self, path, **kwargs):
            calls["save"] = kwargs
            path = Path(path)
            (path / "config.json").write_text("{}")
            (path / "model-00001-of-00001.safetensors").write_bytes(b"weights")

    class FakePeftModel:
        def merge_and_unload(self, **kwargs):
            calls["merge"] = kwargs
            return FakeMergedModel()

    class FakePeftLoader:
        @staticmethod
        def from_pretrained(base, adapter_path, **kwargs):
            calls["adapter"] = (base, adapter_path, kwargs)
            return FakePeftModel()

    class FakeTokenizer:
        def save_pretrained(self, path):
            calls["tokenizer_save"] = str(path)
            (Path(path) / "tokenizer_config.json").write_text("{}")

    class FakeTokenizerLoader:
        @staticmethod
        def from_pretrained(path, **kwargs):
            calls["tokenizer_load"] = (path, kwargs)
            return FakeTokenizer()

    monkeypatch.setattr(
        merge_script,
        "_load_merge_runtime",
        lambda: (FakeTorch, FakeBaseLoader, FakeTokenizerLoader, FakePeftLoader),
    )

    result = merge_script.merge_sft_adapter(
        base_model="Qwen/Qwen3-8B",
        adapter_dir=adapter_dir,
        output_dir=output_dir,
        dtype_name="bfloat16",
        max_shard_size="4GB",
    )

    assert result == output_dir
    assert output_dir.is_dir()
    assert calls["base"] == (
        "Qwen/Qwen3-8B",
        {"torch_dtype": "bf16", "low_cpu_mem_usage": True},
    )
    assert calls["adapter"][1:] == (str(adapter_dir), {"is_trainable": False})
    assert calls["merge"] == {"safe_merge": True, "progressbar": True}
    assert calls["save"] == {"safe_serialization": True, "max_shard_size": "4GB"}
    assert calls["tokenizer_load"] == (str(adapter_dir), {"trust_remote_code": False})

    metadata = json.loads((output_dir / merge_script.MERGE_METADATA_NAME).read_text())
    assert metadata["base_model"] == "Qwen/Qwen3-8B"
    assert metadata["source_adapter"]["r"] == 64
    assert metadata["source_adapter"]["lora_alpha"] == 128
    merge_script.validate_merged_artifact(output_dir)


def test_merge_refuses_existing_output(tmp_path):
    adapter_dir = tmp_path / "adapter"
    output_dir = tmp_path / "merged"
    _write_adapter(adapter_dir)
    output_dir.mkdir()

    with pytest.raises(FileExistsError, match="already exists"):
        merge_script.merge_sft_adapter(
            base_model="Qwen/Qwen3-8B",
            adapter_dir=adapter_dir,
            output_dir=output_dir,
        )
