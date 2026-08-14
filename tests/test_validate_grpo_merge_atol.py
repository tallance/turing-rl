"""Check D's shared-tensor tolerance.

An FSDP2 save re-rounded the 24 frozen ``linear_attn.norm.weight`` tensors of the Qwen3.5-4B
judge checkpoints by exactly one bf16 ULP, which the bit-exact gate reported as a corrupt
backbone (merge jobs 17889/17891). ``--shared_atol`` admits that, bounded and reported --
but the DEFAULT must stay bit-exact, or the gate stops catching a genuinely wrong container.
"""

import json
import subprocess
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
ONE_BF16_ULP = 0.00390625  # 2**-8, the gap between consecutive bf16 values near 1.0

TARGET = "model.layers.0.mlp.gate_proj.weight"
FROZEN = "model.norm.weight"


def _build(tmp_path: Path, hf_base_drift: float) -> dict[str, Path]:
    """A minimal merge that passes A/B/C, with hf_base off the container by `hf_base_drift`."""
    torch.manual_seed(0)
    base_w = torch.randn(3, 4)
    lora_a = torch.randn(1, 4)
    lora_b = torch.randn(3, 1)
    scaling = 2.0  # lora_alpha / r
    frozen = torch.ones(5)

    base = tmp_path / "base"
    dense = tmp_path / "dense"
    hf_base = tmp_path / "hf_base"
    adapter = tmp_path / "hf_base" / "lora_adapter"
    for d in (base, dense, hf_base, adapter):
        d.mkdir(parents=True, exist_ok=True)

    save_file({TARGET: base_w, FROZEN: frozen}, str(base / "model.safetensors"))
    save_file(
        {TARGET: base_w + scaling * (lora_b @ lora_a), FROZEN: frozen},
        str(dense / "model.safetensors"),
    )
    save_file(
        {TARGET: base_w.clone(), FROZEN: frozen + hf_base_drift},
        str(hf_base / "model.safetensors"),
    )
    stem = f"base_model.model.{TARGET[: -len('.weight')]}"
    save_file(
        {f"{stem}.lora_A.weight": lora_a, f"{stem}.lora_B.weight": lora_b},
        str(adapter / "adapter_model.safetensors"),
    )
    (adapter / "adapter_config.json").write_text(json.dumps({"r": 1, "lora_alpha": 2}))
    return {"base": base, "dense": dense, "hf_base": hf_base, "adapter": adapter}


def _run(dirs: dict[str, Path], *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts/validate_grpo_merge.py"),
         "--base", str(dirs["base"]), "--dense", str(dirs["dense"]),
         "--adapter", str(dirs["adapter"]), "--hf_base", str(dirs["hf_base"]),
         "--expect_targets", "1", *extra],
        capture_output=True, text=True, cwd=str(ROOT),
    )


def test_a_clean_merge_passes(tmp_path):
    result = _run(_build(tmp_path, 0.0))
    assert result.returncode == 0, result.stdout + result.stderr


def test_default_is_bit_exact_so_a_one_ulp_drift_still_fails(tmp_path):
    """The regression that matters: if the default ever loosens, a wrong container slides
    through as 'close enough'."""
    result = _run(_build(tmp_path, ONE_BF16_ULP))
    assert result.returncode != 0
    assert "D: hf_base differs from base" in result.stdout + result.stderr


def test_one_ulp_drift_is_admitted_when_explicitly_tolerated(tmp_path):
    result = _run(_build(tmp_path, ONE_BF16_ULP), "--shared_atol", str(ONE_BF16_ULP))
    assert result.returncode == 0, result.stdout + result.stderr
    assert "within --shared_atol" in result.stdout


def test_drift_larger_than_the_tolerance_still_fails(tmp_path):
    """A real backbone mixup is orders of magnitude past one ULP, and must not be excused."""
    result = _run(_build(tmp_path, 0.5), "--shared_atol", str(ONE_BF16_ULP))
    assert result.returncode != 0
    assert "D: hf_base differs from base" in result.stdout + result.stderr
