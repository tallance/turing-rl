"""Recoverability gate for deleting derived GRPO artifacts.

Every test builds a miniature version of the real layout with tiny synthetic safetensors::

    <root>/results/run/models/step4/hf_base/{model.safetensors,lora_adapter/}
    <root>/results/run/models/step4/hf_dense/{model.safetensors,grpo_merge_report.json}

The behaviours pinned here are the ones whose failure loses a model rather than raising:
a dense dir that is NOT base+BA still looks like a perfectly good 19 GB directory, and
``rm -rf hf_base`` takes ``hf_base/lora_adapter`` -- the only step-unique bytes -- with it.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.safe_delete_derived import main  # noqa: E402

R = 4
ALPHA = 2
SCALING = ALPHA / R
MODULES = [f"model.language_model.layers.{i}.mlp.down_proj" for i in range(6)]
EXPECT_TARGETS = len(MODULES)
OUT_DIM, IN_DIM = 8, 12


def _lora_pair(seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    lora_a = torch.randn(R, IN_DIM, generator=generator, dtype=torch.float32)
    lora_b = torch.randn(OUT_DIM, R, generator=generator, dtype=torch.float32)
    return lora_a, lora_b


def _base_weight(seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(1000 + seed)
    return torch.randn(OUT_DIM, IN_DIM, generator=generator, dtype=torch.bfloat16)


@pytest.fixture()
def layout(tmp_path: Path) -> dict:
    """A base container, a step dir with hf_base + hf_dense, and a correctly merged dense."""
    root = tmp_path / "state"
    base_dir = root / "checkpoints/sft/merged_ep3"
    step = root / "results/run/models/step4"
    hf_base = step / "hf_base"
    adapter = hf_base / "lora_adapter"
    hf_dense = step / "hf_dense"
    for path in (base_dir, adapter, hf_dense):
        path.mkdir(parents=True)

    base_weights = {module + ".weight": _base_weight(i) for i, module in enumerate(MODULES)}
    # A non-target tensor, so the key sets are not trivially just the LoRA targets.
    base_weights["lm_head.weight"] = _base_weight(99)
    save_file(base_weights, str(base_dir / "model.safetensors"))
    (base_dir / "config.json").write_text(json.dumps({"model_type": "qwen3_5"}))

    adapter_tensors = {}
    dense_weights = dict(base_weights)
    for i, module in enumerate(MODULES):
        lora_a, lora_b = _lora_pair(i)
        adapter_tensors[f"base_model.model.{module}.lora_A.weight"] = lora_a
        adapter_tensors[f"base_model.model.{module}.lora_B.weight"] = lora_b
        weight = base_weights[module + ".weight"]
        delta = (lora_b @ lora_a) * SCALING
        dense_weights[module + ".weight"] = (weight.float() + delta).to(weight.dtype)
    save_file(adapter_tensors, str(adapter / "adapter_model.safetensors"))
    (adapter / "adapter_config.json").write_text(json.dumps({"r": R, "lora_alpha": ALPHA}))

    # hf_base is verl.model_merger's reconstruction of the backbone: same tensors as the
    # container, minus the mtp.* keys the actor never had.
    save_file(dict(base_weights), str(hf_base / "model.safetensors"))
    (hf_base / "config.json").write_text(json.dumps({"model_type": "qwen3_5"}))
    (hf_base / "merge_provenance.json").write_text(
        json.dumps({"actor_dir": str(root / "gone/actor"), "step": 4, "run_tag": "t"})
    )

    save_file(dense_weights, str(hf_dense / "model.safetensors"))
    (hf_dense / "config.json").write_text(json.dumps({"model_type": "qwen3_5"}))
    (hf_dense / "grpo_merge_report.json").write_text(
        json.dumps({"base": str(base_dir), "adapter": str(adapter), "scaling": SCALING,
                    "n_targets": EXPECT_TARGETS})
    )

    return {
        "root": root, "base": base_dir, "step": step,
        "hf_base": hf_base, "adapter": adapter, "hf_dense": hf_dense,
        "base_weights": base_weights,
    }


def run(layout: dict, *args: str) -> int:
    return main([
        "--allowed-root", str(layout["root"]),
        "--search-root", str(layout["root"]),
        "--expect-targets", str(EXPECT_TARGETS),
        "--sample", "3",
        *args,
    ])


def run_expect_failure(layout: dict, capsys: pytest.CaptureFixture, check: str,
                       *args: str) -> str:
    """Run and assert it failed at ``check``. Asserting only on the exit code would let a test
    pass for the wrong reason -- e.g. a mislaid fixture failing the guard instead of the
    arithmetic it was written to exercise."""
    assert run(layout, "--delete", *args) == 1
    out = capsys.readouterr().out
    assert f": {check}:" in out, out
    return out


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------- happy path


def test_valid_dense_passes_and_deletes(layout: dict) -> None:
    assert run(layout, "--delete", str(layout["hf_dense"])) == 0
    assert not layout["hf_dense"].exists()
    manifest = json.loads((layout["step"] / "hf_dense.deleted.json").read_text())
    assert manifest["deleted"] is True
    assert manifest["adapter"]["path"] == str(layout["adapter"])
    assert manifest["base"]["path"] == str(layout["base"])
    assert len(manifest["verification"]["sampled_modules"]) == 3
    assert all(s["bit_exact"] for s in manifest["verification"]["sampled_modules"])
    assert any("merge_grpo_adapter.py" in cmd for cmd in manifest["rebuild"])
    assert manifest["bytes_reclaimed"] > 0


def test_manifest_rebuild_command_actually_reconstructs_the_dense_model(layout: dict) -> None:
    # The manifest is only honest if its command is the one that puts the bytes back.
    expected = {name: t.clone() for name, t in _load(layout["hf_dense"]).items()}
    assert run(layout, "--delete", str(layout["hf_dense"])) == 0
    manifest = json.loads((layout["step"] / "hf_dense.deleted.json").read_text())

    from scripts.merge_grpo_adapter import main as merge_main

    rebuild = next(c for c in manifest["rebuild"] if "merge_grpo_adapter.py" in c)
    argv = rebuild.split()[2:]  # drop "python scripts/merge_grpo_adapter.py"
    merge_main_argv = [a for a in argv]
    sys_argv = sys.argv
    try:
        sys.argv = ["merge_grpo_adapter.py", *merge_main_argv]
        merge_main()
    finally:
        sys.argv = sys_argv
    rebuilt = _load(layout["hf_dense"])
    assert set(rebuilt) == set(expected)
    assert all(torch.equal(rebuilt[k], expected[k]) for k in expected)


def _load(model_dir: Path) -> dict[str, torch.Tensor]:
    from safetensors import safe_open

    out = {}
    for shard in sorted(model_dir.glob("*.safetensors")):
        with safe_open(str(shard), framework="pt") as handle:
            for key in handle.keys():
                out[key] = handle.get_tensor(key)
    return out


# --------------------------------------------------------------------------- dry run


def test_dry_run_is_the_default(layout: dict, capsys: pytest.CaptureFixture) -> None:
    assert run(layout, str(layout["hf_dense"]), str(layout["hf_base"])) == 0
    assert layout["hf_dense"].is_dir()
    assert layout["hf_base"].is_dir()
    assert layout["adapter"].is_dir()
    assert not (layout["step"] / "hf_dense.deleted.json").exists()
    assert not (layout["step"] / "hf_base.preserved").exists()
    assert "DRY RUN" in capsys.readouterr().out


# --------------------------------------------------------------------------- failures


def test_arithmetic_mismatch_fails_and_deletes_nothing(
    layout: dict, capsys: pytest.CaptureFixture
) -> None:
    # A dense model that is base + 2x the delta: same shapes, same keys, plausible file sizes,
    # and completely wrong. Only the arithmetic catches it.
    tensors = _load(layout["hf_dense"])
    for i, module in enumerate(MODULES):
        lora_a, lora_b = _lora_pair(i)
        weight = layout["base_weights"][module + ".weight"]
        tensors[module + ".weight"] = (weight.float() + 2 * SCALING * (lora_b @ lora_a)).to(weight.dtype)
    save_file(tensors, str(layout["hf_dense"] / "model.safetensors"))

    out = run_expect_failure(layout, capsys, "arith", str(layout["hf_dense"]))
    assert "does NOT reproduce this file" in out
    assert layout["hf_dense"].is_dir()
    assert not (layout["step"] / "hf_dense.deleted.json").exists()


def test_dense_that_never_got_the_delta_fails(
    layout: dict, capsys: pytest.CaptureFixture
) -> None:
    # The classic verl trap: hf_dense is silently just the SFT base.
    save_file(dict(layout["base_weights"]), str(layout["hf_dense"] / "model.safetensors"))
    out = run_expect_failure(layout, capsys, "arith", str(layout["hf_dense"]))
    assert "unchanged from the base" in out
    assert layout["hf_dense"].is_dir()


def test_missing_adapter_fails(layout: dict, capsys: pytest.CaptureFixture) -> None:
    (layout["adapter"] / "adapter_model.safetensors").unlink()
    run_expect_failure(layout, capsys, "adapter", str(layout["hf_dense"]))
    assert layout["hf_dense"].is_dir()


def test_corrupt_adapter_config_fails(layout: dict, capsys: pytest.CaptureFixture) -> None:
    (layout["adapter"] / "adapter_config.json").write_text("{not json")
    run_expect_failure(layout, capsys, "adapter", str(layout["hf_dense"]))
    assert layout["hf_dense"].is_dir()


def test_truncated_adapter_tensors_fail(layout: dict, capsys: pytest.CaptureFixture) -> None:
    # Half the A/B pairs written: still parses, still has a valid config, still merges into
    # *something*. The target count is what catches it.
    tensors = {}
    for i, module in enumerate(MODULES[:3]):
        lora_a, lora_b = _lora_pair(i)
        tensors[f"base_model.model.{module}.lora_A.weight"] = lora_a
        tensors[f"base_model.model.{module}.lora_B.weight"] = lora_b
    save_file(tensors, str(layout["adapter"] / "adapter_model.safetensors"))
    run_expect_failure(layout, capsys, "adapter", str(layout["hf_dense"]))
    assert layout["hf_dense"].is_dir()


def test_adapter_fingerprint_mismatch_fails(
    layout: dict, capsys: pytest.CaptureFixture
) -> None:
    # A stale hf_base -- right shapes, wrong step -- is internally self-consistent and would
    # pass the arithmetic. Only the fingerprint merge_grpo_ckpt.sh recorded catches the mixup.
    (layout["hf_base"] / "merge_provenance.json").write_text(json.dumps({
        "actor_dir": str(layout["root"] / "gone/actor"), "step": 4,
        "adapter_sha256": "0" * 64,
    }))
    out = run_expect_failure(layout, capsys, "adapter", str(layout["hf_dense"]))
    assert "merge_provenance.json" in out
    assert layout["hf_dense"].is_dir()


def test_matching_adapter_fingerprint_is_cross_checked(
    layout: dict, capsys: pytest.CaptureFixture
) -> None:
    digest = sha256(layout["adapter"] / "adapter_model.safetensors")
    (layout["hf_base"] / "merge_provenance.json").write_text(json.dumps({
        "actor_dir": str(layout["root"] / "gone/actor"), "step": 4, "adapter_sha256": digest,
    }))
    assert run(layout, str(layout["hf_dense"])) == 0
    assert f"sha={digest[:12]}" in capsys.readouterr().out


def test_wrong_target_count_fails(layout: dict, capsys: pytest.CaptureFixture) -> None:
    run_expect_failure(layout, capsys, "adapter", "--expect-targets", str(EXPECT_TARGETS + 1),
                       str(layout["hf_dense"]))
    assert layout["hf_dense"].is_dir()


def test_missing_base_container_fails(layout: dict, capsys: pytest.CaptureFixture) -> None:
    (layout["base"] / "model.safetensors").unlink()
    run_expect_failure(layout, capsys, "base", str(layout["hf_dense"]))
    assert layout["hf_dense"].is_dir()


def test_wrong_base_container_fails(layout: dict, capsys: pytest.CaptureFixture) -> None:
    # Right shapes, right keys, different weights -- an easy mix-up between SFT epochs, and
    # one that would produce a silently different model on rebuild.
    other = layout["root"] / "checkpoints/sft/merged_ep2"
    other.mkdir(parents=True)
    save_file({k: v + 1.0 for k, v in layout["base_weights"].items()},
              str(other / "model.safetensors"))
    (other / "config.json").write_text(json.dumps({"model_type": "qwen3_5"}))
    run_expect_failure(layout, capsys, "arith", "--base", str(other), str(layout["hf_dense"]))
    assert layout["hf_dense"].is_dir()


def test_one_bad_target_does_not_block_the_good_one_but_still_exits_nonzero(layout: dict) -> None:
    bad_step = layout["root"] / "results/run/models/step8"
    bad_dense = bad_step / "hf_dense"
    bad_dense.mkdir(parents=True)
    save_file(dict(layout["base_weights"]), str(bad_dense / "model.safetensors"))
    (bad_dense / "grpo_merge_report.json").write_text(
        json.dumps({"base": str(layout["base"]), "adapter": str(layout["adapter"])})
    )

    assert run(layout, "--delete", str(layout["hf_dense"]), str(bad_dense)) == 1
    assert not layout["hf_dense"].exists()
    assert bad_dense.is_dir()


# --------------------------------------------------------------------------- nested adapter


def test_deleting_hf_base_preserves_the_nested_adapter(layout: dict) -> None:
    before = {
        "adapter_model.safetensors": sha256(layout["adapter"] / "adapter_model.safetensors"),
        "adapter_config.json": sha256(layout["adapter"] / "adapter_config.json"),
    }
    shard = layout["hf_base"] / "model.safetensors"
    assert shard.is_file()

    assert run(layout, "--delete", str(layout["hf_base"])) == 0
    assert not layout["hf_base"].exists()

    preserved = layout["step"] / "hf_base.preserved" / "lora_adapter"
    assert preserved.is_dir()
    for name, digest in before.items():
        assert sha256(preserved / name) == digest, f"{name} is not byte-identical after the delete"
    # merge_provenance.json is the only record of which actor produced this step.
    assert (layout["step"] / "hf_base.preserved" / "merge_provenance.json").is_file()

    manifest = json.loads((layout["step"] / "hf_base.deleted.json").read_text())
    assert manifest["adapter"]["path"] == str(preserved)
    assert manifest["preserved"]["destination"] == str(layout["step"] / "hf_base.preserved")


def test_deleting_hf_base_and_hf_dense_together_cites_the_relocated_adapter(layout: dict) -> None:
    assert run(layout, "--delete", str(layout["hf_base"]), str(layout["hf_dense"])) == 0
    preserved = layout["step"] / "hf_base.preserved" / "lora_adapter"
    assert (preserved / "adapter_model.safetensors").is_file()

    # hf_dense's rebuild recipe must point at where the adapter now lives, not at the path it
    # was read from -- that one no longer exists.
    manifest = json.loads((layout["step"] / "hf_dense.deleted.json").read_text())
    assert manifest["adapter"]["path"] == str(preserved)
    assert manifest["adapter"]["relocated_from"] == str(layout["adapter"])
    assert all(str(preserved) in cmd or cmd.startswith("#") for cmd in manifest["rebuild"])


def test_hf_base_that_is_not_a_copy_of_the_container_fails(
    layout: dict, capsys: pytest.CaptureFixture
) -> None:
    tensors = _load(layout["hf_base"])
    tensors["lm_head.weight"] = tensors["lm_head.weight"] + 1.0
    save_file(tensors, str(layout["hf_base"] / "model.safetensors"))
    # Sample everything so the perturbed tensor is certain to be drawn.
    run_expect_failure(layout, capsys, "arith", "--sample", "50", str(layout["hf_base"]))
    assert layout["hf_base"].is_dir()
    assert layout["adapter"].is_dir()


def test_hf_base_with_the_real_verl_key_mangling_and_missing_mtp_keys_passes(
    layout: dict,
) -> None:
    """The shape hf_base actually has on disk, which the synthetic fixture does not.

    ``verl.model_merger`` saves through AutoModelForVision2Seq and comes back nested one level
    too deep (``model.language_model.language_model.language_model.X``), and the actor ran with
    ``mtp_num_hidden_layers=0`` so the container's ``mtp.*`` keys are legitimately absent. Both
    must read as a faithful copy, not as a mismatch.
    """
    mangled = {}
    for key, tensor in _load(layout["hf_base"]).items():
        if key.startswith("model.language_model."):
            key = "model.language_model.language_model.language_model." + key[len("model.language_model."):]
        mangled[key] = tensor
    save_file(mangled, str(layout["hf_base"] / "model.safetensors"))

    container = _load(layout["base"])
    container["mtp.layers.0.weight"] = _base_weight(555)
    save_file(container, str(layout["base"] / "model.safetensors"))

    assert run(layout, "--delete", "--sample", "50", str(layout["hf_base"])) == 0
    assert not layout["hf_base"].exists()
    assert (layout["step"] / "hf_base.preserved" / "lora_adapter" / "adapter_model.safetensors").is_file()


def test_hf_base_missing_a_non_allowlisted_key_fails(
    layout: dict, capsys: pytest.CaptureFixture
) -> None:
    tensors = _load(layout["hf_base"])
    del tensors["lm_head.weight"]
    save_file(tensors, str(layout["hf_base"] / "model.safetensors"))
    out = run_expect_failure(layout, capsys, "arith", str(layout["hf_base"]))
    assert "not a faithful copy" in out
    assert layout["hf_base"].is_dir()


def test_existing_preserve_dir_with_different_content_is_refused(
    layout: dict, capsys: pytest.CaptureFixture
) -> None:
    stale = layout["step"] / "hf_base.preserved"
    (stale / "lora_adapter").mkdir(parents=True)
    (stale / "lora_adapter" / "adapter_config.json").write_text("{}")
    run_expect_failure(layout, capsys, "survives", str(layout["hf_base"]))
    assert layout["hf_base"].is_dir()
    assert (layout["adapter"] / "adapter_model.safetensors").is_file()


# --------------------------------------------------------------------------- path guard


def test_a_failed_preservation_copy_aborts_the_delete(
    layout: dict, capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The quota is what sends anyone here, so the copy can be the thing that fails.

    A half-written adapter followed by a successful rmtree is the single outcome this tool
    exists to prevent, so a refused write must skip the target, not crash mid-delete.
    """
    import scripts.safe_delete_derived as sdd

    def refuse(*args, **kwargs):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(sdd.shutil, "copytree", refuse)
    run_expect_failure(layout, capsys, "preserve", str(layout["hf_base"]))
    assert layout["hf_base"].is_dir()
    assert (layout["adapter"] / "adapter_model.safetensors").is_file()
    assert not (layout["step"] / "hf_base.deleted.json").exists()


def test_target_outside_results_is_refused(
    layout: dict, capsys: pytest.CaptureFixture
) -> None:
    outside = layout["root"] / "checkpoints/sft/hf_dense"
    outside.mkdir(parents=True)
    save_file(dict(layout["base_weights"]), str(outside / "model.safetensors"))
    out = run_expect_failure(layout, capsys, "guard", str(outside))
    assert "no 'results' path component" in out
    assert outside.is_dir()


def test_target_outside_the_allowed_roots_is_refused(
    tmp_path: Path, layout: dict, capsys: pytest.CaptureFixture
) -> None:
    elsewhere = tmp_path / "elsewhere/results/models/step1/hf_dense"
    elsewhere.mkdir(parents=True)
    save_file(dict(layout["base_weights"]), str(elsewhere / "model.safetensors"))
    out = run_expect_failure(layout, capsys, "guard", str(elsewhere))
    assert "outside the allowed roots" in out
    assert elsewhere.is_dir()


def test_unrecognized_directory_name_is_refused(
    layout: dict, capsys: pytest.CaptureFixture
) -> None:
    other = layout["step"] / "raw"
    other.mkdir()
    (other / "keep.txt").write_text("precious")
    out = run_expect_failure(layout, capsys, "guard", str(other))
    assert "unrecognized artifact name" in out
    assert (other / "keep.txt").is_file()


def test_nonexistent_target_fails_without_aborting(
    layout: dict, capsys: pytest.CaptureFixture
) -> None:
    missing = layout["step"] / "hf_dense_typo"
    out = run_expect_failure(layout, capsys, "guard", str(missing), str(layout["hf_dense"]))
    assert "not a directory" in out
    assert not layout["hf_dense"].exists()  # the valid one still went through


def test_symlinked_target_is_refused(layout: dict, capsys: pytest.CaptureFixture) -> None:
    link = layout["step"] / "hf_base"
    link.rename(layout["step"] / "hf_base_real")
    link.symlink_to(layout["step"] / "hf_base_real")
    out = run_expect_failure(layout, capsys, "guard", str(link))
    assert "symlink" in out
    assert (layout["step"] / "hf_base_real" / "lora_adapter").is_dir()


# --------------------------------------------------------------------------- misc


def test_base_recorded_under_a_vanished_work_dir_is_re_anchored(layout: dict) -> None:
    # Real reports record the base as the job saw it: inside results/<run>/work/launcher-*/,
    # which is routinely cleaned up long before anyone reclaims disk.
    (layout["hf_dense"] / "grpo_merge_report.json").write_text(json.dumps({
        "base": str(layout["root"] / "results/run/work/launcher-1/checkpoints/sft/merged_ep3"),
        "adapter": str(layout["adapter"]),
    }))
    assert run(layout, "--delete", str(layout["hf_dense"])) == 0
    manifest = json.loads((layout["step"] / "hf_dense.deleted.json").read_text())
    assert manifest["base"]["path"] == str(layout["base"])


def test_json_report_lists_every_target(layout: dict, tmp_path: Path) -> None:
    report_path = tmp_path / "report.json"
    run(layout, "--json", str(report_path), str(layout["hf_dense"]), str(layout["hf_base"]))
    report = json.loads(report_path.read_text())
    assert report["delete"] is False
    assert {t["kind"] for t in report["targets"]} == {"hf_dense", "hf_base"}
    assert all(t["ok"] and not t["deleted"] for t in report["targets"])
    assert report["bytes_reclaimable"] > 0
