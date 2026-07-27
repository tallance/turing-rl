from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from training.grpo import b0_rollout_sync_hook
from training.grpo import hf_compat_patches
from training.grpo import verl_runtime_patch


def test_optional_legacy_actor_module_can_be_absent():
    with patch.object(
        verl_runtime_patch.importlib.util,
        "find_spec",
        side_effect=ModuleNotFoundError("No module named 'verl.workers.actor'"),
    ):
        verl_runtime_patch._patch_actor_elbo_sft_source()


def test_teacher_forced_logprob_uses_trainer_dataproto_adapter():
    class Output:
        batch = {
            "old_log_probs": np.array([[-0.1, -0.2, -0.3]]),
            "response_mask": np.array([[0, 1, 1]]),
        }

    class Trainer:
        def __init__(self):
            self.calls = 0

        def _compute_old_log_prob(self, fixed_dp):
            self.calls += 1
            assert fixed_dp == {"fixed": True}
            return Output(), 0.0

    trainer = Trainer()
    result = b0_rollout_sync_hook._teacher_forced_logprob(trainer, {"fixed": True})

    assert trainer.calls == 1
    np.testing.assert_allclose(result, [-0.2, -0.3])


def test_runtime_env_wrapper_forwards_verl_09_config_argument():
    calls = []

    def get_runtime_env(*args, **kwargs):
        calls.append((args, kwargs))
        return {"env_vars": {"UPSTREAM": "1"}}

    constants = SimpleNamespace(get_ppo_ray_runtime_env=get_runtime_env)
    with patch.object(
        verl_runtime_patch,
        "_merge_propagated_runtime_env_vars",
        side_effect=lambda runtime_env: runtime_env,
    ):
        verl_runtime_patch._patch_ppo_ray_runtime_env(constants)
        result = constants.get_ppo_ray_runtime_env("config", mode="sync")

    assert calls == [(("config",), {"mode": "sync"})]
    assert result == {"env_vars": {"UPSTREAM": "1"}}


def test_agent_loop_postprocess_wrapper_forwards_verl_09_validate_argument():
    source = Path(verl_runtime_patch.__file__).read_text()
    assert "def patched_postprocess(self, inputs, *args, **kwargs):" in source
    assert "return original_postprocess(self, inputs, *args, **kwargs)" in source


def test_weight_sync_clears_trainer_cache_before_rollout_wakeup():
    source = (
        "        set_expandable_segments(False)\n"
        '        log_gpu_memory_usage("Before resume weights", logger=logger)\n'
        "        if self.config.rollout.free_cache_engine:\n"
        '            await self.rollout.resume(tags=["weights"])\n'
    )
    patched = verl_runtime_patch._insert_pre_resume_cache_clear(source)

    assert "aggressive_empty_cache(force_sync=True)" in patched
    assert patched.index("aggressive_empty_cache") < patched.index("await self.rollout.resume")
    assert verl_runtime_patch._insert_pre_resume_cache_clear(patched) == patched


def test_b0_env_is_injected_into_ray_runtime_env():
    with patch.dict(os.environ, {"B0_ROLLOUT_SYNC": "1", "RL_RUN_DIR": "/tmp/b0"}, clear=False):
        result = verl_runtime_patch._merge_propagated_runtime_env_vars({})

    assert result["env_vars"]["B0_ROLLOUT_SYNC"] == "1"
    assert result["env_vars"]["RL_RUN_DIR"] == "/tmp/b0"


def test_runtime_env_adds_repo_pythonpath_without_repo_root_env():
    expected_root = str(Path(verl_runtime_patch.__file__).resolve().parents[2])
    with patch.dict(os.environ, {"PYTHONPATH": ""}, clear=False):
        os.environ.pop("REPO_ROOT", None)
        result = verl_runtime_patch._with_repo_root_pythonpath({})

    assert result["PYTHONPATH"] == expected_root


def test_agent_loop_score_context_accepts_verl_09_trajectory_signature():
    outputs = [SimpleNamespace(name="first"), SimpleNamespace(name="final")]
    sample_kwargs = {"extra_info": {"post_id": "p1"}}

    output, kwargs = verl_runtime_patch._agent_loop_score_context(
        (outputs,), {"kwargs": sample_kwargs}
    )

    assert output is outputs[-1]
    assert kwargs == sample_kwargs


def test_agent_loop_score_context_keeps_legacy_signature_compatible():
    output = SimpleNamespace(name="legacy")
    sample_kwargs = {"extra_info": {"post_id": "p2"}}

    resolved_output, kwargs = verl_runtime_patch._agent_loop_score_context(
        (output, "prompts", "responses", "attention_mask", "input_ids", "position_ids", sample_kwargs),
        {},
    )

    assert resolved_output is output
    assert kwargs == sample_kwargs


def test_vllm_qwen35_rope_ignore_keys_list_is_accepted():
    import pytest

    pytest.importorskip("transformers")
    qwen35_config = pytest.importorskip("vllm.transformers_utils.configs.qwen3_5")
    assert hf_compat_patches.apply_rope_ignore_keys_compat_patch()

    config = qwen35_config.Qwen3_5TextConfig(
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10_000_000,
            "mrope_interleaved": True,
            "mrope_section": [11, 11, 10],
        }
    )

    assert config.ignore_keys_at_rope_validation == ["mrope_section", "mrope_interleaved"]


def test_attention_padding_falls_back_to_transformers_without_flash_attn():
    import importlib.util
    import pytest

    if importlib.util.find_spec("flash_attn") is not None:
        pytest.skip("standalone flash-attn is installed")
    torch = pytest.importorskip("torch")
    attention_utils = pytest.importorskip("verl.utils.attention_utils")
    pytest.importorskip("transformers.modeling_flash_attention_utils")

    assert verl_runtime_patch._patch_verl_attention_utils_without_flash_attn()
    values = torch.arange(8).reshape(2, 4, 1)
    mask = torch.tensor([[0, 1, 1, 1], [1, 1, 0, 0]])
    unpadded, indices, cu_seqlens, max_seqlen, seqlens = attention_utils.unpad_input(values, mask)

    assert unpadded.flatten().tolist() == [1, 2, 3, 4, 5]
    assert indices.tolist() == [1, 2, 3, 4, 5]
    assert cu_seqlens.tolist() == [0, 3, 5]
    assert int(max_seqlen) == 3
    assert seqlens.tolist() == [3, 2]
