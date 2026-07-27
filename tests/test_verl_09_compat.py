from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from training.grpo import b0_rollout_sync_hook
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


def test_runtime_env_adds_repo_pythonpath_without_repo_root_env():
    expected_root = str(Path(verl_runtime_patch.__file__).resolve().parents[2])
    with patch.dict(os.environ, {"PYTHONPATH": ""}, clear=False):
        os.environ.pop("REPO_ROOT", None)
        result = verl_runtime_patch._with_repo_root_pythonpath({})

    assert result["PYTHONPATH"] == expected_root
