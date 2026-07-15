"""Regression guard: lock the paper-faithful GRPO training params in the base config.

`training/grpo/configs/qwen3_8b_grpo.yaml` is the SSOT for GRPO hyperparameters; the
fresh launcher only overrides data/adapter/batch paths at run time. This test locks the
paper-faithful values (spec S2 fidelity ledger / S6) so no future edit silently drifts them.
It PASSES today and should only fail on future drift.

All locked keys are literals in this file (it composes on veRL's ppo_trainer defaults via
Hydra `defaults:`, but none of the guarded values come from that include), so a plain
yaml.safe_load of the file resolves them directly.
"""
import os

import yaml

# Anchor to repo root so the test works regardless of pytest's cwd.
CFG = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "training", "grpo", "configs", "qwen3_8b_grpo.yaml",
)


def _load():
    with open(CFG) as f:
        return yaml.safe_load(f)


def test_locked_training_params():
    c = _load()
    ar = c["actor_rollout_ref"]
    assert ar["model"]["lora_rank"] == 64 and ar["model"]["lora_alpha"] == 32
    assert ar["actor"]["ppo_epochs"] == 1
    assert ar["actor"]["ppo_mini_batch_size"] == 64
    assert ar["actor"]["use_kl_loss"] is True
    assert float(ar["actor"]["kl_loss_coef"]) == 1e-3
    assert float(ar["actor"]["clip_ratio"]) == 0.2
    assert ar["rollout"]["n"] == 4
    assert float(ar["rollout"]["temperature"]) == 0.6
    assert c["data"]["train_batch_size"] == 64            # match upstream code (paper Table 6 says 128)
    assert c["trainer"]["total_epochs"] == 3
    assert c["algorithm"]["adv_estimator"] == "grpo"


def test_grpo_train_data_path_present_or_skipped():
    # PRISM grpo split used by the launcher override (not the base convokit default).
    # Cluster-only data (gitignored); skip locally when absent -- do NOT assert-True.
    import pytest

    p = "data/prism/full_s42_history_sft40_grpo60_test10/grpo/train.parquet"
    if not os.path.exists(p):
        pytest.skip("PRISM grpo split not present locally (cluster-only)")
    assert os.path.exists(p)
