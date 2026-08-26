# tests/test_temp1_valcard_overrides.py
"""L1 offline guard for the temp-1.0 / model-card-val overfit probe (plan
`before-we-move-on-hazy-puppy`). Catches silent-override typos before any cluster time."""
import pathlib
import pytest

from scripts.rl_grid import temp1_valcard_overrides, TEMP1_VALCARD

OVR = temp1_valcard_overrides()
REWARD_SRC = pathlib.Path("training/grpo/reward.py").read_text()


def test_train_rollout_has_no_truncation():
    # RL rollout must sample from the full policy softmax (on-policy) -> temp 1.0, no top_p/top_k cut.
    assert "actor_rollout_ref.rollout.temperature=1" in OVR
    assert "actor_rollout_ref.rollout.top_p=1" in OVR
    assert "actor_rollout_ref.rollout.top_k=-1" in OVR


def test_val_is_full_model_card():
    for tok in (
        "actor_rollout_ref.rollout.val_kwargs.temperature=0.7",
        "actor_rollout_ref.rollout.val_kwargs.top_p=0.8",
        "actor_rollout_ref.rollout.val_kwargs.top_k=20",
        "actor_rollout_ref.rollout.val_kwargs.do_sample=true",
        "actor_rollout_ref.rollout.val_kwargs.n=1",
    ):
        assert tok in OVR, f"missing val override: {tok}"


def test_val_loop_enabled():
    assert "trainer.test_freq=1" in OVR          # else veRL default -1 = no validation
    assert "trainer.val_before_train=true" in OVR


def test_kl_lr_and_target():
    assert "actor_rollout_ref.actor.kl_loss_coef=0.0001" in OVR
    assert "actor_rollout_ref.actor.optim.lr=0.0001" in OVR
    assert "target_modules=[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]" in OVR
    assert "all-linear" not in OVR               # never LoRA the GDN backbone


def test_train_and_val_temp_differ():
    assert TEMP1_VALCARD["train_temp"] != TEMP1_VALCARD["val_temp"]  # copy-paste guard


def test_word_split_safe():
    # launcher passes EXTRA_OVERRIDES UNQUOTED (word-split on spaces); each token must be space-free.
    assert ", " not in OVR                        # no space after commas in the target list
    assert "  " not in OVR                        # no double spaces
    for tok in OVR.split(" "):
        assert "=" in tok, f"stray token would break word-split: {tok!r}"


def test_reward_split_field_wired_in_source():
    # Source-level (no import) guard that the reward dump can carry the train/val tag.
    assert '"split")' in REWARD_SRC                        # in _REWARD_DUMP_KEYS tuple
    assert "_DUMP_SPLIT.set(" in REWARD_SRC               # set in compute_score
    assert "split=_DUMP_SPLIT.get()" in REWARD_SRC        # passed at the dump call


def test_reward_split_field_behavior():
    # Behavioral check when reward.py is importable (heavy deps may be absent offline -> skip).
    try:
        import training.grpo.reward as r
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"reward.py not importable in this env: {type(e).__name__}")
    assert "split" in r._REWARD_DUMP_KEYS
    assert r._build_reward_dump_row(split="val")["split"] == "val"
    assert r._build_reward_dump_row()["split"] is None       # allowlist -> None when key absent
    assert r._DUMP_SPLIT.get() == "train"                    # dump call uses this default ("train")
