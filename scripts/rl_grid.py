"""Arm-A / Arm-B GRPO overfit grid: KL x LR cells for the proper-checkpoint reward-hack repeat."""
from __future__ import annotations


def _fmt(x: float) -> str:
    # compact tag token: 1e-3 -> "1e3", 1e-4 -> "1e4", 0 -> "0"
    if x == 0:
        return "0"
    exp = round(-__import__("math").log10(x))
    return f"1e{exp}"


def _cells(model: str) -> list[dict]:
    kls = [1e-3, 1e-4, 0.0]
    lrs = [1e-5, 1e-4]
    out = []
    for kl in kls:
        for lr in lrs:
            out.append({"tag": f"{model}_proper_kl{_fmt(kl)}_lr{_fmt(lr)}", "kl": kl, "lr": lr})
    return out


ARM_A_CELLS = _cells("8b")
ARM_B_CELLS = _cells("9b")


def cell_overrides(cell: dict) -> str:
    """Hydra override string for EXTRA_OVERRIDES (KL + LR)."""
    return (
        f"actor_rollout_ref.actor.kl_loss_coef={cell['kl']:g} "
        f"actor_rollout_ref.actor.optim.lr={cell['lr']:g}"
    )


# LoRA target matched to the SFT recipe (attn+MLP; never all-linear -> would hit the GDN backbone).
_ATTN_MLP_TARGET = "actor_rollout_ref.model.target_modules=[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]"

# One-off temp-1.0 exploration probe (kl1e-4/lr1e-4) with a model-card temp-0.7 validation loop.
# Train rollout = full-distribution (temp 1.0, top_p 1.0, top_k -1) = on-policy max exploration.
# Val = Qwen3-8B non-thinking model card (temp 0.7, top_p 0.8, top_k 20; min_p 0 default). See
# docs plan `before-we-move-on-hazy-puppy`.
TEMP1_VALCARD = {
    "tag": "8b_proper_kl1e4_lr1e4_temp1_valcard",
    "kl": 1e-4, "lr": 1e-4,
    "train_temp": 1.0, "train_top_p": 1.0, "train_top_k": -1,
    "val_temp": 0.7, "val_top_p": 0.8, "val_top_k": 20, "val_n": 1,
    "test_freq": 1, "val_before_train": True,
}


def temp1_valcard_overrides() -> str:
    """SSOT Hydra override string for the temp-1.0 / val-card overfit probe (space-separated,
    each token space-free so the launcher's unquoted word-split is safe)."""
    c = TEMP1_VALCARD
    toks = [
        f"actor_rollout_ref.actor.kl_loss_coef={c['kl']:g}",
        f"actor_rollout_ref.actor.optim.lr={c['lr']:g}",
        _ATTN_MLP_TARGET,
        f"actor_rollout_ref.rollout.temperature={c['train_temp']:g}",
        f"actor_rollout_ref.rollout.top_p={c['train_top_p']:g}",
        f"actor_rollout_ref.rollout.top_k={c['train_top_k']}",
        f"actor_rollout_ref.rollout.val_kwargs.temperature={c['val_temp']:g}",
        f"actor_rollout_ref.rollout.val_kwargs.top_p={c['val_top_p']:g}",
        f"actor_rollout_ref.rollout.val_kwargs.top_k={c['val_top_k']}",
        "actor_rollout_ref.rollout.val_kwargs.do_sample=true",
        f"actor_rollout_ref.rollout.val_kwargs.n={c['val_n']}",
        f"trainer.test_freq={c['test_freq']}",
        f"trainer.val_before_train={'true' if c['val_before_train'] else 'false'}",
    ]
    return " ".join(toks)
