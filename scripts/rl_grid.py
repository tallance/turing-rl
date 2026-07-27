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
