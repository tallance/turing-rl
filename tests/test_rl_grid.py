# tests/test_rl_grid.py
from scripts.rl_grid import ARM_A_CELLS, cell_overrides

def test_grid_is_full_kl_by_lr_no_duplicates():
    kls = {1e-3, 1e-4, 0.0}
    lrs = {1e-5, 1e-4}
    got = {(c["kl"], c["lr"]) for c in ARM_A_CELLS}
    assert got == {(k, l) for k in kls for l in lrs}      # 6 cells, full cross
    tags = [c["tag"] for c in ARM_A_CELLS]
    assert len(tags) == len(set(tags)) == 6               # unique tags

def test_tags_encode_model_kl_lr_and_proper():
    for c in ARM_A_CELLS:
        assert c["tag"].startswith("8b_proper_")          # proper-checkpoint runs
    hack = next(c for c in ARM_A_CELLS if c["kl"] == 1e-3 and c["lr"] == 1e-4)
    assert hack["tag"] == "8b_proper_kl1e3_lr1e4"

def test_cell_overrides_string():
    hack = {"tag": "8b_proper_kl1e3_lr1e4", "kl": 1e-3, "lr": 1e-4}
    ovr = cell_overrides(hack)
    assert "actor_rollout_ref.actor.kl_loss_coef=0.001" in ovr
    assert "actor_rollout_ref.actor.optim.lr=0.0001" in ovr
