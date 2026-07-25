# tests/test_submit_arm_a_grid.py
import pathlib
S = pathlib.Path("scripts/slurm/submit_arm_a_grid.sh").read_text()

def test_uses_proper_merged_checkpoint_and_no_early_stop():
    assert "epochsave/merged_ep3" in S            # proper checkpoint, not /merged (buggy)
    assert "OVERFIT_EPOCHS=50" in S
    assert "MODE=overfit" in S and "JUDGE=9b" in S

def test_drives_grid_from_rl_grid_module():
    assert "rl_grid" in S                          # loops the SSOT cells, no hardcoded 6x copy-paste
    assert "rl_generator_run.sh" in S

def test_explicit_lora_target_for_h2_parity():
    # Arm A must set the target explicitly, not inherit all-linear (matches Arm B literally).
    assert "target_modules=[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]" in S
    assert "all-linear" not in S
