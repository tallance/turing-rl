import yaml

from training.sft.lora_sft import LORA_TARGET_MODULES, MODEL_MAP, get_lora_targets


def test_judge_aliases_exist_and_point_at_the_right_bases():
    assert MODEL_MAP["qwen35-4b-judge"] == "Qwen/Qwen3.5-4B"
    assert MODEL_MAP["qwen35-9b-judge"] == "Qwen/Qwen3.5-9B"


def test_judge_aliases_get_the_hybrid_safe_lora_targets():
    """LoRA on the Gated-DeltaNet backbone is destructive on Qwen3.5; the alias must not
    fall through to the qwen3 target list."""
    targets = get_lora_targets("qwen35-9b-judge")
    assert "in_proj_qkv" not in targets
    assert "q_proj" in targets and "gate_proj" in targets


def test_judge_alias_selects_the_qwen35_branch_not_the_qwen3_branch(monkeypatch):
    """qwen3 and qwen3.5 target lists are identical today, so comparing contents
    cannot tell the branches apart. Make them differ, then check which one the alias
    actually resolves to — LoRA on Qwen3.5's Gated-DeltaNet backbone is destructive,
    so falling through to the qwen3 branch must be detectable."""
    monkeypatch.setitem(LORA_TARGET_MODULES, "qwen3.5", ["SENTINEL_35"])
    monkeypatch.setitem(LORA_TARGET_MODULES, "qwen3", ["SENTINEL_3"])
    assert get_lora_targets("qwen35-9b-judge") == ["SENTINEL_35"]
    assert get_lora_targets("qwen35-4b-judge") == ["SENTINEL_35"]


def test_judge_configs_disable_packing_and_stop_token_supervision():
    for alias in ("qwen35_4b_judge", "qwen35_9b_judge"):
        cfg = yaml.safe_load(open(f"training/sft/configs/{alias}_lora.yaml"))
        # Packing concatenates examples and destroys the one-token target boundary.
        assert cfg["packing"] is False
        # Supervising <|im_end|> would make the target two tokens, not one.
        assert cfg["supervise_stop_token"] is False


def test_generator_sft_config_is_untouched():
    cfg = yaml.safe_load(open("training/sft/configs/qwen35_9b_lora.yaml"))
    assert cfg["supervise_stop_token"] is True
