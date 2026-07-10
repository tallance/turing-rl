from training.sft.lora_sft import (
    build_fsdp_kwargs,
    resolve_resume_checkpoint,
    resolve_use_qlora,
    save_kwargs_from_config,
)


def test_resolve_auto_highest(tmp_path):
    (tmp_path/"checkpoint-10").mkdir(); (tmp_path/"checkpoint-70").mkdir()
    assert resolve_resume_checkpoint("auto", str(tmp_path)).endswith("checkpoint-70")


def test_resolve_auto_empty(tmp_path):
    assert resolve_resume_checkpoint("auto", str(tmp_path)) is None


def test_save_kwargs_steps():
    assert save_kwargs_from_config({"save_strategy": "steps", "save_steps": 10, "save_total_limit": 2}) == \
        {"save_strategy": "steps", "save_steps": 10, "save_total_limit": 2}


def test_save_kwargs_default_epoch():
    assert save_kwargs_from_config({}) == {"save_strategy": "epoch"}


def test_resolve_use_qlora_force_over_yaml_false():
    assert resolve_use_qlora({"use_qlora": False}, force_qlora=True, no_qlora=False) is True


def test_resolve_use_qlora_no_qlora_wins():
    assert resolve_use_qlora({"use_qlora": False}, force_qlora=True, no_qlora=True) is False


def test_resolve_use_qlora_yaml_true_no_flags():
    assert resolve_use_qlora({"use_qlora": True}, force_qlora=False, no_qlora=False) is True


def test_resolve_use_qlora_yaml_false_no_flags():
    assert resolve_use_qlora({"use_qlora": False}, force_qlora=False, no_qlora=False) is False


def test_build_fsdp_kwargs_empty():
    assert build_fsdp_kwargs("", None) == {}


def test_build_fsdp_kwargs_with_cls():
    kw = build_fsdp_kwargs("full_shard auto_wrap", "Qwen3DecoderLayer")
    assert kw["fsdp"] == "full_shard auto_wrap"
    assert kw["fsdp_config"]["transformer_layer_cls_to_wrap"] == ["Qwen3DecoderLayer"]
    assert kw["fsdp_config"]["use_orig_params"] is True


def test_build_fsdp_kwargs_no_cls():
    kw = build_fsdp_kwargs("full_shard auto_wrap", None)
    assert "transformer_layer_cls_to_wrap" not in kw["fsdp_config"]
