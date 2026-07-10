from training.sft.lora_sft import resolve_resume_checkpoint, save_kwargs_from_config


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
