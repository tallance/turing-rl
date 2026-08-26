import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval.generate_trained import resolve_adapter_for_run


def test_base_model_flag_forces_no_adapter():
    assert resolve_adapter_for_run("checkpoints/whatever", base_model=True) is None


def test_missing_adapter_raises_when_not_base():
    try:
        resolve_adapter_for_run("/definitely/not/a/checkpoint/dir", base_model=False)
    except ValueError as e:
        assert "No LoRA adapter" in str(e)
    else:
        raise AssertionError("expected ValueError for missing adapter")
