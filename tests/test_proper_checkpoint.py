from scripts.merge_sft_adapter import (
    DEFAULT_ADAPTER_DIR, DEFAULT_OUTPUT_DIR,
    PROPER_ADAPTER_DIR_8B, PROPER_MERGED_DIR_8B,
)

def test_proper_is_distinct_from_buggy():
    assert PROPER_ADAPTER_DIR_8B != DEFAULT_ADAPTER_DIR
    assert PROPER_MERGED_DIR_8B != DEFAULT_OUTPUT_DIR

def test_proper_points_at_epochsave_checkpoint78():
    assert PROPER_ADAPTER_DIR_8B.name == "checkpoint-78"
    assert "epochsave" in str(PROPER_ADAPTER_DIR_8B)
    assert str(DEFAULT_ADAPTER_DIR).endswith("/final")   # buggy stays the old one
