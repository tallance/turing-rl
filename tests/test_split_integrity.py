import os, pytest
from scripts.verify_rl_splits import check_splits

BASE = "data/prism/full_s42_history_sft40_grpo60_test10"

@pytest.mark.skipif(not os.path.exists(f"{BASE}/test.parquet"), reason="PRISM splits not present (cluster-only)")
def test_no_leakage_and_counts():
    r = check_splits(BASE)
    assert r["counts"] == {"grpo/train": (4174, 696), "grpo/val": (705, 696), "test": (880, 128)}
    assert r["overlaps"]["train_test"] == 0
    assert r["overlaps"]["val_test"] == 0
    assert r["overlaps"]["sft_grpo"] == 0
    assert r["overlaps"]["sft_test"] == 0
