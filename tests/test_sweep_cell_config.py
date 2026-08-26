import pytest

from configs.judge_sweep_cells import (
    ANCHOR_CELL,
    SIZE_MAP,
    cell_list,
    extra_cell,
    resolve_cell,
    tp_for_size,
)


def test_tp_lookup():
    # 2nd arg is `quantized`; shape is chosen by footprint = size * (0.5 Int4 | 2.0 bf16),
    # threshold ~30GB/GPU -> fits one GPU (1,8) else whole node (8,1).
    assert tp_for_size(4, False) == (1, 8)    # 8GB bf16
    assert tp_for_size(14, False) == (1, 8)   # 28GB bf16 (just fits)
    assert tp_for_size(27, False) == (8, 1)   # 54GB bf16 -> whole node
    assert tp_for_size(32, False) == (8, 1)   # 64GB bf16 -> whole node
    assert tp_for_size(35, False) == (8, 1)   # 70GB bf16 (non-quantized 35B) -> whole node
    assert tp_for_size(35, True) == (1, 8)    # 17.5GB Int4 -> one GPU
    assert tp_for_size(397, True) == (8, 1)   # 200GB Int4 anchor -> whole node


def test_cell_list_qwen35():
    cells = cell_list("qwen3.5")
    assert len(cells) == 6  # 5 judges (4b/9b/27b/35b/122b) + anchor
    assert any(c["model_id"].endswith("397B-A17B-GPTQ-Int4") for c in cells)


def test_cell_list_qwen3():
    cells = cell_list("qwen3")
    assert len(cells) == 5  # 4 judges + anchor
    # anchor is appended for every family
    assert cells[-1]["cell_name"] == ANCHOR_CELL


def test_size_map_covers_all_cells():
    for c in cell_list("qwen3.5") + cell_list("qwen3"):
        assert c["cell_name"] in SIZE_MAP


def test_opt_in_gemma_cells_have_proven_serving_shapes():
    gemma12 = extra_cell("gemma4-12b")
    assert (gemma12["tp"], gemma12["replicas"], gemma12["concurrency"]) == (1, 8, 4)

    gemma31 = resolve_cell("gemma4-31b")
    assert gemma31["model_id"] == "google/gemma-4-31B-it"
    assert (gemma31["tp"], gemma31["replicas"], gemma31["concurrency"]) == (8, 1, 4)


def test_unknown_opt_in_cell_fails_loudly():
    with pytest.raises(ValueError, match="unknown extra judge cell"):
        extra_cell("gemma4-99b")
