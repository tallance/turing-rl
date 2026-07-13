from configs.judge_sweep_cells import tp_for_size, cell_list, SIZE_MAP, ANCHOR_CELL


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
    assert len(cells) == 5  # 4 judges + anchor (modes handled by launcher)
    assert any(c["model_id"].endswith("397B-A17B-GPTQ-Int4") for c in cells)


def test_cell_list_qwen3():
    cells = cell_list("qwen3")
    assert len(cells) == 5  # 4 judges + anchor
    # anchor is appended for every family
    assert cells[-1]["cell_name"] == ANCHOR_CELL


def test_size_map_covers_all_cells():
    for c in cell_list("qwen3.5") + cell_list("qwen3"):
        assert c["cell_name"] in SIZE_MAP
