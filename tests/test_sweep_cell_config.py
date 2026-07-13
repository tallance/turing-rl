from configs.judge_sweep_cells import tp_for_size, cell_list, SIZE_MAP, ANCHOR_CELL


def test_tp_lookup():
    assert tp_for_size(4, False) == (1, 8)
    assert tp_for_size(14, False) == (1, 8)
    assert tp_for_size(27, False) == (2, 4)
    assert tp_for_size(32, False) == (2, 4)
    assert tp_for_size(35, True) == (1, 8)  # MoE-Int4


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
