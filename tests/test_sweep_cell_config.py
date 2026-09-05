import pytest

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

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


def test_ce_judge_cells_serve_like_the_zero_shot_9b_they_are_compared_against():
    """J1/J2 must match J0's serving shape, or the comparison confounds two variables.

    J0 is the zero-shot `qwen35-9b` family cell; J1/J2 are the same backbone after CE
    training. If the CE cells were served at a different tp/replicas, a difference in their
    scores could come from sharding rather than from the training the eval is measuring.
    """
    j0 = next(c for c in cell_list("qwen3.5") if c["cell_name"] == "qwen35-9b")
    for name in ("9b-ce", "9b-ce2", "9b-ce3", "9b-ce4"):
        cell = resolve_cell(name)
        assert (cell["tp"], cell["replicas"]) == (j0["tp"], j0["replicas"]), name
        assert cell["size_b"] == 9, name
        assert SIZE_MAP[name] == SIZE_MAP["qwen35-9b"], name


def test_ce_judge_cells_point_at_servable_absolute_merges():
    """Absolute, and the _dense merge -- not the _nopack adapter dir.

    vLLM cannot serve a PEFT adapter directory, and the cell is resolved inside a job whose
    checkpoints/ is a symlink. Either mistake surfaces only once an 8-GPU cell has been
    scheduled and the server fails to load, so pin both here.
    """
    for name, stem in (("9b-ce", "judge_qwen35_9b_ce_dense"),
                       ("9b-ce2", "judge_qwen35_9b_ce_iter2_dense"),
                       ("9b-ce3", "judge_qwen35_9b_ce_iter3_dense"),
                       ("9b-ce4", "judge_qwen35_9b_ce_iter4_dense")):
        model_id = resolve_cell(name)["model_id"]
        assert model_id.startswith("/"), f"{name} must be absolute: {model_id}"
        assert model_id.endswith(stem), f"{name} -> {model_id}"
        assert "_nopack" not in model_id, f"{name} points at an adapter dir: {model_id}"


def test_every_opt_in_cell_is_reachable_from_the_launcher():
    """The launcher offers extra_cell_names(), so registering a cell is enough to use it.

    It used to hardcode its own subset, so a newly registered judge was rejected with a
    message blaming the family list -- a confusing failure at submit time.
    """
    from configs.judge_sweep_cells import extra_cell_names

    names = extra_cell_names()
    for required in ("9b-ce", "9b-ce2", "9b-ce3", "9b-ce4", "gemma4-12b", "gemma4-31b"):
        assert required in names, required
    # Enumerable, not a fixed list: every name must actually resolve.
    for name in names:
        assert extra_cell(name)["cell_name"] == name

    launcher = (ROOT / "scripts" / "launch_test_eval.sh").read_text()
    assert "extra_cell(n) for n in extra_cell_names()" in launcher
    for stale in ("extra_cell('gemma4-12b')", "extra_cell('gemma4-31b')"):
        assert stale not in launcher, f"launcher still hardcodes {stale}"


def test_unknown_opt_in_cell_fails_loudly():
    with pytest.raises(ValueError, match="unknown extra judge cell"):
        extra_cell("gemma4-99b")
