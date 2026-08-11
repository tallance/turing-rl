from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CELL_SCRIPT = ROOT / "scripts" / "slurm" / "judge_sweep_cell.sh"


def test_gemma_runtime_is_narrow_and_snapshot_pinned():
    source = CELL_SCRIPT.read_text()
    assert "turing-rl-gemma4-vllm-nightly" in source
    assert "google/gemma-4-12B-it" in source
    assert "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7" in source
    assert "google/gemma-4-31B-it" in source
    assert "842da3794eaa0b77d5f08bae87a17459d91ff475" in source
    assert 'serve "$GEMMA_MODEL_PATH"' in source
    assert '--served-model-name "$MODEL"' in source
    assert "--reasoning-parser gemma4" in source
    assert "--reasoning-parser qwen3" in source
    assert "FLASHINFER_WORKSPACE_BASE" in source
    assert '"image":0' in source
    assert "GEMMA_GPU_MEMORY_UTILIZATION:-0.90" in source


def test_concurrent_replicas_get_disjoint_vllm_internal_port_ranges():
    source = CELL_SCRIPT.read_text()
    assert "VLLM_INTERNAL_PORT_BASE" in source
    assert "VLLM_INTERNAL_PORT_STRIDE" in source
    assert 'internal_port=$((VLLM_INTERNAL_PORT_BASE + i * VLLM_INTERNAL_PORT_STRIDE))' in source
    assert source.count("VLLM_PORT=$internal_port CUDA_VISIBLE_DEVICES=$gpus") == 2
