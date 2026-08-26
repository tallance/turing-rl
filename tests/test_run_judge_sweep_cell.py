import pytest

from scripts.calibration_report import extrapolate_wall_hours
from scripts.run_judge_sweep_cell import (
    _api_key_for_endpoint,
    _final_metadata,
    _raise_on_scoring_errors,
    cell_env,
    cell_output_dirs,
    shard_indices,
)


def test_local_endpoint_does_not_require_secret(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    def fail_if_called():
        raise AssertionError("remote secret resolver called for local vLLM")

    assert _api_key_for_endpoint("http://localhost:8123/v1", fail_if_called) == "EMPTY"


def test_remote_endpoint_keeps_strict_secret_resolution():
    assert _api_key_for_endpoint("https://api.example/v1", lambda: "real-key") == "real-key"


def test_scoring_errors_fail_the_cell_after_collection():
    _raise_on_scoring_errors(0)
    with pytest.raises(RuntimeError, match="3 scoring error"):
        _raise_on_scoring_errors(3)


def test_cell_env_locks_config():
    env = cell_env(model_id="Qwen/Qwen3-8B", mode="off", sampling={"temperature": 0.7},
                   out_dir="/tmp/x")
    assert env["PERSONA_JUDGE_JSON_SCHEMA"] == "1"
    assert env["PERSONA_JUDGE_DUMP_RATE"] == "1.0"
    assert env["PERSONA_JUDGE_ENABLE_THINKING"] == "0"
    assert env["PERSONA_DISABLE_OPENROUTER_EXTRAS"] == "1"
    assert env["JUDGE_MODEL"] == "Qwen/Qwen3-8B"
    assert env["PERSONA_JUDGE_MAX_COMPLETION_TOKENS"] == "8192"


def test_cell_env_thinking_on():
    env = cell_env(model_id="Qwen/Qwen3-8B", mode="on", out_dir="/tmp/x")
    assert env["PERSONA_JUDGE_ENABLE_THINKING"] == "1"


def test_cell_env_does_not_set_sampling():
    # Task-1 froze the policy: no wire override for sampling; vLLM uses each
    # model's generation_config.json defaults. cell_env must NOT emit it.
    env = cell_env(model_id="Qwen/Qwen3-8B", mode="off", sampling={"temperature": 0.7},
                   out_dir="/tmp/x")
    assert "PERSONA_JUDGE_SAMPLING" not in env


def test_cell_env_dump_dirs():
    env = cell_env(model_id="Qwen/Qwen3-8B", mode="off", out_dir="/tmp/x")
    assert env["PERSONA_JUDGE_DUMP_DIR"].endswith("/http")
    assert env["PERSONA_REWARD_DUMP_DIR"].endswith("/reward")


def test_shard():
    assert shard_indices(list(range(10)), endpoint_index=0, num_endpoints=2) == [0, 2, 4, 6, 8]
    assert shard_indices(list(range(10)), endpoint_index=1, num_endpoints=2) == [1, 3, 5, 7, 9]


def test_output_dirs(tmp_path):
    d = cell_output_dirs(str(tmp_path), "qwen3-8b", "off")
    assert d["reward"].endswith("qwen3-8b/off/reward") and d["http"].endswith("qwen3-8b/off/http")


def test_final_metadata_emits_consumer_keys():
    # Producer must emit the keys calibration_report.py reads (n_pairs, wall_seconds).
    base = {"cell_name": "qwen3-8b", "n_pairs_total": 100}
    out = _final_metadata(base, n_pairs=50, wall_seconds=150.0, ended_ts=123.0, ok=48, err=2)
    assert out["n_pairs"] == 50
    assert out["wall_seconds"] == 150.0
    assert out["ended_ts"] == 123.0
    assert out["ok"] == 48 and out["err"] == 2
    # Base fields are preserved.
    assert out["cell_name"] == "qwen3-8b" and out["n_pairs_total"] == 100


def test_final_metadata_roundtrip_gives_finite_projection():
    # The bug: report saw n=0, wall=0 -> req/s=0 -> extrapolation inf -> every cell
    # flagged >4h. With the producer emitting real values the round-trip is finite.
    out = _final_metadata({}, n_pairs=100, wall_seconds=300.0, ended_ts=1.0, ok=100, err=0)
    calls = out["n_pairs"] * 2  # calibration_report models 2 calls per pair
    proj = extrapolate_wall_hours(calls, out["wall_seconds"])
    assert proj != float("inf")
    assert proj > 0
