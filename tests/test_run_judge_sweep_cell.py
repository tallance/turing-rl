import pytest

from scripts.calibration_report import extrapolate_wall_hours
from scripts.run_judge_sweep_cell import (
    _api_key_for_endpoint,
    _final_metadata,
    _raise_on_scoring_errors,
    cell_env,
    cell_output_dirs,
    resolve_prompt_style,
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


def test_output_dirs_fold_in_a_non_default_style(tmp_path):
    # The two arms must not share a directory: the single-token dump is a different
    # schema, and appending it into a full-schema cell corrupts both numbers.
    d = cell_output_dirs(str(tmp_path), "qwen3-8b", "off", "single_token")
    assert d["reward"].endswith("qwen3-8b/off/single_token/reward")
    assert d["http"].endswith("qwen3-8b/off/single_token/http")


def test_prompt_style_defaults_to_full_and_rejects_junk(monkeypatch):
    monkeypatch.delenv("JUDGE_PROMPT_STYLE", raising=False)
    assert resolve_prompt_style() == "full"
    monkeypatch.setenv("JUDGE_PROMPT_STYLE", "single_token")
    assert resolve_prompt_style() == "single_token"
    # An explicit argument wins over the env so the sbatch flag is authoritative.
    assert resolve_prompt_style("full") == "full"
    with pytest.raises(ValueError, match="full|single_token"):
        resolve_prompt_style("freeform")


def test_single_token_env_describes_the_request_that_is_actually_sent():
    """run_metadata.json is copied from this env. The full-schema values would claim a
    37-field JSON constraint, an 8192-token budget and (in an 'on' cell) thinking on --
    none of which the single-token scorer sends."""
    env = cell_env(model_id="Qwen/Qwen3-8B", mode="on", out_dir="/tmp/x", style="single_token")
    assert "PERSONA_JUDGE_JSON_SCHEMA" not in env
    assert env["PERSONA_JUDGE_MAX_COMPLETION_TOKENS"] == "1"
    assert env["PERSONA_JUDGE_ENABLE_THINKING"] == "0"
    assert env["JUDGE_PROMPT_STYLE"] == "single_token"
    assert env["PERSONA_REWARD_DUMP_DIR"] == "/tmp/x/reward"


def test_full_style_env_is_unchanged():
    env = cell_env(model_id="Qwen/Qwen3-8B", mode="on", out_dir="/tmp/x")
    assert env["PERSONA_JUDGE_JSON_SCHEMA"] == "1"
    assert env["PERSONA_JUDGE_MAX_COMPLETION_TOKENS"] == "8192"
    assert env["PERSONA_JUDGE_ENABLE_THINKING"] == "1"
    assert env["JUDGE_PROMPT_STYLE"] == "full"


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


# --------------------------------------------------------------------------- dispatch
#
# The defect this covers: JUDGE_PROMPT_STYLE=single_token passed every guard in the
# launcher and then scored the OLD 37-field protocol, writing the result into a directory
# named single-token -- a plausible-but-wrong number. These tests drive the real
# async_main, so the dispatch is exercised where it actually lives.


def _pair_set(tmp_path):
    import pandas as pd

    path = tmp_path / "pairs.parquet"
    pd.DataFrame(
        [{
            "pair_id": "p0", "generated": "gen turn", "human": "human turn",
            "user_history": "[HUMAN]: past", "context": "[OTHER]: hi",
            "user_id": "u", "post_id": "post", "target_idx": 0,
        }]
    ).to_parquet(path)
    return path


def _run_cell(monkeypatch, tmp_path, style, mode="off"):
    """Run one shard end-to-end with both scorers stubbed; return which one was called."""
    import asyncio

    import training.grpo.reward as reward_module
    from eval import single_token_judge as single_token_module
    from scripts import run_judge_sweep_cell

    called = []

    async def fake_single_token(session, api_key, *args, **kwargs):
        called.append(("single_token", kwargs))
        return {}

    async def fake_full(session, api_key, *args, **kwargs):
        called.append(("full", kwargs))
        return {}

    monkeypatch.setattr(single_token_module, "score_single_token_with_info", fake_single_token)
    monkeypatch.setattr(reward_module, "score_turing_with_info", fake_full)
    monkeypatch.setenv("JUDGE_PROMPT_STYLE", style)
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_judge_sweep_cell.py",
            "--pairs", str(_pair_set(tmp_path)),
            "--endpoints", "http://localhost:8123/v1",
            "--model", "Qwen/Qwen3.5-9B",
            "--thinking_mode", mode,
            "--out_dir", str(tmp_path / "sweep"),
            "--cell_name", "qwen35-9b",
        ],
    )
    asyncio.run(run_judge_sweep_cell.async_main())
    return called


def test_single_token_style_reaches_the_single_token_scorer(monkeypatch, tmp_path):
    called = _run_cell(monkeypatch, tmp_path, "single_token")

    assert [name for name, _ in called] == ["single_token"]
    assert called[0][1]["pair_id"] == "p0"
    # ...and the cell writes under the style-scoped path, not the full-schema one.
    assert (tmp_path / "sweep/qwen35-9b/off/single_token/run_metadata.json").is_file()
    assert not (tmp_path / "sweep/qwen35-9b/off/reward").exists()


def test_default_style_still_reaches_the_reward_path(monkeypatch, tmp_path):
    called = _run_cell(monkeypatch, tmp_path, "full")

    assert [name for name, _ in called] == ["full"]
    assert "pair_id" not in called[0][1]
    assert (tmp_path / "sweep/qwen35-9b/off/run_metadata.json").is_file()


def test_single_token_with_thinking_on_is_refused_before_anything_is_written(
    monkeypatch, tmp_path
):
    """This process creates the mode/style directory and stamps thinking_mode into
    run_metadata.json, so warning and continuing still leaves a whole cell of artifacts
    attributed to the thinking-on arm for a run whose every request pinned
    enable_thinking=False. The launcher's guard does not cover a cell submitted straight
    through snapshot_sbatch.sh."""
    with pytest.raises(SystemExit, match="thinking_mode=on"):
        _run_cell(monkeypatch, tmp_path, "single_token", mode="on")

    assert not (tmp_path / "sweep").exists()


def test_the_metadata_records_the_style_and_the_request_it_describes(monkeypatch, tmp_path):
    import json

    _run_cell(monkeypatch, tmp_path, "single_token")
    meta = json.loads(
        (tmp_path / "sweep/qwen35-9b/off/single_token/run_metadata.json").read_text()
    )

    assert meta["prompt_style"] == "single_token"
    assert meta["json_schema"] is None
    assert meta["max_completion_tokens"] == "1"
    assert meta["enable_thinking"] == "0"
