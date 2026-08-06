import asyncio
import json
import time
from pathlib import Path
from types import SimpleNamespace

import scripts.benchmark_judge_dp_replay as replay_module
from scripts.benchmark_judge_dp_replay import CallResult, build_body, load_prompts, select_prompt


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_load_prompts_selects_first_validation_rows(tmp_path):
    dump = tmp_path / "reward.jsonl"
    rows = [
        {"split": "train", "judge_prompt": "train"},
        {"split": "val", "judge_prompt": "val-1"},
        {"split": "val", "judge_prompt": ""},
        {"split": "val", "judge_prompt": "val-2"},
        {"split": "val", "judge_prompt": "val-3"},
    ]
    dump.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    assert load_prompts(dump, n=2, split="val") == ["val-1", "val-2"]


def test_build_body_matches_production_judge_payload():
    payload = build_body("judge this")

    assert payload["model"] == "Qwen/Qwen3.5-9B"
    assert payload["messages"] == [{"role": "user", "content": "judge this"}]
    assert payload["max_completion_tokens"] == 8192
    assert payload["temperature"] == 0.6
    assert payload["repetition_penalty"] == 1.1
    assert payload["chat_template_kwargs"] == {"enable_thinking": True}
    assert payload["response_format"] == {"type": "json_object"}
    assert payload["provider"] == {
        "order": ["Morph"],
        "allow_fallbacks": False,
    }


def test_select_prompt_cycles_a_fixed_workload():
    prompts = ["a", "b", "c"]

    assert [select_prompt(prompts, i) for i in range(7)] == [
        "a",
        "b",
        "c",
        "a",
        "b",
        "c",
        "a",
    ]


def test_replay_fixed_and_duration_modes_use_bounded_cyclic_workload(tmp_path, monkeypatch):
    dump = tmp_path / "reward.jsonl"
    dump.write_text(
        "\n".join(
            json.dumps({"split": "val", "judge_prompt": prompt})
            for prompt in ["a", "b", "c"]
        )
        + "\n"
    )
    calls = []

    async def fake_one_call(
        session, endpoint, semaphore, call_idx, prompt, model, timeout_s
    ):
        del session, endpoint, semaphore, model, timeout_s
        calls.append((call_idx, prompt))
        await asyncio.sleep(0.002)
        now = time.time()
        return CallResult(
            call_idx=call_idx,
            prompt_sha256="hash",
            started_at=now,
            completed_at=now,
            latency_s=0.0,
            http_ok=True,
            http_status=200,
            prompt_tokens=1,
            completion_tokens=1,
            finish_reason="stop",
            reasoning_chars=0,
            content_chars=2,
        )

    monkeypatch.setattr(replay_module, "one_call", fake_one_call)
    args = SimpleNamespace(
        dumps=dump,
        n=3,
        split="val",
        duration=0.0,
        out=tmp_path / "fixed",
        endpoint="http://localhost:1/v1",
        concurrency=2,
        model="model",
        timeout=1.0,
    )

    fixed_summary = asyncio.run(replay_module.replay(args))

    assert fixed_summary["requests"] == 3
    assert fixed_summary["scheduled_requests"] == 3
    assert [prompt for _, prompt in calls] == ["a", "b", "c"]

    calls.clear()
    args.duration = 0.03
    args.out = tmp_path / "duration"
    duration_summary = asyncio.run(replay_module.replay(args))

    assert duration_summary["requests"] > 3
    assert duration_summary["scheduled_requests"] == duration_summary["requests"]
    assert [prompt for _, prompt in calls[:7]] == ["a", "b", "c", "a", "b", "c", "a"]


def test_slurm_harness_allows_separate_server_environment():
    harness = (REPO_ROOT / "scripts/slurm/judge_dp_replay.sh").read_text()

    assert "SERVER_ENV=${SERVER_ENV:-/home/lancewicki/miniconda3/envs/turing-rl-train}" in harness
    assert "PY_SERVER=$SERVER_ENV/bin/python" in harness
    assert "VLLM=$SERVER_ENV/bin/vllm" in harness
    assert "PY_CLIENT=/home/lancewicki/miniconda3/envs/turing-rl-train/bin/python" in harness
    assert '"$PY_CLIENT" scripts/benchmark_judge_dp_replay.py' in harness


def test_slurm_harness_exposes_frontend_count_and_samples_engine_metrics():
    harness = (REPO_ROOT / "scripts/slurm/judge_dp_replay.sh").read_text()

    assert "API_SERVER_COUNT=${API_SERVER_COUNT:-8}" in harness
    assert '--api-server-count "$API_SERVER_COUNT"' in harness
    assert "METRICS_LOG=$OUT/metrics.log" in harness
    assert 'http://localhost:$PORT/metrics' in harness
    assert "DURATION=${DURATION:-0}" in harness
    assert '--duration "$DURATION"' in harness
