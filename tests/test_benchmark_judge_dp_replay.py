import json
from pathlib import Path

from scripts.benchmark_judge_dp_replay import build_body, load_prompts


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


def test_slurm_harness_allows_separate_server_environment():
    harness = (REPO_ROOT / "scripts/slurm/judge_dp_replay.sh").read_text()

    assert "SERVER_ENV=${SERVER_ENV:-/home/lancewicki/miniconda3/envs/turing-rl-train}" in harness
    assert "PY_SERVER=$SERVER_ENV/bin/python" in harness
    assert "VLLM=$SERVER_ENV/bin/vllm" in harness
    assert "PY_CLIENT=/home/lancewicki/miniconda3/envs/turing-rl-train/bin/python" in harness
    assert '"$PY_CLIENT" scripts/benchmark_judge_dp_replay.py' in harness
