"""Unit tests for the served (self-hosted, thinking-off) CoT client.

Covers only the pure logic (payload shape + round-robin endpoint selection);
no live HTTP is exercised here. The full-run behaviour is validated on the
cluster via scripts/slurm/cot_serve.sh.
"""

from scripts.generate_cot_served import build_cot_payload, pick_endpoint


def test_thinking_off_payload():
    # The client calls build_cot_payload WITHOUT sampling: matching upstream
    # generate_cot.py, no temperature/top_p/top_k/min_p go on the wire — vLLM
    # applies each served model's generation_config.json defaults.
    p = build_cot_payload(
        "Qwen/Qwen3-8B",
        [{"role": "user", "content": "hi"}],
        max_completion_tokens=4096,
    )
    assert p["chat_template_kwargs"] == {"enable_thinking": False}
    assert "reasoning" not in p and "provider" not in p
    assert p["max_completion_tokens"] == 4096
    assert "temperature" not in p  # no sampling override on the wire


def test_sampling_optional_passthrough():
    # sampling stays optional so it CAN be sent if ever needed.
    p = build_cot_payload(
        "Qwen/Qwen3-8B",
        [{"role": "user", "content": "hi"}],
        sampling={"temperature": 0.7},
        max_completion_tokens=4096,
    )
    assert p["temperature"] == 0.7


def test_round_robin():
    eps = ["a", "b", "c"]
    assert [pick_endpoint(eps, i) for i in range(4)] == ["a", "b", "c", "a"]
