"""Guard: judge fidelity env knobs reach the outgoing judge HTTP payload.

The env vars ``PERSONA_JUDGE_SAMPLING`` and ``PERSONA_JUDGE_ENABLE_THINKING``
are read inside ``training.grpo.reward._openai_chat`` (NOT inside
``build_chat_payload``). This test exercises that real call path end to end:
it sets the env vars, invokes the real ``_openai_chat`` (which reads the env,
builds ``sampling`` / ``chat_template_kwargs``, and calls the real
``build_chat_payload`` merge), and captures the exact dict that would be POSTed.

Only the HTTP transport (``post_chat_async``) is stubbed -- that is the network
boundary, not the payload-building logic. The env-reading and the sampling merge
are all real, so a future change that silently drops a knob will fail here.
"""

import asyncio
import json

import training.grpo.reward as reward


def _run_openai_chat_capture(monkeypatch):
    """Call the real ``_openai_chat`` and return the payload it would POST."""
    captured = {}

    async def _fake_post_chat_async(session, payload, *, semaphore, max_retries=None):
        captured["payload"] = payload
        return ""

    # Stub only the network transport; env-reading + build_chat_payload stay real.
    monkeypatch.setattr(reward, "post_chat_async", _fake_post_chat_async)

    async def _drive():
        return await reward._openai_chat(
            session=None,
            messages=[{"role": "user", "content": "hi"}],
            api_key="unused-key",
            model="Qwen/Qwen3.5-9B",
        )

    asyncio.run(_drive())
    assert "payload" in captured, "post_chat_async was never called"
    return captured["payload"]


def test_sampling_and_thinking_reach_payload(monkeypatch):
    monkeypatch.setenv(
        "PERSONA_JUDGE_SAMPLING",
        json.dumps({"repetition_penalty": 1.1, "temperature": 0.6}),
    )
    monkeypatch.setenv("PERSONA_JUDGE_ENABLE_THINKING", "1")

    payload = _run_openai_chat_capture(monkeypatch)

    # Sampling knobs are spread top-level (OpenAI-compatible).
    assert payload.get("repetition_penalty") == 1.1
    assert payload.get("temperature") == 0.6
    # Thinking-on is expressed via chat_template_kwargs.enable_thinking.
    assert payload["chat_template_kwargs"]["enable_thinking"] is True


def test_thinking_off_env_sets_enable_thinking_false(monkeypatch):
    """Guard the inverse: ENABLE_THINKING=0 must emit enable_thinking=False,
    not drop the knob (so a served judge with thinking-off is honored)."""
    monkeypatch.delenv("PERSONA_JUDGE_SAMPLING", raising=False)
    monkeypatch.setenv("PERSONA_JUDGE_ENABLE_THINKING", "0")

    payload = _run_openai_chat_capture(monkeypatch)

    assert payload["chat_template_kwargs"]["enable_thinking"] is False


def test_no_thinking_env_omits_chat_template_kwargs(monkeypatch):
    """Absent env => no chat_template_kwargs (leave the served default alone)."""
    monkeypatch.delenv("PERSONA_JUDGE_SAMPLING", raising=False)
    monkeypatch.delenv("PERSONA_JUDGE_ENABLE_THINKING", raising=False)

    payload = _run_openai_chat_capture(monkeypatch)

    assert "chat_template_kwargs" not in payload
