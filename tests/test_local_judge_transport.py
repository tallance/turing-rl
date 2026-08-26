from __future__ import annotations

import asyncio

import shared.api_client as api_client


class _FakeResponse:
    status = 200
    reason = "OK"
    headers: dict[str, str] = {}

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def raise_for_status(self):
        return None

    async def json(self):
        return {
            "choices": [{
                "message": {"content": "ok"},
                "finish_reason": "stop",
                "logprobs": {"content": [{"token": "A", "top_logprobs": []}]},
            }]
        }


class _FakeSession:
    def __init__(self):
        self.headers = None

    def post(self, url, *, headers, json):
        self.headers = headers
        return _FakeResponse()


def test_async_transport_uses_supplied_local_key_without_secret_lookup(monkeypatch):
    def fail_if_called():
        raise AssertionError("secret resolver called despite an explicit local key")

    monkeypatch.setattr(api_client, "resolve_judge_api_key", fail_if_called)
    monkeypatch.setattr(api_client, "aiohttp", object())
    monkeypatch.setenv("OPENAI_API_BASE", "http://localhost:8123/v1")
    session = _FakeSession()

    result = asyncio.run(
        api_client.post_chat_async(
            session,
            {"model": "local", "messages": []},
            semaphore=asyncio.Semaphore(1),
            api_key="EMPTY",
        )
    )

    assert result == "ok"
    assert session.headers["Authorization"] == "Bearer EMPTY"


def test_choice_transport_returns_the_first_choice_so_logprobs_are_reachable(monkeypatch):
    """The single-token protocol's verdict lives in logprobs, not in the content, so it
    needs the choice object -- and it needs the choice, not the whole response body."""
    monkeypatch.setattr(api_client, "aiohttp", object())
    monkeypatch.setenv("OPENAI_API_BASE", "http://localhost:8123/v1")

    choice = asyncio.run(
        api_client.post_chat_choice_async(
            _FakeSession(),
            {"model": "local", "messages": []},
            semaphore=asyncio.Semaphore(1),
            api_key="EMPTY",
        )
    )

    assert choice["logprobs"]["content"][0]["token"] == "A"
    assert "choices" not in choice


def test_both_async_transports_share_one_telemetry_path(monkeypatch):
    monkeypatch.setattr(api_client, "aiohttp", object())
    monkeypatch.setenv("OPENAI_API_BASE", "http://localhost:8123/v1")

    async def _both():
        payload = {"model": "local", "messages": []}
        await api_client.post_chat_async(
            _FakeSession(), payload, semaphore=asyncio.Semaphore(1), api_key="EMPTY"
        )
        from_content = api_client.get_judge_call_meta()
        await api_client.post_chat_choice_async(
            _FakeSession(), payload, semaphore=asyncio.Semaphore(1), api_key="EMPTY"
        )
        return from_content, api_client.get_judge_call_meta()

    from_content, from_choice = asyncio.run(_both())

    assert from_content["finish_reason"] == "stop"
    assert from_choice["finish_reason"] == "stop"
