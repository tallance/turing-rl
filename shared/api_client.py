"""OpenAI-compatible chat transport."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from contextvars import ContextVar
from typing import Any

try:  # pragma: no cover - exercised in runtime envs
    import aiohttp
except ImportError:  # pragma: no cover
    aiohttp = None  # type: ignore[assignment]

from shared.load_env import get_openai_api_base, get_openai_api_key

_OPENAI_MAX_RETRIES_CAP = 3

# Side-channel for per-call judge telemetry (finish_reason / usage / latency).
# post_chat_async stashes the metadata of its successful call here so callers can
# read it back WITHOUT changing post_chat_async's widely-used ``-> str`` signature.
# Concurrency-safe: asyncio.gather runs each top-level coroutine as its own Task
# with a copied context, so a ``.set()`` inside an awaited callee is visible only
# up that Task's own await-chain.
judge_call_meta: ContextVar = ContextVar("judge_call_meta", default=None)


def get_judge_call_meta() -> dict | None:
    """Return the telemetry stashed by the most recent ``post_chat_async`` call."""
    return judge_call_meta.get()


def get_openai_max_retries(
    *,
    default: int = _OPENAI_MAX_RETRIES_CAP,
    cap: int = _OPENAI_MAX_RETRIES_CAP,
) -> int:
    """Return the bounded retry count."""
    raw_value = os.environ.get("PERSONA_OPENAI_MAX_RETRIES", str(default))
    try:
        configured = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"PERSONA_OPENAI_MAX_RETRIES must be an integer, got {raw_value!r}") from exc
    return max(1, min(cap, configured))


def resolve_judge_api_key() -> str:
    """Resolve the judge API key."""
    get_openai_api_key(extra_env_names=("OPENROUTER_API_KEY",))
    return os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY", "")


def chat_url(api_base: str | None = None) -> str:
    """Return the chat-completions endpoint."""
    return f"{(api_base or get_openai_api_base()).rstrip('/')}/chat/completions"


def openrouter_chat_url() -> str:
    return chat_url()


def openrouter_request_extras(*, reasoning: bool) -> dict[str, Any]:
    """Build OpenRouter routing extras."""
    extras: dict[str, Any] = {}
    provider_order = [
        provider.strip()
        for provider in os.environ.get("OPENROUTER_PROVIDER_ORDER", "Morph").split(",")
        if provider.strip()
    ]
    if provider_order:
        allow_fallbacks = os.environ.get("OPENROUTER_ALLOW_FALLBACKS", "0").strip().lower()
        extras["provider"] = {
            "order": provider_order,
            "allow_fallbacks": allow_fallbacks in {"1", "true", "yes", "on"},
        }
    if reasoning:
        extras["reasoning"] = {"enabled": True}
    return extras


def build_chat_payload(
    *,
    model: str,
    messages: list[dict],
    max_completion_tokens: int,
    response_format: dict | None = None,
    reasoning: bool,
    sampling: dict | None = None,
    chat_template_kwargs: dict | None = None,
) -> dict[str, Any]:
    """Build a chat-completions payload."""
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_completion_tokens": int(max_completion_tokens),
    }
    if response_format:
        payload["response_format"] = response_format
    if sampling:
        payload.update(sampling)  # T/top_p/top_k/min_p top-level (OpenAI-compat)
    if chat_template_kwargs:
        payload["chat_template_kwargs"] = chat_template_kwargs
    if os.environ.get("PERSONA_DISABLE_OPENROUTER_EXTRAS") != "1":
        payload.update(openrouter_request_extras(reasoning=reasoning))
    return payload


def _compact_openai_error_body(body: str, max_chars: int = 1000) -> str:
    compact = " ".join((body or "").split()).strip()
    if len(compact) <= max_chars:
        return compact
    return f"{compact[:max_chars].rstrip()}..."


def _extract_chat_content(data: Any) -> str:
    choices = data.get("choices") if isinstance(data, dict) else None
    first_choice = choices[0] if isinstance(choices, list) and choices else None
    message = first_choice.get("message") if isinstance(first_choice, dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if isinstance(content, str):
        return content
    # OUR PATCH: when reasoning-parser is enabled and the model hits `length`
    # inside <think>, vLLM returns choices[0].message.content=None. Rather than
    # raising (which propagates past the retry loop and kills the whole run),
    # return "" so downstream _extract_json returns None and the reward code
    # falls back to a -0.15 penalty like any other parse failure.
    return ""


def _should_dump_judge(payload: dict) -> bool:
    """Deterministic per-payload sampling gate.

    Reads PERSONA_JUDGE_DUMP_RATE (float, default 0.0 = off). If 0, off.
    If >=1, always on. Otherwise hashes the payload and dumps if the hash
    bucket falls under the rate. Same payload always makes the same decision.
    """
    try:
        rate = float(os.environ.get("PERSONA_JUDGE_DUMP_RATE", "0.0"))
    except ValueError:
        return False
    if rate <= 0:
        return False
    if rate >= 1:
        return True
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    bucket = int.from_bytes(hashlib.md5(blob).digest()[:8], "big") / (1 << 64)
    return bucket < rate


def _dump_judge_response(payload: dict, response: Any, *, latency_ms: float) -> None:
    """Append one JSONL row per judge call to a per-worker file.

    File: ${PERSONA_JUDGE_DUMP_DIR}/judge-{slurm_job}-{pid}.jsonl
    Row : {ts, worker_pid, latency_ms, model, payload_messages, response}

    Safe under aiohttp concurrency: writes are per-process (Ray workers are
    separate PIDs) and a single JSON line is <PIPE_BUF so append is atomic
    on Linux. Any failure is swallowed so dumping never breaks training.
    """
    try:
        dump_dir = os.environ.get("PERSONA_JUDGE_DUMP_DIR")
        if not dump_dir:
            return
        os.makedirs(dump_dir, exist_ok=True)
        job_id = os.environ.get("SLURM_JOB_ID", "nojob")
        path = os.path.join(dump_dir, f"judge-{job_id}-{os.getpid()}.jsonl")
        row = {
            "ts": time.time(),
            "worker_pid": os.getpid(),
            "latency_ms": round(latency_ms, 3),
            "model": payload.get("model"),
            "payload_messages": payload.get("messages"),
            "response": response,
        }
        line = json.dumps(row, ensure_ascii=False, default=str) + "\n"
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)
    except Exception as exc:  # noqa: BLE001 - never fail training over logging
        print(f"[judge_dump] failed to append: {type(exc).__name__}: {exc}", flush=True)


async def post_chat_async(
    session,
    payload: dict,
    *,
    semaphore,
    max_retries: int | None = None,
    api_key: str | None = None,
) -> str:
    """Post a chat request with retries."""
    if aiohttp is None:
        raise ImportError("OpenRouter judge scoring requires aiohttp to be installed")
    resolved_api_key = api_key or resolve_judge_api_key()
    url = openrouter_chat_url()
    headers = {"Authorization": f"Bearer {resolved_api_key}", "Content-Type": "application/json"}
    if max_retries is None:
        max_retries = get_openai_max_retries()
    retry_sleep_seconds = max(0.0, float(os.environ.get("PERSONA_OPENAI_RETRY_SLEEP_SECONDS", "5")))
    for attempt in range(max_retries):
        try:
            retry_after = None
            async with semaphore:
                t0 = time.monotonic()
                async with session.post(url, headers=headers, json=payload) as resp:
                    if resp.status == 429:
                        body = await resp.text()
                        retry_after_header = resp.headers.get("Retry-After", "")
                        retry_after = retry_sleep_seconds
                        print(
                            "[openai] HTTP 429 rate limited "
                            f"on attempt {attempt+1}/{max_retries}; "
                            f"retry_after={retry_after_header or retry_after}; "
                            f"body={_compact_openai_error_body(body)}",
                            flush=True,
                        )
                    else:
                        if resp.status >= 400:
                            body = await resp.text()
                            print(
                                "[openai] HTTP error "
                                f"status={resp.status} reason={resp.reason!r} "
                                f"on attempt {attempt+1}/{max_retries}; "
                                f"body={_compact_openai_error_body(body)}",
                                flush=True,
                            )
                        resp.raise_for_status()
                        data = await resp.json()
                        content = _extract_chat_content(data)
                        latency_ms = (time.monotonic() - t0) * 1000.0
                        # Stash per-call telemetry on the contextvar side-channel.
                        # Defensive: telemetry must never raise inside the HTTP path.
                        try:
                            choices = data.get("choices") if isinstance(data, dict) else None
                            first_choice = choices[0] if isinstance(choices, list) and choices else {}
                            finish_reason = (
                                first_choice.get("finish_reason")
                                if isinstance(first_choice, dict)
                                else None
                            )
                            usage = data.get("usage") if isinstance(data, dict) else None
                            judge_call_meta.set({
                                "latency_ms": latency_ms,
                                "finish_reason": finish_reason,
                                "usage": usage or {},
                            })
                        except Exception:  # noqa: BLE001 - telemetry never breaks the call
                            pass
                        if _should_dump_judge(payload):
                            _dump_judge_response(payload, data, latency_ms=latency_ms)
                        return content
            if retry_after is not None:
                await asyncio.sleep(retry_after)
                continue
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as exc:
            print(f"[openai] {type(exc).__name__} on attempt {attempt+1}/{max_retries}", flush=True)
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(retry_sleep_seconds)
    raise RuntimeError(f"OpenAI API call failed after {max_retries} retries")


def post_chat_sync(
    payload: dict,
    *,
    max_retries: int | None = None,
    api_base: str | None = None,
    api_key: str | None = None,
) -> str:
    """Post a chat request synchronously."""
    resolved_api_key = api_key or resolve_judge_api_key()
    url = chat_url(api_base)
    headers = {"Authorization": f"Bearer {resolved_api_key}", "Content-Type": "application/json"}
    if max_retries is None:
        max_retries = get_openai_max_retries()
    retry_sleep_seconds = max(0.0, float(os.environ.get("PERSONA_OPENAI_RETRY_SLEEP_SECONDS", "5")))
    timeout = float(os.getenv("PERSONA_OPENAI_TIMEOUT_SECONDS", "180"))
    data_bytes = json.dumps(payload).encode("utf-8")
    for attempt in range(max_retries):
        try:
            request = urllib.request.Request(url, data=data_bytes, headers=headers, method="POST")
            with urllib.request.urlopen(request, timeout=timeout) as resp:
                return _extract_chat_content(json.loads(resp.read().decode("utf-8")))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            retry_after = (
                exc.headers.get("Retry-After", "")
                if exc.headers else retry_sleep_seconds
            )
            print(
                f"[openai] HTTP error status={exc.code} on attempt {attempt+1}/{max_retries}; "
                f"retry_after={retry_after}; body={_compact_openai_error_body(body)}",
                flush=True,
            )
            if attempt == max_retries - 1:
                raise
            time.sleep(retry_sleep_seconds)
        except (urllib.error.URLError, ValueError, TimeoutError) as exc:
            print(f"[openai] {type(exc).__name__} on attempt {attempt+1}/{max_retries}", flush=True)
            if attempt == max_retries - 1:
                raise
            time.sleep(retry_sleep_seconds)
    raise RuntimeError(f"OpenAI API call failed after {max_retries} retries")
