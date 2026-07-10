"""Served (self-hosted, thinking-off) CoT client.

Faithful to upstream ``data/sft/generate_cot.py``: same rationalize prompts,
same leakage guard + regen loop, same ``extra_info`` keys. Only the transport
differs — instead of one synchronous OpenRouter call per row with
``reasoning=True``, this posts asynchronously and round-robin to a pool of
self-hosted vLLM replicas with ``chat_template_kwargs={"enable_thinking": False}``
(thinking OFF). No OpenRouter ``reasoning``/``provider`` extras go on the wire.

The pure logic (``build_cot_payload``, ``pick_endpoint``) is unit tested in
``tests/test_generate_cot_served.py``; the full run is exercised on the cluster
via ``scripts/slurm/cot_serve.sh``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:  # pragma: no cover - exercised in runtime envs
    import aiohttp
except ImportError:  # pragma: no cover
    aiohttp = None  # type: ignore[assignment]

# Reuse the confirmed upstream business logic verbatim. Do NOT reimplement.
from data.sft.generate_cot import (  # noqa: E402
    DEFAULT_MAX_COMPLETION_TOKENS,
    RATIONALIZE_SYSTEM_PROMPT,
    RATIONALIZE_USER_TEMPLATE,
    REGEN_NUDGE,
    THINKING_TRACE_SOURCE,
    _as_text,
    _row_context,
    reasoning_leaks_reply,
)

DEFAULT_INPUT = "data/prism/full_s42_history_sft40_grpo60_test10/sft/train.parquet"
DEFAULT_MODEL = "Qwen/Qwen3-8B"
DEFAULT_MAX_REGEN_ATTEMPTS = 10
DEFAULT_CONCURRENCY_PER_ENDPOINT = 16


def build_cot_payload(
    model: str,
    messages: list[dict],
    *,
    sampling: dict | None = None,
    max_completion_tokens: int = DEFAULT_MAX_COMPLETION_TOKENS,
) -> dict[str, Any]:
    """Build a thinking-off chat-completions payload for a served model.

    Only ``model``, ``messages``, ``max_completion_tokens`` and
    ``chat_template_kwargs={"enable_thinking": False}`` are set. If ``sampling``
    is provided its keys (temperature/top_p/top_k/min_p) merge top-level. No
    OpenRouter ``reasoning`` or ``provider`` extras are ever added.
    """
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_completion_tokens": int(max_completion_tokens),
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if sampling:
        payload.update(sampling)
    return payload


def pick_endpoint(endpoints: list[str], i: int) -> str:
    """Round-robin endpoint selection by request index."""
    return endpoints[i % len(endpoints)]


def _build_messages(extra_info: dict[str, Any], ground_truth: str, *, regen: bool) -> list[dict]:
    """Build the rationalize message list, exactly like generate_reasoning_for_row."""
    messages = [
        {"role": "system", "content": RATIONALIZE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": RATIONALIZE_USER_TEMPLATE.format(
                context=_row_context(extra_info), ground_truth=ground_truth
            ),
        },
    ]
    if regen:
        messages.append({"role": "user", "content": REGEN_NUDGE})
    return messages


def _extract_content(data: Any) -> str:
    choices = data.get("choices") if isinstance(data, dict) else None
    first = choices[0] if isinstance(choices, list) and choices else None
    message = first.get("message") if isinstance(first, dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    return content if isinstance(content, str) else ""


def load_endpoints(path: str | Path) -> list[str]:
    """Read one endpoint URL per line (blank lines / comments ignored)."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    endpoints = [ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("#")]
    if not endpoints:
        raise ValueError(f"No endpoints found in {path}")
    return endpoints


async def _post(
    session: "aiohttp.ClientSession",
    endpoint: str,
    payload: dict,
    *,
    api_key: str,
    max_retries: int = 3,
    retry_sleep_seconds: float = 5.0,
) -> str:
    """POST one chat completion to <endpoint>/chat/completions and return content."""
    url = f"{endpoint.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    for attempt in range(max_retries):
        try:
            async with session.post(url, headers=headers, json=payload) as resp:
                if resp.status >= 400:
                    body = await resp.text()
                    print(
                        f"[cot_served] HTTP {resp.status} on {url} "
                        f"attempt {attempt + 1}/{max_retries}; body={body[:500]}",
                        flush=True,
                    )
                    resp.raise_for_status()
                data = await resp.json()
                return _extract_content(data)
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as exc:
            print(
                f"[cot_served] {type(exc).__name__} on {url} "
                f"attempt {attempt + 1}/{max_retries}",
                flush=True,
            )
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(retry_sleep_seconds)
    raise RuntimeError(f"CoT request to {endpoint} failed after {max_retries} retries")


async def _annotate_row(
    session: "aiohttp.ClientSession",
    row: dict[str, Any],
    *,
    endpoint: str,
    model: str,
    max_completion_tokens: int,
    max_regen_attempts: int,
    api_key: str,
    semaphore: "asyncio.Semaphore",
) -> dict[str, Any] | None:
    """Generate one reasoning trace with the leakage-guard regen loop.

    Mirrors generate_reasoning_for_row but async + thinking-off. Returns the
    trace dict (upstream keys) or None if the row has no ground truth.
    """
    extra_info = dict(row.get("extra_info") or {})
    reward_model = dict(row.get("reward_model") or {})
    ground_truth = _as_text(reward_model.get("ground_truth"))
    if not ground_truth.strip():
        return None

    reasoning = ""
    attempts = 0
    leaked = True
    for attempt in range(1, max(1, max_regen_attempts) + 1):
        attempts = attempt
        messages = _build_messages(extra_info, ground_truth, regen=attempt > 1)
        payload = build_cot_payload(
            model, messages, max_completion_tokens=max_completion_tokens
        )
        async with semaphore:
            content = await _post(session, endpoint, payload, api_key=api_key)
        reasoning = (content or "").strip()
        leaked = reasoning_leaks_reply(reasoning, ground_truth)
        if reasoning and not leaked:
            break

    return {
        "ground_truth_reasoning": reasoning,
        "thinking_trace_source": THINKING_TRACE_SOURCE,
        "thinking_trace_model": model,
        "thinking_trace_num_regen_attempts": attempts,
        "thinking_trace_failed_leakage_guard": bool(leaked),
    }


async def generate_cot_served(
    *,
    input_path: str | Path,
    output_path: str | Path,
    endpoints: list[str],
    model: str = DEFAULT_MODEL,
    max_completion_tokens: int = DEFAULT_MAX_COMPLETION_TOKENS,
    max_regen_attempts: int = DEFAULT_MAX_REGEN_ATTEMPTS,
    concurrency_per_endpoint: int = DEFAULT_CONCURRENCY_PER_ENDPOINT,
    sampling: dict | None = None,
) -> dict[str, Any]:
    """Run the served CoT client over one SFT parquet and write outputs."""
    if aiohttp is None:
        raise ImportError("The served CoT client requires aiohttp to be installed")
    import pandas as pd

    input_path = Path(input_path)
    output_path = Path(output_path)
    rows = [dict(r) for r in pd.read_parquet(input_path).to_dict(orient="records")]

    n_endpoints = len(endpoints)
    concurrency = concurrency_per_endpoint * n_endpoints
    semaphore = asyncio.Semaphore(concurrency)
    api_key = os.environ.get("OPENAI_API_KEY", "dummy-self-hosted")

    t0 = time.monotonic()
    timeout = aiohttp.ClientTimeout(total=float(os.getenv("PERSONA_OPENAI_TIMEOUT_SECONDS", "600")))
    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = []
        for i, row in enumerate(rows):
            endpoint = pick_endpoint(endpoints, i)
            tasks.append(
                _annotate_row(
                    session,
                    row,
                    endpoint=endpoint,
                    model=model,
                    max_completion_tokens=max_completion_tokens,
                    max_regen_attempts=max_regen_attempts,
                    api_key=api_key,
                    semaphore=semaphore,
                )
            )
        traces = await asyncio.gather(*tasks)
    wall_s = time.monotonic() - t0

    written = 0
    failed_guard = 0
    skipped = 0
    leak_regen_counts: dict[str, int] = {}
    for row, trace in zip(rows, traces):
        if trace is None:
            skipped += 1
            continue
        extra_info = dict(row.get("extra_info") or {})
        extra_info.update(trace)
        row["extra_info"] = extra_info
        written += 1
        if trace["thinking_trace_failed_leakage_guard"]:
            failed_guard += 1
        key = str(trace["thinking_trace_num_regen_attempts"])
        leak_regen_counts[key] = leak_regen_counts.get(key, 0) + 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(output_path, index=False)

    metadata = {
        "n_rows": len(rows),
        "rows_written": written,
        "rows_failed_leakage_guard": failed_guard,
        "rows_skipped": skipped,
        "endpoints": endpoints,
        "model": model,
        "sampling": sampling,
        "thinking": "off",
        "max_regen_attempts": max_regen_attempts,
        "concurrency": concurrency,
        "wall_s": round(wall_s, 3),
        "leak_regen_counts": leak_regen_counts,
    }
    Path(f"{output_path}.cot_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Served thinking-off CoT client (async, round-robin over vLLM replicas)."
    )
    parser.add_argument("--endpoints", required=True, help="File with one endpoint URL per line.")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Source SFT parquet.")
    parser.add_argument("--out", required=True, help="Destination parquet with ground_truth_reasoning added.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Served model id.")
    parser.add_argument("--max_completion_tokens", type=int, default=DEFAULT_MAX_COMPLETION_TOKENS)
    parser.add_argument("--max_regen_attempts", type=int, default=DEFAULT_MAX_REGEN_ATTEMPTS)
    parser.add_argument("--concurrency_per_endpoint", type=int, default=DEFAULT_CONCURRENCY_PER_ENDPOINT)
    args = parser.parse_args()

    endpoints = load_endpoints(args.endpoints)
    metadata = asyncio.run(
        generate_cot_served(
            input_path=args.input,
            output_path=args.out,
            endpoints=endpoints,
            model=args.model,
            max_completion_tokens=args.max_completion_tokens,
            max_regen_attempts=args.max_regen_attempts,
            concurrency_per_endpoint=args.concurrency_per_endpoint,
        )
    )
    print(json.dumps({"output": args.out, **metadata}, ensure_ascii=False))


if __name__ == "__main__":
    main()
