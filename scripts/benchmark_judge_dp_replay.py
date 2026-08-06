"""Replay production GRPO judge prompts against one OpenAI-compatible endpoint.

This intentionally mirrors the job-14217 judge payload and records compact
per-request telemetry. Response text is not retained because one validation
pass produces several gigabytes of thinking output.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import aiohttp


DEFAULT_MODEL = "Qwen/Qwen3.5-9B"


@dataclass
class CallResult:
    call_idx: int
    prompt_sha256: str
    started_at: float
    completed_at: float
    latency_s: float
    http_ok: bool
    http_status: int | None
    prompt_tokens: int | None
    completion_tokens: int | None
    finish_reason: str | None
    reasoning_chars: int
    content_chars: int
    error_text: str = ""


def load_prompts(dump_path: Path, n: int, split: str = "val") -> list[str]:
    """Load the first ``n`` non-empty judge prompts for ``split``."""
    prompts: list[str] = []
    with dump_path.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("split") != split:
                continue
            prompt = row.get("judge_prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                continue
            prompts.append(prompt)
            if len(prompts) == n:
                break
    return prompts


def build_body(prompt: str, model: str = DEFAULT_MODEL) -> dict[str, Any]:
    """Return the production payload used by GRPO job 14217."""
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_completion_tokens": 8192,
        "response_format": {"type": "json_object"},
        "repetition_penalty": 1.1,
        "temperature": 0.6,
        "chat_template_kwargs": {"enable_thinking": True},
        "provider": {"order": ["Morph"], "allow_fallbacks": False},
    }


def _text_len(value: Any) -> int:
    if value is None:
        return 0
    return len(value) if isinstance(value, str) else len(json.dumps(value, default=str))


async def one_call(
    session: "aiohttp.ClientSession",
    endpoint: str,
    semaphore: asyncio.Semaphore,
    call_idx: int,
    prompt: str,
    model: str,
    timeout_s: float,
) -> CallResult:
    import aiohttp

    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    async with semaphore:
        started_at = time.time()
        try:
            async with session.post(
                endpoint,
                json=build_body(prompt, model),
                headers={"Authorization": "Bearer dummy"},
                timeout=aiohttp.ClientTimeout(total=timeout_s),
            ) as response:
                status = response.status
                try:
                    data = await response.json()
                except (aiohttp.ContentTypeError, json.JSONDecodeError) as exc:
                    body = await response.text()
                    raise RuntimeError(f"non-JSON response: {body[:400]}") from exc
        except (aiohttp.ClientError, asyncio.TimeoutError, RuntimeError) as exc:
            completed_at = time.time()
            return CallResult(
                call_idx=call_idx,
                prompt_sha256=prompt_hash,
                started_at=started_at,
                completed_at=completed_at,
                latency_s=completed_at - started_at,
                http_ok=False,
                http_status=None,
                prompt_tokens=None,
                completion_tokens=None,
                finish_reason=None,
                reasoning_chars=0,
                content_chars=0,
                error_text=f"{type(exc).__name__}: {exc}"[:500],
            )

    completed_at = time.time()
    if status != 200:
        return CallResult(
            call_idx=call_idx,
            prompt_sha256=prompt_hash,
            started_at=started_at,
            completed_at=completed_at,
            latency_s=completed_at - started_at,
            http_ok=False,
            http_status=status,
            prompt_tokens=None,
            completion_tokens=None,
            finish_reason=None,
            reasoning_chars=0,
            content_chars=0,
            error_text=json.dumps(data, default=str)[:500],
        )

    choice = (data.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    usage = data.get("usage") or {}
    reasoning = message.get("reasoning") or message.get("reasoning_content")
    return CallResult(
        call_idx=call_idx,
        prompt_sha256=prompt_hash,
        started_at=started_at,
        completed_at=completed_at,
        latency_s=completed_at - started_at,
        http_ok=True,
        http_status=status,
        prompt_tokens=usage.get("prompt_tokens"),
        completion_tokens=usage.get("completion_tokens"),
        finish_reason=choice.get("finish_reason"),
        reasoning_chars=_text_len(reasoning),
        content_chars=_text_len(message.get("content")),
    )


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize(results: list[CallResult], wall_s: float) -> dict[str, Any]:
    successful = [result for result in results if result.http_ok]
    latencies = [result.latency_s for result in successful]
    completion_tokens = sum(result.completion_tokens or 0 for result in successful)
    prompt_tokens = sum(result.prompt_tokens or 0 for result in successful)
    return {
        "requests": len(results),
        "http_ok": len(successful),
        "http_errors": len(results) - len(successful),
        "wall_s": wall_s,
        "requests_per_s": len(successful) / wall_s if wall_s else 0.0,
        "output_tokens_per_s": completion_tokens / wall_s if wall_s else 0.0,
        "prompt_tokens_total": prompt_tokens,
        "completion_tokens_total": completion_tokens,
        "latency_p50_s": statistics.median(latencies) if latencies else 0.0,
        "latency_p90_s": _percentile(latencies, 0.90),
        "latency_p99_s": _percentile(latencies, 0.99),
        "latency_max_s": max(latencies, default=0.0),
    }


async def replay(args: argparse.Namespace) -> dict[str, Any]:
    import aiohttp

    prompts = load_prompts(args.dumps, args.n, args.split)
    if len(prompts) != args.n:
        raise RuntimeError(
            f"requested {args.n} {args.split!r} prompts but found {len(prompts)} in {args.dumps}"
        )

    args.out.mkdir(parents=True, exist_ok=True)
    requests_path = args.out / "requests.jsonl"
    endpoint = args.endpoint.rstrip("/") + "/chat/completions"
    semaphore = asyncio.Semaphore(args.concurrency)
    connector = aiohttp.TCPConnector(limit=args.concurrency)
    started = time.time()
    results: list[CallResult] = []
    print(
        f"[replay] prompts={len(prompts)} split={args.split} concurrency={args.concurrency} "
        f"endpoint={endpoint}",
        flush=True,
    )

    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            asyncio.create_task(
                one_call(
                    session,
                    endpoint,
                    semaphore,
                    call_idx,
                    prompt,
                    args.model,
                    args.timeout,
                )
            )
            for call_idx, prompt in enumerate(prompts)
        ]
        with requests_path.open("w", encoding="utf-8") as output:
            for completed, task in enumerate(asyncio.as_completed(tasks), start=1):
                result = await task
                results.append(result)
                output.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")
                output.flush()
                if completed % 32 == 0 or completed == len(tasks):
                    elapsed = time.time() - started
                    print(
                        f"[replay] completed={completed}/{len(tasks)} elapsed_s={elapsed:.1f} "
                        f"http_ok={sum(r.http_ok for r in results)}",
                        flush=True,
                    )

    wall_s = time.time() - started
    summary = summarize(results, wall_s)
    summary.update(
        {
            "model": args.model,
            "endpoint": args.endpoint,
            "dump_path": str(args.dumps),
            "split": args.split,
            "concurrency": args.concurrency,
            "timeout_s": args.timeout,
            "first_prompt_sha256": hashlib.sha256(prompts[0].encode("utf-8")).hexdigest(),
            "last_prompt_sha256": hashlib.sha256(prompts[-1].encode("utf-8")).hexdigest(),
        }
    )
    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True, help="base URL ending in /v1")
    parser.add_argument("--dumps", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--split", default="val")
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--timeout", type=float, default=1800.0)
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(replay(parse_args()))
