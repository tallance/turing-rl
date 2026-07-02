"""Smoke test for self-hosted Qwen judge: single calls + concurrent batch."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time

import aiohttp


SAMPLE_USER_HISTORY = """Previous posts by USER:
- "Honestly the new pricing is a joke. Loyal customers get nothing."
- "Tried the trial last week. Was OK but not blown away."
- "Anyone else having issues with the login on mobile?"
"""

SAMPLE_CONTEXT = """[OTHER]: We've heard your feedback on pricing and are introducing a loyalty discount next month — 10% off for accounts older than 2 years.

[HUMAN]:"""

GROUND_TRUTH = "10%, seriously? After all the price hikes that's basically an insult."

CANDIDATE = "Thank you for considering long-time users. I look forward to the loyalty discount and appreciate your responsiveness to community feedback."

JUDGE_PROMPT = f"""You are judging a pairwise Turing test for personalized user simulation.

USER HISTORY:
{SAMPLE_USER_HISTORY}

CONVERSATION CONTEXT:
{SAMPLE_CONTEXT}

Two candidate next responses from [HUMAN]:
- Response A: "{GROUND_TRUTH}"
- Response B: "{CANDIDATE}"

One was written by the real USER, one by an AI. Output a JSON object with fields:
{{"reasoning": "<brief reasoning>", "score": <integer 1-7 where 1 = A is clearly the human and 7 = B is clearly the human>}}

Respond with ONLY the JSON object, no other text."""


async def call_once(session: aiohttp.ClientSession, url: str, model: str, idx: int) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": JUDGE_PROMPT}],
        "max_completion_tokens": 512,
        "temperature": 0.0,
    }
    t0 = time.perf_counter()
    try:
        async with session.post(url, json=payload) as resp:
            data = await resp.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            elapsed = time.perf_counter() - t0
            usage = data.get("usage", {})
            return {
                "idx": idx,
                "status": resp.status,
                "elapsed_s": elapsed,
                "content": content,
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "error": None,
            }
    except Exception as exc:
        return {
            "idx": idx,
            "status": None,
            "elapsed_s": time.perf_counter() - t0,
            "content": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


async def main(args):
    url = f"{args.base_url.rstrip('/')}/chat/completions"
    timeout = aiohttp.ClientTimeout(total=args.timeout)
    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"

    print(f"\n=== Smoke test ===")
    print(f"URL:           {url}")
    print(f"Model:         {args.model}")
    print(f"Single calls:  3")
    print(f"Batch size:    {args.batch_size}\n")

    async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
        # --- single calls ---
        print("--- 3 single calls (sequential) ---")
        for i in range(3):
            res = await call_once(session, url, args.model, i)
            print(f"[{i}] status={res['status']} elapsed={res['elapsed_s']:.2f}s "
                  f"prompt_tokens={res.get('prompt_tokens')} "
                  f"completion_tokens={res.get('completion_tokens')}")
            if res["error"]:
                print(f"    ERROR: {res['error']}")
            else:
                preview = (res["content"] or "")[:300].replace("\n", " ")
                print(f"    content: {preview}")
                # try to parse JSON
                try:
                    parsed = json.loads(res["content"])
                    print(f"    parsed_json: keys={list(parsed.keys())} score={parsed.get('score')}")
                except Exception as e:
                    print(f"    json_parse_failed: {e}")

        # --- concurrent batch ---
        print(f"\n--- {args.batch_size} concurrent calls ---")
        t0 = time.perf_counter()
        tasks = [call_once(session, url, args.model, i) for i in range(args.batch_size)]
        results = await asyncio.gather(*tasks)
        total = time.perf_counter() - t0
        ok = [r for r in results if r["error"] is None and r["status"] == 200]
        err = [r for r in results if r["error"] is not None or r["status"] != 200]
        latencies = sorted(r["elapsed_s"] for r in ok)
        total_completion_tokens = sum((r.get("completion_tokens") or 0) for r in ok)

        print(f"total wall time:     {total:.2f}s")
        print(f"successes:           {len(ok)}/{args.batch_size}")
        print(f"errors:              {len(err)}")
        print(f"throughput:          {len(ok) / total:.2f} req/sec")
        if ok:
            p50 = latencies[len(latencies) // 2]
            p95 = latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))]
            print(f"latency p50/p95:     {p50:.2f}s / {p95:.2f}s")
            print(f"completion tok/sec:  {total_completion_tokens / total:.1f}")
        if err:
            print(f"\nfirst error:")
            print(f"  status={err[0]['status']} error={err[0]['error']}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default=os.environ.get("JUDGE_BASE_URL", "http://localhost:8000/v1"))
    p.add_argument("--model", default=os.environ.get("JUDGE_MODEL", "Qwen/Qwen3.5-397B-A17B-GPTQ-Int4"))
    p.add_argument("--api-key", default=os.environ.get("JUDGE_API_KEY", "dummy"))
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--timeout", type=float, default=600.0)
    args = p.parse_args()
    asyncio.run(main(args))
