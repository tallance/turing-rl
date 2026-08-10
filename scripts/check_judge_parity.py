#!/usr/bin/env python3
"""Check that a judge endpoint still produces parseable ratings for the reward path.

Job 14217 served its judge through ``python -m vllm.entrypoints.openai.api_server``.
MODE=frac10ep10 serves the same vLLM build through the ``vllm serve`` CLI instead,
because the module frontend progressively collapses all traffic onto engine 000 under
sustained load (0.115-0.148 req/s measured during validation, against 0.455 req/s for
`vllm serve` on the same DP-8 topology).

Only the HTTP frontend changes, so judge outputs should be unaffected -- but the judge IS
the reward model, and the existing replay benchmark only ever asserted HTTP 200. It records
reasoning/content character counts and never parses a rating, so "endpoint responds" has
been verified while "reward layer can still read the response" has not. This script closes
that gap by replaying recorded prompts and parsing the replies with the *real* reward
helpers rather than a bespoke parser.

Compares against the ratings the 0.18.0 module frontend recorded for the same prompts.
Judge sampling is stochastic (temperature 0.6), so individual ratings will differ; what
must hold is the parse-failure rate and the rating distribution.

Usage:
  check_judge_parity.py --endpoint http://host:port/v1 --dump <reward_dump.jsonl> \
      --n 50 --out <summary.json>
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
from pathlib import Path

import aiohttp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Import every helper from training.grpo.reward, which is the module the GRPO run actually
# scores with. This is load-bearing: there are TWO _extract_json functions in the tree.
# shared.judge_utils._extract_json calls json.loads with no guard, so it RAISES on malformed
# text and knows nothing about code fences; reward.py defines its own at line 431 that strips
# ```json fences, falls back to the outermost {...} span, and returns None on JSONDecodeError.
# Importing the shared one would both crash on the first malformed reply and overstate the
# parse-failure rate -- i.e. measure a parser the run does not use.
from training.grpo.reward import (  # noqa: E402
    _coerce_turing_rating,
    _extract_json,
    _extract_turing_rating,
)


def load_rows(dump: Path, n: int) -> list[dict]:
    """First n val rows carrying both a prompt and a recorded rating.

    Read line by line: the dump reached 6.9 GB on the full run, so it must never be
    slurped into memory.
    """
    rows = []
    with dump.open() as fh:
        for line in fh:
            try:
                d = json.loads(line)
            except Exception:
                continue
            if d.get("split") != "val" or not d.get("judge_prompt"):
                continue
            recorded = d.get("rating_gen_first")
            if recorded is None:
                recorded = d.get("rating_gt_first")
            if recorded is None:
                continue
            rows.append({"prompt": d["judge_prompt"], "recorded_rating": recorded})
            if len(rows) >= n:
                break
    return rows


async def one(session, endpoint, model, prompt, sem, timeout):
    """Production payload: response_format/thinking/penalty must match reward.py."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_completion_tokens": int(os.environ.get("PERSONA_JUDGE_MAX_COMPLETION_TOKENS", 8192)),
        "temperature": 0.6,
        "repetition_penalty": 1.1,
        "response_format": {"type": "json_object"},
        "chat_template_kwargs": {"enable_thinking": True},
    }
    async with sem:
        try:
            async with session.post(
                f"{endpoint}/chat/completions",
                json=payload,
                headers={"Authorization": "Bearer dummy"},
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as resp:
                status = resp.status
                body = await resp.json() if status == 200 else None
        except Exception as exc:  # noqa: BLE001
            return {"status": None, "error": type(exc).__name__, "outcome": "transport_error"}

    if status != 200 or not body:
        return {"status": status, "outcome": "http_error"}

    text = (body["choices"][0]["message"] or {}).get("content")

    # Mirror reward.py's parse ladder exactly: strict JSON first, then the malformed-text
    # rating recovery, then give up. Anything else would measure a different parser.
    data = _extract_json(text)
    if data is not None:
        outcome = "json_ok"
        rating = _coerce_turing_rating(data.get("rating"))
    else:
        recovered = _extract_turing_rating(text)
        if recovered is not None:
            outcome, rating = "recovered", recovered
        else:
            outcome, rating = "parse_failed", None
    return {"status": status, "outcome": outcome, "rating": rating}


async def main_async(args) -> int:
    rows = load_rows(Path(args.dump), args.n)
    if len(rows) < args.n:
        print(f"WARNING: wanted {args.n} rows, found {len(rows)}", file=sys.stderr)
    if not rows:
        print("ERROR: no usable rows in dump", file=sys.stderr)
        return 2

    sem = asyncio.Semaphore(args.concurrency)
    async with aiohttp.ClientSession() as session:
        results = await asyncio.gather(
            *[one(session, args.endpoint, args.model, r["prompt"], sem, args.timeout) for r in rows]
        )

    counts: dict[str, int] = {}
    for r in results:
        counts[r["outcome"]] = counts.get(r["outcome"], 0) + 1
    new = [r["rating"] for r in results if r.get("rating") is not None]
    old = [r["recorded_rating"] for r in rows]

    parseable = counts.get("json_ok", 0) + counts.get("recovered", 0)
    summary = {
        "endpoint": args.endpoint,
        "n_requested": args.n,
        "n_sent": len(rows),
        "outcomes": counts,
        "parseable": parseable,
        "parse_failure_rate": round(1 - parseable / len(results), 4),
        "rating_count_new": len(new),
        "rating_mean_new": round(statistics.mean(new), 3) if new else None,
        "rating_mean_recorded": round(statistics.mean(old), 3) if old else None,
        "rating_hist_new": {str(v): new.count(v) for v in sorted(set(new))},
        "rating_hist_recorded": {str(v): old.count(v) for v in sorted(set(old))},
    }
    print(json.dumps(summary, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=2))

    # Fail loudly rather than letting a 21 h run start against a judge the reward layer
    # cannot read. The full run's own parse-retry rate was ~11%.
    if summary["parse_failure_rate"] > args.max_parse_failure:
        print(
            f"FAIL: parse_failure_rate {summary['parse_failure_rate']} exceeds "
            f"--max-parse-failure {args.max_parse_failure}",
            file=sys.stderr,
        )
        return 1
    print("PASS: judge responses are parseable by the reward path")
    return 0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--endpoint", required=True, help="base URL ending in /v1")
    p.add_argument("--dump", required=True)
    p.add_argument("--model", default="Qwen/Qwen3.5-9B")
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--timeout", type=float, default=1800.0)
    p.add_argument("--max-parse-failure", type=float, default=0.25)
    p.add_argument("--out", default=None)
    return p.parse_args()


if __name__ == "__main__":
    sys.exit(asyncio.run(main_async(parse_args())))
