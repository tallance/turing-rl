#!/usr/bin/env python
"""Acceptance client for serving a judge to the GRPO *training* reward path.

Written for the gemma-4-12B judge, but model-agnostic: it replays real judge prompts
from an existing reward dump against a live endpoint and reports how the responses
behave under the reward code that training actually runs.

WHY THIS EXISTS
---------------
gemma-4-12B already has a clean record as a judge -- 440/440 pairs parsed at every
checkpoint in results/2026-08-12-test-eval-9b-train10pct-10ep-test50pct-full-schema.
That record does NOT transfer to training, for two independent reasons:

  1. The eval sweep sets PERSONA_JUDGE_JSON_SCHEMA=1 (run_judge_sweep_cell.py:66), the
     37-field ordered schema. The training driver sets nothing, so reward.py falls back
     to {"type": "json_object"}. docs/judge-response-schema.md records gemma-4-31B
     producing 5/16 valid responses and hitting the completion cap in 10/16 cases
     without the ordered schema.
  2. The eval serves gemma as 8 single-GPU servers on 8 ports. Training's reward path
     takes ONE OPENAI_API_BASE, so the judge has to run --data-parallel-size 8. That
     combination has never been exercised.

So this script answers two separate questions, and the --mode flag picks which:

  --mode training : is gemma usable as a training judge AT ALL? Runs with the schema
                    unset, exactly as the training driver would, and reports the
                    parse outcome mix. This is the go/no-go.
  --mode eval     : does the DP-8 serving path reproduce the proven 8-replica path?
                    Runs with the schema enabled so the comparison against the eval's
                    recorded ratings is matched rather than confounded.

PARSE OUTCOMES ARE THREE-WAY, NOT TWO
-------------------------------------
reward.py:584 tries _extract_json first and only falls back to _extract_turing_rating,
so a response can be clean, recovered (malformed JSON but a rating was salvaged by
regex), or failed. Collapsing "recovered" into "ok" would hide a judge that has started
emitting garbage but happens to still contain a parseable digit. All three are reported.

Every helper below is imported from the real reward path rather than reimplemented, so
a change to how training talks to its judge cannot silently diverge from this check.
"""
from __future__ import annotations

import argparse
import asyncio
import glob
import json
import os
import random
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import aiohttp  # noqa: E402

from shared.api_client import (  # noqa: E402
    build_chat_payload,
    get_judge_call_meta,
    post_chat_async,
)
from shared.judge_prompts import TURING_RESPONSE_SCHEMA  # noqa: E402
from shared.judge_utils import (  # noqa: E402
    _extract_json,
    _extract_turing_rating,
)


def resolve_response_format() -> dict:
    """Mirror of reward.py:_resolve_response_format, driven by the same env var."""
    if os.environ.get("PERSONA_JUDGE_JSON_SCHEMA") == "1":
        return {
            "type": "json_schema",
            "json_schema": {"name": "turing_verdict", "schema": TURING_RESPONSE_SCHEMA},
        }
    return {"type": "json_object"}


def load_prompts(dump_glob: str, split: str | None, limit: int, seed: int) -> list[dict]:
    """Pull (key, prompt, recorded score) triples out of reward-*.jsonl dumps.

    The recorded score is what the ORIGINAL judge returned for this exact prompt. In
    --mode eval that is the reference the new serving path must reproduce; in
    --mode training it is the qwen baseline the gemma parse rate is compared against,
    which is why the baseline needs no extra GPU time.
    """
    files = sorted(glob.glob(dump_glob))
    if not files:
        raise SystemExit(f"FAIL: no reward dumps matched {dump_glob}")
    rows: list[dict] = []
    for path in files:
        with open(path) as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if split and rec.get("split") != split:
                    continue
                prompt = rec.get("judge_prompt")
                if not prompt:
                    continue
                rows.append(
                    {
                        "key": f"{rec.get('user_id')}|{rec.get('post_id')}|{rec.get('target_idx')}",
                        "prompt": prompt,
                        "ref_score": rec.get("turing_judge_score_raw"),
                        "ref_finish": rec.get("judge_finish_reason"),
                        "ref_model": rec.get("judge_model"),
                    }
                )
    # Deduplicate by key: a train-split dump repeats each key once per epoch per rollout,
    # and scoring the same prompt 40 times would measure nothing extra.
    seen: dict[str, dict] = {}
    for row in rows:
        seen.setdefault(row["key"], row)
    uniq = list(seen.values())
    uniq.sort(key=lambda r: r["key"])          # deterministic before sampling
    if limit and limit < len(uniq):
        random.Random(seed).shuffle(uniq)
        uniq = uniq[:limit]
        uniq.sort(key=lambda r: r["key"])
    return uniq


async def score_one(session, sem, row: dict, model: str, max_tokens: int) -> dict:
    t0 = time.monotonic()
    out = {"key": row["key"], "ref_score": row["ref_score"]}
    try:
        payload = build_chat_payload(
            model=model,
            messages=[{"role": "user", "content": row["prompt"]}],
            max_completion_tokens=max_tokens,
            response_format=resolve_response_format(),
            reasoning=False,
            sampling=json.loads(os.environ["PERSONA_JUDGE_SAMPLING"])
            if os.environ.get("PERSONA_JUDGE_SAMPLING")
            else None,
            chat_template_kwargs={
                "enable_thinking": os.environ.get("PERSONA_JUDGE_ENABLE_THINKING") == "1"
            }
            if os.environ.get("PERSONA_JUDGE_ENABLE_THINKING") in ("0", "1")
            else None,
        )
        text = await post_chat_async(session, payload, semaphore=sem, api_key="EMPTY")
        meta = get_judge_call_meta() or {}
        out["finish_reason"] = meta.get("finish_reason")
        # Same two-stage logic as reward.py:584 onwards.
        data = _extract_json(text)
        if data is not None:
            out["outcome"] = "clean"
            out["rating"] = data.get("rating")
        else:
            recovered = _extract_turing_rating(text)
            out["outcome"] = "recovered" if recovered is not None else "failed"
            out["rating"] = recovered
        out["chars"] = len(text or "")
    except Exception as exc:  # noqa: BLE001 - a transport error is a smoke result, not a crash
        out["outcome"] = "error"
        out["error"] = f"{type(exc).__name__}: {exc}"
    out["latency_s"] = round(time.monotonic() - t0, 2)
    return out


async def run_pass(rows: list[dict], model: str, concurrency: int, max_tokens: int) -> list[dict]:
    sem = asyncio.Semaphore(concurrency)
    timeout = aiohttp.ClientTimeout(total=int(os.environ.get("PERSONA_OPENAI_TIMEOUT_SECONDS", "1800")))
    async with aiohttp.ClientSession(timeout=timeout) as session:
        return await asyncio.gather(*(score_one(session, sem, r, model, max_tokens) for r in rows))


def summarize(tag: str, results: list[dict], wall_s: float, concurrency: int) -> dict:
    n = len(results)
    outcomes = Counter(r.get("outcome") for r in results)
    finishes = Counter(r.get("finish_reason") for r in results)
    lat = sorted(r["latency_s"] for r in results if "latency_s" in r)
    usable = outcomes["clean"] + outcomes["recovered"]
    summary = {
        "tag": tag,
        "n": n,
        "concurrency": concurrency,
        "clean": outcomes["clean"],
        "recovered": outcomes["recovered"],
        "failed": outcomes["failed"],
        "error": outcomes["error"],
        "usable_rate": round(usable / n, 4) if n else 0.0,
        "hard_fail_rate": round((outcomes["failed"] + outcomes["error"]) / n, 4) if n else 0.0,
        "finish_length_rate": round(finishes.get("length", 0) / n, 4) if n else 0.0,
        "finish_reasons": dict(finishes),
        "wall_s": round(wall_s, 1),
        "req_per_s": round(n / wall_s, 3) if wall_s else 0.0,
        "p50_s": lat[len(lat) // 2] if lat else None,
        "p95_s": lat[int(len(lat) * 0.95)] if lat else None,
    }
    rated = [r["rating"] for r in results if isinstance(r.get("rating"), int)]
    if rated:
        summary["rating_mean"] = round(statistics.fmean(rated), 4)
        summary["rating_hist"] = dict(sorted(Counter(rated).items()))
    return summary


def compare_to_reference(results: list[dict]) -> dict | None:
    """Agreement against the ratings the reference serving path recorded per prompt."""
    pairs = [
        (r["ref_score"], r["rating"])
        for r in results
        if isinstance(r.get("rating"), int) and isinstance(r.get("ref_score"), (int, float))
    ]
    if not pairs:
        return None
    new = [p[1] for p in pairs]
    ref = [p[0] for p in pairs]
    diffs = [abs(a - b) for a, b in zip(new, ref)]
    return {
        "n_compared": len(pairs),
        "ref_mean": round(statistics.fmean(ref), 4),
        "new_mean": round(statistics.fmean(new), 4),
        "mean_delta": round(statistics.fmean(new) - statistics.fmean(ref), 4),
        "exact_agreement": round(sum(d == 0 for d in diffs) / len(diffs), 4),
        "within_1": round(sum(d <= 1 for d in diffs) / len(diffs), 4),
        "mean_abs_diff": round(statistics.fmean(diffs), 4),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--endpoint", required=True, help="e.g. http://10.0.0.1:8321/v1")
    ap.add_argument("--model", required=True, help="model id as advertised by /v1/models")
    ap.add_argument("--dump-glob", required=True, help="reward-*.jsonl to draw prompts from")
    ap.add_argument("--mode", choices=("training", "eval"), required=True)
    ap.add_argument("--split", default=None, help="restrict to this split (e.g. val)")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--repeat", type=int, default=1,
                    help="score the same prompts N times to measure the judge's own noise floor")
    ap.add_argument("--out", required=True, help="per-prompt jsonl output")
    ap.add_argument("--summary", required=True, help="summary json output")
    a = ap.parse_args()

    # The endpoint is passed through the same env var the reward path reads.
    os.environ["OPENAI_API_BASE"] = a.endpoint
    os.environ.setdefault("PERSONA_DISABLE_OPENROUTER_EXTRAS", "1")
    if a.mode == "eval":
        os.environ["PERSONA_JUDGE_JSON_SCHEMA"] = "1"
    else:
        os.environ.pop("PERSONA_JUDGE_JSON_SCHEMA", None)

    rows = load_prompts(a.dump_glob, a.split, a.n, a.seed)
    fmt = resolve_response_format()["type"]
    print(f"[smoke] mode={a.mode} response_format={fmt} prompts={len(rows)} "
          f"concurrency={a.concurrency} repeat={a.repeat}", flush=True)
    ref_models = {r["ref_model"] for r in rows if r.get("ref_model")}
    print(f"[smoke] prompts drawn from dumps judged by: {sorted(ref_models)}", flush=True)

    all_summaries, passes = [], []
    with open(a.out, "w") as handle:
        for i in range(a.repeat):
            t0 = time.monotonic()
            results = asyncio.run(run_pass(rows, a.model, a.concurrency, a.max_tokens))
            wall = time.monotonic() - t0
            for r in results:
                handle.write(json.dumps({**r, "pass": i}) + "\n")
            s = summarize(f"{a.mode}-pass{i}", results, wall, a.concurrency)
            cmp_ref = compare_to_reference(results)
            if cmp_ref:
                s["vs_reference"] = cmp_ref
            all_summaries.append(s)
            passes.append(results)
            print(f"[smoke] pass {i}: {json.dumps(s, indent=2)}", flush=True)

    out: dict = {"mode": a.mode, "endpoint": a.endpoint, "model": a.model,
                 "response_format": fmt, "passes": all_summaries}

    # Two passes over identical prompts give the judge's own run-to-run spread, which is the
    # only defensible tolerance for the reference comparison -- sampling is stochastic at
    # temperature 0.6, so an invented threshold would be arbitrary in either direction.
    if a.repeat >= 2:
        by_key: dict[str, list[int]] = {}
        for results in passes:
            for r in results:
                if isinstance(r.get("rating"), int):
                    by_key.setdefault(r["key"], []).append(r["rating"])
        both = [v for v in by_key.values() if len(v) >= 2]
        if both:
            deltas = [abs(v[0] - v[1]) for v in both]
            means = [s["rating_mean"] for s in all_summaries if "rating_mean" in s]
            out["self_noise"] = {
                "n_keys": len(both),
                "self_exact_agreement": round(sum(d == 0 for d in deltas) / len(deltas), 4),
                "self_within_1": round(sum(d <= 1 for d in deltas) / len(deltas), 4),
                "self_mean_abs_diff": round(statistics.fmean(deltas), 4),
                "pass_mean_spread": round(max(means) - min(means), 4) if len(means) >= 2 else None,
            }
            print(f"[smoke] self-noise: {json.dumps(out['self_noise'], indent=2)}", flush=True)

    with open(a.summary, "w") as handle:
        json.dump(out, handle, indent=2)
    print(f"[smoke] wrote {a.out} and {a.summary}", flush=True)

    # Non-zero only for a transport-level failure; the judgement calls belong to the
    # gate script, which has the reference numbers to compare against.
    worst = max((s["hard_fail_rate"] for s in all_summaries), default=1.0)
    if worst >= 0.5:
        print(f"[smoke] FAIL: hard-failure rate {worst:.2%} -- endpoint is not usable", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
