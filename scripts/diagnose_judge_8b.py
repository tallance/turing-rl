"""8B judge diagnostic: categorize where and why the 8B judge fails.

Loads N real (context, human, generated) triples from a prior 397B judge dump,
then sends each triple through the 8B judge under FOUR payload variants to
isolate what's breaking:

  V1 (prod)   response_format=json_object + reasoning=enabled + tokens=8192
  V2          same but response_format removed (drop constrained decoding)
  V3          same as V1 but reasoning disabled + '/no_think' appended
  V4          same as V1 but max_completion_tokens=16384

Per-call metrics logged to JSONL. Failure categorization uses the SAME code
paths as production (training.grpo.reward._extract_json, shared.judge_utils._extract_turing_rating)
so results reflect what the real reward function would see. Prints a compact
summary table + decision rules at the end.

Usage:
  python scripts/diagnose_judge_8b.py \\
      --url http://a100-010-036:8123/v1 \\
      --dumps /home/lancewicki/tmp/judge_dumps/judge-9239-1968156.jsonl \\
      --n 50 \\
      --out /home/lancewicki/tmp/judge_8b_diag
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import aiohttp

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from training.grpo.reward import _extract_json  # noqa: E402  # canonical JSON extractor
from shared.judge_utils import _extract_turing_rating  # noqa: E402  # regex fallback

REQUIRED_KEYS = (
    "immediate_target_score_a", "immediate_target_score_b",
    "human_goal_score_a", "human_goal_score_b",
    "communication_style_score_a", "communication_style_score_b",
    "rating",
)


@dataclass
class CallResult:
    triple_idx: int
    variant: str
    http_ok: bool
    http_status: int | None
    latency_s: float
    prompt_tokens: int | None
    completion_tokens: int | None
    content_len: int
    reasoning_len: int
    json_parses: bool
    has_required_keys: bool
    has_rating_field: bool
    regex_recovered_rating: int | None
    usable_rating: int | None
    category: str
    rating_397b: int | None
    error_text: str = ""
    content_head: str = ""
    reasoning_head: str = ""


def _coerce_rating(v: Any) -> int | None:
    try:
        r = int(v)
    except (TypeError, ValueError):
        return None
    return r if 1 <= r <= 7 else None


def _categorize(parsed: dict | None, regex_rating: int | None, http_ok: bool) -> tuple[str, int | None]:
    """Return (category, usable_rating).

    Categories:
      ok_full_schema, ok_rating_only, ok_regex_recovered,
      fail_empty_json, fail_malformed_json, fail_http
    """
    if not http_ok:
        return "fail_http", None
    if parsed is None:
        if regex_rating is not None:
            return "ok_regex_recovered", regex_rating
        return "fail_malformed_json", None
    rating = _coerce_rating(parsed.get("rating"))
    has_req = all(k in parsed for k in REQUIRED_KEYS)
    if has_req and rating is not None:
        return "ok_full_schema", rating
    if rating is not None:
        return "ok_rating_only", rating
    if regex_rating is not None:
        return "ok_regex_recovered", regex_rating
    return "fail_empty_json", None


def load_triples(dump_path: Path, n: int) -> list[tuple[str, int | None]]:
    """Return list of (prompt, rating_397b) for first n entries."""
    out: list[tuple[str, int | None]] = []
    with open(dump_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            msgs = d.get("payload_messages") or []
            if not msgs:
                continue
            prompt = msgs[0].get("content") or ""
            if not prompt.strip():
                continue
            resp_content = ((d.get("response", {}).get("choices") or [{}])[0].get("message") or {}).get("content") or ""
            r397 = None
            parsed_397 = _extract_json(resp_content)
            if parsed_397:
                r397 = _coerce_rating(parsed_397.get("rating"))
            out.append((prompt, r397))
            if len(out) >= n:
                break
    return out


def build_body(prompt: str, variant: str) -> dict:
    """Construct the payload for a given variant."""
    if variant == "V3":
        prompt = prompt + "\n\n/no_think"
    body: dict[str, Any] = {
        "model": "Qwen/Qwen3-8B",
        "messages": [{"role": "user", "content": prompt}],
    }
    if variant == "V4":
        body["max_completion_tokens"] = 16384
    else:
        body["max_completion_tokens"] = 8192

    if variant in ("V1", "V4"):
        # Use json_schema (not json_object): source-verified fix. json_object
        # compiles to {"type":"object"}, so {} is legal and Qwen3-8B collapses
        # to that ~50% after </think>. json_schema with required rating field
        # forces the grammar to emit a rating before terminating.
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "turing_verdict",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {"rating": {"type": "integer", "minimum": 1, "maximum": 7}},
                    "required": ["rating"],
                    "additionalProperties": True,
                },
            },
        }
        body["reasoning"] = {"enabled": True}
    elif variant == "V2":
        body["reasoning"] = {"enabled": True}
        # no response_format
    elif variant == "V3":
        body["response_format"] = {"type": "json_object"}
        body["reasoning"] = {"enabled": False}
    else:
        raise ValueError(f"unknown variant {variant}")
    return body


async def call_one(
    session: aiohttp.ClientSession,
    endpoint: str,
    sem: asyncio.Semaphore,
    triple_idx: int,
    prompt: str,
    variant: str,
    rating_397b: int | None,
) -> CallResult:
    body = build_body(prompt, variant)
    async with sem:
        t0 = time.time()
        try:
            async with session.post(
                endpoint,
                json=body,
                headers={"Authorization": "Bearer dummy", "Content-Type": "application/json"},
                timeout=aiohttp.ClientTimeout(total=1200),
            ) as resp:
                status = resp.status
                data = await resp.json()
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            return CallResult(
                triple_idx=triple_idx, variant=variant,
                http_ok=False, http_status=None, latency_s=time.time() - t0,
                prompt_tokens=None, completion_tokens=None,
                content_len=0, reasoning_len=0,
                json_parses=False, has_required_keys=False, has_rating_field=False,
                regex_recovered_rating=None, usable_rating=None,
                category="fail_http", rating_397b=rating_397b,
                error_text=f"{type(exc).__name__}: {exc}",
            )
        lat = time.time() - t0

    http_ok = status == 200
    if not http_ok:
        return CallResult(
            triple_idx=triple_idx, variant=variant,
            http_ok=False, http_status=status, latency_s=lat,
            prompt_tokens=None, completion_tokens=None,
            content_len=0, reasoning_len=0,
            json_parses=False, has_required_keys=False, has_rating_field=False,
            regex_recovered_rating=None, usable_rating=None,
            category="fail_http", rating_397b=rating_397b,
            error_text=str(data)[:400],
        )

    choice = (data.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning") or ""
    usage = data.get("usage") or {}
    parsed = _extract_json(content)
    regex_rating = _extract_turing_rating(content)
    category, usable = _categorize(parsed, regex_rating, http_ok=True)
    return CallResult(
        triple_idx=triple_idx, variant=variant,
        http_ok=True, http_status=200, latency_s=lat,
        prompt_tokens=usage.get("prompt_tokens"),
        completion_tokens=usage.get("completion_tokens"),
        content_len=len(content),
        reasoning_len=len(reasoning),
        json_parses=parsed is not None,
        has_required_keys=bool(parsed and all(k in parsed for k in REQUIRED_KEYS)),
        has_rating_field=bool(parsed and _coerce_rating(parsed.get("rating")) is not None),
        regex_recovered_rating=regex_rating,
        usable_rating=usable,
        category=category,
        rating_397b=rating_397b,
        content_head=content[:400],
        reasoning_head=reasoning[:400],
    )


async def run_all(
    endpoint: str, triples: list[tuple[str, int | None]], variants: list[str], concurrency: int
) -> list[CallResult]:
    sem = asyncio.Semaphore(concurrency)
    connector = aiohttp.TCPConnector(limit=concurrency)
    results: list[CallResult] = []
    total = len(triples) * len(variants)
    done = 0
    async with aiohttp.ClientSession(connector=connector) as session:
        # Interleave variants per triple so all variants make progress together —
        # helps if the judge crashes partway through.
        coros = []
        for i, (prompt, r397) in enumerate(triples):
            for v in variants:
                coros.append(call_one(session, endpoint, sem, i, prompt, v, r397))
        for coro in asyncio.as_completed(coros):
            r = await coro
            results.append(r)
            done += 1
            if done % 10 == 0 or done == total:
                print(f"  [{done}/{total}] triple={r.triple_idx} {r.variant} {r.category} lat={r.latency_s:.1f}s", flush=True)
    return results


def print_summary(results: list[CallResult], variants: list[str]) -> None:
    per_variant: dict[str, list[CallResult]] = {v: [] for v in variants}
    for r in results:
        per_variant[r.variant].append(r)

    print()
    print("=" * 100)
    print("SUMMARY (rows = variants, columns = failure category, % of variant total)")
    print("=" * 100)
    header = f"{'variant':<8} {'n':>4} {'ok_full':>8} {'ok_rating':>10} {'ok_regex':>9} {'usable':>7}  {'empty':>6} {'malfmt':>7} {'http_fail':>10}  {'lat_p50':>8} {'lat_p95':>8}  {'agree397':>9}"
    print(header)
    print("-" * len(header))
    for v in variants:
        rows = per_variant[v]
        n = len(rows)
        if not n:
            continue
        cats = [r.category for r in rows]
        c_full = sum(1 for c in cats if c == "ok_full_schema")
        c_rat = sum(1 for c in cats if c == "ok_rating_only")
        c_reg = sum(1 for c in cats if c == "ok_regex_recovered")
        c_empty = sum(1 for c in cats if c == "fail_empty_json")
        c_mal = sum(1 for c in cats if c == "fail_malformed_json")
        c_http = sum(1 for c in cats if c == "fail_http")
        usable = c_full + c_rat + c_reg
        lats = [r.latency_s for r in rows]
        p50 = statistics.median(lats)
        p95 = statistics.quantiles(lats, n=20)[-1] if len(lats) >= 20 else max(lats)
        # 8B-vs-397B agreement: both must have a usable rating, and they must
        # pick the same "human is..." side. Rating >= 5 means judge picks B,
        # rating <= 3 means A, rating 4 is 'cannot tell'.
        def _side(r: int | None) -> str | None:
            if r is None: return None
            if r <= 3: return "A"
            if r >= 5: return "B"
            return "tie"
        pairs = [(r.usable_rating, r.rating_397b) for r in rows if r.usable_rating is not None and r.rating_397b is not None]
        if pairs:
            agree = sum(1 for a, b in pairs if _side(a) == _side(b)) / len(pairs)
            agree_s = f"{agree:.0%} ({len(pairs)})"
        else:
            agree_s = "n/a"
        print(f"{v:<8} {n:>4} {c_full/n:>7.0%} {c_rat/n:>9.0%} {c_reg/n:>8.0%} {usable/n:>6.0%}  {c_empty/n:>5.0%} {c_mal/n:>6.0%} {c_http/n:>9.0%}  {p50:>7.1f}s {p95:>7.1f}s  {agree_s:>9}")

    print()
    print("=" * 100)
    print("DECISION RULES")
    print("=" * 100)

    def usable_rate(v: str) -> float:
        rows = per_variant.get(v, [])
        if not rows: return 0.0
        return sum(1 for r in rows if r.category.startswith("ok_")) / len(rows)

    v1_ok = usable_rate("V1")
    v2_ok = usable_rate("V2")
    v3_ok = usable_rate("V3")
    v4_ok = usable_rate("V4")
    print(f"  V1 usable = {v1_ok:.0%}   V2 usable = {v2_ok:.0%}   V3 usable = {v3_ok:.0%}   V4 usable = {v4_ok:.0%}")
    print()
    if v1_ok >= 0.90:
        print("  -> V1 (production) >= 90% usable: 8B is fine, proceed to smoke.")
    elif v2_ok >= 0.90 and v2_ok - v1_ok >= 0.10:
        print("  -> Dropping response_format helps materially: prompt/decoding issue.")
        print("     Fix: set PERSONA_JUDGE_DISABLE_RESPONSE_FORMAT=1 or PERSONA_LOCAL_JUDGE_DISABLE_RESPONSE_FORMAT=1 in grpo_smoke_8b.sh.")
    elif v3_ok >= 0.90 and v3_ok - v1_ok >= 0.10:
        print("  -> Turning off thinking helps materially: reasoning is eating the JSON budget.")
        print("     Fix: disable --reasoning-parser deepseek_r1 in judge_serve_8b.sh, or send /no_think in the payload.")
    elif v4_ok >= 0.90 and v4_ok - v1_ok >= 0.10:
        print("  -> Doubling max_completion_tokens to 16384 helps: budget-limited.")
        print("     Fix: export PERSONA_JUDGE_MAX_COMPLETION_TOKENS=16384 in grpo_smoke_8b.sh.")
    else:
        print("  -> No single-knob fix reaches 90% usable. 8B is likely under-capable for this rubric.")
        print("     Options: escalate to a larger judge (Qwen3.5-30B-A3B is cached), or simplify TURING_PROMPT.")

    # 8B-vs-397B semantic sanity: even if V1 succeeds on schema, does it agree?
    v1_rows = per_variant.get("V1", [])
    v1_pairs = [(r.usable_rating, r.rating_397b) for r in v1_rows if r.usable_rating is not None and r.rating_397b is not None]
    if v1_pairs:
        def _side(r: int | None) -> str | None:
            if r is None: return None
            if r <= 3: return "A"
            if r >= 5: return "B"
            return "tie"
        agree = sum(1 for a, b in v1_pairs if _side(a) == _side(b)) / len(v1_pairs)
        print()
        print(f"  V1 8B-vs-397B side-agreement: {agree:.0%} (n={len(v1_pairs)})")
        if agree < 0.60:
            print("  -> WARNING: agreement < 60%. Even when 8B produces valid schema, its verdicts")
            print("     don't track the 397B judge. Frozen-8B hackability results would be suspect.")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=os.environ.get("JUDGE_URL", "http://localhost:8123/v1"))
    parser.add_argument("--dumps", type=Path, default=Path("/home/lancewicki/tmp/judge_dumps/judge-9239-1968156.jsonl"))
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--variants", default="V1,V2,V3,V4")
    parser.add_argument("--out", type=Path, default=Path("/home/lancewicki/tmp/judge_8b_diag"))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    out_path = args.out / f"results-{ts}.jsonl"

    triples = load_triples(args.dumps, args.n)
    if not triples:
        raise SystemExit(f"no triples loaded from {args.dumps}")
    endpoint = f"{args.url.rstrip('/')}/chat/completions"
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    print(f"[diag] endpoint={endpoint}")
    print(f"[diag] dumps={args.dumps}")
    print(f"[diag] n={len(triples)} triples, variants={variants}, concurrency={args.concurrency}")
    print(f"[diag] writing per-call results to {out_path}")
    print()

    results = await run_all(endpoint, triples, variants, args.concurrency)
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

    print_summary(results, variants)


if __name__ == "__main__":
    asyncio.run(main())
