"""Gate probe for the 8B judge before launching the full GRPO smoke.

Renders the actual production Turing pairwise prompt (`shared.judge_prompts.TURING_PROMPT`)
with a small synthetic (persona, history, context, response_a, response_b) case —
so the probe hits the same schema/rubric the generator will send during the smoke.

Prefers reusing a REAL prompt from prior 397B dumps under
/home/lancewicki/tmp/judge_dumps/ (`*.jsonl`) if any exist; falls back to the
synthetic case otherwise.

Sends N repeats with the production payload
(response_format=json_object + reasoning=enabled), then reports:
  - HTTP success rate
  - JSON parse rate (via training/grpo/reward.py:_extract_json)
  - required-key coverage (immediate_target, human_goal, communication_style
    scores for A and B, plus rating)
  - latency p50 / p95

Exits 0 if parse rate >= 0.9 and required keys present in every parsed
response; non-zero otherwise. Wire this as a blocking gate before running
launch_grpo_smoke_8b.sh.

Usage:
  JUDGE_URL=http://<node>:8123/v1 python scripts/probe_judge_8b_real_prompt.py --n 10
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from training.grpo.reward import _extract_json  # noqa: E402  # same parser as prod
from shared.judge_prompts import TURING_PROMPT  # noqa: E402  # same template as prod

REQUIRED_KEYS = (
    "immediate_target_score_a",
    "immediate_target_score_b",
    "human_goal_score_a",
    "human_goal_score_b",
    "communication_style_score_a",
    "communication_style_score_b",
    "rating",
)


SYNTHETIC_CASE = dict(
    persona=(
        "A US-based user who enjoys discussing history and current events. "
        "Writes casually, mixes short opinions with the occasional longer point, "
        "uses lowercase and contractions ('don't', 'i think')."
    ),
    user_history=(
        "[HUMAN] on r/AskHistorians (2023-04-11): "
        "'honestly the WW1 alliance system is way overhyped as the *cause* — "
        "it made things spiral fast but the underlying imperial competition was the real fuel.'\n\n"
        "[HUMAN] on r/politics (2023-05-02): "
        "'i don't think a third party breaks through in the US without ranked choice voting first. "
        "the math just doesn't work.'\n\n"
        "[HUMAN] on r/history (2023-06-19): "
        "'lol the \"great man\" theory keeps refusing to die. structural pressures matter way more than "
        "whether Napoleon happened to have a stomach ache at Waterloo.'"
    ),
    context=(
        "Post on r/AskHistorians: 'Was the fall of the Roman Empire mostly internal decay or external pressure?'"
    ),
    response_a=(
        "It's a false dichotomy imho. The internal decay (currency debasement, tax base collapse, elite "
        "capture) made the empire brittle, and then external pressure from migratory groups broke what "
        "was already cracked. Peter Heather's take is decent here."
    ),
    response_b=(
        "The dichotomy is misleading. Internal factors such as economic instability, political corruption, "
        "and social unrest weakened the Empire, while external pressures from migratory groups exploited "
        "these vulnerabilities. Both factors contributed to the fall."
    ),
    source_copy_watchlist=(
        "## Advisory Watchlist\n\n"
        "### Source-Copy Watchlist\n"
        "No high-risk source-copy warnings for either response.\n"
    ),
)


def load_prompt(dumps_dir: Path) -> tuple[str, str]:
    """Return (prompt, source_label). Prefers a real dump; falls back to synthetic."""
    if dumps_dir.is_dir():
        paths = sorted(glob.glob(str(dumps_dir / "*.jsonl")))
        for path in paths:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    msgs = d.get("payload_messages") or []
                    if msgs and (msgs[0].get("content") or "").strip():
                        return msgs[0]["content"], f"real dump {path}"
    prompt = TURING_PROMPT.format(**SYNTHETIC_CASE)
    return prompt, "synthetic (TURING_PROMPT rendered with in-file case)"


def one_call(endpoint: str, model: str, prompt: str) -> tuple[bool, float, str, dict | None]:
    body = {
        "model": model,
        "max_completion_tokens": 8192,
        "response_format": {"type": "json_object"},
        "reasoning": {"enabled": True},
        "messages": [{"role": "user", "content": prompt}],
    }
    req = urllib.request.Request(
        endpoint,
        data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": "Bearer dummy", "Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=1200) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ConnectionError) as exc:
        return False, time.time() - t0, f"{type(exc).__name__}: {exc}", None
    choice = (data.get("choices") or [{}])[0]
    content = (choice.get("message") or {}).get("content") or ""
    return True, time.time() - t0, content, _extract_json(content)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=os.environ.get("JUDGE_URL", "http://localhost:8123/v1"))
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--dumps", type=Path, default=Path("/home/lancewicki/tmp/judge_dumps"),
                        help="prior 397B dump *.jsonl to lift a real prompt from (falls back to synthetic)")
    parser.add_argument("--n", type=int, default=10, help="how many calls to send")
    parser.add_argument("--min-parse-rate", type=float, default=0.9)
    args = parser.parse_args()

    prompt, source_label = load_prompt(args.dumps)
    endpoint = f"{args.url.rstrip('/')}/chat/completions"
    print(f"[probe] endpoint={endpoint}  model={args.model}  n={args.n}")
    print(f"[probe] prompt source: {source_label}  (len={len(prompt)} chars)")

    latencies: list[float] = []
    http_ok = 0
    parse_ok = 0
    keys_ok = 0
    for i in range(args.n):
        ok, lat, content_or_err, parsed = one_call(endpoint, args.model, prompt)
        latencies.append(lat)
        if not ok:
            print(f"  [{i:02d}] HTTP FAIL after {lat:.1f}s: {content_or_err}")
            continue
        http_ok += 1
        if parsed is None:
            print(f"  [{i:02d}] {lat:5.1f}s HTTP ok, parse FAIL. content head: {content_or_err[:200]!r}")
            continue
        parse_ok += 1
        missing = [k for k in REQUIRED_KEYS if k not in parsed]
        if missing:
            print(f"  [{i:02d}] {lat:5.1f}s parsed but missing keys: {missing}")
            continue
        keys_ok += 1
        print(f"  [{i:02d}] {lat:5.1f}s OK   rating={parsed.get('rating')}  "
              f"a_scores=({parsed.get('immediate_target_score_a')},"
              f"{parsed.get('human_goal_score_a')},"
              f"{parsed.get('communication_style_score_a')})")

    p50 = statistics.median(latencies) if latencies else 0.0
    p95 = statistics.quantiles(latencies, n=20)[-1] if len(latencies) >= 20 else max(latencies, default=0.0)

    print("=" * 72)
    print(f"http ok      : {http_ok}/{args.n}")
    print(f"parses ok    : {parse_ok}/{args.n}")
    print(f"required keys: {keys_ok}/{args.n}")
    print(f"latency p50  : {p50:.1f}s")
    print(f"latency p95  : {p95:.1f}s")
    print("=" * 72)

    parse_rate = parse_ok / args.n if args.n else 0.0
    if parse_rate < args.min_parse_rate:
        print(f"GATE FAIL: parse rate {parse_rate:.2%} < {args.min_parse_rate:.0%}. Do NOT launch smoke.")
        sys.exit(1)
    if keys_ok < parse_ok:
        print("GATE FAIL: some parsed responses missing required rubric keys. Do NOT launch smoke.")
        sys.exit(1)
    print("GATE PASS: proceed to launch_grpo_smoke_8b.sh")


if __name__ == "__main__":
    main()
