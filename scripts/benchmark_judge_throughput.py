"""Throughput benchmark for the 8B Qwen judge at TP=1/2/4/8.

Sweeps client concurrency [1, 4, 16, 64] against each provided endpoint. Uses
real Turing prompts cycled from a prior 397B production dump. Payload matches
the schema-fix production path (response_format=json_schema with required
rating; reasoning enabled). Reports per-(endpoint, concurrency) throughput,
latency, and parse rate, and writes:
  - <out_dir>/results-<ts>.jsonl : one row per HTTP call
  - <out_dir>/report-<ts>.md     : human-readable summary with reproduce info

Usage:
  python scripts/benchmark_judge_throughput.py \\
      --endpoint tp1=http://<node1>:8123/v1 \\
      --endpoint tp2=http://<node2>:8123/v1 \\
      --endpoint tp4=http://<node3>:8123/v1 \\
      --endpoint tp8=http://<node4>:8123/v1 \\
      --dumps /home/lancewicki/tmp/judge_dumps/judge-9239-1968156.jsonl \\
      --n 50 \\
      --concurrency 1,4,16,64 \\
      --out /home/lancewicki/projects/turing-rl/results/judge_throughput
"""
from __future__ import annotations

import argparse
import asyncio
import glob
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
from training.grpo.reward import _extract_json  # noqa: E402


@dataclass
class CallResult:
    endpoint_label: str
    concurrency: int
    call_idx: int
    http_ok: bool
    http_status: int | None
    latency_s: float
    prompt_tokens: int | None
    completion_tokens: int | None
    parses_ok: bool
    rating: int | None
    error_text: str = ""


def load_prompts(dump_path: Path, n: int) -> list[str]:
    """Return first n non-empty user prompts from a judge dump JSONL."""
    out: list[str] = []
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
            p = msgs[0].get("content") or ""
            if p.strip():
                out.append(p)
            if len(out) >= n:
                break
    return out


def build_body(prompt: str, model: str = "Qwen/Qwen3-8B") -> dict:
    """Production-matching payload with schema fix."""
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_completion_tokens": 8192,
        "reasoning": {"enabled": True},
        "response_format": {
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
        },
    }


def _coerce_rating(v: Any) -> int | None:
    try:
        r = int(v)
    except (TypeError, ValueError):
        return None
    return r if 1 <= r <= 7 else None


async def one_call(
    session: aiohttp.ClientSession,
    endpoint_label: str,
    endpoint_url: str,
    concurrency: int,
    sem: asyncio.Semaphore,
    call_idx: int,
    prompt: str,
    model: str = "Qwen/Qwen3-8B",
    timeout_s: float = 1200.0,
) -> CallResult:
    body = build_body(prompt, model=model)
    async with sem:
        t0 = time.time()
        try:
            async with session.post(
                endpoint_url,
                json=body,
                headers={"Authorization": "Bearer dummy", "Content-Type": "application/json"},
                timeout=aiohttp.ClientTimeout(total=timeout_s),
            ) as resp:
                status = resp.status
                data = await resp.json()
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            return CallResult(
                endpoint_label=endpoint_label, concurrency=concurrency, call_idx=call_idx,
                http_ok=False, http_status=None, latency_s=time.time() - t0,
                prompt_tokens=None, completion_tokens=None,
                parses_ok=False, rating=None,
                error_text=f"{type(exc).__name__}: {exc}",
            )
    lat = time.time() - t0
    if status != 200:
        return CallResult(
            endpoint_label=endpoint_label, concurrency=concurrency, call_idx=call_idx,
            http_ok=False, http_status=status, latency_s=lat,
            prompt_tokens=None, completion_tokens=None,
            parses_ok=False, rating=None,
            error_text=str(data)[:400],
        )
    choice = (data.get("choices") or [{}])[0]
    content = (choice.get("message") or {}).get("content") or ""
    usage = data.get("usage") or {}
    parsed = _extract_json(content)
    rating = _coerce_rating(parsed.get("rating")) if parsed else None
    return CallResult(
        endpoint_label=endpoint_label, concurrency=concurrency, call_idx=call_idx,
        http_ok=True, http_status=200, latency_s=lat,
        prompt_tokens=usage.get("prompt_tokens"),
        completion_tokens=usage.get("completion_tokens"),
        parses_ok=parsed is not None,
        rating=rating,
    )


async def measure(
    session: aiohttp.ClientSession,
    endpoint_label: str,
    endpoint_url: str,
    concurrency: int,
    prompts: list[str],
    model: str = "Qwen/Qwen3-8B",
    timeout_s: float = 1200.0,
) -> tuple[float, list[CallResult]]:
    """Fire len(prompts) calls at the given concurrency. Return (wall_seconds, results)."""
    sem = asyncio.Semaphore(concurrency)
    t0 = time.time()
    coros = [
        one_call(session, endpoint_label, endpoint_url, concurrency, sem, i, p, model, timeout_s)
        for i, p in enumerate(prompts)
    ]
    results = await asyncio.gather(*coros)
    wall = time.time() - t0
    return wall, results


def summarize_measurement(
    endpoint_label: str, tp: int | None, concurrency: int,
    wall_s: float, results: list[CallResult],
) -> dict:
    n = len(results)
    http_ok = [r for r in results if r.http_ok]
    parse_ok = [r for r in results if r.parses_ok]
    lats = [r.latency_s for r in http_ok]
    completion_toks = sum((r.completion_tokens or 0) for r in http_ok)
    prompt_toks = sum((r.prompt_tokens or 0) for r in http_ok)
    p50 = statistics.median(lats) if lats else 0.0
    p95 = statistics.quantiles(lats, n=20)[-1] if len(lats) >= 20 else (max(lats) if lats else 0.0)
    return {
        "endpoint_label": endpoint_label,
        "tp": tp,
        "concurrency": concurrency,
        "n": n,
        "http_ok": len(http_ok),
        "parse_ok": len(parse_ok),
        "wall_s": round(wall_s, 2),
        "reqs_per_s": round(len(http_ok) / wall_s, 3) if wall_s > 0 else 0.0,
        "output_toks_per_s": round(completion_toks / wall_s, 1) if wall_s > 0 else 0.0,
        "prompt_toks_total": prompt_toks,
        "completion_toks_total": completion_toks,
        "lat_p50_s": round(p50, 2),
        "lat_p95_s": round(p95, 2),
    }


def _fmt_row(s: dict) -> str:
    return (
        f"{s['endpoint_label']:<6} {s['tp'] or '-':>3}  {s['concurrency']:>4}  "
        f"{s['n']:>3}  {s['wall_s']:>7.1f}s  "
        f"{s['reqs_per_s']:>7.3f}  {s['output_toks_per_s']:>8.1f}  "
        f"{s['lat_p50_s']:>7.1f}  {s['lat_p95_s']:>7.1f}  "
        f"{s['parse_ok']}/{s['n']}"
    )


def print_summary_table(summaries: list[dict]) -> str:
    """Print + return the summary table as a string (for the report)."""
    lines: list[str] = []
    header = f"{'label':<6} {'tp':>3}  {'conc':>4}  {'n':>3}  {'wall':>8}  {'req/s':>7}  {'tok/s':>8}  {'p50s':>7}  {'p95s':>7}  {'parse'}"
    lines.append(header)
    lines.append("-" * len(header))
    for s in summaries:
        lines.append(_fmt_row(s))
    out = "\n".join(lines)
    print()
    print(out)
    print()
    return out


def write_report(
    report_path: Path, summaries: list[dict], meta: dict, table_str: str
) -> None:
    lines = [
        f"# 8B Judge Throughput Sweep — {meta['timestamp']}",
        "",
        "## What we ran",
        "",
        f"- Model: `{meta['model']}`",
        f"- Reasoning parser: `--reasoning-parser qwen3`",
        f"- Payload: `response_format=json_schema` (required rating), `reasoning.enabled=true`, `max_completion_tokens=8192`",
        f"- TP configs: {meta['tps']}",
        f"- Client concurrency sweep: {meta['concurrencies']}",
        f"- Prompts per measurement: {meta['n']} (cycled from `{meta['dumps']}`)",
        f"- Port: 8123 (each judge on its own node)",
        f"- Total wall-clock: {meta['total_wall_s']:.1f}s",
        "",
        "## Endpoints",
        "",
    ]
    for ep in meta["endpoints"]:
        lines.append(f"- **{ep['label']}** (TP={ep['tp']}): job `{ep['job_id']}` on `{ep['node']}`, warmup `{ep['warmup_s']:.1f}s`, url `{ep['url']}`")
    lines += [
        "",
        "## Timing table",
        "",
        "```",
        table_str,
        "```",
        "",
        "## How to reproduce",
        "",
        "```",
        f"bash /home/lancewicki/projects/turing-rl/scripts/launch_judge_throughput_sweep.sh",
        "```",
        "",
        "This launches four sbatch jobs (one per TP), waits for them to warm, then runs `scripts/benchmark_judge_throughput.py` against all four.",
        "",
        "**Scripts involved**:",
        "- `/home/lancewicki/projects/turing-rl/scripts/slurm/judge_serve_8b_tp.sh` — parameterized judge (TP env var)",
        "- `/home/lancewicki/projects/turing-rl/scripts/benchmark_judge_throughput.py` — client benchmark",
        "- `/home/lancewicki/projects/turing-rl/scripts/launch_judge_throughput_sweep.sh` — orchestrator",
        "",
        "**Tweak knobs** (edit orchestrator or pass CLI):",
        "- TP set: edit the `TPS` array in the orchestrator",
        "- Concurrency levels: `--concurrency 1,4,16,64` on the benchmark",
        "- Prompt count per measurement: `--n 50`",
        "",
        "**To keep judges alive after the sweep** (skip auto-scancel): comment out the `trap … EXIT` line in `launch_judge_throughput_sweep.sh`; the judges' `--time` is 2h so they stay up until the walltime expires or you scancel them manually.",
    ]
    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")


async def async_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", action="append", required=True,
                        help="label=url  e.g. tp1=http://a100-010-036:8123/v1  (repeatable)")
    parser.add_argument("--dumps", type=Path, default=Path("/home/lancewicki/tmp/judge_dumps/judge-9239-1968156.jsonl"))
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--concurrency", default="1,4,16,64",
                        help="comma-separated concurrency levels to sweep")
    parser.add_argument("--out", type=Path, default=Path("/home/lancewicki/projects/turing-rl/results/judge_throughput"))
    parser.add_argument("--meta-json", type=Path, default=None,
                        help="optional JSON file with per-endpoint metadata (job_id, node, warmup_s, tp) for the report")
    parser.add_argument("--model", default="Qwen/Qwen3-8B",
                        help="model id sent in the payload (must match what the endpoint serves)")
    parser.add_argument("--timeout", type=float, default=1200.0,
                        help="per-request client timeout in seconds")
    args = parser.parse_args()

    endpoints = []
    for pair in args.endpoint:
        if "=" not in pair:
            raise SystemExit(f"--endpoint must be label=url, got {pair!r}")
        label, url = pair.split("=", 1)
        endpoints.append((label.strip(), url.strip().rstrip("/") + "/chat/completions"))
    concurrencies = [int(c) for c in args.concurrency.split(",") if c.strip()]
    args.out.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    results_path = args.out / f"results-{ts}.jsonl"
    report_path = args.out / f"report-{ts}.md"

    prompts = load_prompts(args.dumps, args.n)
    if len(prompts) < args.n:
        print(f"[bench] warning: only {len(prompts)} prompts available, requested {args.n}", flush=True)
    print(f"[bench] loaded {len(prompts)} prompts from {args.dumps}")
    print(f"[bench] endpoints: {[l for l,_ in endpoints]}")
    print(f"[bench] concurrencies: {concurrencies}")

    # Optional per-endpoint metadata from the orchestrator
    endpoint_meta = {}
    if args.meta_json and args.meta_json.exists():
        endpoint_meta = json.loads(args.meta_json.read_text())

    all_results: list[CallResult] = []
    summaries: list[dict] = []
    t0_all = time.time()
    connector = aiohttp.TCPConnector(limit=max(concurrencies) * 2)
    async with aiohttp.ClientSession(connector=connector) as session:
        for label, url in endpoints:
            for conc in concurrencies:
                print(f"[bench] measuring endpoint={label} concurrency={conc} n={len(prompts)} ...", flush=True)
                wall, results = await measure(session, label, url, conc, prompts, args.model, args.timeout)
                all_results.extend(results)
                tp = endpoint_meta.get(label, {}).get("tp")
                s = summarize_measurement(label, tp, conc, wall, results)
                summaries.append(s)
                print(f"  -> {s['reqs_per_s']:.2f} req/s, {s['output_toks_per_s']:.0f} tok/s, "
                      f"p50={s['lat_p50_s']:.1f}s, parse={s['parse_ok']}/{s['n']}",
                      flush=True)
    total_wall = time.time() - t0_all

    # Persist per-call results
    with open(results_path, "w") as f:
        for r in all_results:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

    # Summary table + report
    table_str = print_summary_table(summaries)

    meta = {
        "timestamp": ts,
        "model": args.model,
        "tps": sorted({endpoint_meta.get(l, {}).get("tp") for l, _ in endpoints if endpoint_meta.get(l, {}).get("tp") is not None}),
        "concurrencies": concurrencies,
        "n": len(prompts),
        "dumps": str(args.dumps),
        "total_wall_s": total_wall,
        "endpoints": [
            {
                "label": l,
                "url": u.replace("/chat/completions", ""),
                "tp": endpoint_meta.get(l, {}).get("tp"),
                "job_id": endpoint_meta.get(l, {}).get("job_id"),
                "node": endpoint_meta.get(l, {}).get("node"),
                "warmup_s": endpoint_meta.get(l, {}).get("warmup_s", 0.0),
            }
            for l, u in endpoints
        ],
    }
    write_report(report_path, summaries, meta, table_str)
    print(f"[bench] wrote {results_path}")
    print(f"[bench] wrote {report_path}")


if __name__ == "__main__":
    asyncio.run(async_main())
