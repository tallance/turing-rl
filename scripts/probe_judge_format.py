"""Zero-shot judge format probe across three decoding regimes.

Phase 0 gate for judge-only RLVR. veRL cannot constrain rollout decoding, so the number
that matters is how often an *unconstrained* model emits a usable 37-field verdict. The
json_schema and json_object arms are the controls: comparing accuracy across the three
says whether the format scaffold buys verdict quality or only parseability.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections import Counter
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from shared.api_client import (
    build_chat_payload,
    get_judge_call_meta,
    post_chat_async,
    resolve_judge_api_key,
)
from shared.judge_prompts import TURING_RESPONSE_SCHEMA
from training.grpo.judge_reward import directional_task_reward
from training.grpo.judge_verdict import parse_judge_verdict

REGIMES: tuple[str, ...] = ("json_schema", "json_object", "freeform")


def response_format_for_regime(regime: str) -> dict | None:
    """Map a probe regime to the response_format the request should carry."""
    if regime == "json_schema":
        return {
            "type": "json_schema",
            "json_schema": {"name": "turing_verdict", "schema": TURING_RESPONSE_SCHEMA},
        }
    if regime == "json_object":
        return {"type": "json_object"}
    if regime == "freeform":
        return None
    raise ValueError(f"unknown regime {regime!r}; expected one of {REGIMES}")


def even_limit(limit: int) -> int:
    """Round a row limit down to an even number.

    The pairs parquet stores each pair twice, in both A/B orders. An odd limit takes one
    order without its partner, which biases the slot-bias and order-consistency numbers the
    probe exists to produce.
    """
    return max(0, limit - (limit % 2))


def probe_record(completion: str | None, finish_reason: str, human_is_b: bool) -> dict[str, Any]:
    """Score one probe completion into the fields the gate cares about."""
    verdict = parse_judge_verdict(completion)
    correct = (
        directional_task_reward(verdict.rating, human_is_b) if verdict.recovered else 0.0
    )
    return {
        "rung": verdict.recovery_rung,
        "recovered": float(verdict.recovered),
        "fmt_json_valid": float(verdict.fmt_json_valid),
        "fmt_all_fields": float(verdict.fmt_all_fields),
        "fmt_arith": float(verdict.fmt_arith),
        "format_score": verdict.format_score,
        "correct": float(correct == 1.0),
        "acc": float(correct),
        "truncated": 1.0 if finish_reason == "length" else 0.0,
        "rating": verdict.rating if verdict.recovered else None,
        "completion_chars": len(completion or ""),
    }


def summarize_probe(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate probe records into the Phase 0 gate table."""
    if not records:
        return {"n": 0}

    def mean(key: str) -> float:
        return sum(float(r[key]) for r in records) / len(records)

    return {
        "n": len(records),
        "fmt_all_fields_rate": mean("fmt_all_fields"),
        "fmt_json_valid_rate": mean("fmt_json_valid"),
        "fmt_arith_rate": mean("fmt_arith"),
        "format_score_mean": mean("format_score"),
        "recovered_rate": mean("recovered"),
        "accuracy": mean("acc"),
        "truncation_rate": mean("truncated"),
        "completion_chars_mean": mean("completion_chars"),
        "rung_counts": dict(Counter(r["rung"] for r in records)),
    }


def dump_row(model: str, row: dict, record: dict[str, Any], *, regime: str) -> dict[str, Any]:
    """One long-format row for scripts/analyze_judge_training.py.

    ``regime`` is recorded because a verdict from the freeform arm and a verdict from the
    forced-schema arm are not the same measurement, and nothing downstream of this CSV can
    tell them apart otherwise.
    """
    extra = row["extra_info"]
    return {
        "model": model,
        "regime": regime,
        "pair_id": extra["pair_id"],
        "order": extra["order"],
        "rating": record["rating"],
        "human_is_b": bool(extra["human_is_b"]),
    }


async def _run_regime(
    rows: list[dict], regime: str, model: str, max_tokens: int
) -> list[tuple[dict, dict]]:
    """Score every row in one decoding regime; returns (record, source_row) pairs.

    The payload is assembled exactly as reward.py::_openai_chat does, so the probe
    exercises the same sampling, thinking flag and transport as the real judge path.
    """
    import aiohttp

    api_key = resolve_judge_api_key()
    response_format = response_format_for_regime(regime)
    sampling_raw = os.environ.get("PERSONA_JUDGE_SAMPLING")
    sampling = json.loads(sampling_raw) if sampling_raw else None
    thinking = os.environ.get("PERSONA_JUDGE_ENABLE_THINKING")
    chat_template_kwargs = {"enable_thinking": thinking == "1"} if thinking in ("0", "1") else None

    semaphore = asyncio.Semaphore(int(os.environ.get("JUDGE_PROBE_CONCURRENCY", "8")))
    timeout = aiohttp.ClientTimeout(
        total=float(os.environ.get("PERSONA_OPENAI_TIMEOUT_SECONDS", "1800"))
    )

    async def _one(row: dict) -> tuple[dict, dict]:
        payload = build_chat_payload(
            model=model,
            messages=[{"role": "user", "content": row["prompt"][0]["content"]}],
            max_completion_tokens=max_tokens,
            response_format=response_format,
            reasoning=False,
            sampling=sampling,
            chat_template_kwargs=chat_template_kwargs,
        )
        content = await post_chat_async(session, payload, semaphore=semaphore, api_key=api_key)
        # Telemetry for THIS call: post_chat_async stashes it in a ContextVar, and each
        # gather task carries its own context copy, so this is not cross-talk.
        meta = get_judge_call_meta() or {}
        return (
            probe_record(
                content,
                str(meta.get("finish_reason", "")),
                bool(row["extra_info"]["human_is_b"]),
            ),
            row,
        )

    async with aiohttp.ClientSession(timeout=timeout) as session:
        return list(await asyncio.gather(*(_one(row) for row in rows)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Zero-shot judge format probe")
    parser.add_argument("--pairs_parquet", required=True, help="judge-format parquet (Task 2)")
    parser.add_argument("--model", required=True)
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--dump_csv", default=None,
                        help="Long-format per-row CSV for scripts/analyze_judge_training.py")
    parser.add_argument("--model_label", default=None,
                        help="Name to record in --dump_csv (defaults to --model)")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--max_tokens", type=int, default=8192)
    parser.add_argument("--regimes", nargs="+", default=list(REGIMES))
    parser.add_argument("--dump_regime", default="json_schema",
                        help="Which regime --dump_csv records. Defaults to the forced-schema "
                             "arm, which is the published comparison; it is NOT whichever "
                             "regime happens to be last in --regimes.")
    args = parser.parse_args()

    for regime in args.regimes:
        response_format_for_regime(regime)  # fail fast on a typo
    if args.dump_csv and args.dump_regime not in args.regimes:
        raise SystemExit(
            f"--dump_regime {args.dump_regime!r} is not among --regimes {args.regimes}"
        )

    # The parquet emits each pair twice, in both A/B orders, adjacently. An odd head() takes
    # one unpaired row and silently skews pred_b_rate and order_consistency.
    limit = even_limit(args.limit)
    if limit != args.limit:
        print(f"--limit {args.limit} is odd; using {limit} to keep A/B orders paired", flush=True)

    rows = pd.read_parquet(args.pairs_parquet).head(limit).to_dict("records")
    label = args.model_label or args.model
    summaries: dict[str, Any] = {}
    dump_rows: list[dict[str, Any]] = []
    for regime in args.regimes:
        results = asyncio.run(_run_regime(rows, regime, args.model, args.max_tokens))
        records = [record for record, _row in results]
        summaries[regime] = summarize_probe(records)
        print(f"[{regime}] {json.dumps(summaries[regime], sort_keys=True)}", flush=True)
        if args.dump_csv and regime == args.dump_regime:
            dump_rows = [dump_row(label, row, record, regime=regime) for record, row in results]

    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as handle:
        json.dump({"model": args.model, "n_rows": len(rows), "regimes": summaries}, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"Wrote {args.out_json}")

    if args.dump_csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.dump_csv)), exist_ok=True)
        pd.DataFrame(dump_rows).to_csv(args.dump_csv, index=False)
        print(f"Wrote {len(dump_rows)} rows -> {args.dump_csv} (regime={args.dump_regime})")


if __name__ == "__main__":
    main()
