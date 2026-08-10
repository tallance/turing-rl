#!/usr/bin/env python3
"""Paired Gemma 4 judge smoke test for minimal versus full JSON Schema.

The two variants reuse the same recorded judge messages, request seed, sampling
settings, model server, and completion budget.  Only ``response_format`` differs.
All raw API responses are retained so structural failures can be audited.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


MODEL = "google/gemma-4-31B-it"
CURRENT_SCHEMA = {
    "type": "object",
    "properties": {
        "rating": {"type": "integer", "minimum": 1, "maximum": 7},
    },
    "required": ["rating"],
    "additionalProperties": True,
}


def _string() -> dict[str, Any]:
    return {"type": "string"}


def _number(minimum: float, maximum: float) -> dict[str, Any]:
    return {"type": "number", "minimum": minimum, "maximum": maximum}


# Property insertion order intentionally matches the prompt's requested order.
FULL_PROPERTIES: dict[str, dict[str, Any]] = {
    "immediate_target_a": _string(),
    "immediate_target_score_a": _number(0.0, 1.0),
    "immediate_target_b": _string(),
    "immediate_target_score_b": _number(0.0, 1.0),
    "human_goal_a": _string(),
    "human_goal_score_a": _number(0.0, 1.0),
    "human_goal_b": _string(),
    "human_goal_score_b": _number(0.0, 1.0),
    "communication_style_a": _string(),
    "communication_style_score_a": _number(0.0, 1.0),
    "communication_style_b": _string(),
    "communication_style_score_b": _number(0.0, 1.0),
    "base_score_a": _number(0.0, 3.0),
    "base_score_b": _number(0.0, 3.0),
    "response_a_score": _number(0.0, 3.0),
    "response_b_score": _number(0.0, 3.0),
    "score_gap": _number(-3.0, 3.0),
    "response_a_source_copy": _string(),
    "source_copy_penalty_a": _number(0.0, 1.0),
    "response_b_source_copy": _string(),
    "source_copy_penalty_b": _number(0.0, 1.0),
    "response_a_wrong_target_or_role": _string(),
    "wrong_target_or_role_penalty_a": _number(0.0, 1.0),
    "response_b_wrong_target_or_role": _string(),
    "wrong_target_or_role_penalty_b": _number(0.0, 1.0),
    "response_a_unsupported_adversarial_reframing": _string(),
    "unsupported_adversarial_reframing_penalty_a": _number(0.0, 1.0),
    "response_b_unsupported_adversarial_reframing": _string(),
    "unsupported_adversarial_reframing_penalty_b": _number(0.0, 1.0),
    "response_a_assistant_like": _string(),
    "assistant_like_penalty_a": _number(0.0, 1.0),
    "response_b_assistant_like": _string(),
    "assistant_like_penalty_b": _number(0.0, 1.0),
    "penalty_a": _number(0.0, 3.0),
    "penalty_b": _number(0.0, 3.0),
    "reasoning": _string(),
    "rating": {"type": "integer", "minimum": 1, "maximum": 7},
}
FULL_SCHEMA = {
    "type": "object",
    "properties": FULL_PROPERTIES,
    "required": list(FULL_PROPERTIES),
    "additionalProperties": False,
}
SCHEMAS = {"current_minimal": CURRENT_SCHEMA, "full_prompt_schema": FULL_SCHEMA}


def load_recorded_messages(path: Path) -> list[list[dict[str, Any]]]:
    rows: list[list[dict[str, Any]]] = []
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_no}") from exc
            messages = row.get("payload_messages")
            if isinstance(messages, list) and messages:
                rows.append(messages)
    if not rows:
        raise ValueError(f"no payload_messages found in {path}")
    return rows


def evenly_spaced_indices(total: int, count: int) -> list[int]:
    if count < 1:
        raise ValueError("count must be positive")
    if count > total:
        raise ValueError(f"requested {count} prompts, but only {total} are available")
    if count == 1:
        return [0]
    return [round(i * (total - 1) / (count - 1)) for i in range(count)]


def build_body(
    messages: list[dict[str, Any]], variant: str, model: str, seed: int
) -> dict[str, Any]:
    return {
        "model": model,
        "messages": messages,
        "max_completion_tokens": 8192,
        "repetition_penalty": 1.1,
        "temperature": 0.6,
        "seed": seed,
        "chat_template_kwargs": {"enable_thinking": True},
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": f"turing_{variant}",
                "schema": SCHEMAS[variant],
            },
        },
    }


def _parse_root_pairs(content: Any) -> tuple[dict[str, Any] | None, list[str], list[str]]:
    if not isinstance(content, str):
        return None, [], []
    try:
        pairs = json.loads(content, object_pairs_hook=lambda value: value)
    except json.JSONDecodeError:
        return None, [], []
    if not isinstance(pairs, list) or any(not isinstance(item, tuple) for item in pairs):
        return None, [], []
    keys = [key for key, _ in pairs]
    duplicates = sorted(key for key, count in Counter(keys).items() if count > 1)
    return dict(pairs), keys, duplicates


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def schema_valid(parsed: dict[str, Any] | None, variant: str) -> bool:
    if parsed is None:
        return False
    rating = parsed.get("rating")
    if not isinstance(rating, int) or isinstance(rating, bool) or not 1 <= rating <= 7:
        return False
    if variant == "current_minimal":
        return True
    if list(parsed) != list(FULL_PROPERTIES):
        return False
    for key, spec in FULL_PROPERTIES.items():
        value = parsed[key]
        if spec["type"] == "string":
            if not isinstance(value, str):
                return False
        elif spec["type"] == "integer":
            if not isinstance(value, int) or isinstance(value, bool):
                return False
        elif not _is_number(value):
            return False
        if "minimum" in spec and value < spec["minimum"]:
            return False
        if "maximum" in spec and value > spec["maximum"]:
            return False
    return True


def formulas_valid(parsed: dict[str, Any] | None) -> bool | None:
    if parsed is None or any(key not in parsed for key in FULL_PROPERTIES):
        return None
    try:
        expected = {
            "base_score_a": parsed["immediate_target_score_a"]
            + parsed["human_goal_score_a"]
            + parsed["communication_style_score_a"],
            "base_score_b": parsed["immediate_target_score_b"]
            + parsed["human_goal_score_b"]
            + parsed["communication_style_score_b"],
            "penalty_a": (
                parsed["source_copy_penalty_a"]
                + parsed["wrong_target_or_role_penalty_a"]
                + parsed["unsupported_adversarial_reframing_penalty_a"]
                + parsed["assistant_like_penalty_a"]
            )
            / 4
            * 3,
            "penalty_b": (
                parsed["source_copy_penalty_b"]
                + parsed["wrong_target_or_role_penalty_b"]
                + parsed["unsupported_adversarial_reframing_penalty_b"]
                + parsed["assistant_like_penalty_b"]
            )
            / 4
            * 3,
            "score_gap": parsed["response_b_score"] - parsed["response_a_score"],
        }
        return all(abs(float(parsed[key]) - float(value)) <= 0.011 for key, value in expected.items())
    except (TypeError, ValueError):
        return False


def call_one(
    endpoint: str,
    messages: list[dict[str, Any]],
    prompt_index: int,
    variant: str,
    model: str,
    seed: int,
    timeout_s: float,
) -> dict[str, Any]:
    body = build_body(messages, variant, model, seed)
    encoded = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=encoded,
        headers={"Authorization": "Bearer dummy", "Content-Type": "application/json"},
        method="POST",
    )
    started = time.time()
    status: int | None = None
    response: dict[str, Any] | None = None
    error = ""
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as handle:
            status = handle.status
            response = json.loads(handle.read())
    except urllib.error.HTTPError as exc:
        status = exc.code
        error = exc.read().decode("utf-8", errors="replace")[:2000]
    except Exception as exc:  # noqa: BLE001 - preserve per-call diagnostics
        error = f"{type(exc).__name__}: {exc}"[:2000]
    latency_s = time.time() - started

    choice = ((response or {}).get("choices") or [{}])[0]
    message = choice.get("message") or {}
    content = message.get("content")
    reasoning = message.get("reasoning") or message.get("reasoning_content") or ""
    parsed, keys, duplicate_keys = _parse_root_pairs(content)
    usage = (response or {}).get("usage") or {}
    rating = parsed.get("rating") if parsed else None
    if not isinstance(rating, int) or isinstance(rating, bool) or not 1 <= rating <= 7:
        rating = None
    normalized_rating_keys = [
        key for key in keys if key != "rating" and key.strip().casefold() == "rating"
    ]
    return {
        "prompt_index": prompt_index,
        "prompt_sha256": hashlib.sha256(
            json.dumps(messages, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest(),
        "variant": variant,
        "seed": seed,
        "http_status": status,
        "http_ok": status == 200 and response is not None,
        "error": error,
        "latency_s": latency_s,
        "finish_reason": choice.get("finish_reason"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "content_chars": len(content) if isinstance(content, str) else 0,
        "reasoning_chars": len(reasoning) if isinstance(reasoning, str) else 0,
        "valid_json": parsed is not None,
        "schema_valid": schema_valid(parsed, variant),
        "formulas_valid": formulas_valid(parsed),
        "rating": rating,
        "key_count": len(keys),
        "keys": keys,
        "first_key": keys[0] if keys else None,
        "last_key": keys[-1] if keys else None,
        "duplicate_keys": duplicate_keys,
        "normalized_rating_keys": normalized_rating_keys,
        "response": response,
    }


def _numeric_stats(rows: list[dict[str, Any]], key: str) -> dict[str, float | int | None]:
    values = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]
    if not values:
        return {"n": 0, "min": None, "median": None, "mean": None, "max": None}
    return {
        "n": len(values),
        "min": min(values),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "max": max(values),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_variant: dict[str, Any] = {}
    for variant in SCHEMAS:
        selected = [row for row in rows if row["variant"] == variant]
        by_variant[variant] = {
            "n": len(selected),
            "http_ok": sum(row["http_ok"] for row in selected),
            "valid_json": sum(row["valid_json"] for row in selected),
            "schema_valid": sum(row["schema_valid"] for row in selected),
            "rating_present": sum(row["rating"] is not None for row in selected),
            "reasoning_nonempty": sum(row["reasoning_chars"] > 0 for row in selected),
            "rating_first": sum(row["first_key"] == "rating" for row in selected),
            "rating_last": sum(row["last_key"] == "rating" for row in selected),
            "duplicate_key_responses": sum(bool(row["duplicate_keys"]) for row in selected),
            "near_duplicate_rating_key_responses": sum(
                bool(row["normalized_rating_keys"]) for row in selected
            ),
            "formula_valid": sum(row["formulas_valid"] is True for row in selected),
            "formula_invalid": sum(row["formulas_valid"] is False for row in selected),
            "finish_reasons": dict(Counter(str(row["finish_reason"]) for row in selected)),
            "latency_s": _numeric_stats(selected, "latency_s"),
            "completion_tokens": _numeric_stats(selected, "completion_tokens"),
            "content_chars": _numeric_stats(selected, "content_chars"),
            "reasoning_chars": _numeric_stats(selected, "reasoning_chars"),
            "key_count": _numeric_stats(selected, "key_count"),
        }

    paired: list[dict[str, Any]] = []
    for prompt_index in sorted({row["prompt_index"] for row in rows}):
        match = {row["variant"]: row for row in rows if row["prompt_index"] == prompt_index}
        if set(match) != set(SCHEMAS):
            continue
        current = match["current_minimal"]
        full = match["full_prompt_schema"]
        paired.append(
            {
                "prompt_index": prompt_index,
                "seed": current["seed"],
                "current_rating": current["rating"],
                "full_rating": full["rating"],
                "rating_agrees": current["rating"] is not None
                and current["rating"] == full["rating"],
                "current_schema_valid": current["schema_valid"],
                "full_schema_valid": full["schema_valid"],
                "current_completion_tokens": current["completion_tokens"],
                "full_completion_tokens": full["completion_tokens"],
            }
        )
    comparable = [row for row in paired if row["current_rating"] is not None and row["full_rating"] is not None]
    return {
        "variants": by_variant,
        "paired": {
            "n": len(paired),
            "both_ratings_present": len(comparable),
            "rating_agreement": sum(row["rating_agrees"] for row in comparable),
            "rows": paired,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--seed-base", type=int, default=310400)
    parser.add_argument("--timeout-s", type=float, default=1200.0)
    args = parser.parse_args()

    all_messages = load_recorded_messages(args.input)
    indices = evenly_spaced_indices(len(all_messages), args.n)
    selected = [(index, all_messages[index]) for index in indices]
    args.out.mkdir(parents=True, exist_ok=True)

    source_sha256 = hashlib.sha256(args.input.read_bytes()).hexdigest()
    requests_path = args.out / "requests.jsonl"
    with requests_path.open("w", encoding="utf-8") as fh:
        for index, messages in selected:
            fh.write(json.dumps({"prompt_index": index, "messages": messages}, ensure_ascii=False) + "\n")

    manifest = {
        "created_at": time.time(),
        "model": args.model,
        "endpoint": args.endpoint,
        "source": str(args.input),
        "source_sha256": source_sha256,
        "available_prompts": len(all_messages),
        "selected_indices": indices,
        "n_prompts": args.n,
        "concurrency": args.concurrency,
        "seed_base": args.seed_base,
        "timeout_s": args.timeout_s,
        "common_payload": {
            "max_completion_tokens": 8192,
            "repetition_penalty": 1.1,
            "temperature": 0.6,
            "chat_template_kwargs": {"enable_thinking": True},
        },
        "schemas": SCHEMAS,
        "deployed_sha": os.environ.get("DEPLOYED_SHA"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    work = []
    for index, messages in selected:
        seed = args.seed_base + index
        for variant in SCHEMAS:
            work.append((messages, index, variant, seed))

    rows: list[dict[str, Any]] = []
    started = time.time()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [
            pool.submit(
                call_one,
                args.endpoint,
                messages,
                index,
                variant,
                args.model,
                seed,
                args.timeout_s,
            )
            for messages, index, variant, seed in work
        ]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(
                f"prompt={row['prompt_index']:02d} variant={row['variant']} "
                f"status={row['http_status']} finish={row['finish_reason']} "
                f"json={row['valid_json']} schema={row['schema_valid']} "
                f"tokens={row['completion_tokens']} latency={row['latency_s']:.1f}s",
                flush=True,
            )
    rows.sort(key=lambda row: (row["prompt_index"], row["variant"]))
    with (args.out / "responses.jsonl").open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = summarize(rows)
    report["wall_s"] = time.time() - started
    report["artifacts"] = {
        name: hashlib.sha256((args.out / name).read_bytes()).hexdigest()
        for name in ("manifest.json", "requests.jsonl", "responses.jsonl")
    }
    (args.out / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)
    return 0 if all(row["http_ok"] for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
