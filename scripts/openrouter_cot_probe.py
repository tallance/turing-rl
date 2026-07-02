"""One-off smoke test against OpenRouter to verify what qwen/qwen3-8b actually returns.

Uses the exact payload structure that data/sft/generate_cot.py uses, with a real
PRISM smoke row. Dumps the full raw response so we can compare to what self-hosted
vLLM will give us.

Usage:
  OPENROUTER_API_KEY=sk-or-v1-... python scripts/openrouter_cot_probe.py
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402

# Match data/sft/generate_cot.py constants verbatim
RATIONALIZE_SYSTEM_PROMPT = (
    "You reconstruct a Reddit user's private reasoning. You are given the "
    "conversation context and the reply the user actually wrote. Write the user's "
    "first-person, step-by-step reasoning that leads naturally to that exact "
    "reply: what they noticed in the context, their intent, stance, and tone. Do "
    "not quote or restate the reply verbatim; reason about it. Output only the "
    "reasoning, with no preamble and no copy of the reply."
)
RATIONALIZE_USER_TEMPLATE = (
    "[CONTEXT]\n{context}\n\n"
    "[THE USER'S ACTUAL REPLY]\n{ground_truth}\n\n"
    "Write the user's reasoning that leads to this reply."
)


def _as_text(value) -> str:
    return "" if value is None else str(value)


def _row_context(extra_info: dict) -> str:
    sections = []
    persona = _as_text(extra_info.get("persona")).strip()
    history = _as_text(extra_info.get("user_history")).strip()
    context = _as_text(extra_info.get("context") or extra_info.get("thread_context")).strip()
    if persona:
        sections.append(f"[PERSONA]\n{persona}")
    if history:
        sections.append(f"[USER HISTORY]\n{history}")
    if context:
        sections.append(f"[CURRENT CONTEXT]\n{context}")
    return "\n\n".join(sections)


def main() -> int:
    api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: set OPENROUTER_API_KEY", file=sys.stderr)
        return 2

    # Pull one row from the smoke parquet
    parquet_path = REPO_ROOT / "data/prism/history_smoke/train.parquet"
    df = pd.read_parquet(parquet_path)
    row = df.iloc[0].to_dict()
    extra_info = dict(row.get("extra_info") or {})
    reward_model = dict(row.get("reward_model") or {})
    ground_truth = _as_text(reward_model.get("ground_truth"))

    messages = [
        {"role": "system", "content": RATIONALIZE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": RATIONALIZE_USER_TEMPLATE.format(
                context=_row_context(extra_info), ground_truth=ground_truth
            ),
        },
    ]

    # Matches build_chat_payload + openrouter_request_extras exactly:
    # model, messages, max_completion_tokens, provider (optional), reasoning.
    # Provider list configurable via OPENROUTER_PROVIDER_ORDER env (comma-sep, empty=skip).
    payload = {
        "model": "qwen/qwen3-8b",
        "messages": messages,
        "max_completion_tokens": 4096,
        "reasoning": {"enabled": True},
    }
    provider_env = os.environ.get("OPENROUTER_PROVIDER_ORDER", "Morph")
    provider_list = [p.strip() for p in provider_env.split(",") if p.strip()]
    if provider_list:
        payload["provider"] = {"order": provider_list, "allow_fallbacks": False}

    print("=" * 72)
    print("PAYLOAD (excluding messages content):")
    redacted = {k: v for k, v in payload.items() if k != "messages"}
    redacted["messages"] = [{"role": m["role"], "content_len": len(m["content"])} for m in messages]
    print(json.dumps(redacted, indent=2))
    print("=" * 72)

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=180) as resp:
        body = resp.read().decode("utf-8")

    data = json.loads(body)

    print("FULL RAW RESPONSE (json):")
    print(json.dumps(data, indent=2, ensure_ascii=False))
    print("=" * 72)

    # Targeted diagnostics
    choice = (data.get("choices") or [{}])[0]
    message = choice.get("message", {})
    content = message.get("content") or ""
    print("DIAGNOSTICS:")
    print(f"  top-level keys           : {sorted(data.keys())}")
    print(f"  choices[0] keys          : {sorted(choice.keys())}")
    print(f"  message keys             : {sorted(message.keys())}")
    print(f"  has 'reasoning' field?   : {'reasoning' in message}")
    print(f"  has 'reasoning_content'? : {'reasoning_content' in message}")
    print(f"  content length           : {len(content)} chars")
    print(f"  content has <think>?     : {'<think>' in content}")
    print(f"  content has </think>?    : {'</think>' in content}")
    if "reasoning" in message:
        r = message["reasoning"] or ""
        print(f"  reasoning length         : {len(r)} chars")
        print(f"  reasoning has <think>?   : {'<think>' in r}")
    print(f"  usage                    : {data.get('usage')}")
    if "provider" in data:
        print(f"  provider routed          : {data.get('provider')}")
    print("=" * 72)

    print("CONTENT FIRST 800 CHARS:")
    print(content[:800])
    print()
    print("CONTENT LAST 400 CHARS:")
    print(content[-400:])
    if "reasoning" in message and message["reasoning"]:
        print("=" * 72)
        print("REASONING FIELD FIRST 800 CHARS:")
        print(message["reasoning"][:800])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
