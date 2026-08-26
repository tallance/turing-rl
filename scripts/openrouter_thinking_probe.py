"""Ask OpenRouter whether reasoning:{enabled:true} is honored for qwen/qwen3-8b.

Small toy prompt (no parquet dep). Prints a classification block so we can tell
if OpenRouter populated `.reasoning`, left <think> inline in `.content`, or
silently ignored the flag entirely.
"""

from __future__ import annotations

import getpass
import json
import os
import sys
import urllib.error
import urllib.request

API_KEY = os.environ.get("OPENROUTER_API_KEY") or getpass.getpass("OpenRouter API key: ").strip()
if not API_KEY:
    print("ERROR: no API key provided (paste got eaten?). Try:", file=sys.stderr)
    print("  OPENROUTER_API_KEY=sk-or-v1-... python3 " + sys.argv[0], file=sys.stderr)
    sys.exit(2)
print(f"[probe] using key of length {len(API_KEY)} starting with {API_KEY[:8]!r}")

PAYLOAD = {
    "model": "qwen/qwen3-8b",
    "max_completion_tokens": 4096,
    "reasoning": {"enabled": True},
    "messages": [
        {
            "role": "system",
            "content": "You reconstruct a user's private reasoning that led to a Reddit reply.",
        },
        {
            "role": "user",
            "content": (
                "[CONTEXT] Someone posted: I love hiking! Any beginner trails near Denver?\n"
                "[REPLY] Check out Mount Falcon, super gentle grade and stunning views. "
                "I did it my first month here.\n"
                "Write the user's reasoning that leads to this reply."
            ),
        },
    ],
}

req = urllib.request.Request(
    "https://openrouter.ai/api/v1/chat/completions",
    data=json.dumps(PAYLOAD).encode("utf-8"),
    headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
    method="POST",
)
try:
    with urllib.request.urlopen(req, timeout=180) as resp:
        data = json.loads(resp.read().decode("utf-8"))
except urllib.error.HTTPError as exc:
    body = exc.read().decode("utf-8", errors="replace")
    print(f"[probe] HTTP {exc.code} {exc.reason}", file=sys.stderr)
    print(f"[probe] response body: {body}", file=sys.stderr)
    sys.exit(1)

choice = data["choices"][0]
msg = choice["message"]
content = msg.get("content") or ""
reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""

print("=" * 60)
print(f"finish_reason        : {choice.get('finish_reason')}")
print(f"message keys         : {sorted(msg.keys())}")
print(f"content len          : {len(content)}")
print(f"reasoning len        : {len(reasoning)}")
print(f"<think> in content   : {'<think>' in content}")
print(f"</think> in content  : {'</think>' in content}")
print(f"usage                : {data.get('usage')}")
print(f"provider             : {data.get('provider')}")
print("=" * 60)
print("CONTENT (first 500 chars):")
print(content[:500])
print()
print("=" * 60)
print("REASONING (first 500 chars):")
print(reasoning[:500] if reasoning else "(empty)")
