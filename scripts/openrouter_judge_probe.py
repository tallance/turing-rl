"""Tiny probe: does the paper's judge call (json_object + reasoning:enabled) get thinking?

Sends ONE request to OpenRouter's qwen/qwen3.5-397b-a17b with:
  - response_format = {"type": "json_object"}
  - reasoning = {"enabled": true}

Prints whether .content has clean JSON, whether .reasoning is populated,
and shows the shape so we can decide if our self-hosted vLLM is faithful.

Usage:
  OPENROUTER_API_KEY=sk-or-... python scripts/openrouter_judge_probe.py

Cost: ~$0.001-0.005 for one call.
"""
import json
import os
import sys
import urllib.request

api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
if not api_key:
    print("ERROR: set OPENROUTER_API_KEY", file=sys.stderr)
    sys.exit(2)

# Minimal Turing-style prompt (much shorter than real, but same shape:
# rubric, response_format request, JSON schema).
payload = {
    "model": "qwen/qwen3.5-397b-a17b",
    "max_completion_tokens": 1024,
    "response_format": {"type": "json_object"},
    "reasoning": {"enabled": True},
    "messages": [
        {
            "role": "user",
            "content": (
                "Decide which candidate reply is more likely written by a human vs an AI.\n\n"
                "Context: The user asked about gun control policy.\n\n"
                "Response A: 'Guns don\\'t kill people, people do. We need mental health support not more laws.'\n\n"
                "Response B: 'Gun control is a complex topic with valid arguments on both sides. "
                "It requires balancing individual rights with public safety concerns.'\n\n"
                "Return JSON: {\"rating\": <1-7, lower means A more human>, "
                "\"explanation\": <one sentence>}."
            ),
        }
    ],
}

url = "https://openrouter.ai/api/v1/chat/completions"
req = urllib.request.Request(
    url,
    data=json.dumps(payload).encode("utf-8"),
    headers={
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    },
    method="POST",
)
with urllib.request.urlopen(req, timeout=120) as resp:
    data = json.loads(resp.read().decode("utf-8"))

print("=" * 70)
print("Full response (pretty):")
print(json.dumps(data, indent=2, ensure_ascii=False))
print("=" * 70)

choice = data["choices"][0]
msg = choice["message"]
content = msg.get("content") or ""
reasoning = msg.get("reasoning") or ""

print()
print("DIAGNOSTICS:")
print(f"  model returned : {data.get('model')}")
print(f"  provider       : {data.get('provider')}")
print(f"  message keys   : {sorted(msg.keys())}")
print(f"  content len    : {len(content)} chars")
print(f"  reasoning len  : {len(reasoning)} chars")
print(f"  content parses : ", end="")
try:
    parsed = json.loads(content)
    print(f"YES, keys={sorted(parsed.keys())}")
except Exception as exc:
    print(f"NO ({type(exc).__name__}: {exc})")
print(f"  usage          : {data.get('usage')}")

print()
print("=== VERDICT ===")
if reasoning and len(reasoning) > 50:
    print(f"REASONING IS POPULATED ({len(reasoning)} chars).")
    print("→ Paper judge DOES have thinking. Our self-hosted judge (no --reasoning-parser)")
    print("  is DIVERGING from paper. We should figure out how to enable it.")
else:
    print("REASONING IS EMPTY or trivial.")
    print("→ Paper judge effectively has thinking OFF when json_object is set.")
    print("  Our current judge behavior is FAITHFUL. Task closed.")
