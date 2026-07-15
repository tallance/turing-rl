"""Faithful replica of the PAPER's judge call (upstream 6aaecfb reward path).

Unlike openrouter_judge_probe.py (which forced reasoning:{enabled:true}), this
mirrors what the paper's code actually sent:
  - response_format = {"type": "json_object"}
  - reasoning=False  -> NO `reasoning` field added (see openrouter_request_extras)
  - provider.order = ["Morph"]  (upstream default OPENROUTER_PROVIDER_ORDER=Morph)

Question it answers: with the paper's exact call, does the 397B judge still
think (inline <think> in content, or a populated .reasoning), or not?

Usage:
  OPENROUTER_API_KEY=sk-or-... python3 scripts/openrouter_judge_probe_faithful.py
  # optionally override provider(s):
  OPENROUTER_PROVIDER_ORDER=Morph,Novita OPENROUTER_API_KEY=... python3 scripts/...
"""
import json
import os
import sys
import urllib.request

api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
if not api_key:
    print("ERROR: set OPENROUTER_API_KEY", file=sys.stderr)
    sys.exit(2)

provider_order = [p.strip() for p in os.environ.get("OPENROUTER_PROVIDER_ORDER", "Morph").split(",") if p.strip()]

payload = {
    "model": "qwen/qwen3.5-397b-a17b",
    "max_completion_tokens": 1024,
    "response_format": {"type": "json_object"},
    # NOTE: no "reasoning" field on purpose — this is the reasoning=False path.
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
if provider_order:
    payload["provider"] = {"order": provider_order}

req = urllib.request.Request(
    "https://openrouter.ai/api/v1/chat/completions",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(req, timeout=120) as resp:
    data = json.loads(resp.read().decode("utf-8"))

choice = data["choices"][0]
msg = choice["message"]
content = msg.get("content") or ""
reasoning = msg.get("reasoning") or ""
usage = data.get("usage") or {}
reasoning_tokens = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")

print("=" * 70)
print(f"requested provider order : {provider_order or '(none / OpenRouter default)'}")
print(f"served provider          : {data.get('provider')}")
print(f"model returned           : {data.get('model')}")
print(f"finish_reason            : {choice.get('finish_reason')}")
print(f"message keys             : {sorted(msg.keys())}")
print(f"content len              : {len(content)} chars")
print(f"reasoning len            : {len(reasoning)} chars")
print(f"<think> in content       : {'<think>' in content}")
print(f"reasoning_tokens (usage) : {reasoning_tokens}")
print(f"completion_tokens        : {usage.get('completion_tokens')}")
print("content parses as JSON    : ", end="")
try:
    parsed = json.loads(content)
    print(f"YES, keys={sorted(parsed.keys())}")
except Exception as exc:
    print(f"NO ({type(exc).__name__}: {exc})")
print("=" * 70)
print("CONTENT (first 400):")
print(content[:400])
print()
print("REASONING (first 400):")
print(reasoning[:400] if reasoning else "(empty)")
print("=" * 70)
print("=== VERDICT (paper-faithful call: reasoning=False, provider=Morph) ===")
thinks = bool(reasoning and len(reasoning) > 50) or ("<think>" in content) or bool(reasoning_tokens and reasoning_tokens > 20)
if thinks:
    print("Judge STILL THINKS even with reasoning=False → thinking is the model/template")
    print("default; the paper's judge effectively ran thinking-ON.")
else:
    print("Judge does NOT think with reasoning=False → paper's judge ran thinking-OFF.")
