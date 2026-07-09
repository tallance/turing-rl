"""Probe vLLM judge with three payload variants to see what makes .content
and .reasoning behave sanely.

Assumes a Qwen3.5-397B-A17B-GPTQ-Int4 judge is running at $JUDGE_URL
(default http://localhost:8000/v1) with --reasoning-parser <something>.

Prints, for each variant:
  - message keys
  - content len + first 200 chars + parses-as-JSON?
  - reasoning len + first 200 chars + parses-as-JSON?
  - usage token counts
"""
import argparse
import json
import os
import sys
import urllib.request

PROMPT = (
    "Decide which candidate reply is more likely written by a human vs an AI.\n\n"
    "Context: The user asked about gun control policy.\n\n"
    "Response A: 'Guns don\\'t kill people, people do. We need mental health support not more laws.'\n\n"
    "Response B: 'Gun control is a complex topic with valid arguments on both sides. "
    "It requires balancing individual rights with public safety concerns.'\n\n"
    "Return JSON: {\"rating\": <1-7, lower means A more human>, "
    "\"explanation\": <one sentence>}."
)


def call(url: str, body: dict, label: str) -> None:
    print("=" * 72)
    print(f"VARIANT: {label}")
    print(f"  payload keys sent: {sorted([k for k in body if k not in ('messages',)])}")
    print("=" * 72)
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": "Bearer dummy", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        print(f"  HTTP ERROR: {type(exc).__name__}: {exc}")
        return

    choice = (data.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning") or ""
    print(f"  message keys : {sorted(msg.keys())}")
    print(f"  content len  : {len(content)}")
    print(f"  reasoning len: {len(reasoning)}")
    print(f"  usage        : {data.get('usage')}")

    def _parses(s: str) -> str:
        s = s.strip()
        try:
            obj = json.loads(s)
            return f"YES, keys={sorted(obj.keys())}"
        except json.JSONDecodeError as e:
            return f"NO ({e})"

    print(f"  content parses: {_parses(content)}")
    print(f"  reasoning parses: {_parses(reasoning)}")
    print(f"  content [first 400]:\n    {content[:400]!r}")
    print(f"  reasoning [first 400]:\n    {reasoning[:400]!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=os.environ.get("JUDGE_URL", "http://localhost:8000/v1"))
    args = parser.parse_args()

    base = {
        "model": "Qwen/Qwen3.5-397B-A17B-GPTQ-Int4",
        "max_completion_tokens": 2048,
        "messages": [{"role": "user", "content": PROMPT}],
    }

    endpoint = f"{args.url.rstrip('/')}/chat/completions"

    # A0: verl's actual payload — json_object AND reasoning:enabled
    call(endpoint, {**base, "response_format": {"type": "json_object"}, "reasoning": {"enabled": True}},
         "A0 (verl actual: json_object + reasoning:enabled)")

    # A1: drop json_object, keep reasoning
    call(endpoint, {**base, "reasoning": {"enabled": True}},
         "A1 (option A: no json_object, reasoning:enabled)")

    # B0: json_object only, no reasoning field
    call(endpoint, {**base, "response_format": {"type": "json_object"}},
         "B0 (control: json_object only, no reasoning:enabled)")

    # X: no extras (pure default)
    call(endpoint, base, "X (default: no response_format, no reasoning)")


if __name__ == "__main__":
    main()
