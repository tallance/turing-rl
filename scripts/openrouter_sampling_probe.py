"""One-off: probe what sampling OpenRouter/Morph applies to Qwen3 by default.

Run from the Mac only. Requires OPENROUTER_API_KEY in env (personal $10 account).
NOT part of any cluster path. Records the completion-length / token-usage
distributions (and any echoed params) for thinking-on vs thinking-off, so we can
replicate the same sampling server-side instead of guessing.
"""
import argparse, json, os, statistics, urllib.request
from pathlib import Path

URL = "https://openrouter.ai/api/v1/chat/completions"
PROMPT = "Reply with one short sentence about the weather."


def call(reasoning_enabled: bool) -> dict:
    payload = {
        "model": "qwen/qwen3-8b",
        "messages": [{"role": "user", "content": PROMPT}],
        "max_completion_tokens": 512,
        "provider": {"order": ["Morph"], "allow_fallbacks": True},
    }
    if reasoning_enabled:
        payload["reasoning"] = {"enabled": True}
    req = urllib.request.Request(
        URL,
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", default="results/2026-07-08-judge-sweep/derived/sampling_fidelity.md")
    args = ap.parse_args()
    rows = {"on": [], "off": []}
    for mode, enabled in (("on", True), ("off", False)):
        for _ in range(args.n):
            d = call(enabled)
            u = d.get("usage", {})
            rows[mode].append({
                "completion_tokens": u.get("completion_tokens"),
                "content_len": len((d["choices"][0]["message"].get("content") or "")),
                "params_echo": d.get("provider") or d.get("generation_config"),
            })
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        f.write("# OpenRouter Qwen3 sampling fidelity probe\n\n")
        for mode in ("on", "off"):
            lens = [r["completion_tokens"] for r in rows[mode] if r["completion_tokens"]]
            f.write(f"## thinking-{mode} (n={len(rows[mode])})\n")
            if lens:
                f.write(f"- completion_tokens: mean={statistics.mean(lens):.0f} "
                        f"min={min(lens)} max={max(lens)}\n")
            f.write(f"- sample params echo: {json.dumps(rows[mode][0]['params_echo'])}\n\n")
        f.write("## DECISION\n\nFrozen sampling to replicate server-side (fill after review):\n"
                "- thinking-on: T=?, top_p=?, top_k=?, min_p=?\n"
                "- thinking-off: T=?, top_p=?, top_k=?, min_p=?\n")


if __name__ == "__main__":
    main()
