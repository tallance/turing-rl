#!/usr/bin/env python3
"""Prompt in -> text out from a frontier model, via Meta's Claude Code binary.

The binary at /usr/local/bin/claude carries Meta auth, compliance safeguards and
quota accounting, and routes to Vertex. Running it with `-p --output-format json`
is a plain single-turn completion, so no SDK is needed (claude_agent_sdk just
shells out to this same binary).

    from eval.claude_call import ask
    ask("What is 2+2?")                          # -> "4"
    python3 eval/claude_call.py "What is 2+2?"   # CLI
    python3 eval/claude_call.py                  # self-check

Mac only: the cluster's /usr/local/bin/claude is the FAIR Cloud container
sandbox (egress allowlist: llama.com), not a usable completion endpoint.
"""

import json
import os
import subprocess
import sys

CLAUDE = "/usr/local/bin/claude"


def ask(prompt, model="opus", system=None, timeout=300):
    """Single-turn, no-tools completion. Returns the model's text."""
    cmd = [
        CLAUDE,
        "--bare",  # skip hooks, plugins, CLAUDE.md: ~3s startup instead of ~28s
        "-p",
        prompt,
        "--model",
        model,
        "--allowed-tools",
        "",
        "--max-turns",
        "1",
        "--output-format",
        "json",
    ]
    if system:
        cmd += ["--system-prompt", system]
    # ponytail: macOS forbids nested sandbox_exec, so a call made from inside a
    # Claude Code session dies at startup unless the inner sandbox is dropped.
    # Only skip it in that case -- a standalone run keeps the sandbox.
    if os.environ.get("CLAUDECODE") and sys.platform == "darwin":
        cmd.insert(1, "--dangerously-disable-osx-sandbox")

    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(f"claude exited {proc.returncode}: {proc.stderr.strip()}")
    result = json.loads(proc.stdout)  # banner goes to stderr, JSON to stdout
    if result.get("is_error"):
        raise RuntimeError(f"claude error: {result.get('result')}")
    return result["result"]


if __name__ == "__main__":
    if len(sys.argv) > 1:
        print(ask(" ".join(sys.argv[1:])))
    else:
        assert ask("Reply with exactly: PONG") == "PONG"
        assert ask("What is 2+2? Reply with the number only.") == "4"
        assert ask(
            "Say the word.", system="Always reply with exactly: BANANA"
        ) == "BANANA"
        print("ok")
