#!/usr/bin/env python3
"""Score a real eval example with Opus, using the repo's production judge prompt.

    python3 eval/opus_judge_example.py                 # record 221, writes JSON
    python3 eval/opus_judge_example.py --index 75

Rebuilds TURING_PROMPT from a val_dump.jsonl record and sends it to Opus via
eval.claude_call.ask, then parses it with the production parser. The rebuilt
prompt is asserted byte-identical to the `judge_prompt` the incumbent judge
actually saw, so the only thing that changes is which model answers it.

Rating is 1-7 on Response B minus Response A: >4 means B looked more human.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.claude_call import EFFORT_LEVELS, ask  # noqa: E402
from eval.metrics import _parse_turing_response  # noqa: E402
from shared.judge_prompts import TURING_PROMPT  # noqa: E402
from shared.judge_utils import (  # noqa: E402
    build_source_copy_warning,
    format_source_copy_watchlist,
)

DUMP = Path.home() / "Projects/turing-rl/results/9b_half_kl1e4_lr1e4_temp1/val_dump.jsonl"
OUT = Path.home() / "Projects/turing-rl/results/2026-09-04-opus-judge-smoke"


def build_prompt(rec):
    """Rebuild the pairwise Turing prompt exactly as the eval pipeline does."""
    generated_is_b = rec["generated_is_b"]
    response_a = rec["ground_truth"] if generated_is_b else rec["response"]
    response_b = rec["response"] if generated_is_b else rec["ground_truth"]
    warnings = [
        build_source_copy_warning(
            r, user_history=rec["user_history"], thread_context=rec["context"]
        )
        for r in (response_a, response_b)
    ]
    prompt = TURING_PROMPT.format(
        user_history=rec["user_history"],
        context=rec["context"],
        response_a=response_a,
        response_b=response_b,
        source_copy_watchlist=format_source_copy_watchlist(
            warnings, item_label="Response", labels=["Response A", "Response B"]
        ),
    )
    return prompt, response_a, response_b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, default=221)
    ap.add_argument("--dump", type=Path, default=DUMP)
    ap.add_argument("--out", type=Path, default=OUT)
    # Pin this for any run you want to compare across: unset, the binary picks
    # its own level (~800 thinking tokens here), and low/max move real numbers.
    ap.add_argument("--effort", choices=EFFORT_LEVELS, default=None)
    args = ap.parse_args()

    with open(args.dump) as f:
        rec = json.loads(f.readlines()[args.index])

    prompt, response_a, response_b = build_prompt(rec)
    if rec.get("judge_prompt") and prompt != rec["judge_prompt"]:
        raise AssertionError("rebuilt prompt differs from the stored judge_prompt")

    envelope = ask(prompt, full=True, effort=args.effort)
    raw = envelope["result"]
    verdict = _parse_turing_response(raw)
    if verdict.get("parse_error"):
        raise RuntimeError(f"could not parse Opus output: {raw[:500]}")

    human_side = "B" if not rec["generated_is_b"] else "A"
    opus_side = "B" if verdict["rating"] > 4 else "A" if verdict["rating"] < 4 else None

    out = {
        "record_index": args.index,
        "dump": str(args.dump),
        "human_side": human_side,
        "response_a": response_a,
        "response_b": response_b,
        "incumbent_judge": rec.get("judge_model"),
        "incumbent_rating": rec.get("rating_gen_first"),
        "opus_rating": verdict["rating"],
        "opus_picked_human": opus_side,
        "opus_correct": opus_side == human_side,
        "opus_verdict": verdict,
        "opus_raw": raw,
        "effort": args.effort,
        "cost_usd": envelope.get("total_cost_usd"),
        "usage": envelope.get("usage"),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"opus_verdict_{args.index}.json"
    path.write_text(json.dumps(out, indent=2))

    print(f"human is {human_side}; opus rating {verdict['rating']} -> picked {opus_side} "
          f"({'correct' if out['opus_correct'] else 'WRONG'}); "
          f"incumbent {rec.get('judge_model')} rated {rec.get('rating_gen_first')}; "
          f"${envelope.get('total_cost_usd'):.4f}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
