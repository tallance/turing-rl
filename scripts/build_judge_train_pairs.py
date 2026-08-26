"""Build the judge-training parquet from k-sample inference + a source parquet.

Differs from ``scripts/build_judge_pairs.py`` in three ways: it keeps *every*
generation rather than only the first, it emits each pair in *both* A/B orders, and
it writes veRL training rows (rendered prompt + ``"A"``/``"B"`` label) rather than a
flat pair table for offline judging.

The ``.meta.json`` written next to the parquet carries the rendered-prompt length
distribution. That is the only measurement of how big these prompts actually are, and
``data.max_prompt_length`` in ``training/grpo/configs/qwen35_judge_grpo.yaml`` must be set
from it before any training submit: veRL's ``filter_overlong_prompts`` silently DROPS
over-budget rows from train and val alike.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import re
import sys
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from data.judge.slice import select_slice
from shared.judge_prompts import TURING_PROMPT, TURING_SINGLE_TOKEN_PROMPT
from shared.judge_utils import (
    build_source_copy_warning,
    format_source_copy_watchlist,
    sanitize_prompt_text,
)
from shared.prompt_utils import parse_reasoning_and_response

DATA_SOURCE = "prism_judge"
# Rough chars-per-token for Qwen3.5 on English prose. Every derived field is named
# ``*_tokens_est_*`` so nobody mistakes it for a tokenizer count.
CHARS_PER_TOKEN_ESTIMATE = 3.9
DEFAULT_PROMPT_BUDGET_TOKENS = 10240
_TARGET_LIST_KEYS = ("test_targets", "test_results")
_GENERATION_LIST_KEYS = ("generations", "outputs")
_RAW_TEXT_KEYS = ("raw_completion", "text", "response")

# parse_reasoning_and_response splits on the FIRST <reasoning> block; a small fraction of
# generations emit a stray trailing tag after the response. Same cleanup as build_judge_pairs.
_REASONING_BLOCK_RE = re.compile(r"<reasoning>.*?</reasoning>", re.DOTALL)
_REASONING_TAG_RE = re.compile(r"</?reasoning>")


def _key(user_id: Any, post_id: Any, target_idx: Any) -> tuple[str, str, str]:
    return (str(user_id), str(post_id), str(target_idx))


def _first_present(container: dict, keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in container and container[key] is not None:
            return container[key]
    return None


def _extract_raw_text(generation: Any) -> str:
    if isinstance(generation, str):
        return generation
    if isinstance(generation, dict):
        for key in _RAW_TEXT_KEYS:
            value = generation.get(key)
            if isinstance(value, str) and value:
                return value
    return ""


def _clean_generation(raw: str) -> str:
    text = parse_reasoning_and_response(raw)[1]
    text = _REASONING_BLOCK_RE.sub("", text)
    text = _REASONING_TAG_RE.sub("", text)
    return text.strip()


def flatten_all_generations(inference: Any) -> dict[tuple[str, str, str], list[str]]:
    """Flatten the inference pickle to ``{(user, post, target_idx): [gen, ...]}``."""
    if isinstance(inference, dict):
        user_results = list(inference.values())
    elif isinstance(inference, list):
        user_results = inference
    else:
        raise ValueError(f"Unexpected inference pickle type: {type(inference)!r}")

    flat: dict[tuple[str, str, str], list[str]] = {}
    for user_result in user_results:
        if not isinstance(user_result, dict):
            raise ValueError(f"Unexpected user_result type: {type(user_result)!r}")
        targets = _first_present(user_result, _TARGET_LIST_KEYS)
        if targets is None:
            continue
        user_id_fallback = user_result.get("user_id")
        for target in targets:
            generations = _first_present(target, _GENERATION_LIST_KEYS)
            if not generations:
                continue
            cleaned = [_clean_generation(_extract_raw_text(g)) for g in generations]
            key = _key(
                target.get("user_id", user_id_fallback),
                target.get("post_id"),
                target.get("target_idx"),
            )
            flat[key] = [text for text in cleaned if text]
    return flat


def _percentile(sorted_values: list[int], q: float) -> int:
    """Nearest-rank percentile over an already-sorted list."""
    if not sorted_values:
        return 0
    index = math.ceil(q * len(sorted_values)) - 1
    return sorted_values[min(len(sorted_values) - 1, max(0, index))]


def prompt_length_stats(prompts: list[str], *, budget_tokens: int) -> dict[str, Any]:
    """Length distribution of the rendered prompts, plus how many blow the budget.

    ``max_prompt_length`` cannot be chosen from the template alone: the empty rendered
    template is already ~5.3k estimated tokens before any user history, context or candidate
    responses are substituted in. These numbers are what that choice must be made from.
    """
    chars = sorted(len(p) for p in prompts)
    budget_chars = budget_tokens * CHARS_PER_TOKEN_ESTIMATE

    def est(value: int) -> float:
        return round(value / CHARS_PER_TOKEN_ESTIMATE, 1)

    stats: dict[str, Any] = {
        "prompt_budget_tokens": int(budget_tokens),
        "chars_per_token_estimate": CHARS_PER_TOKEN_ESTIMATE,
        "n_over_budget": sum(1 for c in chars if c > budget_chars),
    }
    for label, value in (
        ("p50", _percentile(chars, 0.50)),
        ("p95", _percentile(chars, 0.95)),
        ("max", chars[-1] if chars else 0),
    ):
        stats[f"prompt_chars_{label}"] = int(value)
        stats[f"prompt_tokens_est_{label}"] = est(value)
    stats["over_budget_rate"] = (stats["n_over_budget"] / len(chars)) if chars else 0.0
    return stats


_PROMPT_TEMPLATES = {
    "full": TURING_PROMPT,
    "single_token": TURING_SINGLE_TOKEN_PROMPT,
}


def render_turing_prompt(
    *,
    user_history: str,
    context: str,
    response_a: str,
    response_b: str,
    prompt_style: str = "full",
) -> str:
    """Render the judge prompt exactly as the reward path does at eval time.

    "Exactly" includes the control-character strip the reward path applies to the four
    content fields before formatting (reward.py::_score_pairwise_likert_with_info); without
    it a history holding a stray \\x0b would render a different prompt here than at eval.

    ``prompt_style`` selects the template: "full" (default) is the rubric-and-schema
    TURING_PROMPT every existing caller expects; "single_token" is TURING_SINGLE_TOKEN_PROMPT,
    which shares the same header and inputs but asks only for a bare A/B verdict.
    """
    try:
        template = _PROMPT_TEMPLATES[prompt_style]
    except KeyError:
        raise ValueError(
            f"prompt_style must be one of {sorted(_PROMPT_TEMPLATES)}, got {prompt_style!r}"
        ) from None
    user_history = sanitize_prompt_text(user_history)
    context = sanitize_prompt_text(context)
    response_a = sanitize_prompt_text(response_a)
    response_b = sanitize_prompt_text(response_b)
    warning_a = build_source_copy_warning(
        response_a, user_history=user_history, thread_context=context
    )
    warning_b = build_source_copy_warning(
        response_b, user_history=user_history, thread_context=context
    )
    return template.format(
        persona="",
        user_history=user_history,
        context=context,
        response_a=response_a,
        response_b=response_b,
        source_copy_watchlist=format_source_copy_watchlist(
            [warning_a, warning_b],
            item_label="Response",
            labels=["Response A", "Response B"],
        ),
    )


def build_judge_rows(
    source_df: pd.DataFrame,
    generations: dict[tuple[str, str, str], list[str]],
    *,
    lo: float,
    hi: float,
    limit: int | None,
    split: str,
    prompt_budget_tokens: int = DEFAULT_PROMPT_BUDGET_TOKENS,
    prompt_style: str = "full",
) -> tuple[pd.DataFrame, dict]:
    """Build veRL judge-training rows: two per (context, generation), one per order."""
    sliced = select_slice(source_df, lo=lo, hi=hi, limit=limit)

    rows: list[dict[str, Any]] = []
    prompts: list[str] = []
    missing: list[tuple[str, str, str]] = []
    n_generations = 0
    for record in sliced.to_dict("records"):
        extra = record.get("extra_info") or {}
        user_id = extra.get("user_id")
        post_id = extra.get("post_id")
        target_idx = extra.get("target_idx")
        key = _key(user_id, post_id, target_idx)
        if key not in generations or not generations[key]:
            missing.append(key)
            continue
        human = (record.get("reward_model") or {}).get("ground_truth", "")
        user_history = extra.get("user_history", "")
        context = extra.get("context", extra.get("thread_context", ""))

        for gen_idx, generated in enumerate(generations[key]):
            n_generations += 1
            pair_id = f"{user_id}::{post_id}::{target_idx}::g{gen_idx}"
            # human_a: the human occupies slot A. human_b: the human occupies slot B.
            for order, human_is_b in (("human_a", False), ("human_b", True)):
                response_a = generated if human_is_b else human
                response_b = human if human_is_b else generated
                rendered = render_turing_prompt(
                    user_history=user_history,
                    context=context,
                    response_a=response_a,
                    response_b=response_b,
                    prompt_style=prompt_style,
                )
                prompts.append(rendered)
                rows.append(
                    {
                        "data_source": DATA_SOURCE,
                        "prompt": [{"role": "user", "content": rendered}],
                        "reward_model": {
                            "style": "rule",
                            "ground_truth": "B" if human_is_b else "A",
                        },
                        "extra_info": {
                            "row_id": f"{pair_id}::{order}",
                            "pair_id": pair_id,
                            "user_id": user_id,
                            "post_id": post_id,
                            "target_idx": target_idx,
                            "gen_idx": gen_idx,
                            "order": order,
                            "human_is_b": human_is_b,
                            "split": split,
                        },
                    }
                )

    assert not missing, (
        f"{len(missing)} sliced rows have no generation; first few: {missing[:5]}"
    )

    df = pd.DataFrame(rows)
    human_is_b = [r["human_is_b"] for r in df["extra_info"]] if len(df) else []
    assert not human_is_b or sum(human_is_b) * 2 == len(human_is_b), (
        "human_is_b must be exactly balanced by construction"
    )

    meta = {
        "n_rows": int(len(df)),
        "n_contexts": int(len(sliced)),
        "n_generations": n_generations,
        "slice_lo": lo,
        "slice_hi": hi,
        "limit": limit,
        "split": split,
        "prompt_style": prompt_style,
        "human_is_b_rate": (sum(human_is_b) / len(human_is_b)) if human_is_b else 0.0,
    }
    meta.update(prompt_length_stats(prompts, budget_tokens=prompt_budget_tokens))
    return df, meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the judge-training pair parquet")
    parser.add_argument("--inference_pkl", required=True)
    parser.add_argument("--source_parquet", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--slice_lo", type=float, default=0.0)
    parser.add_argument("--slice_hi", type=float, default=0.1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--prompt_budget_tokens",
        type=int,
        default=DEFAULT_PROMPT_BUDGET_TOKENS,
        help="Token budget the emitted prompts are measured against; only affects the "
             "n_over_budget/over_budget_rate fields in the .meta.json. Set it to whatever "
             "data.max_prompt_length the training config currently declares.",
    )
    parser.add_argument(
        "--prompt-style", choices=["full", "single_token"], default="full",
        help="Judge prompt template. single_token drops the rubric and asks for one letter.",
    )
    args = parser.parse_args()

    with open(args.inference_pkl, "rb") as handle:
        inference = pickle.load(handle)
    source_df = pd.read_parquet(args.source_parquet)

    df, meta = build_judge_rows(
        source_df,
        flatten_all_generations(inference),
        lo=args.slice_lo,
        hi=args.slice_hi,
        limit=args.limit,
        split=args.split,
        prompt_budget_tokens=args.prompt_budget_tokens,
        prompt_style=args.prompt_style,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    df.to_parquet(args.out, index=False)
    meta["inference_pkl"] = os.path.abspath(args.inference_pkl)
    meta["source_parquet"] = os.path.abspath(args.source_parquet)
    with open(os.path.splitext(args.out)[0] + ".meta.json", "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"Wrote {len(df)} rows -> {args.out}")
    print(json.dumps(meta, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
