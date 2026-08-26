"""Build the (human, generated) judge pair-set from heldout inference + test parquet.

Flattens the pickle emitted by ``eval/generate_trained.py`` into one row per
heldout target, aligned to ``test.parquet`` by ``(user_id, post_id, target_idx)``.

Real pickle structure (from ``generate_trained.py``, see ``main``/
``generate_for_user_results_vllm``):

    results = {user_id: user_result, ...}                # dict keyed by user_id
    user_result["test_targets"] = [target_result, ...]   # per-target list
    target_result["generations"] = [gen, ...]            # parsed generations
    gen = {"reasoning": ..., "response": ..., "raw_completion": <raw text>, ...}

We parse the *raw* completion with ``parse_reasoning_and_response`` and take the
response (index 1) so the generated text is derived identically to a fresh parse.
Fallback keys (``test_results`` / ``outputs``, and str-or-dict generation entries
with ``raw_completion`` / ``text`` / ``response``) are handled defensively.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import sys
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from shared.prompt_utils import parse_reasoning_and_response

COLUMNS = [
    "pair_id",
    "user_id",
    "post_id",
    "target_idx",
    "user_history",
    "context",
    "persona",
    "human",
    "generated",
]

# Preference order for keys that hold the per-user list of targets / the
# per-target list of generations / the raw generation text.
_TARGET_LIST_KEYS = ("test_targets", "test_results")
_GENERATION_LIST_KEYS = ("generations", "outputs")
_RAW_TEXT_KEYS = ("raw_completion", "text", "response")


def _key(user_id: Any, post_id: Any, target_idx: Any) -> tuple[str, str, str]:
    """Normalized alignment key so int/str mismatches don't break the join."""
    return (str(user_id), str(post_id), str(target_idx))


def _extract_raw_text(generation: Any) -> str:
    """Pull the raw completion text from a generation entry (dict or str)."""
    if isinstance(generation, str):
        return generation
    if isinstance(generation, dict):
        for key in _RAW_TEXT_KEYS:
            value = generation.get(key)
            if isinstance(value, str) and value:
                return value
        return ""
    return ""


# parse_reasoning_and_response() splits on the FIRST <reasoning>...</reasoning> block.
# A small fraction of generations emit a stray trailing </reasoning> (or a second block)
# AFTER the response, which the primary parse leaves in the text. These are malformed
# model artifacts, not part of the user turn, and must not leak into the judged pair.
_REASONING_BLOCK_RE = re.compile(r"<reasoning>.*?</reasoning>", re.DOTALL)
_REASONING_TAG_RE = re.compile(r"</?reasoning>")


def _strip_reasoning_residue(text: str) -> str:
    """Remove any residual reasoning blocks/stray tags left after the primary parse."""
    text = _REASONING_BLOCK_RE.sub("", text)  # drop any complete stray block first
    text = _REASONING_TAG_RE.sub("", text)    # then any lone/unmatched tag
    return text


def _first_present(container: dict, keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in container and container[key] is not None:
            return container[key]
    return None


def flatten_inference(inference: Any) -> dict[tuple[str, str, str], str]:
    """Flatten the inference pickle to {(user, post, target_idx): raw_text}."""
    # The pickle is a dict keyed by user_id; tolerate a bare list of user_results.
    if isinstance(inference, dict):
        user_results = list(inference.values())
    elif isinstance(inference, list):
        user_results = inference
    else:
        raise ValueError(f"Unexpected inference pickle type: {type(inference)!r}")

    flat: dict[tuple[str, str, str], str] = {}
    for user_result in user_results:
        if not isinstance(user_result, dict):
            raise ValueError(f"Unexpected user_result type: {type(user_result)!r}")
        targets = _first_present(user_result, _TARGET_LIST_KEYS)
        if targets is None:
            continue
        user_id_fallback = user_result.get("user_id")
        for target in targets:
            user_id = target.get("user_id", user_id_fallback)
            post_id = target.get("post_id")
            target_idx = target.get("target_idx")
            generations = _first_present(target, _GENERATION_LIST_KEYS)
            if not generations:
                # No generation for this target; skip so the parquet-side
                # completeness assertion surfaces the gap explicitly.
                continue
            raw = _extract_raw_text(generations[0])
            flat[_key(user_id, post_id, target_idx)] = raw
    return flat


def build_pairs(
    inference_pkl_path: str,
    test_parquet_path: str,
) -> tuple[pd.DataFrame, dict]:
    """Build the aligned (human, generated) pair-set DataFrame + metadata."""
    with open(inference_pkl_path, "rb") as handle:
        inference = pickle.load(handle)
    flat = flatten_inference(inference)

    test_df = pd.read_parquet(test_parquet_path)

    rows: list[dict[str, Any]] = []
    missing: list[tuple[str, str, str]] = []
    residue_stripped = 0
    for record in test_df.to_dict("records"):
        extra_info = record.get("extra_info") or {}
        reward_model = record.get("reward_model") or {}
        user_id = extra_info.get("user_id")
        post_id = extra_info.get("post_id")
        target_idx = extra_info.get("target_idx")
        key = _key(user_id, post_id, target_idx)
        if key not in flat:
            missing.append(key)
            continue
        raw = flat[key]
        generated_raw = parse_reasoning_and_response(raw)[1]
        if _REASONING_TAG_RE.search(generated_raw):
            residue_stripped += 1
        generated = _strip_reasoning_residue(generated_raw).strip()
        human = reward_model.get("ground_truth", "")
        rows.append(
            {
                "pair_id": f"{user_id}::{post_id}::{target_idx}",
                "user_id": user_id,
                "post_id": post_id,
                "target_idx": target_idx,
                "user_history": extra_info.get("user_history", ""),
                "context": extra_info.get("context", extra_info.get("thread_context", "")),
                "persona": extra_info.get("persona", ""),
                "human": human,
                "generated": generated,
            }
        )

    assert not missing, (
        f"{len(missing)} heldout rows have no matching inference generation; "
        f"first few: {missing[:5]}"
    )

    df = pd.DataFrame(rows, columns=COLUMNS)

    # Post-strip sanity: no residual reasoning tags should survive (see _strip_reasoning_residue).
    residual = df["generated"].str.contains("<reasoning>|</reasoning>", regex=True, na=False)
    assert not residual.any(), (
        f"{int(residual.sum())} generated rows still contain <reasoning> tags after stripping"
    )
    if residue_stripped:
        print(
            f"NOTE: stripped residual reasoning tags/blocks from {residue_stripped}/{len(rows)} "
            "generation(s) (malformed model artifacts, e.g. a stray trailing </reasoning>).",
            flush=True,
        )

    n_pairs = len(df)
    exact_match_count = int((df["human"] == df["generated"]).sum())
    exact_match_frac = (exact_match_count / n_pairs) if n_pairs else 0.0
    if exact_match_frac > 0.01:
        print(
            f"WARN: exact_match_frac={exact_match_frac:.4f} "
            f"({exact_match_count}/{n_pairs}) exceeds 0.01 -- generator may be "
            "copying the ground-truth human turn.",
            flush=True,
        )

    meta = {
        "n_pairs": n_pairs,
        "exact_match_count": exact_match_count,
        "exact_match_frac": exact_match_frac,
        "reasoning_residue_stripped": residue_stripped,
        "inference_pkl": os.path.abspath(inference_pkl_path),
        "test_parquet": os.path.abspath(test_parquet_path),
    }
    return df, meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the judge (human, generated) pair-set")
    parser.add_argument("--inference_pkl", required=True, help="Heldout inference pickle")
    parser.add_argument("--test_parquet", required=True, help="GRPO-format heldout test parquet")
    parser.add_argument("--out", required=True, help="Output parquet path for the pair-set")
    args = parser.parse_args()

    df, meta = build_pairs(args.inference_pkl, args.test_parquet)

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    df.to_parquet(args.out, index=False)

    meta_path = os.path.splitext(args.out)[0] + ".meta.json"
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)

    print(f"Wrote {len(df)} pairs -> {args.out}")
    print(f"Metadata -> {meta_path}")
    print(
        f"exact_match_count={meta['exact_match_count']} "
        f"exact_match_frac={meta['exact_match_frac']:.4f}"
    )


if __name__ == "__main__":
    main()
