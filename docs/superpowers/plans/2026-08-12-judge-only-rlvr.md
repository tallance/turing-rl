# Judge-Only RLVR Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train Qwen3.5 2B / 4B / 9B discriminators with GRPO to identify which of two candidate turns a real human wrote, against a frozen pair set generated from the 9B SFT checkpoint.

**Architecture:** A deterministic hash carves 10% of the GRPO train split; the frozen `merged_ep3` generator produces k=4 fake turns per context; each (human, fake) pair is rendered into the existing 37-field `TURING_PROMPT` in both A/B orders and written as a veRL parquet whose `reward_model.ground_truth` is the label `"A"` or `"B"`. A new local reward function parses the rollout's verdict, recovers a 1–7 rating through a four-rung fallback ladder, and scores it under one of two arms. No judge server is involved — the label is known by construction.

**Tech Stack:** Python 3.12, pandas/pyarrow, veRL 0.9 (env `turing-rl-rl-qwen35`), vLLM, Hydra configs, pytest, Slurm.

**Spec:** `docs/superpowers/specs/2026-08-12-judge-only-rlvr-design.md`

## Global Constraints

- **Test command:** `/Users/lancewicki/miniforge3/bin/python -m pytest` from the repo root. All tasks in this plan are Mac-local and CPU-only; nothing here needs a GPU.
- **VCS is git, not Sapling.** Commit with `git commit`, never `sl commit`.
- **Never invoke `sbatch` directly.** `scripts/snapshot_sbatch.sh` is the only maintained gateway, reached through `scripts/cluster_launch.sh`. Run the `preflight-job-check` skill before any submission.
- **Commit before cluster execution.** Dirty-tree cluster runs are prohibited; every run maps to a clean commit.
- **Judge sampling is pinned** for every judge invocation: `PERSONA_JUDGE_SAMPLING='{"repetition_penalty":1.1,"temperature":0.6}'`, `PERSONA_JUDGE_MAX_COMPLETION_TOKENS=8192`, `PERSONA_OPENAI_TIMEOUT_SECONDS=1800`, `PERSONA_JUDGE_ENABLE_THINKING=1`, reasoning parser `qwen3`.
- **Generator sampling is pinned:** `T=0.7`, `top_p=0.8`, `top_k=20`, `max_tokens=1024`. Do not use `eval/generate_trained.py`'s older prism default of `T=0.6 / top_p=1.0 / top_k=-1`.
- **Eval set** is the frozen 880-pair heldout set: `data/prism/full_s42_history_sft40_grpo60_test10/test.parquet`.
- **Slice bounds for judge iteration 1** are `[0.0, 0.1)`, capped at **416 contexts**.
- **Do not LoRA the Gated-DeltaNet backbone** on any Qwen3.5 model. Target modules are exactly `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`, with `exclude_modules='.*(visual|mtp).*'`.
- **The 37 required verdict fields come from `shared/judge_prompts.TURING_RESPONSE_PROPERTIES`.** Never hard-code the field list or the number 37.

### Deviation from the spec, decided at plan time

Spec §6 lists `order_consistency` as a training metric. It cannot be computed there: a pair's two A/B orders are rendered as two separate prompts, so they receive different GRPO group ids, and re-pairing them inside a batch would be fragile and order-dependent. **`order_consistency` moves to the offline eval analysis (Task 10).** In-training position bias is tracked by `judge_pred_b`, whose batch mean sits at 0.5 for an unbiased judge. Group-health metrics are unaffected and are computed in-training (Task 5).

---

## File Structure

**New files**

| Path | Responsibility |
|---|---|
| `data/judge/__init__.py` | package marker |
| `data/judge/slice.py` | deterministic hash slicing — the only place the slice rule lives |
| `scripts/build_judge_train_pairs.py` | inference pickle + source parquet → veRL judge-training parquet |
| `training/grpo/judge_verdict.py` | parse one rollout completion → rating + four format components |
| `training/grpo/judge_reward.py` | veRL `compute_score` for judge training; both reward arms |
| `training/grpo/configs/qwen35_judge_grpo.yaml` | veRL config for judge GRPO |
| `scripts/probe_judge_format.py` | three-regime zero-shot format probe |
| `scripts/analyze_judge_training.py` | eval-side accuracy / order-consistency analysis |
| `scripts/slurm/judge_format_probe.sh` | Slurm wrapper for the probe |
| `scripts/slurm/judge_train_gen.sh` | Slurm wrapper for pair generation |
| `scripts/slurm/judge_grpo_train.sh` | Slurm wrapper for judge GRPO |
| `tests/test_judge_slice.py`, `tests/test_build_judge_train_pairs.py`, `tests/test_judge_verdict.py`, `tests/test_judge_reward.py`, `tests/test_judge_metric_patch.py`, `tests/test_judge_grpo_config.py`, `tests/test_probe_judge_format.py`, `tests/test_analyze_judge_training.py` | unit tests, one per module |

**Modified files**

| Path | Change |
|---|---|
| `training/grpo/verl_metric_patch.py` | discover `judge_`-prefixed reward keys; add group-health metrics |
| `training/grpo/reward.py` | add a third `response_format` mode so the probe can send none |

Judge code is deliberately separate from `training/grpo/reward.py`. The generator reward carries a judge HTTP client, length penalties, source-copy bookkeeping and a 1105-line surface; the judge reward shares none of it. Two focused modules beat one branching one.

---

## Task 1: Deterministic hash slice

**Files:**
- Create: `data/judge/__init__.py`
- Create: `data/judge/slice.py`
- Test: `tests/test_judge_slice.py`

**Interfaces:**
- Consumes: nothing (leaf module)
- Produces: `slice_key(user_id, post_id, target_idx) -> str`; `slice_fraction(user_id, post_id, target_idx) -> float`; `in_slice(user_id, post_id, target_idx, *, lo: float, hi: float) -> bool`; `select_slice(df: pd.DataFrame, *, lo: float, hi: float, limit: int | None = None) -> pd.DataFrame`

- [ ] **Step 1: Write the failing test**

Create `tests/test_judge_slice.py`:

```python
"""Unit tests for deterministic judge-data slicing."""

import pandas as pd
import pytest

from data.judge.slice import in_slice, select_slice, slice_fraction, slice_key


def _rows(n: int) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "data_source": "prism",
                "prompt": [{"role": "user", "content": f"p{i}"}],
                "reward_model": {"ground_truth": f"gt{i}"},
                "extra_info": {"user_id": f"u{i % 7}", "post_id": f"p{i}", "target_idx": i % 3},
            }
            for i in range(n)
        ]
    )


def test_slice_key_is_the_documented_triple():
    assert slice_key("u1", "p2", 3) == "u1::p2::3"


def test_slice_fraction_is_deterministic():
    assert slice_fraction("u1", "p2", 3) == slice_fraction("u1", "p2", 3)


def test_slice_fraction_is_in_unit_interval():
    for i in range(500):
        u = slice_fraction(f"u{i}", f"p{i}", i)
        assert 0.0 <= u < 1.0


def test_adjacent_slices_are_disjoint():
    for i in range(500):
        args = (f"u{i}", f"p{i}", i)
        first = in_slice(*args, lo=0.0, hi=0.1)
        second = in_slice(*args, lo=0.1, hi=0.2)
        assert not (first and second)


def test_slice_covers_roughly_the_requested_fraction():
    hits = sum(1 for i in range(4174) if in_slice(f"u{i}", f"p{i}", i, lo=0.0, hi=0.1))
    assert 340 <= hits <= 500


def test_select_slice_is_independent_of_row_order():
    df = _rows(300)
    forward = select_slice(df, lo=0.0, hi=0.25)
    shuffled = select_slice(df.iloc[::-1].reset_index(drop=True), lo=0.0, hi=0.25)
    assert [r["post_id"] for r in forward["extra_info"]] == [
        r["post_id"] for r in shuffled["extra_info"]
    ]


def test_limit_truncates_deterministically():
    df = _rows(300)
    full = select_slice(df, lo=0.0, hi=1.0)
    capped = select_slice(df, lo=0.0, hi=1.0, limit=10)
    assert len(capped) == 10
    assert list(capped["extra_info"]) == list(full["extra_info"])[:10]


def test_bad_bounds_raise():
    with pytest.raises(ValueError):
        in_slice("u", "p", 0, lo=0.5, hi=0.5)
    with pytest.raises(ValueError):
        in_slice("u", "p", 0, lo=-0.1, hi=0.5)


def test_non_dict_extra_info_raises():
    df = pd.DataFrame([{"extra_info": "nope"}])
    with pytest.raises(TypeError):
        select_slice(df, lo=0.0, hi=1.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/lancewicki/miniforge3/bin/python -m pytest tests/test_judge_slice.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'data.judge'`

- [ ] **Step 3: Write minimal implementation**

Create `data/judge/__init__.py` as an empty file.

Create `data/judge/slice.py`:

```python
"""Deterministic hash slicing for judge-training data.

Membership is a pure function of ``(user_id, post_id, target_idx)``: no seed, no
ordering dependence, and no dependence on how many rows precede a given row. Future
alternating iterations can therefore draw disjoint slices from the same pool without
storing or recomputing anything.

The triple is hashed rather than ``user_id`` alone because users hold varying row
counts, so a 10% user slice would be lumpy. The corpus is already partitioned by user
one level up, in ``data/prism/split_data.py``.
"""

from __future__ import annotations

import hashlib
from typing import Any

import pandas as pd


def slice_key(user_id: Any, post_id: Any, target_idx: Any) -> str:
    """Canonical identity string for one target turn."""
    return f"{user_id}::{post_id}::{target_idx}"


def slice_fraction(user_id: Any, post_id: Any, target_idx: Any) -> float:
    """Map a row identity to a stable ``u`` in ``[0.0, 1.0)``."""
    digest = hashlib.blake2b(
        slice_key(user_id, post_id, target_idx).encode("utf-8"), digest_size=8
    ).digest()
    return int.from_bytes(digest, "big") / 2**64


def _validate_bounds(lo: float, hi: float) -> None:
    if not 0.0 <= lo < hi <= 1.0:
        raise ValueError(f"slice bounds must satisfy 0 <= lo < hi <= 1, got lo={lo} hi={hi}")


def in_slice(user_id: Any, post_id: Any, target_idx: Any, *, lo: float, hi: float) -> bool:
    """True when this row's hash falls in the half-open interval ``[lo, hi)``."""
    _validate_bounds(lo, hi)
    return lo <= slice_fraction(user_id, post_id, target_idx) < hi


def select_slice(
    df: pd.DataFrame, *, lo: float, hi: float, limit: int | None = None
) -> pd.DataFrame:
    """Return rows whose ``extra_info`` identity hashes into ``[lo, hi)``.

    Rows come back in ascending hash order, so ``limit`` truncates deterministically
    instead of depending on the input row order.
    """
    _validate_bounds(lo, hi)
    if df.empty:
        return df.copy()

    fractions: list[float] = []
    for extra in df["extra_info"]:
        if not isinstance(extra, dict):
            raise TypeError(f"extra_info must be a dict, got {type(extra)!r}")
        fractions.append(
            slice_fraction(extra.get("user_id"), extra.get("post_id"), extra.get("target_idx"))
        )

    out = df.assign(_slice_u=fractions)
    out = out.loc[(out["_slice_u"] >= lo) & (out["_slice_u"] < hi)]
    out = out.sort_values("_slice_u", kind="stable")
    if limit is not None:
        out = out.head(limit)
    return out.drop(columns=["_slice_u"]).reset_index(drop=True)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/Users/lancewicki/miniforge3/bin/python -m pytest tests/test_judge_slice.py -q`
Expected: PASS, 9 tests

- [ ] **Step 5: Commit**

```bash
git add data/judge/__init__.py data/judge/slice.py tests/test_judge_slice.py
git commit -m "feat: deterministic hash slicing for judge-training data"
```

---

## Task 2: Judge pair builder

Turns the k-generation inference pickle plus the source parquet into a veRL training parquet. Two rows per (context, generation): one with the human in slot A, one in slot B.

**Files:**
- Create: `scripts/build_judge_train_pairs.py`
- Test: `tests/test_build_judge_train_pairs.py`

**Interfaces:**
- Consumes: `data.judge.slice.select_slice`; `shared.judge_prompts.TURING_PROMPT`; `shared.judge_utils.build_source_copy_warning`, `shared.judge_utils.format_source_copy_watchlist`; `shared.prompt_utils.parse_reasoning_and_response`
- Produces: `flatten_all_generations(inference: Any) -> dict[tuple[str, str, str], list[str]]`; `render_turing_prompt(*, user_history: str, context: str, response_a: str, response_b: str) -> str`; `build_judge_rows(source_df: pd.DataFrame, generations: dict, *, lo: float, hi: float, limit: int | None, split: str) -> tuple[pd.DataFrame, dict]`

- [ ] **Step 1: Write the failing test**

Create `tests/test_build_judge_train_pairs.py`:

```python
"""Unit tests for the judge-training pair builder."""

import pandas as pd
import pytest

from scripts.build_judge_train_pairs import (
    build_judge_rows,
    flatten_all_generations,
    render_turing_prompt,
)


def _inference(n_gens: int = 2):
    return {
        "u1": {
            "user_id": "u1",
            "test_targets": [
                {
                    "user_id": "u1",
                    "post_id": "p1",
                    "target_idx": 0,
                    "generations": [
                        {"raw_completion": f"<reasoning>r</reasoning>[HUMAN]: fake {i}"}
                        for i in range(n_gens)
                    ],
                }
            ],
        }
    }


def _source_df():
    return pd.DataFrame(
        [
            {
                "data_source": "prism",
                "prompt": [{"role": "user", "content": "ignored"}],
                "reward_model": {"ground_truth": "real human turn"},
                "extra_info": {
                    "user_id": "u1",
                    "post_id": "p1",
                    "target_idx": 0,
                    "user_history": "hist",
                    "context": "ctx",
                },
            }
        ]
    )


def test_flatten_keeps_every_generation():
    flat = flatten_all_generations(_inference(n_gens=3))
    assert flat[("u1", "p1", "0")] == ["fake 0", "fake 1", "fake 2"]


def test_render_places_each_response_in_its_slot():
    prompt = render_turing_prompt(
        user_history="hist", context="ctx", response_a="AAA", response_b="BBB"
    )
    assert prompt.index("AAA") < prompt.index("BBB")
    assert "<|Response A|>" in prompt and "<|Response B|>" in prompt


def test_two_rows_per_generation_one_per_order():
    df, meta = build_judge_rows(
        _source_df(), flatten_all_generations(_inference(n_gens=3)),
        lo=0.0, hi=1.0, limit=None, split="train",
    )
    assert len(df) == 6
    assert meta["n_contexts"] == 1
    assert meta["n_generations"] == 3


def test_human_side_is_exactly_balanced():
    df, _ = build_judge_rows(
        _source_df(), flatten_all_generations(_inference(n_gens=4)),
        lo=0.0, hi=1.0, limit=None, split="train",
    )
    human_is_b = [r["human_is_b"] for r in df["extra_info"]]
    assert sum(human_is_b) * 2 == len(human_is_b)


def test_ground_truth_names_the_slot_holding_the_human():
    df, _ = build_judge_rows(
        _source_df(), flatten_all_generations(_inference(n_gens=1)),
        lo=0.0, hi=1.0, limit=None, split="train",
    )
    for _, row in df.iterrows():
        human_is_b = row["extra_info"]["human_is_b"]
        assert row["reward_model"]["ground_truth"] == ("B" if human_is_b else "A")
        text = row["prompt"][0]["content"]
        a_start = text.index("<|Response A|>")
        b_start = text.index("<|Response B|>")
        human_at = text.index("real human turn")
        assert (human_at > b_start) == human_is_b
        assert (b_start > human_at > a_start) != human_is_b


def test_row_ids_are_unique():
    df, _ = build_judge_rows(
        _source_df(), flatten_all_generations(_inference(n_gens=4)),
        lo=0.0, hi=1.0, limit=None, split="train",
    )
    row_ids = [r["row_id"] for r in df["extra_info"]]
    assert len(set(row_ids)) == len(row_ids)


def test_split_tag_is_propagated():
    df, _ = build_judge_rows(
        _source_df(), flatten_all_generations(_inference(n_gens=1)),
        lo=0.0, hi=1.0, limit=None, split="val",
    )
    assert all(r["split"] == "val" for r in df["extra_info"])


def test_missing_generation_raises():
    with pytest.raises(AssertionError):
        build_judge_rows(_source_df(), {}, lo=0.0, hi=1.0, limit=None, split="train")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/lancewicki/miniforge3/bin/python -m pytest tests/test_build_judge_train_pairs.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.build_judge_train_pairs'`

- [ ] **Step 3: Write minimal implementation**

Create `scripts/build_judge_train_pairs.py`:

```python
"""Build the judge-training parquet from k-sample inference + a source parquet.

Differs from ``scripts/build_judge_pairs.py`` in three ways: it keeps *every*
generation rather than only the first, it emits each pair in *both* A/B orders, and
it writes veRL training rows (rendered prompt + ``"A"``/``"B"`` label) rather than a
flat pair table for offline judging.
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

from data.judge.slice import select_slice
from shared.judge_prompts import TURING_PROMPT
from shared.judge_utils import build_source_copy_warning, format_source_copy_watchlist
from shared.prompt_utils import parse_reasoning_and_response

DATA_SOURCE = "prism_judge"
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


def render_turing_prompt(
    *, user_history: str, context: str, response_a: str, response_b: str
) -> str:
    """Render TURING_PROMPT exactly as the reward path does at eval time."""
    warning_a = build_source_copy_warning(
        response_a, user_history=user_history, thread_context=context
    )
    warning_b = build_source_copy_warning(
        response_b, user_history=user_history, thread_context=context
    )
    return TURING_PROMPT.format(
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
) -> tuple[pd.DataFrame, dict]:
    """Build veRL judge-training rows: two per (context, generation), one per order."""
    sliced = select_slice(source_df, lo=lo, hi=hi, limit=limit)

    rows: list[dict[str, Any]] = []
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
                rows.append(
                    {
                        "data_source": DATA_SOURCE,
                        "prompt": [
                            {
                                "role": "user",
                                "content": render_turing_prompt(
                                    user_history=user_history,
                                    context=context,
                                    response_a=response_a,
                                    response_b=response_b,
                                ),
                            }
                        ],
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
        "human_is_b_rate": (sum(human_is_b) / len(human_is_b)) if human_is_b else 0.0,
    }
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/Users/lancewicki/miniforge3/bin/python -m pytest tests/test_build_judge_train_pairs.py -q`
Expected: PASS, 8 tests

- [ ] **Step 5: Commit**

```bash
git add scripts/build_judge_train_pairs.py tests/test_build_judge_train_pairs.py
git commit -m "feat: judge-training pair builder with k generations and both A/B orders"
```

---

## Task 3: Verdict parsing and the rating-recovery ladder

Turns one rollout completion into a rating plus four independent format components. This is the module that makes "malformed but parseable still earns task credit" true.

**Files:**
- Create: `training/grpo/judge_verdict.py`
- Test: `tests/test_judge_verdict.py`

**Interfaces:**
- Consumes: `shared.judge_prompts.TURING_RESPONSE_PROPERTIES`; `shared.judge_utils._coerce_turing_rating`, `_extract_turing_rating`, `_rating_from_turing_score_gap`
- Produces: `TURING_FIELDS: tuple[str, ...]`; `extract_json_object(text: str | None) -> dict | None`; `JudgeVerdict` dataclass with fields `rating: int | None`, `recovery_rung: str`, `fmt_json_valid: bool`, `fmt_all_fields: bool`, `fmt_arith: bool`, `fmt_rating_range: bool` and properties `recovered: bool`, `format_score: float`; `derive_rating(data: dict) -> tuple[int, float]`; `parse_judge_verdict(completion: str | None) -> JudgeVerdict`

**Do not use `shared.judge_utils._extract_json` here.** It calls `json.loads` bare, so it
*raises* `JSONDecodeError` on non-JSON input instead of returning `None`, and it requires the
entire string to be one JSON object. Unconstrained rollouts routinely wrap the verdict in prose
or a ```json fence. `reward.py::_extract_json` is the tolerant variant, but importing `reward.py`
drags in aiohttp, veRL and the judge HTTP client for what is a pure-text helper — so
`judge_verdict.py` carries its own copy and stays importable in a bare test environment.

Recovery rungs, in order: `"dimensions"` → `"score_gap"` → `"rating_field"` → `"rating_text"` → `"none"`. The spec names four rungs; `rating_field` splits into a JSON variant and a raw-text-regex variant, which are worth distinguishing in the metrics.

- [ ] **Step 1: Write the failing test**

Create `tests/test_judge_verdict.py`:

````python
"""Unit tests for judge verdict parsing and the rating-recovery ladder."""

import json

from shared.judge_prompts import TURING_RESPONSE_PROPERTIES
from training.grpo.judge_verdict import (
    TURING_FIELDS,
    derive_rating,
    extract_json_object,
    parse_judge_verdict,
)


def _verdict(**overrides) -> dict:
    """A well-formed verdict where B scores 3.0 and A scores 0.0 -> gap 3.0 -> rating 7."""
    data = {}
    for name, schema in TURING_RESPONSE_PROPERTIES.items():
        if schema["type"] == "string":
            data[name] = "text"
        elif schema["type"] == "integer":
            data[name] = 4
        else:
            data[name] = 0.0
    data["immediate_target_score_b"] = 1.0
    data["human_goal_score_b"] = 1.0
    data["communication_style_score_b"] = 1.0
    data["base_score_a"] = 0.0
    data["base_score_b"] = 3.0
    data["penalty_a"] = 0.0
    data["penalty_b"] = 0.0
    data["response_a_score"] = 0.0
    data["response_b_score"] = 3.0
    data["score_gap"] = 3.0
    data["rating"] = 7
    data.update(overrides)
    return data


def test_turing_fields_come_from_the_schema():
    assert TURING_FIELDS == tuple(TURING_RESPONSE_PROPERTIES)


def test_extract_json_object_handles_a_fenced_block():
    assert extract_json_object('```json\n{"rating": 5}\n```') == {"rating": 5}


def test_extract_json_object_handles_prose_around_the_object():
    assert extract_json_object('here you go: {"rating": 5} hope that helps') == {"rating": 5}


def test_extract_json_object_returns_none_instead_of_raising():
    assert extract_json_object("no json here") is None
    assert extract_json_object(None) is None
    assert extract_json_object("[1, 2, 3]") is None


def test_derive_rating_recomputes_gap_and_rating():
    rating, gap = derive_rating(_verdict())
    assert rating == 7
    assert gap == 3.0


def test_derive_rating_applies_the_penalty_formula():
    # All four B penalties at 1.0 -> penalty_b = (4/4)*3 = 3.0 -> b_score = max(0, 3-3) = 0.
    data = _verdict(
        source_copy_penalty_b=1.0,
        wrong_target_or_role_penalty_b=1.0,
        unsupported_adversarial_reframing_penalty_b=1.0,
        assistant_like_penalty_b=1.0,
    )
    rating, gap = derive_rating(data)
    assert gap == 0.0
    assert rating == 4


def test_perfect_verdict_scores_every_format_component():
    v = parse_judge_verdict(json.dumps(_verdict()))
    assert v.rating == 7
    assert v.recovery_rung == "dimensions"
    assert v.fmt_json_valid and v.fmt_all_fields and v.fmt_arith and v.fmt_rating_range
    assert v.format_score == 1.0


def test_missing_field_loses_all_fields_but_keeps_the_rating():
    data = _verdict()
    del data["reasoning"]
    v = parse_judge_verdict(json.dumps(data))
    assert v.rating == 7
    assert v.recovery_rung == "dimensions"
    assert v.fmt_json_valid and not v.fmt_all_fields
    assert v.recovered


def test_extra_field_loses_the_all_fields_component():
    v = parse_judge_verdict(json.dumps(_verdict(surprise="nope")))
    assert not v.fmt_all_fields
    assert v.rating == 7


def test_bad_arithmetic_loses_only_the_arith_component():
    v = parse_judge_verdict(json.dumps(_verdict(score_gap=-3.0, rating=1)))
    assert v.rating == 7  # derived from the dimensions, not the model's own claim
    assert not v.fmt_arith
    assert v.fmt_json_valid and v.fmt_all_fields


def test_score_gap_rung_when_dimensions_are_absent():
    v = parse_judge_verdict(json.dumps({"score_gap": 1.5, "reasoning": "x"}))
    assert v.recovery_rung == "score_gap"
    assert v.rating == 6
    assert not v.fmt_arith


def test_rating_field_rung_when_only_the_rating_survives():
    v = parse_judge_verdict(json.dumps({"rating": 2, "reasoning": "x"}))
    assert v.recovery_rung == "rating_field"
    assert v.rating == 2
    assert v.fmt_rating_range


def test_rating_text_rung_when_json_is_unparseable():
    v = parse_judge_verdict('the verdict is "rating": 6 and that is final')
    assert v.recovery_rung == "rating_text"
    assert v.rating == 6
    assert not v.fmt_json_valid


def test_unrecoverable_completion():
    v = parse_judge_verdict("total gibberish with no verdict")
    assert v.rating is None
    assert v.recovery_rung == "none"
    assert not v.recovered
    assert v.format_score == 0.0


def test_none_completion_is_unrecoverable():
    assert parse_judge_verdict(None).recovery_rung == "none"


def test_out_of_range_rating_field_fails_the_range_component():
    v = parse_judge_verdict(json.dumps({"rating": 99, "reasoning": "x"}))
    assert not v.fmt_rating_range
    assert v.recovery_rung == "none"
````

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/lancewicki/miniforge3/bin/python -m pytest tests/test_judge_verdict.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'training.grpo.judge_verdict'`

- [ ] **Step 3: Write minimal implementation**

Create `training/grpo/judge_verdict.py`:

````python
"""Parse one judge rollout into a rating plus independent format components.

veRL cannot constrain rollout decoding (no schema field exists in 0.7.0-0.9.0.dev), so
training rollouts are free-form and frequently imperfect. A rating is therefore
recovered through a fallback ladder, and format quality is scored *separately* rather
than gating the task reward: a malformed-but-parseable verdict still earns task credit,
because it still contains a prediction worth scoring.
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from shared.judge_prompts import TURING_RESPONSE_PROPERTIES
from shared.judge_utils import (
    _coerce_turing_rating,
    _extract_turing_rating,
    _rating_from_turing_score_gap,
)

TURING_FIELDS: tuple[str, ...] = tuple(TURING_RESPONSE_PROPERTIES)

_JSON_FENCE_RE = re.compile(r"```json\s*(.*?)\s*```", re.DOTALL)


def extract_json_object(text: str | None) -> dict | None:
    """Pull a JSON object out of free-form completion text.

    Mirrors ``reward.py::_extract_json``: tolerates a ```json fence or prose around the
    object, and returns None rather than raising when nothing parses. Duplicated rather
    than imported because ``reward.py`` pulls in aiohttp and veRL at import time, and
    this module must stay importable anywhere.
    """
    if not isinstance(text, str):
        return None
    text = text.strip()
    if not text:
        return None
    match = _JSON_FENCE_RE.search(text)
    if match:
        text = match.group(1)
    else:
        brace_start = text.find("{")
        brace_end = text.rfind("}")
        if brace_start != -1 and brace_end != -1:
            text = text[brace_start : brace_end + 1]
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None

_DIMENSION_FIELDS_A = (
    "immediate_target_score_a",
    "human_goal_score_a",
    "communication_style_score_a",
)
_DIMENSION_FIELDS_B = (
    "immediate_target_score_b",
    "human_goal_score_b",
    "communication_style_score_b",
)
_PENALTY_FIELDS_A = (
    "source_copy_penalty_a",
    "wrong_target_or_role_penalty_a",
    "unsupported_adversarial_reframing_penalty_a",
    "assistant_like_penalty_a",
)
_PENALTY_FIELDS_B = (
    "source_copy_penalty_b",
    "wrong_target_or_role_penalty_b",
    "unsupported_adversarial_reframing_penalty_b",
    "assistant_like_penalty_b",
)
_PRIMITIVE_FIELDS = (
    _DIMENSION_FIELDS_A + _DIMENSION_FIELDS_B + _PENALTY_FIELDS_A + _PENALTY_FIELDS_B
)

# The prompt asks for one-decimal scores; 0.05 tolerates rounding without excusing
# a model that simply asserts numbers unrelated to its own dimension scores.
ARITH_TOLERANCE = 0.05


@dataclass(frozen=True)
class JudgeVerdict:
    """One parsed rollout: what it predicted, and how well-formed it was."""

    rating: int | None
    recovery_rung: str
    fmt_json_valid: bool
    fmt_all_fields: bool
    fmt_arith: bool
    fmt_rating_range: bool

    @property
    def recovered(self) -> bool:
        return self.rating is not None

    @property
    def format_score(self) -> float:
        components = (
            self.fmt_json_valid,
            self.fmt_all_fields,
            self.fmt_arith,
            self.fmt_rating_range,
        )
        return sum(1.0 for c in components if c) / len(components)


def _as_float(value: object) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0


def derive_rating(data: dict) -> tuple[int, float]:
    """Recompute score_gap and rating from the primitive dimension + penalty fields.

    Mirrors the arithmetic stated in TURING_PROMPT, and matches reward.py's rule that
    model-emitted arithmetic is never authoritative when the primitives are present.
    """
    base_a = sum(_as_float(data.get(f)) for f in _DIMENSION_FIELDS_A)
    base_b = sum(_as_float(data.get(f)) for f in _DIMENSION_FIELDS_B)
    penalty_a = sum(_as_float(data.get(f)) for f in _PENALTY_FIELDS_A) / 4.0 * 3.0
    penalty_b = sum(_as_float(data.get(f)) for f in _PENALTY_FIELDS_B) / 4.0 * 3.0
    score_a = max(0.0, base_a - penalty_a)
    score_b = max(0.0, base_b - penalty_b)
    score_gap = score_b - score_a
    return _rating_from_turing_score_gap(score_gap), score_gap


def _arithmetic_is_consistent(data: dict, rating: int, score_gap: float) -> bool:
    """True when the model's own stated totals match what its primitives imply."""
    base_a = sum(_as_float(data.get(f)) for f in _DIMENSION_FIELDS_A)
    base_b = sum(_as_float(data.get(f)) for f in _DIMENSION_FIELDS_B)
    penalty_a = sum(_as_float(data.get(f)) for f in _PENALTY_FIELDS_A) / 4.0 * 3.0
    penalty_b = sum(_as_float(data.get(f)) for f in _PENALTY_FIELDS_B) / 4.0 * 3.0
    expected = {
        "base_score_a": base_a,
        "base_score_b": base_b,
        "penalty_a": penalty_a,
        "penalty_b": penalty_b,
        "response_a_score": max(0.0, base_a - penalty_a),
        "response_b_score": max(0.0, base_b - penalty_b),
        "score_gap": score_gap,
    }
    for name, want in expected.items():
        if name not in data:
            return False
        if abs(_as_float(data.get(name)) - want) > ARITH_TOLERANCE:
            return False
    return _coerce_turing_rating(data.get("rating")) == rating


def parse_judge_verdict(completion: str | None) -> JudgeVerdict:
    """Recover a rating and score format quality from one rollout completion."""
    text = completion if isinstance(completion, str) else ""
    data = extract_json_object(text)

    if not isinstance(data, dict):
        recovered = _extract_turing_rating(text)
        return JudgeVerdict(
            rating=recovered,
            recovery_rung="rating_text" if recovered is not None else "none",
            fmt_json_valid=False,
            fmt_all_fields=False,
            fmt_arith=False,
            fmt_rating_range=False,
        )

    fmt_all_fields = set(data) == set(TURING_FIELDS)
    explicit_rating = _coerce_turing_rating(data.get("rating"))
    fmt_rating_range = explicit_rating is not None

    if all(field in data for field in _PRIMITIVE_FIELDS):
        rating, score_gap = derive_rating(data)
        return JudgeVerdict(
            rating=rating,
            recovery_rung="dimensions",
            fmt_json_valid=True,
            fmt_all_fields=fmt_all_fields,
            fmt_arith=_arithmetic_is_consistent(data, rating, score_gap),
            fmt_rating_range=fmt_rating_range,
        )

    if "score_gap" in data:
        return JudgeVerdict(
            rating=_rating_from_turing_score_gap(_as_float(data.get("score_gap"))),
            recovery_rung="score_gap",
            fmt_json_valid=True,
            fmt_all_fields=fmt_all_fields,
            fmt_arith=False,
            fmt_rating_range=fmt_rating_range,
        )

    if explicit_rating is not None:
        return JudgeVerdict(
            rating=explicit_rating,
            recovery_rung="rating_field",
            fmt_json_valid=True,
            fmt_all_fields=fmt_all_fields,
            fmt_arith=False,
            fmt_rating_range=True,
        )

    recovered = _extract_turing_rating(text)
    return JudgeVerdict(
        rating=recovered,
        recovery_rung="rating_text" if recovered is not None else "none",
        fmt_json_valid=True,
        fmt_all_fields=fmt_all_fields,
        fmt_arith=False,
        fmt_rating_range=False,
    )
````

- [ ] **Step 4: Run test to verify it passes**

Run: `/Users/lancewicki/miniforge3/bin/python -m pytest tests/test_judge_verdict.py -q`
Expected: PASS, 16 tests

- [ ] **Step 5: Commit**

```bash
git add training/grpo/judge_verdict.py tests/test_judge_verdict.py
git commit -m "feat: judge verdict parsing with rating-recovery ladder and format components"
```

---

## Task 4: Judge reward function

The veRL entry point. Both reward arms, the 0.9/0.1 split, and every per-sample metric.

**Files:**
- Create: `training/grpo/judge_reward.py`
- Test: `tests/test_judge_reward.py`

**Interfaces:**
- Consumes: `training.grpo.judge_verdict.parse_judge_verdict`, `JudgeVerdict`
- Produces: `ARM_DIRECTIONAL = "directional"`; `ARM_GRADED = "graded"`; `resolve_arm() -> str`; `directional_task_reward(rating: int, human_is_b: bool) -> float`; `graded_task_reward(rating: int, human_is_b: bool) -> float`; `task_reward(rating: int, human_is_b: bool, arm: str) -> float`; `async compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs) -> dict`

Every metric key emitted is prefixed `judge_` so Task 5's discovery rule picks it up.

- [ ] **Step 1: Write the failing test**

Create `tests/test_judge_reward.py`:

```python
"""Unit tests for the judge GRPO reward."""

import asyncio
import json

import pytest

from shared.judge_prompts import TURING_RESPONSE_PROPERTIES
from training.grpo.judge_reward import (
    ARM_DIRECTIONAL,
    ARM_GRADED,
    compute_score,
    directional_task_reward,
    graded_task_reward,
    resolve_arm,
    task_reward,
)


def _verdict_json(rating: int) -> str:
    """A verdict whose primitives derive exactly the requested rating."""
    gap_for = {1: -3.0, 2: -1.5, 3: -0.5, 4: 0.0, 5: 0.5, 6: 1.5, 7: 3.0}
    gap = gap_for[rating]
    data = {}
    for name, schema in TURING_RESPONSE_PROPERTIES.items():
        data[name] = "text" if schema["type"] == "string" else 0.0
    # Put the whole gap on one side; each dimension is capped at 1.0.
    if gap >= 0:
        for i, field in enumerate(
            ("immediate_target_score_b", "human_goal_score_b", "communication_style_score_b")
        ):
            data[field] = max(0.0, min(1.0, gap - i))
    else:
        for i, field in enumerate(
            ("immediate_target_score_a", "human_goal_score_a", "communication_style_score_a")
        ):
            data[field] = max(0.0, min(1.0, -gap - i))
    base_a = sum(data[f] for f in ("immediate_target_score_a", "human_goal_score_a", "communication_style_score_a"))
    base_b = sum(data[f] for f in ("immediate_target_score_b", "human_goal_score_b", "communication_style_score_b"))
    data["base_score_a"] = base_a
    data["base_score_b"] = base_b
    data["penalty_a"] = 0.0
    data["penalty_b"] = 0.0
    data["response_a_score"] = base_a
    data["response_b_score"] = base_b
    data["score_gap"] = base_b - base_a
    data["rating"] = rating
    return json.dumps(data)


def _score(solution: str, ground_truth: str, arm: str = ARM_DIRECTIONAL) -> dict:
    return asyncio.run(
        compute_score(
            "prism_judge", solution, ground_truth, {"row_id": "r", "split": "train"}, arm=arm
        )
    )


def test_directional_rewards_a_correct_confident_call():
    assert directional_task_reward(7, human_is_b=True) == 1.0
    assert directional_task_reward(1, human_is_b=False) == 1.0


def test_directional_punishes_a_wrong_call():
    assert directional_task_reward(7, human_is_b=False) == 0.0
    assert directional_task_reward(1, human_is_b=True) == 0.0


def test_directional_pays_half_for_a_tie():
    assert directional_task_reward(4, human_is_b=True) == 0.5
    assert directional_task_reward(4, human_is_b=False) == 0.5


def test_graded_reward_values_match_the_spec():
    assert graded_task_reward(7, human_is_b=True) == pytest.approx(1.0)
    assert graded_task_reward(6, human_is_b=True) == pytest.approx(0.9722, abs=1e-4)
    assert graded_task_reward(5, human_is_b=True) == pytest.approx(0.8889, abs=1e-4)
    assert graded_task_reward(4, human_is_b=True) == pytest.approx(0.75)
    assert graded_task_reward(1, human_is_b=True) == pytest.approx(0.0)


def test_graded_reward_is_symmetric_under_swapping_the_human_side():
    for rating in range(1, 8):
        mirrored = 8 - rating
        assert graded_task_reward(rating, human_is_b=True) == pytest.approx(
            graded_task_reward(mirrored, human_is_b=False)
        )


def test_task_reward_dispatches_on_arm():
    assert task_reward(4, True, ARM_DIRECTIONAL) == 0.5
    assert task_reward(4, True, ARM_GRADED) == pytest.approx(0.75)


def test_unknown_arm_raises():
    with pytest.raises(ValueError):
        task_reward(4, True, "nonsense")


def test_resolve_arm_defaults_to_directional(monkeypatch):
    monkeypatch.delenv("JUDGE_REWARD_ARM", raising=False)
    assert resolve_arm() == ARM_DIRECTIONAL


def test_resolve_arm_rejects_an_unknown_env_value(monkeypatch):
    monkeypatch.setenv("JUDGE_REWARD_ARM", "nonsense")
    with pytest.raises(ValueError):
        resolve_arm()


def test_compute_score_totals_task_and_format():
    result = _score(_verdict_json(7), "B")
    assert result["judge_task_reward"] == 1.0
    assert result["judge_format_score"] == 1.0
    assert result["score"] == pytest.approx(1.0)
    assert result["judge_acc"] == 1.0
    assert result["judge_rating"] == 7
    assert result["judge_pred_b"] == 1.0
    assert result["judge_human_is_b"] == 1.0


def test_compute_score_handles_a_ground_truth_of_a():
    result = _score(_verdict_json(1), "A")
    assert result["judge_acc"] == 1.0
    assert result["judge_human_is_b"] == 0.0
    assert result["judge_pred_b"] == 0.0


def test_unrecoverable_verdict_scores_zero_but_still_reports():
    result = _score("gibberish", "B")
    assert result["judge_task_reward"] == 0.0
    assert result["judge_format_score"] == 0.0
    assert result["score"] == 0.0
    assert result["judge_recovered"] == 0.0
    assert result["judge_rung_none"] == 1.0


def test_malformed_but_parseable_still_earns_task_reward():
    data = json.loads(_verdict_json(7))
    del data["reasoning"]
    result = _score(json.dumps(data), "B")
    assert result["judge_task_reward"] == 1.0
    assert result["judge_fmt_all_fields"] == 0.0
    assert result["score"] == pytest.approx(0.9 + 0.1 * 0.75)


def test_tie_is_reported_and_excluded_from_strict_accuracy():
    result = _score(_verdict_json(4), "B")
    assert result["judge_tie"] == 1.0
    assert result["judge_acc"] == 0.5
    assert result["judge_correct_strict"] == 0.0


def test_rating_histogram_is_one_hot():
    result = _score(_verdict_json(5), "B")
    assert result["judge_rating_5"] == 1.0
    assert sum(result[f"judge_rating_{i}"] for i in range(1, 8)) == 1.0


def test_graded_arm_changes_the_total():
    result = _score(_verdict_json(5), "B", arm=ARM_GRADED)
    assert result["judge_task_reward"] == pytest.approx(0.8889, abs=1e-4)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/lancewicki/miniforge3/bin/python -m pytest tests/test_judge_reward.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'training.grpo.judge_reward'`

- [ ] **Step 3: Write minimal implementation**

Create `training/grpo/judge_reward.py`:

```python
"""veRL reward for judge-only RLVR.

The label is known by construction, so this reward is entirely local: no judge server,
no HTTP, no external model. That is the structural difference from the generator's
reward path, where judge calls dominated wall-clock.

Reward = JUDGE_TASK_WEIGHT * task + JUDGE_FORMAT_WEIGHT * format. Format is a minor,
*additive* term rather than a gate: at eval time every model runs under forced schema
decoding, so format is free there and the published comparison is on accuracy alone.
Format matters only as a training-time scaffold for unconstrained rollouts.
"""

from __future__ import annotations

import os
import sys
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from training.grpo.judge_verdict import JudgeVerdict, parse_judge_verdict

ARM_DIRECTIONAL = "directional"
ARM_GRADED = "graded"
ARMS = (ARM_DIRECTIONAL, ARM_GRADED)

DEFAULT_TASK_WEIGHT = 0.9
DEFAULT_FORMAT_WEIGHT = 0.1

RECOVERY_RUNGS = ("dimensions", "score_gap", "rating_field", "rating_text", "none")

_TIE_RATING = 4


def resolve_arm() -> str:
    """Read the reward arm from the environment, defaulting to directional."""
    arm = os.environ.get("JUDGE_REWARD_ARM", ARM_DIRECTIONAL)
    if arm not in ARMS:
        raise ValueError(f"JUDGE_REWARD_ARM must be one of {ARMS}, got {arm!r}")
    return arm


def _weights() -> tuple[float, float]:
    return (
        float(os.environ.get("JUDGE_TASK_WEIGHT", DEFAULT_TASK_WEIGHT)),
        float(os.environ.get("JUDGE_FORMAT_WEIGHT", DEFAULT_FORMAT_WEIGHT)),
    )


def directional_task_reward(rating: int, human_is_b: bool) -> float:
    """1 for the right side, 0 for the wrong side, 0.5 for a tie."""
    if rating == _TIE_RATING:
        return 0.5
    return 1.0 if (rating > _TIE_RATING) == human_is_b else 0.0


def graded_task_reward(rating: int, human_is_b: bool) -> float:
    """1 - (p - y)^2 with p = (rating - 1) / 6, a graded version of the directional arm."""
    p = (rating - 1) / 6.0
    y = 1.0 if human_is_b else 0.0
    return 1.0 - (p - y) ** 2


def task_reward(rating: int, human_is_b: bool, arm: str) -> float:
    if arm == ARM_DIRECTIONAL:
        return directional_task_reward(rating, human_is_b)
    if arm == ARM_GRADED:
        return graded_task_reward(rating, human_is_b)
    raise ValueError(f"unknown reward arm {arm!r}; expected one of {ARMS}")


def _metrics(verdict: JudgeVerdict, human_is_b: bool, arm: str) -> dict[str, float]:
    """Per-sample metrics. Every key is judge_-prefixed so verl_metric_patch finds it."""
    rating = verdict.rating
    task = task_reward(rating, human_is_b, arm) if verdict.recovered else 0.0
    task_weight, format_weight = _weights()
    total = task_weight * task + format_weight * verdict.format_score

    is_tie = bool(rating == _TIE_RATING)
    if not verdict.recovered:
        acc = 0.0
        correct_strict = 0.0
        pred_b = 0.0
        p = 0.5
    else:
        acc = directional_task_reward(rating, human_is_b)
        correct_strict = 1.0 if acc == 1.0 else 0.0
        pred_b = 1.0 if rating > _TIE_RATING else 0.0
        p = (rating - 1) / 6.0

    y = 1.0 if human_is_b else 0.0
    metrics: dict[str, float] = {
        "score": total,
        "total_score": total,
        "judge_total": total,
        "judge_task_reward": task,
        "judge_format_score": verdict.format_score,
        "judge_fmt_json_valid": float(verdict.fmt_json_valid),
        "judge_fmt_all_fields": float(verdict.fmt_all_fields),
        "judge_fmt_arith": float(verdict.fmt_arith),
        "judge_fmt_rating_range": float(verdict.fmt_rating_range),
        "judge_acc": acc,
        "judge_correct_strict": correct_strict,
        "judge_tie": 1.0 if (verdict.recovered and is_tie) else 0.0,
        "judge_brier": (p - y) ** 2,
        "judge_conf": 2.0 * abs(p - 0.5),
        "judge_rating": float(rating) if verdict.recovered else 0.0,
        "judge_pred_b": pred_b,
        "judge_human_is_b": y,
        "judge_recovered": float(verdict.recovered),
    }
    for rung in RECOVERY_RUNGS:
        metrics[f"judge_rung_{rung}"] = 1.0 if verdict.recovery_rung == rung else 0.0
    for value in range(1, 8):
        metrics[f"judge_rating_{value}"] = 1.0 if rating == value else 0.0
    return metrics


async def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict | None = None,
    **kwargs: Any,
) -> dict:
    """Score one judge rollout. ``ground_truth`` is the slot holding the human: "A"/"B"."""
    _ = data_source
    extra_info = extra_info or {}
    label = str(ground_truth).strip().upper()
    if label not in ("A", "B"):
        raise ValueError(f"judge ground_truth must be 'A' or 'B', got {ground_truth!r}")
    human_is_b = label == "B"

    arm = kwargs.get("arm") or resolve_arm()
    verdict = parse_judge_verdict(solution_str)
    return _metrics(verdict, human_is_b, arm)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/Users/lancewicki/miniforge3/bin/python -m pytest tests/test_judge_reward.py -q`
Expected: PASS, 16 tests

- [ ] **Step 5: Commit**

```bash
git add training/grpo/judge_reward.py tests/test_judge_reward.py
git commit -m "feat: judge GRPO reward with directional and graded arms"
```

---

## Task 5: wandb metric wiring

Two changes to `verl_metric_patch.py`: make `judge_`-prefixed reward keys discoverable, and add group-health metrics that need the whole batch.

**Files:**
- Modify: `training/grpo/verl_metric_patch.py` (`_RATE_METRIC_NAMES` ~line 134, `_collect_reward_metric_names` ~line 203, `append_custom_reward_metrics` ~line 218)
- Test: `tests/test_judge_metric_patch.py`

**Interfaces:**
- Consumes: `training.grpo.judge_reward` metric key names
- Produces: `append_judge_group_metrics(metrics: dict, batch: Any) -> None`; `_collect_reward_metric_names` additionally returns any key starting with `judge_`

- [ ] **Step 1: Write the failing test**

Create `tests/test_judge_metric_patch.py`:

```python
"""Unit tests for judge-metric discovery and group-health metrics."""

from types import SimpleNamespace

from training.grpo.verl_metric_patch import (
    _collect_reward_metric_names,
    append_judge_group_metrics,
)


def _batch(uids, totals, corrects):
    return SimpleNamespace(
        non_tensor_batch={
            "uid": uids,
            "reward_extra_info": [
                {"judge_total": t, "judge_correct_strict": c}
                for t, c in zip(totals, corrects)
            ],
        }
    )


def test_judge_keys_are_discovered():
    names = _collect_reward_metric_names(
        {"reward_extra_info": {"judge_acc": [1.0], "judge_fmt_arith": [0.0]}}
    )
    assert "judge_acc" in names
    assert "judge_fmt_arith" in names


def test_existing_format_and_length_discovery_still_works():
    names = _collect_reward_metric_names(
        {"reward_extra_info": {"format_score": [1.0], "length_ratio": [1.0]}}
    )
    assert "format_score" in names
    assert "length_ratio" in names


def test_group_metrics_flag_a_degenerate_all_correct_group():
    metrics = {}
    append_judge_group_metrics(metrics, _batch(["g1"] * 4, [1.0] * 4, [1.0] * 4))
    assert metrics["judge_group/all_equal_rate"] == 1.0
    assert metrics["judge_group/all_correct_rate"] == 1.0
    assert metrics["judge_group/all_wrong_rate"] == 0.0


def test_group_metrics_flag_a_degenerate_all_wrong_group():
    metrics = {}
    append_judge_group_metrics(metrics, _batch(["g1"] * 4, [0.1] * 4, [0.0] * 4))
    assert metrics["judge_group/all_equal_rate"] == 1.0
    assert metrics["judge_group/all_wrong_rate"] == 1.0


def test_a_mixed_group_is_not_degenerate():
    metrics = {}
    append_judge_group_metrics(metrics, _batch(["g1"] * 4, [1.0, 0.0, 1.0, 0.0], [1.0, 0.0, 1.0, 0.0]))
    assert metrics["judge_group/all_equal_rate"] == 0.0
    assert metrics["judge_group/all_correct_rate"] == 0.0
    assert metrics["judge_group/all_wrong_rate"] == 0.0


def test_rates_average_over_groups():
    metrics = {}
    append_judge_group_metrics(
        metrics,
        _batch(["g1", "g1", "g2", "g2"], [1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 1.0, 0.0]),
    )
    assert metrics["judge_group/all_equal_rate"] == 0.5
    assert metrics["judge_group/n_groups"] == 2


def test_missing_judge_keys_are_a_no_op():
    metrics = {}
    append_judge_group_metrics(metrics, SimpleNamespace(non_tensor_batch={}))
    assert metrics == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/lancewicki/miniforge3/bin/python -m pytest tests/test_judge_metric_patch.py -q`
Expected: FAIL with `ImportError: cannot import name 'append_judge_group_metrics'`

- [ ] **Step 3: Write minimal implementation**

In `training/grpo/verl_metric_patch.py`, replace the body of `_collect_reward_metric_names` (currently lines 203-215) so both loops also accept the `judge_` prefix:

```python
_DISCOVERED_KEY_PREFIXES = ("format_", "length_", "judge_")


def _collect_reward_metric_names(non_tensor_batch: dict[str, Any]) -> set[str]:
    metric_names = set(_REWARD_KEY_ALIASES)
    for key in non_tensor_batch:
        key = str(key)
        if key.startswith(_DISCOVERED_KEY_PREFIXES):
            metric_names.add(key)
    reward_extra_info = non_tensor_batch.get("reward_extra_info")
    if isinstance(reward_extra_info, dict):
        for key in reward_extra_info:
            key = str(key)
            if key.startswith(_DISCOVERED_KEY_PREFIXES):
                metric_names.add(key)
    return metric_names
```

Add these judge 0/1 metrics to `_RATE_METRIC_NAMES` (currently lines 134-146), so they log as rates:

```python
_RATE_METRIC_NAMES = {
    "meaningful_thinking",
    "source_copy",
    "wrong_perspective",
    "assistant_like_response",
    "unjustified_code_switching_response",
    "wrong_target_or_role_response",
    "unsupported_adversarial_reframing_response",
    "format_human_prefix",
    "format_nonempty_reasoning",
    "format_no_post_human_thinking",
    "format_reasoning_schema",
    "judge_correct_strict",
    "judge_tie",
    "judge_recovered",
    "judge_pred_b",
    "judge_human_is_b",
    "judge_fmt_json_valid",
    "judge_fmt_all_fields",
    "judge_fmt_arith",
    "judge_fmt_rating_range",
}
```

Add `append_judge_group_metrics` immediately after `append_custom_reward_metrics`, and call it from the end of `append_custom_reward_metrics`:

```python
def append_judge_group_metrics(metrics: dict[str, Any], batch: Any) -> None:
    """Group-health metrics for judge GRPO.

    A group whose rollouts all earn the same reward yields zero advantage and so
    contributes no gradient. With a near-deterministic classifier this can dominate,
    which is the signal that DAPO-style dynamic sampling is needed. Grouping needs the
    whole batch, so it cannot live in the per-sample reward function.
    """
    non_tensor_batch = getattr(batch, "non_tensor_batch", None)
    if not isinstance(non_tensor_batch, dict):
        return
    uids = non_tensor_batch.get("uid")
    if uids is None:
        return
    totals = _extract_reward_metric_values(non_tensor_batch, "judge_total")
    corrects = _extract_reward_metric_values(non_tensor_batch, "judge_correct_strict")
    if totals is None or corrects is None:
        return

    grouped_totals: dict[str, list[float]] = defaultdict(list)
    grouped_correct: dict[str, list[float]] = defaultdict(list)
    for uid, total, correct in zip(uids, totals, corrects):
        grouped_totals[str(uid)].append(float(total))
        grouped_correct[str(uid)].append(float(correct))

    n_groups = len(grouped_totals)
    if not n_groups:
        return
    all_equal = sum(1 for v in grouped_totals.values() if max(v) - min(v) < 1e-9)
    all_correct = sum(1 for v in grouped_correct.values() if all(c == 1.0 for c in v))
    all_wrong = sum(1 for v in grouped_correct.values() if all(c == 0.0 for c in v))

    metrics["judge_group/n_groups"] = float(n_groups)
    metrics["judge_group/all_equal_rate"] = all_equal / n_groups
    metrics["judge_group/all_correct_rate"] = all_correct / n_groups
    metrics["judge_group/all_wrong_rate"] = all_wrong / n_groups
```

At the end of `append_custom_reward_metrics`, add the call:

```python
    append_judge_group_metrics(metrics, batch)
```

`defaultdict` is **not** currently imported by this module — its imports are `importlib.util`,
`os`, `re`, `typing.Any` and `numpy`. Add `from collections import defaultdict` at the top.

- [ ] **Step 4: Run test to verify it passes**

Run: `/Users/lancewicki/miniforge3/bin/python -m pytest tests/test_judge_metric_patch.py -q`
Expected: PASS, 7 tests

- [ ] **Step 5: Run the full suite to check nothing regressed**

Run: `/Users/lancewicki/miniforge3/bin/python -m pytest tests/ -q`
Expected: PASS — no previously-passing test may fail. `verl_metric_patch.py` is shared with the generator path, so this check is not optional.

- [ ] **Step 6: Commit**

```bash
git add training/grpo/verl_metric_patch.py tests/test_judge_metric_patch.py
git commit -m "feat: discover judge_ reward metrics and add GRPO group-health metrics"
```

---

## Task 6: veRL config for judge GRPO

**Files:**
- Create: `training/grpo/configs/qwen35_judge_grpo.yaml`
- Test: `tests/test_judge_grpo_config.py`

**Interfaces:**
- Consumes: `training/grpo/judge_reward.py` (as `custom_reward_function.path`)
- Produces: Hydra config name `qwen35_judge_grpo`, passed as `--config-name` by Task 8's launcher

The single most important override here is `target_modules`. The shared base sets
`all-linear`, which on a Qwen3.5 hybrid model would attach LoRA to the Gated-DeltaNet
backbone — destructive per arXiv:2604.22127. The explicit attention+MLP list is mandatory
for all three sizes, and the test locks it.

- [ ] **Step 1: Write the failing test**

Create `tests/test_judge_grpo_config.py`:

```python
"""Regression guard for the judge GRPO config.

Locks the values that make this a *judge* run rather than a generator run: the local
reward function, thinking-on, the long-prompt budget, and — most importantly — a
target_modules list that never touches the Gated-DeltaNet backbone.
"""

import os

import yaml

CFG = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "training", "grpo", "configs", "qwen35_judge_grpo.yaml",
)

# Gated-DeltaNet backbone projections. LoRA on any of these is destructive.
GDN_MODULES = ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj")


def _load():
    with open(CFG) as handle:
        return yaml.safe_load(handle)


def test_uses_the_local_judge_reward():
    c = _load()
    assert c["custom_reward_function"]["path"] == "training/grpo/judge_reward.py"
    assert c["custom_reward_function"]["name"] == "compute_score"


def test_thinking_is_enabled():
    assert _load()["data"]["apply_chat_template_kwargs"]["enable_thinking"] is True


def test_target_modules_never_touch_the_deltanet_backbone():
    modules = _load()["actor_rollout_ref"]["model"]["target_modules"]
    assert isinstance(modules, list), "must be an explicit list, never the base's all-linear"
    assert set(modules) == {
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
    }
    for banned in GDN_MODULES:
        assert banned not in modules


def test_vision_and_mtp_are_excluded():
    assert _load()["actor_rollout_ref"]["model"]["exclude_modules"] == ".*(visual|mtp).*"


def test_prompt_and_response_budgets_fit_the_context_window():
    c = _load()
    data = c["data"]
    rollout = c["actor_rollout_ref"]["rollout"]
    assert data["max_prompt_length"] + data["max_response_length"] <= rollout["max_model_len"]
    assert data["max_response_length"] == rollout["response_length"]


def test_optimiser_follows_the_9b_recipe():
    c = _load()
    actor = c["actor_rollout_ref"]["actor"]
    assert float(actor["optim"]["lr"]) == 1e-4
    assert float(actor["kl_loss_coef"]) == 1e-4
    assert actor["use_kl_loss"] is True
    assert float(c["actor_rollout_ref"]["rollout"]["temperature"]) == 1.0


def test_validation_sampling_is_narrower_than_training():
    val = _load()["actor_rollout_ref"]["rollout"]["val_kwargs"]
    assert float(val["temperature"]) == 0.7
    assert float(val["top_p"]) == 0.8
    assert val["top_k"] == 20
    assert val["n"] == 1


def test_fsdp2_is_selected_for_actor_and_ref():
    c = _load()
    assert c["actor_rollout_ref"]["actor"]["strategy"] == "fsdp2"
    assert c["actor_rollout_ref"]["ref"]["strategy"] == "fsdp2"


def test_reward_is_declared_the_way_verl_09_actually_reads_it():
    """veRL 0.9's V1 controller ignores the legacy top-level block.

    A config carrying only `custom_reward_function` runs V1, never calls the reward, and
    scores every rollout 0 without erroring. The working 9B launcher disables V1 and uses
    the nested `reward.custom_reward_function` block; both must be present.
    """
    c = _load()
    assert c["trainer"]["use_v1"] is False
    assert c["reward"]["custom_reward_function"]["path"] == "training/grpo/judge_reward.py"
    assert c["reward"]["custom_reward_function"]["name"] == "compute_score"
    assert c["custom_reward_function"]["path"] == "training/grpo/judge_reward.py"


def test_rollout_overrides_the_8b_generator_hardware_profile():
    """The parent is an 8B generator config; these three are wrong for a judge run."""
    rollout = _load()["actor_rollout_ref"]["rollout"]
    assert rollout["tensor_model_parallel_size"] == 1
    assert float(rollout["gpu_memory_utilization"]) == 0.55
    # Judge prompts (~6k tokens) exceed the 4096 batched-token cap.
    assert rollout["enable_chunked_prefill"] is True


def test_judge_runs_log_to_their_own_wandb_project():
    assert _load()["trainer"]["project_name"] == "grpo-judge"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/lancewicki/miniforge3/bin/python -m pytest tests/test_judge_grpo_config.py -q`
Expected: FAIL with `FileNotFoundError` on `qwen35_judge_grpo.yaml`

- [ ] **Step 3: Write minimal implementation**

Create `training/grpo/configs/qwen35_judge_grpo.yaml`:

```yaml
# veRL GRPO config for judge-only RLVR (Qwen3.5 2B / 4B / 9B discriminators).
#
# Composes the shared 8B base for structure, then overrides what the judge side needs
# differently:
#   - reward is training/grpo/judge_reward.py: local, label-verifiable, no judge server
#   - thinking ON, matching what generator RL actually ran against
#   - long prompts: a rendered TURING_PROMPT is ~6k tokens against ~1k for a generator prompt
#   - the 9B optimiser recipe (lr 1e-4, kl 1e-4, rollout T=1.0), not the 8B-era defaults
#
# actor_rollout_ref.model.path is set per run by scripts/slurm/judge_grpo_train.sh.
defaults:
  - qwen3_8b_grpo
  - _self_

data:
  train_files: data/prism/judge/iter1/train.parquet
  val_files: data/prism/judge/iter1/val.parquet
  train_batch_size: 64
  # The rendered prompt is ~22k chars (~6k tokens); the headroom absorbs long user histories.
  max_prompt_length: 10240
  # Thinking plus a 37-field verdict. Deliberately below the 8192 eval budget to bound
  # rollout cost; judge_truncation_rate from the Task 7 probe says whether this is too tight.
  max_response_length: 6144
  truncation: left
  filter_overlong_prompts: true
  apply_chat_template_kwargs:
    enable_thinking: true

actor_rollout_ref:
  model:
    # MUST stay an explicit list. The base config says all-linear, which on a Qwen3.5
    # hybrid would attach LoRA to the Gated-DeltaNet backbone (destructive; arXiv:2604.22127).
    target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
    exclude_modules: '.*(visual|mtp).*'
    lora_rank: 64
    lora_alpha: 32
    lora_adapter_path: null
    enable_gradient_checkpointing: true
  actor:
    ppo_mini_batch_size: 64
    ppo_micro_batch_size_per_gpu: 1
    use_kl_loss: true
    kl_loss_coef: 0.0001
    strategy: fsdp2
    use_dynamic_bsz: false
    optim:
      lr: 0.0001
  ref:
    strategy: fsdp2
  rollout:
    n: 4
    temperature: 1.0
    top_p: 1.0
    top_k: -1
    max_model_len: 16384
    max_num_batched_tokens: 16384
    response_length: 6144
    # The parent is an 8B *generator* profile. These three must be overridden or the judge
    # run inherits values that are wrong for it, the failure mode that cost jobs 15143 and
    # 13634. TP=1 matches the working 9B recipe (a 2B/4B judge has no reason to shard);
    # chunked prefill is REQUIRED because judge prompts (~6k tokens, budget 10240) exceed
    # the 4096 batched-token cap; 0.55 is the memory fraction the 9B recipe actually ran.
    tensor_model_parallel_size: 1
    gpu_memory_utilization: 0.55
    enable_chunked_prefill: true
    val_kwargs:
      temperature: 0.7
      top_p: 0.8
      top_k: 20
      do_sample: true
      n: 1

critic:
  optim:
    lr: 0.0001

# Declared BOTH ways on purpose. veRL 0.9's V1 controller does not migrate the legacy
# top-level `custom_reward_function` block (see scripts/slurm/rl_generator_train_9b.sh),
# so a config carrying only the legacy block runs V1, never calls the reward, and scores
# every rollout 0 with no error at all. `trainer.use_v1: false` plus the nested
# `reward.custom_reward_function` block is what the working 9B run uses. This lives in the
# config rather than a submit-time override string for the same reason the 9B recipe does:
# values that can be dropped from EXTRA_OVERRIDES eventually are.
custom_reward_function:
  path: training/grpo/judge_reward.py
  name: compute_score

reward:
  custom_reward_function:
    path: training/grpo/judge_reward.py
    name: compute_score

trainer:
  use_v1: false
  project_name: grpo-judge
  default_local_dir: results/grpo/checkpoints_judge
  experiment_name: qwen35-judge-grpo
  total_epochs: 3
```

`max_num_seqs` and `enforce_eager` are deliberately left inherited from the parent: the
working 9B launcher does not override either, and it runs 12.5k-token prompts, so the
inherited values are proven in practice rather than merely plausible.

- [ ] **Step 4: Run test to verify it passes**

Run: `/Users/lancewicki/miniforge3/bin/python -m pytest tests/test_judge_grpo_config.py -q`
Expected: PASS, 11 tests

- [ ] **Step 5: Commit**

```bash
git add training/grpo/configs/qwen35_judge_grpo.yaml tests/test_judge_grpo_config.py
git commit -m "feat: veRL config for judge GRPO with GDN-safe LoRA targets"
```

---

## Task 7: Three-regime zero-shot format probe

The Phase 0 gate. Measures what unconstrained rollouts actually look like, which is the
only thing that predicts whether GRPO has format signal to learn from.

**Files:**
- Modify: `training/grpo/reward.py` (`_resolve_response_format`, ~line 345)
- Create: `scripts/probe_judge_format.py`
- Test: `tests/test_probe_judge_format.py`

**Interfaces:**
- Consumes: `training.grpo.judge_verdict.parse_judge_verdict`; `training.grpo.judge_reward.directional_task_reward`; `shared.api_client.post_chat_async`
- Produces: `REGIMES: tuple[str, ...]`; `response_format_for_regime(regime: str) -> dict | None`; `probe_record(completion: str | None, finish_reason: str, human_is_b: bool) -> dict`; `dump_row(model: str, row: dict, record: dict) -> dict`; `summarize_probe(records: list[dict]) -> dict`

This script does double duty. Besides the Phase 0 gate, its `--dump_csv` output is the
long-format CSV that Task 10 analyses — so scoring a trained judge or a zero-shot baseline
on the 880-pair eval set is the same command with `--regimes json_schema`. Nothing else in
the plan produces that CSV.

- [ ] **Step 1: Write the failing test**

Create `tests/test_probe_judge_format.py`:

```python
"""Unit tests for the zero-shot judge format probe.

Network calls are out of scope here; these lock the regime mapping and the summary
arithmetic, which are the parts that decide the Phase 0 gate.
"""

import json

import pytest

from shared.judge_prompts import TURING_RESPONSE_PROPERTIES
from scripts.probe_judge_format import (
    REGIMES,
    dump_row,
    probe_record,
    response_format_for_regime,
    summarize_probe,
)


def _full_verdict(rating: int = 7) -> str:
    data = {}
    for name, schema in TURING_RESPONSE_PROPERTIES.items():
        data[name] = "text" if schema["type"] == "string" else 0.0
    data["immediate_target_score_b"] = 1.0
    data["human_goal_score_b"] = 1.0
    data["communication_style_score_b"] = 1.0
    data["base_score_a"] = 0.0
    data["base_score_b"] = 3.0
    data["penalty_a"] = 0.0
    data["penalty_b"] = 0.0
    data["response_a_score"] = 0.0
    data["response_b_score"] = 3.0
    data["score_gap"] = 3.0
    data["rating"] = rating
    return json.dumps(data)


def test_regimes_are_the_three_documented_ones():
    assert REGIMES == ("json_schema", "json_object", "freeform")


def test_freeform_sends_no_response_format():
    assert response_format_for_regime("freeform") is None


def test_json_object_sends_the_loose_constraint():
    assert response_format_for_regime("json_object") == {"type": "json_object"}


def test_json_schema_sends_the_full_ordered_schema():
    fmt = response_format_for_regime("json_schema")
    assert fmt["type"] == "json_schema"
    assert fmt["json_schema"]["schema"]["required"] == list(TURING_RESPONSE_PROPERTIES)


def test_unknown_regime_raises():
    with pytest.raises(ValueError):
        response_format_for_regime("nonsense")


def test_probe_record_scores_a_well_formed_verdict():
    record = probe_record(_full_verdict(7), "stop", human_is_b=True)
    assert record["fmt_all_fields"] == 1.0
    assert record["recovered"] == 1.0
    assert record["correct"] == 1.0
    assert record["truncated"] == 0.0
    assert record["rung"] == "dimensions"


def test_probe_record_marks_a_length_stop_as_truncated():
    assert probe_record(_full_verdict(), "length", human_is_b=True)["truncated"] == 1.0


def test_probe_record_handles_an_unusable_completion():
    record = probe_record("nothing useful", "stop", human_is_b=True)
    assert record["recovered"] == 0.0
    assert record["fmt_all_fields"] == 0.0
    assert record["correct"] == 0.0


def test_summary_averages_the_gate_metrics():
    records = [
        probe_record(_full_verdict(7), "stop", human_is_b=True),
        probe_record("nothing useful", "stop", human_is_b=True),
    ]
    summary = summarize_probe(records)
    assert summary["n"] == 2
    assert summary["fmt_all_fields_rate"] == 0.5
    assert summary["recovered_rate"] == 0.5
    assert summary["accuracy"] == 0.5
    assert summary["truncation_rate"] == 0.0
    assert summary["rung_counts"]["dimensions"] == 1
    assert summary["rung_counts"]["none"] == 1


def test_summary_of_no_records_is_empty_not_a_crash():
    assert summarize_probe([])["n"] == 0


def test_dump_row_emits_the_five_canonical_analysis_columns():
    row = {
        "prompt": [{"role": "user", "content": "x"}],
        "extra_info": {"pair_id": "p1::g0", "order": "human_b", "human_is_b": True},
    }
    record = probe_record(_full_verdict(7), "stop", human_is_b=True)
    assert dump_row("qwen35-4b", row, record) == {
        "model": "qwen35-4b",
        "pair_id": "p1::g0",
        "order": "human_b",
        "rating": 7,
        "human_is_b": True,
    }


def test_dump_row_carries_a_null_rating_when_nothing_was_recovered():
    row = {
        "prompt": [{"role": "user", "content": "x"}],
        "extra_info": {"pair_id": "p1::g0", "order": "human_a", "human_is_b": False},
    }
    record = probe_record("garbage", "stop", human_is_b=False)
    assert dump_row("m", row, record)["rating"] is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/lancewicki/miniforge3/bin/python -m pytest tests/test_probe_judge_format.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.probe_judge_format'`

- [ ] **Step 3a: Add the third response-format mode**

In `training/grpo/reward.py`, replace `_resolve_response_format` (currently lines 345-355) with:

```python
def _resolve_response_format() -> dict | None:
    """Select the decoding constraint for a judge call.

    ``PERSONA_JUDGE_JSON_SCHEMA=1``     -> full 37-field ordered schema (the eval regime)
    ``PERSONA_JUDGE_JSON_SCHEMA=none``  -> no constraint at all (the training-rollout regime)
    unset or anything else              -> ``{"type": "json_object"}`` (valid JSON, free fields)

    The "none" mode exists for scripts/probe_judge_format.py. veRL builds SamplingParams
    directly and never sends response_format, so probing through a constrained path would
    report near-total compliance and say nothing about real rollouts.
    """
    mode = os.environ.get("PERSONA_JUDGE_JSON_SCHEMA")
    if mode == "1":
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "turing_verdict",
                "schema": TURING_RESPONSE_SCHEMA,
            },
        }
    # `mode is not None` is load-bearing: str(None).strip().lower() == "none", so the
    # obvious `str(mode).strip().lower() == "none"` would make the UNSET env take this
    # branch and silently drop response_format for every existing generator run.
    if mode is not None and mode.strip().lower() == "none":
        return None
    return {"type": "json_object"}
```

This is additive: unset and `=1` behave exactly as before, so no existing run changes.
`build_chat_payload` already omits the key when the value is falsy (`shared/api_client.py:102`).

- [ ] **Step 3b: Write the probe**

Create `scripts/probe_judge_format.py`:

```python
"""Zero-shot judge format probe across three decoding regimes.

Phase 0 gate for judge-only RLVR. veRL cannot constrain rollout decoding, so the number
that matters is how often an *unconstrained* model emits a usable 37-field verdict. The
json_schema and json_object arms are the controls: comparing accuracy across the three
says whether the format scaffold buys verdict quality or only parseability.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections import Counter
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from shared.api_client import (
    build_chat_payload,
    get_judge_call_meta,
    post_chat_async,
    resolve_judge_api_key,
)
from shared.judge_prompts import TURING_RESPONSE_SCHEMA
from training.grpo.judge_reward import directional_task_reward
from training.grpo.judge_verdict import parse_judge_verdict

REGIMES: tuple[str, ...] = ("json_schema", "json_object", "freeform")


def response_format_for_regime(regime: str) -> dict | None:
    """Map a probe regime to the response_format the request should carry."""
    if regime == "json_schema":
        return {
            "type": "json_schema",
            "json_schema": {"name": "turing_verdict", "schema": TURING_RESPONSE_SCHEMA},
        }
    if regime == "json_object":
        return {"type": "json_object"}
    if regime == "freeform":
        return None
    raise ValueError(f"unknown regime {regime!r}; expected one of {REGIMES}")


def probe_record(completion: str | None, finish_reason: str, human_is_b: bool) -> dict[str, Any]:
    """Score one probe completion into the fields the gate cares about."""
    verdict = parse_judge_verdict(completion)
    correct = (
        directional_task_reward(verdict.rating, human_is_b) if verdict.recovered else 0.0
    )
    return {
        "rung": verdict.recovery_rung,
        "recovered": float(verdict.recovered),
        "fmt_json_valid": float(verdict.fmt_json_valid),
        "fmt_all_fields": float(verdict.fmt_all_fields),
        "fmt_arith": float(verdict.fmt_arith),
        "format_score": verdict.format_score,
        "correct": float(correct == 1.0),
        "acc": float(correct),
        "truncated": 1.0 if finish_reason == "length" else 0.0,
        "rating": verdict.rating if verdict.recovered else None,
        "completion_chars": len(completion or ""),
    }


def summarize_probe(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate probe records into the Phase 0 gate table."""
    if not records:
        return {"n": 0}

    def mean(key: str) -> float:
        return sum(float(r[key]) for r in records) / len(records)

    return {
        "n": len(records),
        "fmt_all_fields_rate": mean("fmt_all_fields"),
        "fmt_json_valid_rate": mean("fmt_json_valid"),
        "fmt_arith_rate": mean("fmt_arith"),
        "format_score_mean": mean("format_score"),
        "recovered_rate": mean("recovered"),
        "accuracy": mean("acc"),
        "truncation_rate": mean("truncated"),
        "completion_chars_mean": mean("completion_chars"),
        "rung_counts": dict(Counter(r["rung"] for r in records)),
    }


def dump_row(model: str, row: dict, record: dict[str, Any]) -> dict[str, Any]:
    """One long-format row for scripts/analyze_judge_training.py."""
    extra = row["extra_info"]
    return {
        "model": model,
        "pair_id": extra["pair_id"],
        "order": extra["order"],
        "rating": record["rating"],
        "human_is_b": bool(extra["human_is_b"]),
    }


async def _run_regime(
    rows: list[dict], regime: str, model: str, max_tokens: int
) -> list[tuple[dict, dict]]:
    """Score every row in one decoding regime; returns (record, source_row) pairs.

    The payload is assembled exactly as reward.py::_openai_chat does, so the probe
    exercises the same sampling, thinking flag and transport as the real judge path.
    """
    import aiohttp

    api_key = resolve_judge_api_key()
    response_format = response_format_for_regime(regime)
    sampling_raw = os.environ.get("PERSONA_JUDGE_SAMPLING")
    sampling = json.loads(sampling_raw) if sampling_raw else None
    thinking = os.environ.get("PERSONA_JUDGE_ENABLE_THINKING")
    chat_template_kwargs = {"enable_thinking": thinking == "1"} if thinking in ("0", "1") else None

    semaphore = asyncio.Semaphore(int(os.environ.get("JUDGE_PROBE_CONCURRENCY", "8")))
    timeout = aiohttp.ClientTimeout(
        total=float(os.environ.get("PERSONA_OPENAI_TIMEOUT_SECONDS", "1800"))
    )

    async def _one(row: dict) -> tuple[dict, dict]:
        payload = build_chat_payload(
            model=model,
            messages=[{"role": "user", "content": row["prompt"][0]["content"]}],
            max_completion_tokens=max_tokens,
            response_format=response_format,
            reasoning=False,
            sampling=sampling,
            chat_template_kwargs=chat_template_kwargs,
        )
        content = await post_chat_async(session, payload, semaphore=semaphore, api_key=api_key)
        # Telemetry for THIS call: post_chat_async stashes it in a ContextVar, and each
        # gather task carries its own context copy, so this is not cross-talk.
        meta = get_judge_call_meta() or {}
        return (
            probe_record(
                content,
                str(meta.get("finish_reason", "")),
                bool(row["extra_info"]["human_is_b"]),
            ),
            row,
        )

    async with aiohttp.ClientSession(timeout=timeout) as session:
        return list(await asyncio.gather(*(_one(row) for row in rows)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Zero-shot judge format probe")
    parser.add_argument("--pairs_parquet", required=True, help="judge-format parquet (Task 2)")
    parser.add_argument("--model", required=True)
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--dump_csv", default=None,
                        help="Long-format per-row CSV for scripts/analyze_judge_training.py")
    parser.add_argument("--model_label", default=None,
                        help="Name to record in --dump_csv (defaults to --model)")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--max_tokens", type=int, default=8192)
    parser.add_argument("--regimes", nargs="+", default=list(REGIMES))
    args = parser.parse_args()

    for regime in args.regimes:
        response_format_for_regime(regime)  # fail fast on a typo

    rows = pd.read_parquet(args.pairs_parquet).head(args.limit).to_dict("records")
    label = args.model_label or args.model
    summaries: dict[str, Any] = {}
    dump_rows: list[dict[str, Any]] = []
    for regime in args.regimes:
        results = asyncio.run(_run_regime(rows, regime, args.model, args.max_tokens))
        records = [record for record, _row in results]
        summaries[regime] = summarize_probe(records)
        print(f"[{regime}] {json.dumps(summaries[regime], sort_keys=True)}", flush=True)
        if args.dump_csv and regime == args.regimes[-1]:
            dump_rows = [dump_row(label, row, record) for record, row in results]

    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as handle:
        json.dump({"model": args.model, "n_rows": len(rows), "regimes": summaries}, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"Wrote {args.out_json}")

    if args.dump_csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.dump_csv)), exist_ok=True)
        pd.DataFrame(dump_rows).to_csv(args.dump_csv, index=False)
        print(f"Wrote {len(dump_rows)} rows -> {args.dump_csv} (regime={args.regimes[-1]})")


if __name__ == "__main__":
    main()
```

Signatures verified against the repo: `post_chat_async(session, payload, *, semaphore,
max_retries=None, api_key=None) -> str` (`shared/api_client.py:187`) returns only the content
string, with per-call telemetry in `get_judge_call_meta()` (`shared/api_client.py:33`).

- [ ] **Step 4: Run test to verify it passes**

Run: `/Users/lancewicki/miniforge3/bin/python -m pytest tests/test_probe_judge_format.py -q`
Expected: PASS, 12 tests

- [ ] **Step 5: Run the full suite**

Run: `/Users/lancewicki/miniforge3/bin/python -m pytest tests/ -q`
Expected: PASS. `reward.py` is shared with the generator path; `tests/test_judge_payload.py` and `tests/test_reward_env_payload.py` in particular exercise the response-format logic.

- [ ] **Step 6: Commit**

```bash
git add training/grpo/reward.py scripts/probe_judge_format.py tests/test_probe_judge_format.py
git commit -m "feat: three-regime zero-shot judge format probe"
```

---

## Task 8: Slurm launchers

Three wrappers, all submitted through the snapshot gateway. Their tests are static
assertions in the style of `tests/test_rl_9b_launcher.py` — they lock the pinned env so a
future edit cannot silently drop it, the way job 15143 lost its hyperparameters.

**Files:**
- Create: `scripts/slurm/judge_format_probe.sh`
- Create: `scripts/slurm/judge_train_gen.sh`
- Create: `scripts/slurm/judge_grpo_train.sh`
- Test: `tests/test_judge_launchers.py`

**Interfaces:**
- Consumes: `scripts/probe_judge_format.py`, `scripts/build_judge_train_pairs.py`, `training/grpo/configs/qwen35_judge_grpo.yaml`, `eval/generate_trained.py`
- Produces: three sbatch-able scripts driven by `JUDGE_MODEL_PATH`, `JUDGE_REWARD_ARM`, `JUDGE_RUN_TAG`

- [ ] **Step 1: Write the failing test**

Create `tests/test_judge_launchers.py`:

```python
"""Static guards on the judge Slurm launchers."""

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROBE = os.path.join(ROOT, "scripts", "slurm", "judge_format_probe.sh")
GEN = os.path.join(ROOT, "scripts", "slurm", "judge_train_gen.sh")
TRAIN = os.path.join(ROOT, "scripts", "slurm", "judge_grpo_train.sh")
ALL = (PROBE, GEN, TRAIN)


def _text(path: str) -> str:
    with open(path) as handle:
        return handle.read()


def test_every_launcher_clears_the_stale_v2_proxy_vars():
    for path in ALL:
        assert "unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY" in _text(path)


def test_every_launcher_runs_from_the_snapshot_roots():
    for path in ALL:
        text = _text(path)
        assert "TURING_RL_WORK_ROOT" in text
        assert "cluster_job_bootstrap.sh" in text


def test_no_launcher_calls_sbatch_directly():
    for path in ALL:
        for line in _text(path).splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            assert not stripped.startswith("sbatch "), f"{path}: direct sbatch is forbidden"


def test_generation_pins_the_documented_sampling():
    text = _text(GEN)
    assert "GEN_TEMPERATURE=${GEN_TEMPERATURE:-0.7}" in text
    assert "GEN_TOP_P=${GEN_TOP_P:-0.8}" in text
    assert "GEN_TOP_K=${GEN_TOP_K:-20}" in text
    assert "GEN_MAX_TOKENS=${GEN_MAX_TOKENS:-1024}" in text


def test_generation_defaults_to_the_sft_ep3_checkpoint():
    assert "merged_ep3" in _text(GEN)


def test_probe_uses_the_freeform_capable_script():
    text = _text(PROBE)
    assert "scripts/probe_judge_format.py" in text
    assert "freeform" in text


def test_training_names_the_judge_config_and_arm():
    text = _text(TRAIN)
    assert "qwen35_judge_grpo" in text
    assert "JUDGE_REWARD_ARM" in text
    assert "REWARD_METRIC" not in text, "judge training must not inherit the generator reward switch"


def test_training_uses_the_qwen35_verl09_environment():
    assert "turing-rl-rl-qwen35" in _text(TRAIN)


def test_training_enables_judge_thinking():
    assert "PERSONA_JUDGE_ENABLE_THINKING=1" in _text(TRAIN)


def test_training_never_re_enables_the_v1_controller():
    """veRL 0.9 V1 ignores the reward config; the yaml pins use_v1=false.

    A launcher override flipping it back on would silently disable the judge reward and
    score every rollout 0 without erroring.
    """
    assert "trainer.use_v1=True" not in _text(TRAIN)
    assert "trainer.use_v1=true" not in _text(TRAIN)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/lancewicki/miniforge3/bin/python -m pytest tests/test_judge_launchers.py -q`
Expected: FAIL with `FileNotFoundError` on `judge_format_probe.sh`

- [ ] **Step 3a: Write the generation launcher**

Create `scripts/slurm/judge_train_gen.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=judge_gen
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/home/lancewicki/projects/turing-rl/logs/judge_gen-%j.out
#SBATCH --partition=a100
#SBATCH --account=rfai
#
# Generate the fake turns for judge training, then build the both-orders pair parquet.
# Sampling matches how the frozen 880 eval pairs were produced, so train and eval pairs
# are distribution-matched.
set -uo pipefail
source "${TURING_RL_CODE_ROOT:?}/scripts/cluster_job_bootstrap.sh"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY
export HF_HOME=/home/lancewicki/data/hf_cache HF_HUB_CACHE=/home/lancewicki/data/hf_cache
export HF_HUB_DISABLE_XET=1 PYTHONUNBUFFERED=1
export TMPDIR=/home/lancewicki/tmp/build PIP_CACHE_DIR=/home/lancewicki/tmp/pip-cache
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR"

REPO=${TURING_RL_WORK_ROOT:?}
cd "$REPO" || exit 2
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
PY=/home/lancewicki/miniconda3/envs/turing-rl-train/bin/python

MERGED_EP3=${MERGED_EP3:-$REPO/checkpoints/sft/qwen35_9b_prism_full_s42_bf16_fsdp_nopack_epochsave/merged_ep3}
SPLIT=${SPLIT:-train}
SLICE_LO=${SLICE_LO:-0.0}
SLICE_HI=${SLICE_HI:-0.1}
LIMIT=${LIMIT:-416}
GEN_NUM=${GEN_NUM:-4}
OUT_DIR=${OUT_DIR:-$REPO/data/prism/judge/iter1}

DATA_BASE=$TURING_RL_INPUT_DATA_ROOT/prism/full_s42_history_sft40_grpo60_test10/grpo
case "$SPLIT" in
  train) SOURCE_PARQUET=${SOURCE_PARQUET:-$DATA_BASE/train.parquet} ;;
  val)   SOURCE_PARQUET=${SOURCE_PARQUET:-$DATA_BASE/val.parquet}; SLICE_LO=0.0; SLICE_HI=1.0; LIMIT=0; GEN_NUM=1 ;;
  *) echo "ERROR: SPLIT must be train or val, got $SPLIT" >&2; exit 2 ;;
esac

# Generation sampling: Qwen3.5 model card = job 13634 val_kwargs = how the 880 eval pairs were made.
export GEN_TEMPERATURE=${GEN_TEMPERATURE:-0.7}
export GEN_TOP_P=${GEN_TOP_P:-0.8}
export GEN_TOP_K=${GEN_TOP_K:-20}
export GEN_MAX_TOKENS=${GEN_MAX_TOKENS:-1024}

PKL=$OUT_DIR/raw/${SPLIT}_generations.pkl
mkdir -p "$OUT_DIR/raw"

echo "=== judge gen: split=$SPLIT slice=[$SLICE_LO,$SLICE_HI) limit=$LIMIT k=$GEN_NUM ==="
echo "=== model=$MERGED_EP3 sampling T=$GEN_TEMPERATURE top_p=$GEN_TOP_P top_k=$GEN_TOP_K ==="

$PY -u -m eval.generate_trained --base_model --model_id "$MERGED_EP3" \
  --test_parquet "$SOURCE_PARQUET" --output "$PKL" --gen_num "$GEN_NUM" \
  --temperature "$GEN_TEMPERATURE" --top_p "$GEN_TOP_P" --top_k "$GEN_TOP_K" \
  --max_tokens "$GEN_MAX_TOKENS" --backend vllm \
  --vllm_max_model_len "${GEN_MAX_MODEL_LEN:-13524}" \
  --vllm_truncate_prompt_tokens "${GEN_TRUNCATE_PROMPT_TOKENS:-12500}" || exit 3

LIMIT_ARG=()
[ "$LIMIT" -gt 0 ] && LIMIT_ARG=(--limit "$LIMIT")
$PY -u scripts/build_judge_train_pairs.py \
  --inference_pkl "$PKL" --source_parquet "$SOURCE_PARQUET" \
  --out "$OUT_DIR/$SPLIT.parquet" \
  --slice_lo "$SLICE_LO" --slice_hi "$SLICE_HI" --split "$SPLIT" "${LIMIT_ARG[@]}"
```

Flags verified against `eval/generate_trained.py`: `--base_model`, `--model_id`,
`--test_parquet`, `--output`, `--gen_num`, `--temperature`, `--top_p`, `--top_k`,
`--max_tokens`, `--backend`, `--vllm_max_model_len` and `--vllm_truncate_prompt_tokens` all
exist. The shape matches `scripts/slurm/generator_infer.sh`.

- [ ] **Step 3b: Write the probe launcher**

Create `scripts/slurm/judge_format_probe.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=judge_probe
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:8
#SBATCH --mem=256G
#SBATCH --time=06:00:00
#SBATCH --output=/home/lancewicki/projects/turing-rl/logs/judge_probe-%j.out
#SBATCH --partition=a100
#SBATCH --account=rfai
#
# Phase 0 gate: serve one candidate judge and probe it in three decoding regimes.
# The freeform arm is the one that matters -- it is the only regime that matches what
# veRL rollouts will actually produce.
set -uo pipefail
source "${TURING_RL_CODE_ROOT:?}/scripts/cluster_job_bootstrap.sh"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY
export HF_HOME=/home/lancewicki/data/hf_cache HF_HUB_CACHE=/home/lancewicki/data/hf_cache
export HF_HUB_DISABLE_XET=1 PYTHONUNBUFFERED=1

REPO=${TURING_RL_WORK_ROOT:?}
cd "$REPO" || exit 2
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
PY=/home/lancewicki/miniconda3/envs/turing-rl-train/bin/python

JUDGE_MODEL=${JUDGE_MODEL:?set JUDGE_MODEL, e.g. Qwen/Qwen3.5-4B}
PAIRS=${PAIRS:-$REPO/data/prism/judge/iter1/val.parquet}
OUT_JSON=${OUT_JSON:-$REPO/results/judge-format-probe/$(basename "$JUDGE_MODEL").json}
LIMIT=${LIMIT:-200}
REGIMES=${REGIMES:-"json_schema json_object freeform"}

export PERSONA_JUDGE_SAMPLING='{"repetition_penalty":1.1,"temperature":0.6}'
export PERSONA_JUDGE_ENABLE_THINKING=1
export PERSONA_JUDGE_MAX_COMPLETION_TOKENS=8192
export PERSONA_OPENAI_TIMEOUT_SECONDS=1800

# Serve the candidate judge, wait for its endpoint file (written only after model-verified
# health -- see judge_serve_9b_replicas.sh), then point the OpenAI-compatible probe client
# at it. Without this, resolve_judge_api_key()/get_openai_api_base() fall through to the
# real OpenAI endpoint instead of our vLLM server. Serving shape follows
# configs/judge_sweep_cells.py: <=30GB footprint -> TP=1 with 8 replicas.
ENDPOINT_FILE=${JUDGE_ENDPOINT_FILE:-$REPO/logs/judge_probe_endpoint-${SLURM_JOB_ID}.txt}
rm -f "$ENDPOINT_FILE"
MODEL=$JUDGE_MODEL JUDGE_ENDPOINT_FILE=$ENDPOINT_FILE \
  bash "$REPO/scripts/slurm/judge_serve_9b_replicas.sh" &
SERVE_PID=$!
trap 'kill $SERVE_PID 2>/dev/null || true' EXIT

echo "waiting for judge endpoint (up to 60 min warmup)..."
ok=0
for t in $(seq 1 1800); do
  [ -s "$ENDPOINT_FILE" ] && { ok=1; break; }
  kill -0 "$SERVE_PID" 2>/dev/null || { echo "ERROR: judge serve step died before publishing endpoint" >&2; exit 3; }
  sleep 2
done
[ $ok -eq 1 ] || { echo "TIMEOUT waiting for judge endpoint" >&2; exit 4; }
export OPENAI_API_BASE=$(cat "$ENDPOINT_FILE")
echo "judge endpoint: $OPENAI_API_BASE"

# shellcheck disable=SC2086
$PY -u scripts/probe_judge_format.py \
  --pairs_parquet "$PAIRS" --model "$JUDGE_MODEL" \
  --out_json "$OUT_JSON" --limit "$LIMIT" --regimes $REGIMES
```

- [ ] **Step 3c: Write the training launcher**

Create `scripts/slurm/judge_grpo_train.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=judge_grpo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --mem=512G
#SBATCH --time=3-00:00:00
#SBATCH --output=/home/lancewicki/projects/turing-rl/logs/judge_grpo-%j.out
#SBATCH --partition=a100
#SBATCH --account=rfai
#
# Judge-only GRPO. No judge server: the reward is local and label-verifiable, so unlike
# the generator runs there is nothing to serve and nothing to tear down.
set -uo pipefail
source "${TURING_RL_CODE_ROOT:?}/scripts/cluster_job_bootstrap.sh"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY
export HF_HOME=/home/lancewicki/data/hf_cache HF_HUB_CACHE=/home/lancewicki/data/hf_cache
export HF_HUB_DISABLE_XET=1 PYTHONUNBUFFERED=1
export TMPDIR=/home/lancewicki/tmp/build PIP_CACHE_DIR=/home/lancewicki/tmp/pip-cache
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR"

REPO=${TURING_RL_WORK_ROOT:?}
cd "$REPO" || exit 2
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
# Qwen3.5 needs transformers 5.x + veRL 0.9; this is the env both 9B GRPO runs used.
PY=/home/lancewicki/miniconda3/envs/turing-rl-rl-qwen35/bin/python
$PY -c 'import transfer_queue' || {
  echo "ERROR: veRL 0.9 env requires TransferQueue==0.1.8" >&2
  exit 2
}

JUDGE_MODEL_PATH=${JUDGE_MODEL_PATH:?set JUDGE_MODEL_PATH, e.g. Qwen/Qwen3.5-4B}
export JUDGE_REWARD_ARM=${JUDGE_REWARD_ARM:?set JUDGE_REWARD_ARM to directional or graded}
case "$JUDGE_REWARD_ARM" in
  directional|graded) ;;
  *) echo "ERROR: JUDGE_REWARD_ARM must be directional or graded, got $JUDGE_REWARD_ARM" >&2; exit 2 ;;
esac
export JUDGE_TASK_WEIGHT=${JUDGE_TASK_WEIGHT:-0.9}
export JUDGE_FORMAT_WEIGHT=${JUDGE_FORMAT_WEIGHT:-0.1}
# The judge REASONS before answering; this mirrors what generator RL ran against.
export PERSONA_JUDGE_ENABLE_THINKING=1

DATA_DIR=${DATA_DIR:-$REPO/data/prism/judge/iter1}
TRAIN_FILE=${TRAIN_FILE:-$DATA_DIR/train.parquet}
VAL_FILE=${VAL_FILE:-$DATA_DIR/val.parquet}
RUN_TAG=${JUDGE_RUN_TAG:-$(basename "$JUDGE_MODEL_PATH")_${JUDGE_REWARD_ARM}}
CKPT_DIR=${CKPT_DIR:-$REPO/results/grpo/judge/$RUN_TAG/checkpoints}

echo "=== judge GRPO: model=$JUDGE_MODEL_PATH arm=$JUDGE_REWARD_ARM tag=$RUN_TAG ==="
echo "=== train=$TRAIN_FILE val=$VAL_FILE ckpt=$CKPT_DIR host=$(hostname) date=$(date) ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

OVR=(
  actor_rollout_ref.model.path="$JUDGE_MODEL_PATH"
  # TP is the one rollout knob that legitimately varies by model size; everything else
  # (chunked prefill, gpu_memory_utilization, use_v1, reward routing) is pinned in
  # qwen35_judge_grpo.yaml so it cannot be dropped from a submit-time string.
  actor_rollout_ref.rollout.tensor_model_parallel_size=${RL_ROLLOUT_TP:-1}
  actor_rollout_ref.actor.fsdp_config.fsdp_size=${RL_NGPUS:-8}
  actor_rollout_ref.actor.fsdp_config.param_offload=True
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
  actor_rollout_ref.actor.fsdp_config.offload_policy=True
  actor_rollout_ref.ref.fsdp_config.param_offload=True
  actor_rollout_ref.ref.fsdp_config.offload_policy=True
  data.train_files="$TRAIN_FILE"
  data.val_files="$VAL_FILE"
  trainer.default_local_dir="$CKPT_DIR"
  trainer.experiment_name="qwen35-judge-grpo-$RUN_TAG"
  trainer.project_name=grpo-judge
)

$PY -u -m training.grpo.run_verl_main_ppo \
  --config-name qwen35_judge_grpo "${OVR[@]}" ${EXTRA_OVERRIDES:-}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/Users/lancewicki/miniforge3/bin/python -m pytest tests/test_judge_launchers.py -q`
Expected: PASS, 10 tests

- [ ] **Step 5: Commit**

```bash
chmod +x scripts/slurm/judge_format_probe.sh scripts/slurm/judge_train_gen.sh scripts/slurm/judge_grpo_train.sh
git add scripts/slurm/judge_format_probe.sh scripts/slurm/judge_train_gen.sh scripts/slurm/judge_grpo_train.sh tests/test_judge_launchers.py
git commit -m "feat: Slurm launchers for judge probe, pair generation and GRPO"
```

---

## Task 9: Overfit gate

R0 from the spec. A handful of pairs, trained to saturation, to prove the loop learns
before anything expensive runs.

**Files:**
- Create: `scripts/build_judge_overfit.py`
- Create: `scripts/judge_overfit_gate.py`
- Test: `tests/test_judge_overfit.py`

**Interfaces:**
- Consumes: the Task 2 judge parquet
- Produces: `build_judge_overfit(src: str, out: str, n_pairs: int = 8) -> pd.DataFrame`; `read_metric_series(jsonl_path: str, key: str) -> list[float]`; `gate_verdict(series: list[float], *, threshold: float, tail: int) -> dict`

`n_pairs` counts *pairs*, and each contributes both orders, so the row count is `2 * n_pairs`
and the human side stays exactly balanced — an overfit subset that lost that balance would
let the model win by always answering one slot.

- [ ] **Step 1: Write the failing test**

Create `tests/test_judge_overfit.py`:

```python
"""Unit tests for the judge overfit subset builder and the gate check."""

import json

import pandas as pd
import pytest

from scripts.build_judge_overfit import build_judge_overfit
from scripts.judge_overfit_gate import gate_verdict, read_metric_series


def _judge_parquet(tmp_path, n_pairs: int = 10):
    rows = []
    for i in range(n_pairs):
        for order, human_is_b in (("human_a", False), ("human_b", True)):
            rows.append(
                {
                    "data_source": "prism_judge",
                    "prompt": [{"role": "user", "content": f"prompt {i}"}],
                    "reward_model": {"style": "rule", "ground_truth": "B" if human_is_b else "A"},
                    "extra_info": {
                        "row_id": f"p{i}::{order}",
                        "pair_id": f"p{i}",
                        "order": order,
                        "human_is_b": human_is_b,
                    },
                }
            )
    path = tmp_path / "train.parquet"
    pd.DataFrame(rows).to_parquet(path)
    return str(path)


def test_overfit_subset_keeps_both_orders_of_each_pair(tmp_path):
    out = str(tmp_path / "overfit.parquet")
    df = build_judge_overfit(_judge_parquet(tmp_path), out, n_pairs=3)
    assert len(df) == 6
    assert df["extra_info"].map(lambda e: e["pair_id"]).nunique() == 3


def test_overfit_subset_stays_balanced(tmp_path):
    out = str(tmp_path / "overfit.parquet")
    df = build_judge_overfit(_judge_parquet(tmp_path), out, n_pairs=4)
    sides = [e["human_is_b"] for e in df["extra_info"]]
    assert sum(sides) * 2 == len(sides)


def test_overfit_subset_is_written_to_disk(tmp_path):
    out = str(tmp_path / "overfit.parquet")
    build_judge_overfit(_judge_parquet(tmp_path), out, n_pairs=2)
    assert len(pd.read_parquet(out)) == 4


def test_requesting_more_pairs_than_exist_raises(tmp_path):
    with pytest.raises(ValueError):
        build_judge_overfit(_judge_parquet(tmp_path, n_pairs=2), str(tmp_path / "o.parquet"), n_pairs=99)


def test_read_metric_series_pulls_one_key(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_text(
        "\n".join(json.dumps({"step": s, "judge_acc": s / 10}) for s in range(5)) + "\n"
    )
    assert read_metric_series(str(path), "judge_acc") == [0.0, 0.1, 0.2, 0.3, 0.4]


def test_read_metric_series_skips_rows_missing_the_key(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_text(json.dumps({"step": 0}) + "\n" + json.dumps({"judge_acc": 1.0}) + "\n")
    assert read_metric_series(str(path), "judge_acc") == [1.0]


def test_gate_passes_when_the_tail_saturates():
    verdict = gate_verdict([0.5, 0.6, 0.9, 0.98, 1.0], threshold=0.95, tail=2)
    assert verdict["passed"] is True
    assert verdict["tail_mean"] == pytest.approx(0.99)


def test_gate_fails_when_the_tail_stays_low():
    verdict = gate_verdict([0.5, 0.5, 0.52, 0.51], threshold=0.95, tail=2)
    assert verdict["passed"] is False


def test_gate_fails_on_an_empty_series():
    assert gate_verdict([], threshold=0.95, tail=2)["passed"] is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/lancewicki/miniforge3/bin/python -m pytest tests/test_judge_overfit.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.build_judge_overfit'`

- [ ] **Step 3a: Write the subset builder**

Create `scripts/build_judge_overfit.py`:

```python
"""Cut a tiny, side-balanced overfit subset from the judge training parquet.

Selection is by *pair*, keeping both A/B orders together, so the human side stays exactly
balanced. An unbalanced overfit set would let the model saturate by always answering one
slot, which would pass the gate while proving nothing.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd


def build_judge_overfit(src: str, out: str, n_pairs: int = 8) -> pd.DataFrame:
    df = pd.read_parquet(src)
    required = ["data_source", "prompt", "reward_model", "extra_info"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"missing veRL columns: {missing}")

    pair_ids = list(dict.fromkeys(e["pair_id"] for e in df["extra_info"]))
    if n_pairs > len(pair_ids):
        raise ValueError(f"requested {n_pairs} pairs but only {len(pair_ids)} exist in {src}")
    keep = set(pair_ids[:n_pairs])

    subset = df.loc[df["extra_info"].map(lambda e: e["pair_id"] in keep)].reset_index(drop=True)
    sides = [e["human_is_b"] for e in subset["extra_info"]]
    if sum(sides) * 2 != len(sides):
        raise ValueError("overfit subset is not side-balanced; both orders must be present")

    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    subset.to_parquet(out, index=False)
    return subset


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the judge overfit subset")
    parser.add_argument("--src", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n_pairs", type=int, default=8)
    args = parser.parse_args()
    subset = build_judge_overfit(args.src, args.out, args.n_pairs)
    print(f"wrote {len(subset)} rows ({args.n_pairs} pairs x 2 orders) -> {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3b: Write the gate check**

Create `scripts/judge_overfit_gate.py`:

```python
"""Decide whether the judge overfit run actually learned.

Reads a veRL metrics JSONL and checks that the tail of the training-accuracy series has
saturated. A flat series near 0.5 means the loop is wired but learning nothing; a flat
series near the tie payoff means the model found the hedge instead of the signal.
"""

from __future__ import annotations

import argparse
import json


def read_metric_series(jsonl_path: str, key: str) -> list[float]:
    """Pull one metric out of a veRL metrics JSONL, skipping rows that lack it."""
    series: list[float] = []
    with open(jsonl_path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict) and key in record:
                series.append(float(record[key]))
    return series


def gate_verdict(series: list[float], *, threshold: float, tail: int) -> dict:
    """Pass when the mean of the last ``tail`` points clears ``threshold``."""
    if not series:
        return {"passed": False, "reason": "no metric points found", "tail_mean": 0.0, "n": 0}
    window = series[-tail:] if tail > 0 else series
    tail_mean = sum(window) / len(window)
    passed = tail_mean >= threshold
    return {
        "passed": passed,
        "reason": "saturated" if passed else f"tail_mean {tail_mean:.4f} < {threshold}",
        "tail_mean": tail_mean,
        "n": len(series),
        "first": series[0],
        "last": series[-1],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Judge overfit gate check")
    parser.add_argument("--metrics_jsonl", required=True)
    parser.add_argument("--key", default="judge_acc")
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument("--tail", type=int, default=3)
    args = parser.parse_args()

    verdict = gate_verdict(
        read_metric_series(args.metrics_jsonl, args.key),
        threshold=args.threshold,
        tail=args.tail,
    )
    print(json.dumps(verdict, indent=2, sort_keys=True))
    return 0 if verdict["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/Users/lancewicki/miniforge3/bin/python -m pytest tests/test_judge_overfit.py -q`
Expected: PASS, 9 tests

- [ ] **Step 5: Commit**

```bash
git add scripts/build_judge_overfit.py scripts/judge_overfit_gate.py tests/test_judge_overfit.py
git commit -m "feat: judge overfit subset builder and gate check"
```

---

## Task 10: Eval analysis

Produces the headline table: accuracy per model, plus the diagnostics that say whether an
accuracy gain is real discrimination or an artifact.

**Files:**
- Create: `scripts/analyze_judge_training.py`
- Test: `tests/test_analyze_judge_training.py`

**Interfaces:**
- Consumes: a long-format CSV with columns `model`, `pair_id`, `order`, `rating`, `human_is_b`
- Produces: `summarize_judge_eval(df: pd.DataFrame) -> pd.DataFrame`; `order_consistency(df: pd.DataFrame) -> pd.DataFrame`

**Why the input is our own CSV and not the old sweep dump.** The existing wide
`judge_rating_pairs.csv` has one `r_<judge>` column per judge and *no side label*, because
the older eval picks the side with `_stable_turing_generated_is_b`, which hashes the
response text into the decision. The side therefore cannot be recovered after the fact.
Scoring every model — trained and zero-shot baseline alike — through the Task 2
both-orders parquet gives an explicit `human_is_b` per row and makes `order_consistency`
computable at all. This is the same conclusion the spec reached in §4 when it required all
baselines to be re-run.

- [ ] **Step 1: Write the failing test**

Create `tests/test_analyze_judge_training.py`:

```python
"""Unit tests for judge eval analysis."""

import pandas as pd
import pytest

from scripts.analyze_judge_training import order_consistency, summarize_judge_eval


def _rows(records):
    return pd.DataFrame(
        [
            {"model": m, "pair_id": p, "order": o, "rating": r, "human_is_b": h}
            for m, p, o, r, h in records
        ]
    )


def test_a_perfect_judge_scores_one():
    df = _rows([("m", "p1", "human_a", 1, False), ("m", "p1", "human_b", 7, True)])
    summary = summarize_judge_eval(df).set_index("model")
    assert summary.loc["m", "accuracy"] == 1.0
    assert summary.loc["m", "tie_rate"] == 0.0
    assert summary.loc["m", "n"] == 2


def test_an_always_tie_judge_scores_half():
    df = _rows([("m", "p1", "human_a", 4, False), ("m", "p1", "human_b", 4, True)])
    summary = summarize_judge_eval(df).set_index("model")
    assert summary.loc["m", "accuracy"] == 0.5
    assert summary.loc["m", "tie_rate"] == 1.0


def test_a_slot_b_biased_judge_is_caught_by_pred_b_rate():
    df = _rows([("m", "p1", "human_a", 7, False), ("m", "p1", "human_b", 7, True)])
    summary = summarize_judge_eval(df).set_index("model")
    assert summary.loc["m", "accuracy"] == 0.5
    assert summary.loc["m", "pred_b_rate"] == 1.0


def test_brier_is_reported():
    df = _rows([("m", "p1", "human_a", 1, False)])
    summary = summarize_judge_eval(df).set_index("model")
    assert summary.loc["m", "brier"] == pytest.approx(0.0)


def test_models_are_summarised_separately():
    df = _rows(
        [
            ("good", "p1", "human_a", 1, False),
            ("good", "p1", "human_b", 7, True),
            ("bad", "p1", "human_a", 7, False),
            ("bad", "p1", "human_b", 1, True),
        ]
    )
    summary = summarize_judge_eval(df).set_index("model")
    assert summary.loc["good", "accuracy"] == 1.0
    assert summary.loc["bad", "accuracy"] == 0.0


def test_order_consistency_is_one_when_both_orders_agree():
    # Both orders name the human: correct in each presentation.
    df = _rows([("m", "p1", "human_a", 1, False), ("m", "p1", "human_b", 7, True)])
    result = order_consistency(df).set_index("model")
    assert result.loc["m", "order_consistency"] == 1.0
    assert result.loc["m", "n_pairs"] == 1


def test_order_consistency_is_zero_for_a_fixed_slot_answer():
    # Always says B regardless of where the human is: inconsistent across orders.
    df = _rows([("m", "p1", "human_a", 7, False), ("m", "p1", "human_b", 7, True)])
    assert order_consistency(df).set_index("model").loc["m", "order_consistency"] == 0.0


def test_order_consistency_ignores_pairs_missing_an_order():
    df = _rows([("m", "p1", "human_a", 1, False)])
    assert order_consistency(df).set_index("model").loc["m", "n_pairs"] == 0


def test_unrecovered_verdicts_do_not_crash_the_summary():
    """probe_judge_format writes rating=None for unrecoverable verdicts -> NaN on CSV."""
    df = _rows([("m", "p1", "human_a", 1, False), ("m", "p1", "human_b", 7, True)])
    df.loc[len(df)] = {
        "model": "m", "pair_id": "p2", "order": "human_a",
        "rating": float("nan"), "human_is_b": False,
    }
    summary = summarize_judge_eval(df).set_index("model")
    assert summary.loc["m", "n"] == 2
    assert summary.loc["m", "accuracy"] == 1.0


def test_summary_reports_how_much_of_the_eval_set_was_unrecoverable():
    """Accuracy over a shrunken, non-random subset flatters a low-compliance model."""
    df = _rows([("m", "p1", "human_a", 1, False)])
    for pair in ("p2", "p3", "p4"):
        df.loc[len(df)] = {
            "model": "m", "pair_id": pair, "order": "human_a",
            "rating": float("nan"), "human_is_b": False,
        }
    summary = summarize_judge_eval(df).set_index("model")
    assert summary.loc["m", "n"] == 1
    assert summary.loc["m", "n_total"] == 4
    assert summary.loc["m", "unrecovered_rate"] == pytest.approx(0.75)
    # The flattering part: perfect accuracy on the quarter it managed to parse.
    assert summary.loc["m", "accuracy"] == 1.0


def test_order_consistency_drops_unrecovered_rows():
    df = _rows([("m", "p1", "human_a", 1, False)])
    df.loc[len(df)] = {
        "model": "m", "pair_id": "p1", "order": "human_b",
        "rating": float("nan"), "human_is_b": True,
    }
    # p1's second order is unrecoverable, so the pair is incomplete, not consistent.
    assert order_consistency(df).set_index("model").loc["m", "n_pairs"] == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/lancewicki/miniforge3/bin/python -m pytest tests/test_analyze_judge_training.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.analyze_judge_training'`

- [ ] **Step 3: Write minimal implementation**

Create `scripts/analyze_judge_training.py`:

```python
"""Summarise judge eval runs: accuracy plus the diagnostics that qualify it.

Accuracy alone cannot distinguish a judge that discriminates from one that always answers
the same slot, or one that hedges every call. pred_b_rate catches the first, tie_rate the
second, and order_consistency catches a judge whose verdict flips when the same pair is
presented the other way round.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

TIE_RATING = 4
REQUIRED_COLUMNS = ("model", "pair_id", "order", "rating", "human_is_b")


def _check_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")


def _row_accuracy(rating: int, human_is_b: bool) -> float:
    if rating == TIE_RATING:
        return 0.5
    return 1.0 if (rating > TIE_RATING) == bool(human_is_b) else 0.0


def _drop_unrecovered(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with no rating.

    probe_judge_format.py::dump_row writes rating=None when a verdict could not be
    recovered, which becomes NaN on the CSV round trip. Those rows carry no signal about
    which side the judge picked, so they are excluded rather than left to blow up
    `int(nan)` downstream.
    """
    return df.dropna(subset=["rating"]).copy()


def summarize_judge_eval(df: pd.DataFrame) -> pd.DataFrame:
    """One row per model: accuracy, ties, slot bias, confidence, Brier, and coverage.

    `unrecovered_rate` is not decoration. Accuracy is computed only over verdicts that
    parsed, and that subset is NOT random — it is the cases the model handled well. A model
    with poor format compliance therefore gets a flattering accuracy over a shrunken
    sample, which is exactly the artifact this table exists to expose. Read accuracy
    together with unrecovered_rate or not at all.
    """
    _check_columns(df)
    n_total = df.groupby("model", sort=True).size()
    work = _drop_unrecovered(df)
    work["_acc"] = [
        _row_accuracy(int(r), bool(h)) for r, h in zip(work["rating"], work["human_is_b"])
    ]
    work["_tie"] = (work["rating"] == TIE_RATING).astype(float)
    work["_pred_b"] = (work["rating"] > TIE_RATING).astype(float)
    p = (work["rating"] - 1) / 6.0
    y = work["human_is_b"].astype(float)
    work["_brier"] = (p - y) ** 2
    work["_conf"] = 2.0 * (p - 0.5).abs()

    grouped = work.groupby("model", sort=True)
    summary = pd.DataFrame(
        {
            "n": grouped.size(),
            "accuracy": grouped["_acc"].mean(),
            "tie_rate": grouped["_tie"].mean(),
            "pred_b_rate": grouped["_pred_b"].mean(),
            "brier": grouped["_brier"].mean(),
            "conf_mean": grouped["_conf"].mean(),
            "rating_mean": grouped["rating"].mean(),
        }
    )
    # Coverage: how much of each model's eval set is actually behind its accuracy.
    summary["n_total"] = n_total.reindex(summary.index).fillna(0).astype(int)
    summary["unrecovered_rate"] = 1.0 - (summary["n"] / summary["n_total"])
    return summary.reset_index()


def order_consistency(df: pd.DataFrame) -> pd.DataFrame:
    """Fraction of pairs whose two presentations name the same side as human.

    Only pairs with both orders present are counted; a pair seen once cannot be
    self-inconsistent. Ties count as disagreement with everything, including another tie,
    because a tie names no side.
    """
    _check_columns(df)
    rows = []
    for model, model_df in df.groupby("model", sort=True):
        consistent = 0
        total = 0
        for _pair_id, pair_df in model_df.groupby("pair_id", sort=True):
            orders = {str(o): (int(r), bool(h)) for o, r, h in
                      zip(pair_df["order"], pair_df["rating"], pair_df["human_is_b"])}
            if "human_a" not in orders or "human_b" not in orders:
                continue
            total += 1
            calls = []
            for order in ("human_a", "human_b"):
                rating, human_is_b = orders[order]
                if rating == TIE_RATING:
                    calls.append(None)
                else:
                    # Did the judge name the slot that actually holds the human?
                    calls.append((rating > TIE_RATING) == human_is_b)
            if calls[0] is not None and calls[0] == calls[1]:
                consistent += 1
        rows.append(
            {
                "model": model,
                "n_pairs": total,
                "order_consistency": (consistent / total) if total else 0.0,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarise judge eval results")
    parser.add_argument("--eval_csv", required=True, help="long-format CSV of judge verdicts")
    parser.add_argument("--out_csv", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.eval_csv)
    summary = summarize_judge_eval(df).merge(order_consistency(df), on="model", how="left")
    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    summary.to_csv(args.out_csv, index=False)
    print(summary.to_markdown(index=False))
    print(f"\nWrote {args.out_csv}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/Users/lancewicki/miniforge3/bin/python -m pytest tests/test_analyze_judge_training.py -q`
Expected: PASS, 12 tests

- [ ] **Step 5: Run the full suite one last time**

Run: `/Users/lancewicki/miniforge3/bin/python -m pytest tests/ -q`
Expected: PASS, all tests

- [ ] **Step 6: Commit**

```bash
git add scripts/analyze_judge_training.py tests/test_analyze_judge_training.py
git commit -m "feat: judge eval analysis with order-consistency and slot-bias diagnostics"
```

---

## Execution order and gates

Tasks 1–7 and 9–10 are Mac-local and CPU-only. Only these steps touch the cluster, and
each is gated on the previous one:

1. **After Task 8**, commit, then generate the training and val pairs
   (`SPLIT=train` then `SPLIT=val` through `judge_train_gen.sh`). Check the emitted
   `.meta.json`: `human_is_b_rate` must be exactly 0.5, `n_contexts` must be 416.
2. **Run the Phase 0 probe** for 2B / 4B / 9B. **Gate:** freeform `fmt_all_fields_rate`
   around 0.5 or better. If a model comes in near zero, stop. The remedy is the spec's
   self-distilled format-SFT fallback (§7), which is *not* implemented by this plan — it is
   contingency work and needs its own plan written against whatever the probe actually shows.
3. **Run R0**, the overfit gate, on the 4B. **Gate:** `judge_overfit_gate.py` exits 0.
4. **Run R1**: {2B, 4B, 9B} × {directional, graded}. Watch
   `judge_group/all_equal_rate` from the first steps — if it is high, the groups are
   degenerate and DAPO-style dynamic sampling is needed before the results mean anything.
   Watch `judge_tie` for the hedging collapse.
5. **Score every model** — trained and zero-shot baselines — through the Task 2
   both-orders parquet built from the 880-pair heldout set, then run Task 10's analysis.

Per repo policy: run the `preflight-job-check` skill before every submission, keep
concurrency under ~10 jobs, and never `scancel` a job this session did not submit.

---

## Plan self-review

Recorded so the implementer knows what was and was not verified when this was written.

**Every Python block in this plan was extracted and executed before the plan was
committed.** All 93 tests across tasks 1–7, 9 and 10 pass, and the per-task counts stated in
each "Expected: PASS, N tests" line are the counts actually observed: slice 9, pairs 8,
verdict 16, reward 16, metric patch 7, config 8, probe 12, overfit 9, analysis 8. The
implementer is still expected to do the TDD cycle — the point of running it here was to make
sure they are not debugging *my* typos. Task 8's launcher test is the exception: it asserts
against shell scripts, so it can only pass once those files exist.

Four defects were found and fixed this way:

1. `shared.judge_utils._extract_json` calls `json.loads` bare, so it raises on non-JSON
   rather than returning `None`, and it demands the whole string be one object. Replaced with
   a local tolerant extractor in `judge_verdict.py`.
2. `defaultdict` is not imported in `verl_metric_patch.py`; the plan now says to add it
   rather than assuming it is there.
3. `post_chat_async(session, payload, *, semaphore, ...)` returns only the content string and
   needs a session and semaphore — the probe's original call site was invented and wrong.
4. Literal triple-backticks inside ```python fences silently truncated two code blocks in the
   markdown; those blocks now use four-backtick fences.

**Spec coverage.** Every section of the spec maps to a task: §3.1→1, §3.2→8, §3.3→2,
§3.4→8 and the execution order, §4→7 (`--dump_csv`) and the execution order, §5→3 and 4,
§6→4 and 5, §7→7, §8→9 and the execution order, §10→deferred. Two documented gaps, both
deliberate: `order_consistency` moves from training to eval analysis (see the header), and
the format-SFT fallback is contingency work needing its own plan.

**One coverage gap was found and closed:** nothing originally produced the long-format CSV
that Task 10 consumes. Rather than add an eleventh task, `probe_judge_format.py` gained
`--dump_csv`, so the same script serves as both the Phase 0 gate and the eval driver for
trained models and baselines alike.

**Not verified here, and worth checking on first use:** the `#SBATCH` resource shapes in
Task 8 are estimates, not measurements — particularly the 8-GPU allocation and 3-day walltime
for judge GRPO, and the 300-second serve warmup in the probe launcher.

