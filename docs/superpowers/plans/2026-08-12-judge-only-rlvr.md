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
