# Judge-Model Comparison Experiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the tooling and run the judge-model sweep described in `docs/superpowers/specs/2026-07-08-judge-sweep-design.md`: verify the PRISM split is paper-faithful, generate SFT CoT + train the LoRA SFT generator, produce 880 (human, generated) pairs from the heldout split, then sweep 5 judge sizes × 2 thinking modes with `PERSONA_JUDGE_JSON_SCHEMA=1` and produce a metrics table + plots.

**Architecture:** Serial pipeline with parallelizable branches. New tooling under `scripts/` and `scripts/slurm/`. One patch to `training/sft/lora_sft.py` for checkpoint auto-resume. Immutable raw artifacts + regeneratable derived tree under `results/2026-07-08-judge-sweep/`. Reuse upstream code paths (`data/prism/build.py`, `data/sft/build_sft_jsonl.py`, `eval/generate_trained.py`) unmodified where possible.

**Tech Stack:** Python 3.12, vLLM 0.23 (`judge-vllm` env for offline, `turing-rl-train` env for online), TRL SFTTrainer + PEFT LoRA, Slurm on 3-node A100 40GB cluster, pandas/pyarrow for parquet, matplotlib for plots.

**Repo root:** `/storage/home/lancewicki/projects/turing-rl` (referenced as `$REPO` in commands).

**Conda envs:**
- `/home/lancewicki/miniconda3/envs/judge-vllm/bin/python` — vllm 0.23, no pandas.
- `/home/lancewicki/miniconda3/envs/turing-rl-train/bin/python` — pandas, torch, training stack, vllm 0.18.

**Key paths (must exist before starting):**
- Spec: `docs/superpowers/specs/2026-07-08-judge-sweep-design.md`
- Split parquets: `data/prism/full_s42_history_sft40_grpo60_test10/{sft/train,grpo/train,grpo/val,test}.parquet`
- HF cache: `/home/lancewicki/data/hf_cache/models--Qwen--Qwen3-8B` (base) and `models--Qwen--Qwen3.5-397B-A17B-GPTQ-Int4` (anchor).

**Commit style** (from recent git log): lower-case, short, imperative; no scope prefixes like `feat:`.

---

## Task 1: Add split-verification helper (raw-PRISM loader)

**Files:**
- Create: `tests/prism_verification_helpers.py`

- [ ] **Step 1: Write the file**

```python
"""Helpers for tests/test_prism_split_verification.py.

Loads the raw HuggingFace PRISM dataset (cached at
/home/lancewicki/data/hf_cache/datasets--HannahRoseKirk--prism-alignment)
and builds (user_id, conversation_id, turn_idx) -> raw_reply lookups
used by verification test 6.
"""
from __future__ import annotations

import os
from typing import Any

# Force offline; the raw dataset is already cached.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HOME", "/home/lancewicki/data/hf_cache")
os.environ.setdefault("HF_DATASETS_CACHE", "/home/lancewicki/data/hf_cache/datasets")


def load_raw_prism_replies() -> dict[tuple[str, str, int], str]:
    """Return {(user_id, conversation_id, turn_idx): reply_text} for every [HUMAN] turn.

    turn_idx counts [HUMAN] turns within a conversation, starting at 0.
    """
    from datasets import load_dataset

    ds = load_dataset("HannahRoseKirk/prism-alignment", "conversations", split="train")
    out: dict[tuple[str, str, int], str] = {}
    for row in ds:
        user_id = str(row["user_id"])
        conversation_id = str(row["conversation_id"])
        turn_idx = 0
        for turn in row.get("conversation_turns", []) or []:
            if str(turn.get("role") or "").lower() != "user":
                continue
            reply = str(turn.get("content") or "")
            out[(user_id, conversation_id, turn_idx)] = reply
            turn_idx += 1
    return out


def extra_info_key(extra_info: dict[str, Any]) -> tuple[str, str, int]:
    """Build the raw-PRISM lookup key from a row's extra_info."""
    return (
        str(extra_info["raw_user_id"]),
        str(extra_info["post_id"]),
        int(extra_info["target_idx"]),
    )
```

- [ ] **Step 2: Sanity-run it to confirm the raw dataset loads**

Run: `/home/lancewicki/miniconda3/envs/turing-rl-train/bin/python -c "from tests.prism_verification_helpers import load_raw_prism_replies; d = load_raw_prism_replies(); print('replies:', len(d)); print('sample keys:', list(d.keys())[:3])"`
Expected: prints a count (~thousands) and 3 sample keys. If it fails with a KeyError on `conversation_turns`, inspect one raw row to find the actual field name.

- [ ] **Step 3: Commit**

```bash
git add tests/prism_verification_helpers.py
git commit -m "add prism-raw-reply helper for split verification"
```

---

## Task 2: PRISM split verification pytest (7 checks)

**Files:**
- Create: `tests/test_prism_split_verification.py`

- [ ] **Step 1: Write the test file**

```python
"""Row-level verification of the paper-faithful PRISM split.

Runs against data/prism/full_s42_history_sft40_grpo60_test10/.
Complements tests/test_prism_split_determinism.py (which only checks
counts and byte-determinism).
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import pandas as pd
import pytest

from tests.prism_verification_helpers import extra_info_key, load_raw_prism_replies

REPO_ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = REPO_ROOT / "data" / "prism" / "full_s42_history_sft40_grpo60_test10"

EXPECTED_COUNTS = {
    "sft/train.parquet":  {"rows": 3272, "users": 464},
    "grpo/train.parquet": {"rows": 4174, "users": 696},
    "grpo/val.parquet":   {"rows": 705,  "users": 696},
    "test.parquet":       {"rows": 880,  "users": 128},
}


def _load(relpath: str) -> pd.DataFrame:
    p = SPLIT_DIR / relpath
    if not p.exists():
        pytest.skip(f"missing {p}; rebuild the split first")
    return pd.read_parquet(p)


def _users(df: pd.DataFrame) -> set[str]:
    return {str(e["user_id"]) for e in df["extra_info"]}


def test_1_files_exist():
    for name in list(EXPECTED_COUNTS) + ["split_metadata.json"]:
        assert (SPLIT_DIR / name).exists(), f"missing {name}"


@pytest.mark.parametrize("relpath,expected", list(EXPECTED_COUNTS.items()))
def test_2_row_counts_match(relpath: str, expected: dict[str, int]):
    df = _load(relpath)
    assert len(df) == expected["rows"], f"{relpath}: rows {len(df)} != {expected['rows']}"


@pytest.mark.parametrize("relpath,expected", list(EXPECTED_COUNTS.items()))
def test_3_user_counts_match(relpath: str, expected: dict[str, int]):
    df = _load(relpath)
    got = len(_users(df))
    assert got == expected["users"], f"{relpath}: users {got} != {expected['users']}"


def test_4_user_disjointness_row_level():
    sft = _users(_load("sft/train.parquet"))
    grpo = _users(_load("grpo/train.parquet"))
    heldout = _users(_load("test.parquet"))
    assert sft & grpo == set(), f"sft ∩ grpo not empty: {list(sft & grpo)[:5]}"
    assert sft & heldout == set(), f"sft ∩ heldout not empty: {list(sft & heldout)[:5]}"
    assert grpo & heldout == set(), f"grpo ∩ heldout not empty: {list(grpo & heldout)[:5]}"


@pytest.mark.parametrize("relpath", list(EXPECTED_COUNTS))
def test_5_prompt_schema_wellformed(relpath: str):
    df = _load(relpath)
    rng = random.Random(42)
    idxs = rng.sample(range(len(df)), min(50, len(df)))
    for i in idxs:
        row = df.iloc[i]
        prompt = row["prompt"]
        assert isinstance(prompt, (list, tuple)) or hasattr(prompt, "__iter__"), \
            f"{relpath}[{i}] prompt not a list"
        prompt_list = list(prompt)
        assert len(prompt_list) > 0, f"{relpath}[{i}] empty prompt"
        for msg in prompt_list:
            assert isinstance(msg, dict) or hasattr(msg, "keys"), \
                f"{relpath}[{i}] prompt msg not dict"
            assert "role" in msg and "content" in msg, \
                f"{relpath}[{i}] prompt msg missing role/content"
        rm = row["reward_model"]
        gt = str(rm["ground_truth"])
        assert gt.strip(), f"{relpath}[{i}] empty ground_truth"
        extra = row["extra_info"]
        for k in ("user_id", "post_id", "target_idx", "user_history", "context"):
            assert k in extra, f"{relpath}[{i}] extra_info missing {k}"
        assert row["data_source"] == "prism_alignment_user_sim", \
            f"{relpath}[{i}] data_source {row['data_source']!r}"


def test_6_heldout_groundtruth_matches_raw_prism():
    df = _load("test.parquet")
    raw = load_raw_prism_replies()
    rng = random.Random(42)
    idxs = rng.sample(range(len(df)), 20)
    mismatches: list[str] = []
    for i in idxs:
        row = df.iloc[i]
        try:
            key = extra_info_key(dict(row["extra_info"]))
        except (KeyError, TypeError) as exc:
            mismatches.append(f"row {i}: bad extra_info ({exc})")
            continue
        raw_reply = raw.get(key)
        got = str(row["reward_model"]["ground_truth"])
        if raw_reply is None:
            mismatches.append(f"row {i} key {key} not in raw PRISM")
        elif raw_reply != got:
            mismatches.append(f"row {i} key {key}: raw != split (raw[:80]={raw_reply[:80]!r})")
    assert not mismatches, "heldout ground_truth mismatches:\n" + "\n".join(mismatches)


def test_7_no_text_leak_heldout_from_sft_targets():
    sft = _load("sft/train.parquet")
    heldout = _load("test.parquet")
    sft_targets = {str(rm["ground_truth"]) for rm in sft["reward_model"]}
    long_targets = {t for t in sft_targets if len(t) >= 60}
    rng = random.Random(42)
    idxs = rng.sample(range(len(heldout)), 20)
    leaks: list[str] = []
    for i in idxs:
        row = heldout.iloc[i]
        gt = str(row["reward_model"]["ground_truth"])
        for t in long_targets:
            if t in gt:
                leaks.append(f"heldout[{i}] contains SFT target verbatim: {t[:80]!r}")
                break
    assert not leaks, "text-leak candidates:\n" + "\n".join(leaks)
```

- [ ] **Step 2: Run the tests**

Run: `cd /storage/home/lancewicki/projects/turing-rl && /home/lancewicki/miniconda3/envs/turing-rl-train/bin/python -m pytest tests/test_prism_split_verification.py -v`
Expected: all tests pass. If test_6 fails on the raw-lookup key structure, inspect `raw` dict keys vs `extra_info_key` and adjust the helper.

- [ ] **Step 3: Commit**

```bash
git add tests/test_prism_split_verification.py
git commit -m "add prism split row-level verification"
```

---

## Task 3: Byte-diff verification orchestrator

**Files:**
- Create: `scripts/verify_prism_split.sh`

- [ ] **Step 1: Write the orchestrator**

```bash
#!/bin/bash
# Rebuild PRISM split in a temp dir using upstream-unmodified code and byte-diff
# against the current data/prism/full_s42_history_sft40_grpo60_test10/ parquets.
# Then run the pytest verification suite. Runs on the login pod (no GPU).

set -uo pipefail

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY

REPO=/home/lancewicki/projects/turing-rl
PY=/home/lancewicki/miniconda3/envs/turing-rl-train/bin/python
TMP=$(mktemp -d /tmp/prism-verify.XXXXXX)
CURRENT_BUILD=$REPO/data/prism/full_s42_history
CURRENT_SPLIT=$REPO/data/prism/full_s42_history_sft40_grpo60_test10
FRESH_BUILD=$TMP/full_s42_history
FRESH_SPLIT=$TMP/full_s42_history_sft40_grpo60_test10
OUT_DIR=$REPO/results/2026-07-08-judge-sweep/derived
mkdir -p "$FRESH_BUILD" "$FRESH_SPLIT" "$OUT_DIR"

export HF_HOME=/home/lancewicki/data/hf_cache
export HF_HUB_CACHE=/home/lancewicki/data/hf_cache
export HF_DATASETS_CACHE=/home/lancewicki/data/hf_cache/datasets
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

echo "=== rebuild PRISM (upstream data.prism.build, same config as build_prism_full_s42.sh) ==="
cd "$REPO"
$PY -u -m data.prism.build \
    --output      "$FRESH_BUILD/train.parquet" \
    --val_output  "$FRESH_BUILD/val.parquet" \
    --test_output "$FRESH_BUILD/test.parquet" \
    --conditioning_mode history \
    --shuffle_rows || { echo "build failed"; exit 2; }

echo "=== re-split (upstream data.prism.split_data, default fracs from spec) ==="
$PY -u -m data.prism.split_data \
    --input-dir  "$FRESH_BUILD" \
    --output-dir "$FRESH_SPLIT" || { echo "split failed"; exit 2; }

echo "=== byte-diff current vs fresh ==="
STATUS=0
REPORT=$OUT_DIR/split_verification.md
{
    echo "# PRISM split verification report"
    echo ""
    echo "Date: $(date -Iseconds)"
    echo "Fresh rebuild in: \`$TMP\`"
    echo ""
    echo "## Byte-diff (current vs fresh)"
    echo ""
    echo '| File | current sha256 | fresh sha256 | match |'
    echo '|---|---|---|---|'
} > "$REPORT"

for f in train.parquet val.parquet test.parquet \
         sft/train.parquet grpo/train.parquet grpo/val.parquet test.parquet; do
    case "$f" in
        train.parquet|val.parquet|test.parquet)
            cur="$CURRENT_BUILD/$f"; fresh="$FRESH_BUILD/$f" ;;
        *)
            cur="$CURRENT_SPLIT/$f"; fresh="$FRESH_SPLIT/$f" ;;
    esac
    if [ ! -f "$cur" ] || [ ! -f "$fresh" ]; then
        echo "| $f | MISSING | MISSING | ❌ |" >> "$REPORT"
        STATUS=1
        continue
    fi
    cur_sum=$(sha256sum "$cur" | cut -d' ' -f1)
    fresh_sum=$(sha256sum "$fresh" | cut -d' ' -f1)
    if [ "$cur_sum" = "$fresh_sum" ]; then
        echo "| $f | \`${cur_sum:0:12}\` | \`${fresh_sum:0:12}\` | ✅ |" >> "$REPORT"
    else
        echo "| $f | \`${cur_sum:0:12}\` | \`${fresh_sum:0:12}\` | ❌ |" >> "$REPORT"
        STATUS=1
    fi
done

echo "" >> "$REPORT"
echo "## Pytest verification suite" >> "$REPORT"
echo "" >> "$REPORT"
if $PY -m pytest tests/test_prism_split_verification.py -v > "$OUT_DIR/split_verification_pytest.out" 2>&1; then
    echo "All 7 checks passed. See \`split_verification_pytest.out\`." >> "$REPORT"
else
    echo "Failures. See \`split_verification_pytest.out\`." >> "$REPORT"
    STATUS=1
fi

echo ""
echo "=== report: $REPORT ==="
cat "$REPORT"

rm -rf "$TMP"
exit $STATUS
```

- [ ] **Step 2: Make it executable**

Run: `chmod +x /storage/home/lancewicki/projects/turing-rl/scripts/verify_prism_split.sh`

- [ ] **Step 3: Run it**

Run: `bash /storage/home/lancewicki/projects/turing-rl/scripts/verify_prism_split.sh`
Expected: exits 0, prints report with all ✅ and pytest passing. If byte-diff shows mismatches, inspect the diff and rebuild from the fresh copy before proceeding.

- [ ] **Step 4: Commit**

```bash
git add scripts/verify_prism_split.sh
git commit -m "add prism split byte-diff verification orchestrator"
```

---

## Task 4: Patch lora_sft.py with --resume_from_checkpoint auto

**Files:**
- Modify: `training/sft/lora_sft.py`
- Modify: `our_patches.md`

- [ ] **Step 1: Add the CLI argument to lora_sft.py**

Find the last `parser.add_argument(...)` block (around line 240 — `--exit_after_trainer_build`) and add this argument immediately after it:

```python
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Resume LoRA training from a saved checkpoint. Pass 'auto' to "
            "auto-detect the highest-numbered checkpoint under output_dir, "
            "or pass an explicit checkpoint dir path. If unset or if 'auto' "
            "finds no checkpoints, training starts fresh."
        ),
    )
```

- [ ] **Step 2: Wire it into trainer.train()**

Find the `trainer.train()` call (around line 413) and replace it plus the surrounding logging with:

```python
    resume_arg: str | bool | None = None
    if args.resume_from_checkpoint == "auto":
        import glob as _glob
        candidates = sorted(
            _glob.glob(os.path.join(output_dir, "checkpoint-*")),
            key=lambda p: int(p.rsplit("-", 1)[-1]) if p.rsplit("-", 1)[-1].isdigit() else -1,
        )
        if candidates:
            resume_arg = candidates[-1]
            log(f"Resuming from checkpoint: {resume_arg}")
        else:
            log("--resume_from_checkpoint=auto: no checkpoints found, starting fresh")
    elif args.resume_from_checkpoint:
        resume_arg = args.resume_from_checkpoint
        log(f"Resuming from checkpoint: {resume_arg}")

    if resume_arg is not None:
        trainer.train(resume_from_checkpoint=resume_arg)
    else:
        trainer.train()
```

- [ ] **Step 3: Sanity check the file compiles**

Run: `/home/lancewicki/miniconda3/envs/turing-rl-train/bin/python -c "import ast; ast.parse(open('/storage/home/lancewicki/projects/turing-rl/training/sft/lora_sft.py').read())"`
Expected: no output (successful parse).

- [ ] **Step 4: Sanity check the CLI arg is exposed**

Run: `cd /storage/home/lancewicki/projects/turing-rl && /home/lancewicki/miniconda3/envs/turing-rl-train/bin/python -m training.sft.lora_sft --help 2>&1 | grep -A2 resume_from_checkpoint`
Expected: prints the `--resume_from_checkpoint` help text.

- [ ] **Step 5: Document in our_patches.md**

Append this section to `/storage/home/lancewicki/projects/turing-rl/our_patches.md`:

```markdown

---

## PERSISTENT: `training/sft/lora_sft.py` — `--resume_from_checkpoint auto`

- **Original**: `trainer.train()` called unconditionally; no CLI resume support.
- **Patched**: added `--resume_from_checkpoint` CLI arg. When set to `auto`,
  globs `${output_dir}/checkpoint-*`, picks the highest step, and passes it
  to `trainer.train(resume_from_checkpoint=path)`. When set to a path, uses
  that path directly. When unset or when `auto` finds no checkpoints, starts
  fresh (upstream behavior).
- **Why**: paper-faithful SFT is ~12-16h on 8× A100 40GB. Upstream saves
  epoch checkpoints (`save_strategy="epoch"`) but the launcher can't resume
  from them. Adding auto-resume turns a mid-training crash from "restart from
  scratch, lose 12h" into "restart from latest epoch, lose ≤5h". Zero
  fidelity impact: happy-path is unchanged (resume_arg is None when nothing
  to resume, so `trainer.train()` runs as upstream).
- **Where the flag is set**: `scripts/slurm/sft_full.sh` always passes
  `--resume_from_checkpoint auto`. First launch: no checkpoints → fresh.
  Re-launch after crash: highest checkpoint-N → resumed.
- **Reverted**: no. Persistent; opt-in via CLI flag, no-op when unset.
```

- [ ] **Step 6: Commit**

```bash
git add training/sft/lora_sft.py our_patches.md
git commit -m "add --resume_from_checkpoint auto to lora_sft.py"
```

---

## Task 5: Batched offline CoT generation script

**Files:**
- Create: `scripts/generate_cot_batched.py`

- [ ] **Step 1: Write the script**

```python
"""Batched offline CoT generation via vLLM LLM.generate.

Paper-faithful (Section 4.1 of the design spec): Qwen3-8B thinking-off,
sampling from Qwen3 model-card thinking-off defaults, chat template with
enable_thinking=False. Preserves the leakage-guard + regen loop from
data/sft/generate_cot.py (which uses HTTP, whereas this uses in-process
vLLM for throughput).

Usage:
  python scripts/generate_cot_batched.py \
    --input data/prism/full_s42_history_sft40_grpo60_test10/sft/train.parquet \
    --output data/sft/prism_full_s42_sft_cot.parquet \
    --tensor_parallel_size 1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from data.sft.generate_cot import (
    RATIONALIZE_SYSTEM_PROMPT,
    RATIONALIZE_USER_TEMPLATE,
    REGEN_NUDGE,
    THINKING_TRACE_SOURCE,
    _as_text,
    _row_context,
    reasoning_leaks_reply,
)

MODEL_ID = "Qwen/Qwen3-8B"
SAMPLING = dict(temperature=0.7, top_p=0.8, top_k=20, min_p=0.0, max_tokens=4096)
DEFAULT_MAX_REGEN_ATTEMPTS = 10
LEAKAGE_NGRAM_SIZE = 5
LEAKAGE_MAX_MATCH_TOKENS = 5


def build_messages(extra_info: dict[str, Any], ground_truth: str, *, nudge: bool) -> list[dict[str, str]]:
    msgs = [
        {"role": "system", "content": RATIONALIZE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": RATIONALIZE_USER_TEMPLATE.format(
                context=_row_context(extra_info),
                ground_truth=ground_truth,
            ),
        },
    ]
    if nudge:
        msgs.append({"role": "user", "content": REGEN_NUDGE})
    return msgs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    p.add_argument("--max_model_len", type=int, default=8192)
    p.add_argument("--max_regen_attempts", type=int, default=DEFAULT_MAX_REGEN_ATTEMPTS)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    from vllm import LLM, SamplingParams

    rows = pd.read_parquet(args.input).to_dict(orient="records")
    rows = [dict(r) for r in rows]
    print(f"[cot-batched] loaded {len(rows)} rows from {args.input}", flush=True)

    llm = LLM(
        model=MODEL_ID,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        dtype="bfloat16",
    )
    sampling_params = SamplingParams(**SAMPLING)
    tokenizer = llm.get_tokenizer()

    # Track per-row state: current best reasoning + attempts + leakage flag.
    reasonings: list[str] = [""] * len(rows)
    attempts_used: list[int] = [0] * len(rows)
    leaked_final: list[bool] = [True] * len(rows)

    def render(msgs: list[dict[str, str]]) -> str:
        return tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    t0 = time.time()
    for attempt in range(1, args.max_regen_attempts + 1):
        pending: list[int] = [i for i, ok in enumerate(leaked_final) if ok]
        if not pending:
            break
        print(f"[cot-batched] attempt {attempt}: {len(pending)} pending", flush=True)
        prompts: list[str] = []
        for i in pending:
            row = rows[i]
            extra = dict(row.get("extra_info") or {})
            gt = _as_text((row.get("reward_model") or {}).get("ground_truth"))
            msgs = build_messages(extra, gt, nudge=(attempt > 1))
            prompts.append(render(msgs))
        outputs = llm.generate(prompts, sampling_params)
        for local_i, out in enumerate(outputs):
            i = pending[local_i]
            text = (out.outputs[0].text or "").strip()
            gt = _as_text((rows[i].get("reward_model") or {}).get("ground_truth"))
            leaked = reasoning_leaks_reply(
                text, gt,
                ngram_size=LEAKAGE_NGRAM_SIZE,
                max_match_tokens=LEAKAGE_MAX_MATCH_TOKENS,
            )
            reasonings[i] = text
            attempts_used[i] = attempt
            leaked_final[i] = leaked and bool(text)

    dt = time.time() - t0
    print(f"[cot-batched] generation done in {dt:.1f}s", flush=True)

    # Write back into rows.
    n_empty = 0
    for i, row in enumerate(rows):
        extra = dict(row.get("extra_info") or {})
        extra["ground_truth_reasoning"] = reasonings[i]
        extra["thinking_trace_source"] = THINKING_TRACE_SOURCE
        extra["thinking_trace_model"] = MODEL_ID
        extra["thinking_trace_num_regen_attempts"] = attempts_used[i]
        extra["thinking_trace_failed_leakage_guard"] = bool(leaked_final[i])
        row["extra_info"] = extra
        if not reasonings[i].strip():
            n_empty += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(args.output, index=False)

    meta = {
        "thinking_trace_model": MODEL_ID,
        "sampling": SAMPLING,
        "enable_thinking": False,
        "max_regen_attempts": args.max_regen_attempts,
        "leakage_ngram_size": LEAKAGE_NGRAM_SIZE,
        "leakage_max_match_tokens": LEAKAGE_MAX_MATCH_TOKENS,
        "rows_written": len(rows),
        "rows_failed_leakage_guard": sum(leaked_final),
        "rows_empty_reasoning": n_empty,
        "wall_seconds": dt,
    }
    (args.output.with_suffix(args.output.suffix + ".cot_metadata.json")).write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"[cot-batched] wrote {args.output} and metadata; {n_empty} empty rows, "
          f"{sum(leaked_final)} still leaked", flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Syntax check**

Run: `/home/lancewicki/miniconda3/envs/judge-vllm/bin/python -c "import ast; ast.parse(open('/storage/home/lancewicki/projects/turing-rl/scripts/generate_cot_batched.py').read())"`
Expected: no output.

- [ ] **Step 3: Smoke-run against the 138-row smoke parquet (1 GPU, 5 min)**

You'll need to sbatch this on 1 GPU. Create the ad-hoc srun test:

Run: `srun -A rfai -p a100 --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=00:20:00 --pty bash -c 'unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY; export HF_HOME=/home/lancewicki/data/hf_cache HF_HUB_CACHE=/home/lancewicki/data/hf_cache; /home/lancewicki/miniconda3/envs/judge-vllm/bin/python /storage/home/lancewicki/projects/turing-rl/scripts/generate_cot_batched.py --input /storage/home/lancewicki/projects/turing-rl/data/prism/history_smoke/train.parquet --output /tmp/cot_smoke_check.parquet'`

Expected: writes `/tmp/cot_smoke_check.parquet` and prints "wrote ... 0 empty rows" (or few). Inspect one row:
`/home/lancewicki/miniconda3/envs/turing-rl-train/bin/python -c "import pandas as pd; d = pd.read_parquet('/tmp/cot_smoke_check.parquet'); print(d.iloc[0]['extra_info']['ground_truth_reasoning'][:400])"`
Expected: prints coherent rationalization text, no `<think>` tags.

- [ ] **Step 4: Commit**

```bash
git add scripts/generate_cot_batched.py
git commit -m "add offline batched CoT generation script (thinking-off)"
```

---

## Task 6: CoT parity test (batched vs served, 20 rows)

**Files:**
- Create: `scripts/cot_parity_test.py`

- [ ] **Step 1: Write the parity test**

```python
"""20-row parity test: batched offline CoT vs served-mode CoT.

Both use Qwen3-8B thinking-off (drop --reasoning-parser qwen3, set
enable_thinking=False, sampling T=0.7/top_p=0.8/top_k=20/min_p=0).
Emits a pass/fail report to results/2026-07-08-judge-sweep/derived/.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from data.sft.generate_cot import (
    RATIONALIZE_SYSTEM_PROMPT,
    RATIONALIZE_USER_TEMPLATE,
    _as_text,
    _row_context,
    reasoning_leaks_reply,
)
from scripts.generate_cot_batched import (
    LEAKAGE_MAX_MATCH_TOKENS,
    LEAKAGE_NGRAM_SIZE,
    MODEL_ID,
    SAMPLING,
    build_messages,
)


def run_batched(rows: list[dict], tp: int = 1) -> list[str]:
    from vllm import LLM, SamplingParams
    llm = LLM(model=MODEL_ID, tensor_parallel_size=tp,
              gpu_memory_utilization=0.85, max_model_len=8192, dtype="bfloat16")
    sp = SamplingParams(**SAMPLING)
    tok = llm.get_tokenizer()
    prompts = []
    for row in rows:
        extra = dict(row.get("extra_info") or {})
        gt = _as_text((row.get("reward_model") or {}).get("ground_truth"))
        msgs = build_messages(extra, gt, nudge=False)
        prompts.append(tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        ))
    outs = llm.generate(prompts, sp)
    return [(o.outputs[0].text or "").strip() for o in outs]


def run_served(rows: list[dict], base_url: str) -> list[str]:
    import requests
    results = []
    for row in rows:
        extra = dict(row.get("extra_info") or {})
        gt = _as_text((row.get("reward_model") or {}).get("ground_truth"))
        msgs = build_messages(extra, gt, nudge=False)
        body = {
            "model": MODEL_ID,
            "messages": msgs,
            "temperature": SAMPLING["temperature"],
            "top_p": SAMPLING["top_p"],
            "top_k": SAMPLING["top_k"],
            "max_completion_tokens": SAMPLING["max_tokens"],
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        }
        r = requests.post(f"{base_url}/chat/completions", json=body, timeout=120)
        r.raise_for_status()
        data = r.json()
        text = (data.get("choices", [{}])[0].get("message", {}).get("content") or "").strip()
        results.append(text)
    return results


def perspective(text: str) -> str:
    head = text.lstrip().lower()
    if head.startswith(("i ", "my ", "me ", "i'", "im ", "ive ")):
        return "first"
    if head.startswith(("the user", "they ", "this user", "the human")):
        return "third"
    return "other"


def summarize(name: str, texts: list[str]) -> dict:
    lens = [len(t) for t in texts]
    return {
        "name": name,
        "n": len(texts),
        "empty": sum(1 for t in texts if not t.strip()),
        "len_p25": int(statistics.quantiles(lens, n=4)[0]) if lens else 0,
        "len_p50": int(statistics.median(lens)) if lens else 0,
        "len_p75": int(statistics.quantiles(lens, n=4)[2]) if lens else 0,
        "perspective": {p: sum(1 for t in texts if perspective(t) == p) for p in ("first", "third", "other")},
    }


def leakage_flags(rows: list[dict], texts: list[str]) -> list[bool]:
    flags = []
    for row, text in zip(rows, texts):
        gt = _as_text((row.get("reward_model") or {}).get("ground_truth"))
        flags.append(reasoning_leaks_reply(
            text, gt,
            ngram_size=LEAKAGE_NGRAM_SIZE,
            max_match_tokens=LEAKAGE_MAX_MATCH_TOKENS,
        ))
    return flags


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, type=Path,
                   help="Source SFT parquet (typically sft/train.parquet)")
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--served_url", type=str, required=True,
                   help="e.g. http://a100-XXX-XXX:8000/v1")
    p.add_argument("--out", type=Path, required=True,
                   help="Report path (typically results/2026-07-08-judge-sweep/derived/cot_parity_report.md)")
    p.add_argument("--fail_on_diff", action="store_true",
                   help="Exit 1 if perspective distributions differ by >30% or empty rate >10%")
    args = p.parse_args()

    df = pd.read_parquet(args.input)
    rng = random.Random(args.seed)
    idxs = rng.sample(range(len(df)), args.n)
    rows = [dict(df.iloc[i]) for i in idxs]

    print(f"[parity] running batched on {args.n} rows", flush=True)
    t0 = time.time()
    b_texts = run_batched(rows, tp=1)
    t_batched = time.time() - t0
    print(f"[parity] batched done in {t_batched:.1f}s", flush=True)

    print(f"[parity] running served on {args.n} rows (url={args.served_url})", flush=True)
    t0 = time.time()
    s_texts = run_served(rows, args.served_url)
    t_served = time.time() - t0
    print(f"[parity] served done in {t_served:.1f}s", flush=True)

    b_summ = summarize("batched", b_texts)
    s_summ = summarize("served", s_texts)
    b_leak = leakage_flags(rows, b_texts)
    s_leak = leakage_flags(rows, s_texts)

    ok_empty = b_summ["empty"] == 0 and s_summ["empty"] == 0
    ok_leak = sum(b_leak) == 0 and sum(s_leak) == 0
    persp_diff = max(
        abs(b_summ["perspective"][p] - s_summ["perspective"][p]) for p in b_summ["perspective"]
    )
    ok_persp = persp_diff <= max(3, args.n // 3)
    ok = ok_empty and ok_leak and ok_persp

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        fh.write("# CoT batched-vs-served parity report\n\n")
        fh.write(f"n = {args.n}, seed = {args.seed}\n\n")
        fh.write(f"batched wall = {t_batched:.1f}s; served wall = {t_served:.1f}s\n\n")
        fh.write("## Summaries\n\n")
        fh.write(f"```\n{json.dumps({'batched': b_summ, 'served': s_summ}, indent=2)}\n```\n\n")
        fh.write(f"batched leaked = {sum(b_leak)}/{args.n}; served leaked = {sum(s_leak)}/{args.n}\n\n")
        fh.write(f"perspective count difference (max over categories): {persp_diff}\n\n")
        fh.write(f"## Verdict: {'PASS' if ok else 'FAIL'}\n\n")
        if not ok:
            fh.write(f"- empty_ok={ok_empty}, leak_ok={ok_leak}, perspective_ok={ok_persp}\n")
        fh.write("\n## Side-by-side (first 5)\n\n")
        for k in range(min(5, args.n)):
            fh.write(f"### Row {idxs[k]}\n\n")
            fh.write(f"**batched:** {b_texts[k][:400]!r}\n\n")
            fh.write(f"**served:**  {s_texts[k][:400]!r}\n\n")

    print(f"[parity] report: {args.out}", flush=True)
    if args.fail_on_diff and not ok:
        print("[parity] FAIL — see report", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Syntax check**

Run: `/home/lancewicki/miniconda3/envs/judge-vllm/bin/python -c "import ast; ast.parse(open('/storage/home/lancewicki/projects/turing-rl/scripts/cot_parity_test.py').read())"`
Expected: no output.

- [ ] **Step 3: Commit**

```bash
git add scripts/cot_parity_test.py
git commit -m "add cot batched-vs-served 20-row parity test"
```

---

## Task 7: Slurm launcher for CoT parity test

**Files:**
- Create: `scripts/slurm/cot_parity.sh`

- [ ] **Step 1: Write the launcher**

```bash
#!/bin/bash
#SBATCH --job-name=cot_parity
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --time=00:45:00
#SBATCH --output=/home/lancewicki/projects/turing-rl/logs/cot_parity-%j.out
#SBATCH --partition=a100
#SBATCH --account=rfai

# 20-row batched-vs-served CoT parity test.
# Starts a vLLM server on GPU 1, runs the batched path in-process on GPU 0.

set -uo pipefail

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY

export HF_HOME=/home/lancewicki/data/hf_cache
export HF_HUB_CACHE=/home/lancewicki/data/hf_cache
export PYTHONUNBUFFERED=1
export VLLM_LOGGING_LEVEL=INFO

REPO=/home/lancewicki/projects/turing-rl
PY=/home/lancewicki/miniconda3/envs/judge-vllm/bin/python
MODEL=Qwen/Qwen3-8B
PORT=8125
SFT_PARQUET=$REPO/data/prism/full_s42_history_sft40_grpo60_test10/sft/train.parquet
OUT=$REPO/results/2026-07-08-judge-sweep/derived/cot_parity_report.md
mkdir -p "$(dirname "$OUT")"

echo "=== start served-mode Qwen3-8B on GPU 1 (thinking-off; NO --reasoning-parser) ==="
CUDA_VISIBLE_DEVICES=1 $PY -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --download-dir /home/lancewicki/data/hf_cache \
    --tensor-parallel-size 1 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.85 \
    --dtype bfloat16 \
    --host 0.0.0.0 --port $PORT \
    > /tmp/vllm_parity_server-$SLURM_JOB_ID.log 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT

echo "=== wait for server to accept /v1/models ==="
for i in $(seq 1 180); do
    if curl -s -m 2 http://localhost:$PORT/v1/models >/dev/null 2>&1; then
        echo "server ready after ${i}s"
        break
    fi
    sleep 1
done

echo "=== run batched on GPU 0 + call served on GPU 1 ==="
cd "$REPO"
CUDA_VISIBLE_DEVICES=0 $PY scripts/cot_parity_test.py \
    --input "$SFT_PARQUET" \
    --n 20 \
    --served_url "http://localhost:$PORT/v1" \
    --out "$OUT" \
    --fail_on_diff
RC=$?
echo "=== parity exit: $RC ==="
exit $RC
```

- [ ] **Step 2: Make executable, submit, tail log**

```bash
chmod +x /storage/home/lancewicki/projects/turing-rl/scripts/slurm/cot_parity.sh
JOB=$(sbatch --parsable /storage/home/lancewicki/projects/turing-rl/scripts/slurm/cot_parity.sh)
echo "job: $JOB"
tail -f /home/lancewicki/projects/turing-rl/logs/cot_parity-$JOB.out
```

Expected: exits 0, `results/2026-07-08-judge-sweep/derived/cot_parity_report.md` contains `## Verdict: PASS`.
Gate: **do not proceed to Task 8 unless Task 7 passes.**

- [ ] **Step 3: Commit**

```bash
git add scripts/slurm/cot_parity.sh
git commit -m "add cot parity slurm launcher"
```

---

## Task 8: Slurm launcher for full CoT generation

**Files:**
- Create: `scripts/slurm/cot_batched.sh`

- [ ] **Step 1: Write the launcher**

```bash
#!/bin/bash
#SBATCH --job-name=cot_full
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH --mem=512G
#SBATCH --time=02:00:00
#SBATCH --output=/home/lancewicki/projects/turing-rl/logs/cot_full-%j.out
#SBATCH --partition=a100
#SBATCH --account=rfai

# Full 3272-row CoT generation via batched offline vLLM, TP=8 on one node.
# Note: TP=8 not 8xDP for simplicity; vLLM's continuous batching saturates
# well and we don't need multi-replica orchestration.

set -uo pipefail

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY

export HF_HOME=/home/lancewicki/data/hf_cache
export HF_HUB_CACHE=/home/lancewicki/data/hf_cache
export PYTHONUNBUFFERED=1
export VLLM_LOGGING_LEVEL=INFO

REPO=/home/lancewicki/projects/turing-rl
PY=/home/lancewicki/miniconda3/envs/judge-vllm/bin/python
IN=$REPO/data/prism/full_s42_history_sft40_grpo60_test10/sft/train.parquet
OUT=$REPO/data/sft/prism_full_s42_sft_cot.parquet

echo "=== CoT full ($IN -> $OUT) ==="
date
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
cd "$REPO"

$PY scripts/generate_cot_batched.py \
    --input "$IN" \
    --output "$OUT" \
    --tensor_parallel_size 8 \
    --gpu_memory_utilization 0.85 \
    --max_model_len 8192
RC=$?
echo "=== exit: $RC ==="
exit $RC
```

- [ ] **Step 2: Submit and watch**

```bash
chmod +x /storage/home/lancewicki/projects/turing-rl/scripts/slurm/cot_batched.sh
JOB=$(sbatch --parsable /storage/home/lancewicki/projects/turing-rl/scripts/slurm/cot_batched.sh)
tail -f /home/lancewicki/projects/turing-rl/logs/cot_full-$JOB.out
```

Expected: exits 0, `data/sft/prism_full_s42_sft_cot.parquet` exists with 3272 rows. Metadata JSON reports 0 rows with `thinking_trace_failed_leakage_guard=true` (or ≤5).

- [ ] **Step 3: Sanity-check the output**

Run: `/home/lancewicki/miniconda3/envs/turing-rl-train/bin/python -c "import pandas as pd, json; d = pd.read_parquet('/storage/home/lancewicki/projects/turing-rl/data/sft/prism_full_s42_sft_cot.parquet'); print('rows:', len(d)); print('sample reasoning:'); print(d.iloc[0]['extra_info']['ground_truth_reasoning'][:400]); print('has <think>:', '<think>' in d.iloc[0]['extra_info']['ground_truth_reasoning'])"`
Expected: 3272 rows, coherent third-person rationalization, no `<think>` tags.

- [ ] **Step 4: Commit the launcher (output parquet stays out of git — it's data)**

```bash
git add scripts/slurm/cot_batched.sh
git commit -m "add full-cot batched slurm launcher"
```

---

## Task 9: Build SFT JSONL from full CoT parquet

**Files:** (no new files; uses upstream `data/sft/build_sft_jsonl.py`)

- [ ] **Step 1: Run the upstream JSONL builder**

Run: `cd /storage/home/lancewicki/projects/turing-rl && /home/lancewicki/miniconda3/envs/turing-rl-train/bin/python -m data.sft.build_sft_jsonl --input_parquet data/sft/prism_full_s42_sft_cot.parquet --output_jsonl data/sft/prism_full_s42_sft_cot.jsonl`
Expected: prints "Wrote 3272 SFT examples to ..." and the file exists.

- [ ] **Step 2: Sanity-check the JSONL format**

Run: `head -1 /storage/home/lancewicki/projects/turing-rl/data/sft/prism_full_s42_sft_cot.jsonl | /home/lancewicki/miniconda3/envs/turing-rl-train/bin/python -c "import json, sys; d = json.loads(sys.stdin.read()); print('msg count:', len(d['messages'])); print('last role:', d['messages'][-1]['role']); print('last content head:', d['messages'][-1]['content'][:200])"`
Expected: `msg count: 3`, `last role: assistant`, content starts with `<reasoning>` and contains `[HUMAN]:`.

- [ ] **Step 3: Line count matches**

Run: `wc -l /storage/home/lancewicki/projects/turing-rl/data/sft/prism_full_s42_sft_cot.jsonl`
Expected: `3272`.

No commit — no code changed, just data produced.

---

## Task 10: SFT full-training slurm launcher (with auto-resume)

**Files:**
- Create: `scripts/slurm/sft_full.sh`

- [ ] **Step 1: Write the launcher**

```bash
#!/bin/bash
#SBATCH --job-name=sft_full
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH --mem=512G
#SBATCH --time=20:00:00
#SBATCH --output=/home/lancewicki/projects/turing-rl/logs/sft_full-%j.out
#SBATCH --partition=a100
#SBATCH --account=rfai

# Full paper-faithful LoRA SFT on the 3272-row CoT parquet.
# 8x A100 40GB, effective BS=128 (1 per-device * 16 grad_accum * 8 GPUs).
# max_seq_length=8192 per paper Table 5. Auto-resumes from the latest
# checkpoint under OUT/ on re-launch (see our_patches.md).

set -uo pipefail

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY

if [ -f /home/lancewicki/projects/turing-rl/.env ]; then
  set -a; source /home/lancewicki/projects/turing-rl/.env; set +a
fi

export HF_HUB_DISABLE_XET=1
export HF_HOME=/home/lancewicki/data/hf_cache
export HF_HUB_CACHE=/home/lancewicki/data/hf_cache
export PYTHONUNBUFFERED=1
export WANDB_MODE=online
export WANDB_PROJECT=turing-rl
export WANDB_RUN_GROUP=sft-full
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1

PY=/home/lancewicki/miniconda3/envs/turing-rl-train/bin/python
REPO=/home/lancewicki/projects/turing-rl
DATA=$REPO/data/sft/prism_full_s42_sft_cot.jsonl
OUT=$REPO/checkpoints/sft/qwen3_8b_prism_full_s42
YAML=$REPO/training/sft/configs/qwen3_8b_lora.yaml
YAML_BAK=${YAML}.full-bak
mkdir -p "$OUT"

# Snapshot YAML, swap report_to: none -> report_to: wandb. Restored on exit.
restore_yaml() {
  if [ -f "$YAML_BAK" ]; then mv -f "$YAML_BAK" "$YAML"; echo "[cleanup] restored $YAML"; fi
}
trap restore_yaml EXIT
cp -p "$YAML" "$YAML_BAK"
sed -i 's/^report_to: none$/report_to: wandb/' "$YAML"

echo "=== SFT full launch ==="
date; nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

cd "$REPO"
$PY -u -m training.sft.lora_sft \
    --model Qwen/Qwen3-8B \
    --data_path "$DATA" \
    --output_dir "$OUT" \
    --config "$YAML" \
    --max_seq_length 8192 \
    --resume_from_checkpoint auto
RC=$?
echo "=== SFT exit: $RC ==="
exit $RC
```

Note: check `training/sft/lora_sft.py:150` for the exact `--config` flag name (or `--config_yaml`). If different, adjust the invocation above.

- [ ] **Step 2: Verify the CLI flags match**

Run: `cd /storage/home/lancewicki/projects/turing-rl && /home/lancewicki/miniconda3/envs/turing-rl-train/bin/python -m training.sft.lora_sft --help 2>&1 | grep -E "^\s*--(config|model|data_path|output_dir|max_seq|resume)"`
Expected: shows all of `--config`, `--model`, `--data_path`, `--output_dir`, `--max_seq_length`, `--resume_from_checkpoint`. If `--config` isn't there, look at the actual flag name in the argparse block (top of `parse_args()`) and update `scripts/slurm/sft_full.sh` accordingly.

- [ ] **Step 3: Submit**

```bash
chmod +x /storage/home/lancewicki/projects/turing-rl/scripts/slurm/sft_full.sh
JOB=$(sbatch --parsable /storage/home/lancewicki/projects/turing-rl/scripts/slurm/sft_full.sh)
echo "SFT job: $JOB"
tail -f /home/lancewicki/projects/turing-rl/logs/sft_full-$JOB.out
```

Expected: after ~12-16h, exit 0, adapter saved to `checkpoints/sft/qwen3_8b_prism_full_s42/final/adapter_config.json`. Loss curve visible in wandb.

If the job dies mid-training, just re-submit the same script — it auto-resumes.

- [ ] **Step 4: Commit**

```bash
git add scripts/slurm/sft_full.sh
git commit -m "add full sft slurm launcher with auto-resume"
```

---

## Task 11: Heldout inference slurm launcher

**Files:**
- Create: `scripts/slurm/heldout_inference.sh`

- [ ] **Step 1: Write the launcher**

```bash
#!/bin/bash
#SBATCH --job-name=heldout_inf
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home/lancewicki/projects/turing-rl/logs/heldout_inf-%j.out
#SBATCH --partition=a100
#SBATCH --account=rfai

# Run the SFT'd Qwen3-8B (LoRA adapter) on the 880-row heldout parquet
# with paper Table 4 PRISM sampling (T=0.6, top_p=1.0, top_k=-1,
# pres_pen=0.5, max_tokens=2048). One generation per row.

set -uo pipefail
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY
export HF_HOME=/home/lancewicki/data/hf_cache
export HF_HUB_CACHE=/home/lancewicki/data/hf_cache
export PYTHONUNBUFFERED=1

PY=/home/lancewicki/miniconda3/envs/turing-rl-train/bin/python
REPO=/home/lancewicki/projects/turing-rl
CKPT=$REPO/checkpoints/sft/qwen3_8b_prism_full_s42
TEST=$REPO/data/prism/full_s42_history_sft40_grpo60_test10/test.parquet
OUT_DIR=$REPO/results/2026-07-08-judge-sweep/raw/generator
OUT=$OUT_DIR/heldout_inference.pkl
mkdir -p "$OUT_DIR"

echo "=== heldout inference ==="
date; nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

cd "$REPO"
$PY -u -m eval.generate_trained \
    --checkpoint_dir "$CKPT" \
    --test_parquet "$TEST" \
    --model_id Qwen/Qwen3-8B \
    --gen_num 1 \
    --output "$OUT" \
    --conditioning_mode history \
    --vllm_tensor_parallel_size 1 \
    --vllm_gpu_memory_utilization 0.6 \
    --vllm_max_num_seqs 32
RC=$?
echo "=== exit: $RC ==="

# Emit metadata sidecar.
$PY -c "
import json, os
meta = {
    'checkpoint_dir': '$CKPT',
    'test_parquet': '$TEST',
    'base_model': 'Qwen/Qwen3-8B',
    'gen_num': 1,
    'sampling': 'paper Table 4 PRISM defaults from eval/generate_trained.py',
    'output': '$OUT',
    'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
}
with open('$OUT_DIR/heldout_inference_metadata.json', 'w') as f:
    json.dump(meta, f, indent=2)
print('wrote metadata sidecar')
"
exit $RC
```

- [ ] **Step 2: Submit after SFT completes**

```bash
chmod +x /storage/home/lancewicki/projects/turing-rl/scripts/slurm/heldout_inference.sh
JOB=$(sbatch --parsable /storage/home/lancewicki/projects/turing-rl/scripts/slurm/heldout_inference.sh)
tail -f /home/lancewicki/projects/turing-rl/logs/heldout_inf-$JOB.out
```

Expected: ~1h, exit 0, `results/2026-07-08-judge-sweep/raw/generator/heldout_inference.pkl` exists.

- [ ] **Step 3: Sanity-check the pickle**

Run: `/home/lancewicki/miniconda3/envs/turing-rl-train/bin/python -c "import pickle; d = pickle.load(open('/storage/home/lancewicki/projects/turing-rl/results/2026-07-08-judge-sweep/raw/generator/heldout_inference.pkl', 'rb')); print(type(d)); print('n users:', len(d)); u0 = d[0]; print('user keys:', list(u0.keys())[:10]); tr = u0.get('test_results', u0)[:1]; print('sample target keys:', list(tr[0].keys())[:15] if tr else 'n/a')"`
Expected: prints structure. The pickle is a list of per-user dicts; each has `test_results` with per-target generations.

- [ ] **Step 4: Commit**

```bash
git add scripts/slurm/heldout_inference.sh
git commit -m "add heldout-inference slurm launcher"
```

---

## Task 12: Pair-set construction script

**Files:**
- Create: `scripts/build_judge_pairs.py`

- [ ] **Step 1: Write the script**

```python
"""Build the 880-row (human, generated) pair parquet from heldout inference.

Reads the pickle from eval/generate_trained.py output and the raw heldout
test.parquet, joins them on (user_id, post_id, target_idx), strips the
<reasoning>...</reasoning> envelope from generator output, runs sanity
checks, and writes results/2026-07-08-judge-sweep/raw/pairs/prism_heldout_880.parquet.
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from shared.prompt_utils import parse_reasoning_and_response


def _flatten_user_results(user_results: list) -> list[dict]:
    """Flatten per-user-generation records into one row per target."""
    out = []
    for user in user_results:
        for tr in user.get("test_results", []) or []:
            out.append(tr)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inference_pkl", type=Path, required=True)
    p.add_argument("--test_parquet", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with args.inference_pkl.open("rb") as fh:
        user_results = pickle.load(fh)
    trs = _flatten_user_results(user_results)
    print(f"[pairs] flattened {len(trs)} target records", flush=True)

    test = pd.read_parquet(args.test_parquet)
    test_idx: dict[tuple, dict] = {}
    for _, row in test.iterrows():
        extra = dict(row["extra_info"])
        key = (str(extra["user_id"]), str(extra["post_id"]), int(extra["target_idx"]))
        test_idx[key] = {
            "prompt": row["prompt"],
            "ground_truth": str(row["reward_model"]["ground_truth"]),
            "user_history": str(extra.get("user_history", "")),
            "context": str(extra.get("context", extra.get("thread_context", ""))),
            "persona": str(extra.get("persona", "")),
            "user_id": str(extra["user_id"]),
            "post_id": str(extra["post_id"]),
            "target_idx": int(extra["target_idx"]),
        }
    print(f"[pairs] indexed {len(test_idx)} heldout rows", flush=True)

    pairs = []
    missing_keys = 0
    empty_generated = 0
    residual_reasoning = 0
    identical = 0
    for tr in trs:
        gens = tr.get("generations") or tr.get("outputs") or []
        raw = ""
        if gens:
            first = gens[0]
            raw = first.get("text") if isinstance(first, dict) else str(first)
        raw = str(raw or "")
        # Take the last output for the target.
        extra = tr.get("extra_info") or tr
        try:
            key = (
                str(extra.get("user_id") or tr.get("user_id")),
                str(extra.get("post_id") or tr.get("post_id")),
                int(extra.get("target_idx") if extra.get("target_idx") is not None else tr.get("target_idx")),
            )
        except (KeyError, TypeError, ValueError):
            missing_keys += 1
            continue
        target = test_idx.get(key)
        if target is None:
            missing_keys += 1
            continue

        parsed = parse_reasoning_and_response(raw)
        generated = (parsed[1] if isinstance(parsed, tuple) else parsed).strip()
        if "<reasoning>" in generated or "</reasoning>" in generated:
            residual_reasoning += 1
            continue
        if not generated:
            empty_generated += 1
            continue
        if generated == target["ground_truth"]:
            identical += 1
            continue

        pairs.append({
            "pair_id": f"{key[0]}::{key[1]}::{key[2]}",
            "user_id": key[0],
            "post_id": key[1],
            "target_idx": key[2],
            "user_history": target["user_history"],
            "context": target["context"],
            "persona": target["persona"],
            "human": target["ground_truth"],
            "generated": generated,
        })

    print(f"[pairs] built {len(pairs)} pairs; missing_keys={missing_keys}, "
          f"empty={empty_generated}, residual_reasoning={residual_reasoning}, "
          f"identical={identical}", flush=True)

    assert len(pairs) == 880, f"expected 880 pairs, got {len(pairs)}"
    assert missing_keys == 0, f"{missing_keys} rows had no matching heldout key"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(pairs).to_parquet(args.out, index=False)
    print(f"[pairs] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Syntax check**

Run: `/home/lancewicki/miniconda3/envs/turing-rl-train/bin/python -c "import ast; ast.parse(open('/storage/home/lancewicki/projects/turing-rl/scripts/build_judge_pairs.py').read())"`
Expected: no output.

- [ ] **Step 3: Run it against the heldout inference pkl**

Run:
```
cd /storage/home/lancewicki/projects/turing-rl && \
/home/lancewicki/miniconda3/envs/turing-rl-train/bin/python scripts/build_judge_pairs.py \
    --inference_pkl results/2026-07-08-judge-sweep/raw/generator/heldout_inference.pkl \
    --test_parquet data/prism/full_s42_history_sft40_grpo60_test10/test.parquet \
    --out results/2026-07-08-judge-sweep/raw/pairs/prism_heldout_880.parquet
```
Expected: exit 0, "built 880 pairs" printed, parquet exists.

If the assertion fails, inspect the output pkl structure (`tr` keys may differ from what I assumed). The most likely fixes are: (a) `generations` field is called `outputs`; (b) parsed_response is already exposed as `parsed_response`. Adjust extraction accordingly.

- [ ] **Step 4: Sanity-check the parquet**

Run: `/home/lancewicki/miniconda3/envs/turing-rl-train/bin/python -c "import pandas as pd; d = pd.read_parquet('/storage/home/lancewicki/projects/turing-rl/results/2026-07-08-judge-sweep/raw/pairs/prism_heldout_880.parquet'); print('rows:', len(d)); print('cols:', list(d.columns)); r = d.iloc[0]; print('human[:100]:', r['human'][:100]); print('generated[:100]:', r['generated'][:100])"`
Expected: 880 rows, expected columns, both text fields non-empty.

- [ ] **Step 5: Commit**

```bash
git add scripts/build_judge_pairs.py
git commit -m "add judge-pair builder from heldout inference"
```

---

## Task 13: Family-selection 4B throughput smoke

**Files:**
- Create: `scripts/slurm/family_smoke.sh`
- Create: `docs/superpowers/decisions/2026-07-08-family-decision.md` (stub; filled by this task's output)

- [ ] **Step 1: Write the launcher**

```bash
#!/bin/bash
#SBATCH --job-name=family_smoke
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=/home/lancewicki/projects/turing-rl/logs/family_smoke-%j.out
#SBATCH --partition=a100
#SBATCH --account=rfai

# Qwen3-4B vs Qwen3.5-4B throughput + agreement smoke.
# Same 50 real Turing prompts, TP=1, both thinking modes, concurrencies 1/8/32.

set -uo pipefail
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY
export HF_HOME=/home/lancewicki/data/hf_cache
export HF_HUB_CACHE=/home/lancewicki/data/hf_cache
export PYTHONUNBUFFERED=1
export VLLM_LOGGING_LEVEL=INFO

REPO=/home/lancewicki/projects/turing-rl
PY=/home/lancewicki/miniconda3/envs/judge-vllm/bin/python
DUMPS=/home/lancewicki/tmp/judge_dumps/judge-9239-1968156.jsonl
OUT_DIR=$REPO/results/2026-07-08-judge-sweep/raw/family_smoke/qwen3_vs_qwen35_4b
mkdir -p "$OUT_DIR"

start_vllm() {
    local model=$1 port=$2 device=$3 mode=$4
    local parser_args=""
    if [ "$mode" = "on" ]; then
        parser_args="--reasoning-parser qwen3"
    fi
    CUDA_VISIBLE_DEVICES=$device $PY -m vllm.entrypoints.openai.api_server \
        --model "$model" \
        --download-dir /home/lancewicki/data/hf_cache \
        --tensor-parallel-size 1 \
        --max-model-len 8192 \
        --gpu-memory-utilization 0.85 \
        --dtype bfloat16 \
        $parser_args \
        --host 0.0.0.0 --port $port \
        > "$OUT_DIR/vllm-${model//\//_}-${mode}.log" 2>&1 &
}

wait_url() {
    local url=$1
    for i in $(seq 1 240); do
        curl -s -m 2 "$url/models" >/dev/null 2>&1 && { echo "ready: $url"; return 0; }
        sleep 1
    done
    echo "TIMEOUT: $url" >&2; return 1
}

# Run each mode sequentially to avoid GPU/port bookkeeping complexity.
for MODE in off on; do
    echo "=== mode=$MODE ==="
    start_vllm "Qwen/Qwen3-4B"   8130 0 "$MODE"; PID_A=$!
    start_vllm "Qwen/Qwen3.5-4B" 8131 1 "$MODE"; PID_B=$!
    trap "kill $PID_A $PID_B 2>/dev/null || true" EXIT
    wait_url "http://localhost:8130/v1" || exit 2
    wait_url "http://localhost:8131/v1" || exit 2
    cd "$REPO"
    $PY scripts/benchmark_judge_throughput.py \
        --endpoint qwen3=http://localhost:8130/v1 \
        --endpoint qwen35=http://localhost:8131/v1 \
        --dumps "$DUMPS" \
        --n 50 \
        --concurrency 1,8,32 \
        --out "$OUT_DIR/mode-$MODE"
    kill $PID_A $PID_B 2>/dev/null || true
    sleep 5
done

# Write the decision stub. A human reads the report and fills in the winner.
cat > $REPO/docs/superpowers/decisions/2026-07-08-family-decision.md <<'EOF'
# Judge family decision (2026-07-08 judge sweep)

## What the smoke measured
Qwen3-4B vs Qwen3.5-4B, same 50 real Turing prompts from a prior dump,
TP=1, thinking-on and thinking-off, concurrencies 1/8/32.

## Raw reports
- Thinking-off: `results/2026-07-08-judge-sweep/raw/family_smoke/qwen3_vs_qwen35_4b/mode-off/report-*.md`
- Thinking-on: `results/2026-07-08-judge-sweep/raw/family_smoke/qwen3_vs_qwen35_4b/mode-on/report-*.md`

## Decision
**Family:** _[fill in: qwen3 or qwen35]_

**Sizes for full sweep:**
- If qwen3: 4B / 8B / 14B / 32B (dense).
- If qwen35: 4B / 9B / 27B / 35B-A3B-GPTQ-Int4 (mixed dense + MoE).

## Rationale
_[fill in: winning tokens/sec, parse rate parity check, agreement spot-check if run]_

## Fallback
Per spec R6: default to Qwen3.5 if the smoke is inconclusive.
EOF
echo "wrote decision stub"
```

- [ ] **Step 2: Submit and wait for results**

```bash
mkdir -p /storage/home/lancewicki/projects/turing-rl/docs/superpowers/decisions
chmod +x /storage/home/lancewicki/projects/turing-rl/scripts/slurm/family_smoke.sh
JOB=$(sbatch --parsable /storage/home/lancewicki/projects/turing-rl/scripts/slurm/family_smoke.sh)
tail -f /home/lancewicki/projects/turing-rl/logs/family_smoke-$JOB.out
```

Expected: exit 0, both `mode-off/report-*.md` and `mode-on/report-*.md` exist and show throughput/latency tables for both models.

- [ ] **Step 3: Read reports and fill in the family-decision doc**

Read the two report files, pick the winner (higher req/s at concurrency=32 with ≥95% parse rate; if tied, default to Qwen3.5 per spec R6), then edit `docs/superpowers/decisions/2026-07-08-family-decision.md` to fill in **Family**, **Sizes**, and **Rationale**.

- [ ] **Step 4: Commit**

```bash
git add scripts/slurm/family_smoke.sh docs/superpowers/decisions/2026-07-08-family-decision.md
git commit -m "add family-selection 4b smoke and decision doc"
```

---

## Task 14: Parameterized judge-serve slurm script (for sweep + calibration)

**Files:**
- Create: `scripts/slurm/judge_serve_sweep.sh`

- [ ] **Step 1: Write the script**

```bash
#!/bin/bash
#SBATCH --job-name=judge_serve
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --time=12:00:00
#SBATCH --output=/home/lancewicki/projects/turing-rl/logs/judge_serve_sweep-%j.out
#SBATCH --partition=a100
#SBATCH --account=rfai

# Parameterized judge server for the sweep.
# Env vars (all required):
#   MODEL          e.g. Qwen/Qwen3-8B or Qwen/Qwen3.5-397B-A17B-GPTQ-Int4
#   TP             tensor-parallel size (must match --gres=gpu:N at sbatch time)
#   REPLICAS       number of parallel vLLM processes on this node (fills the node)
#   THINKING_MODE  "on" or "off"
#   PORT_BASE      base port (each replica gets PORT_BASE + i)
#   MAX_MODEL_LEN  vLLM --max-model-len (default 32768)
#
# Notes:
#   - REPLICAS x TP must be <= 8 (GPUs per node).
#   - THINKING_MODE=on adds --reasoning-parser qwen3.
#   - Each replica is pinned to a contiguous GPU slice via CUDA_VISIBLE_DEVICES.

set -uo pipefail
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY

for var in MODEL TP REPLICAS THINKING_MODE PORT_BASE; do
    if [ -z "${!var:-}" ]; then echo "ERROR: env var $var must be set" >&2; exit 2; fi
done
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}

export HF_HOME=/home/lancewicki/data/hf_cache
export HF_HUB_CACHE=/home/lancewicki/data/hf_cache
export PYTHONUNBUFFERED=1
export VLLM_LOGGING_LEVEL=INFO
export HF_HUB_DISABLE_XET=1

PY=/home/lancewicki/miniconda3/envs/judge-vllm/bin/python
PARSER_ARGS=""
if [ "$THINKING_MODE" = "on" ]; then
    PARSER_ARGS="--reasoning-parser qwen3"
fi

if [ $((REPLICAS * TP)) -gt 8 ]; then
    echo "ERROR: REPLICAS ($REPLICAS) * TP ($TP) > 8" >&2; exit 2
fi

echo "=== judge server: $MODEL, TP=$TP, replicas=$REPLICAS, thinking=$THINKING_MODE ==="
date; hostname; hostname -I
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

PIDS=()
for i in $(seq 0 $((REPLICAS - 1))); do
    gpu_start=$((i * TP))
    gpu_end=$((gpu_start + TP - 1))
    gpus=$(seq -s, $gpu_start $gpu_end)
    port=$((PORT_BASE + i))
    log=$OUT_DIR/vllm_server/replica_${i}.log
    mkdir -p "$(dirname "$log")"
    echo "starting replica $i on GPUs [$gpus] port $port -> $log"
    CUDA_VISIBLE_DEVICES=$gpus $PY -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" \
        --download-dir /home/lancewicki/data/hf_cache \
        --tensor-parallel-size "$TP" \
        --max-model-len "$MAX_MODEL_LEN" \
        --gpu-memory-utilization 0.85 \
        --dtype bfloat16 \
        $PARSER_ARGS \
        --host 0.0.0.0 --port $port \
        > "$log" 2>&1 &
    PIDS+=($!)
done

cleanup() {
    echo "cleanup: killing ${#PIDS[@]} replicas"
    for pid in "${PIDS[@]}"; do kill $pid 2>/dev/null || true; done
}
trap cleanup EXIT

# Wait for all replicas to answer /v1/models.
for i in $(seq 0 $((REPLICAS - 1))); do
    port=$((PORT_BASE + i))
    for tries in $(seq 1 600); do
        curl -s -m 2 http://localhost:$port/v1/models >/dev/null 2>&1 && break
        sleep 2
    done
    curl -s -m 2 http://localhost:$port/v1/models >/dev/null 2>&1 \
        || { echo "TIMEOUT replica $i port $port"; exit 3; }
    echo "replica $i ready"
done
echo "=== all $REPLICAS replicas ready ==="

# Stay alive until walltime.
wait
```

Note: this script assumes `$OUT_DIR` is set by the calling script (calibration or sweep-cell launcher exports it). If run standalone, replace `$OUT_DIR/vllm_server/replica_${i}.log` with `/tmp/vllm_replica_${i}-$SLURM_JOB_ID.log`.

- [ ] **Step 2: Syntax check with shellcheck (optional if available)**

Run: `bash -n /storage/home/lancewicki/projects/turing-rl/scripts/slurm/judge_serve_sweep.sh`
Expected: no output.

- [ ] **Step 3: Commit**

```bash
chmod +x /storage/home/lancewicki/projects/turing-rl/scripts/slurm/judge_serve_sweep.sh
git add scripts/slurm/judge_serve_sweep.sh
git commit -m "add parameterized judge-serve script for sweep"
```

---

## Task 15: Judge sweep client (one cell)

**Files:**
- Create: `scripts/run_judge_sweep_cell.py`

- [ ] **Step 1: Write the client**

```python
"""Run one (judge, thinking-mode) cell of the sweep against N replica endpoints.

Iterates 880 pairs x 2 orderings = 1760 judge calls. Round-robins requests
across the endpoint URLs. Writes reward-layer dumps to
results/2026-07-08-judge-sweep/raw/sweep/<judge>/<mode>/reward/ and HTTP dumps
to .../http/. Also emits run_metadata.json.

Reuses training/grpo/reward.py:_score_pairwise_likert_with_info so the exact
same code path used at GRPO training time is used here.
"""
from __future__ import annotations

import argparse
import asyncio
import contextvars
import json
import os
import random
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

# We import the reward module so its `_openai_chat` uses our environment
# (JUDGE_MODEL, PERSONA_JUDGE_JSON_SCHEMA, dump env vars).
from shared.judge_prompts import TURING_PROMPT


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pairs", type=Path, required=True)
    p.add_argument("--endpoints", type=str, required=True,
                   help="comma-separated URLs like http://host:8123/v1,http://host:8124/v1")
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--thinking_mode", choices=["on", "off"], required=True)
    p.add_argument("--out_dir", type=Path, required=True,
                   help="Cell output dir, e.g. .../raw/sweep/qwen3-8b/off/")
    p.add_argument("--concurrency_per_endpoint", type=int, default=16)
    p.add_argument("--max_pairs", type=int, default=None,
                   help="Cap on pair count (for calibration).")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def sampling_params_for_mode(mode: str) -> dict:
    if mode == "on":
        return {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0}
    return {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0}


async def main_async() -> None:
    args = parse_args()
    endpoints = [u.strip().rstrip("/") for u in args.endpoints.split(",") if u.strip()]
    if not endpoints:
        print("no endpoints", file=sys.stderr); sys.exit(2)

    # Configure environment BEFORE importing reward, so its module-level env
    # reads pick up our values.
    os.environ["JUDGE_MODEL"] = args.model
    os.environ["PERSONA_JUDGE_JSON_SCHEMA"] = "1"
    os.environ["PERSONA_JUDGE_MAX_COMPLETION_TOKENS"] = "8192"
    os.environ["PERSONA_JUDGE_DUMP_RATE"] = "1.0"
    os.environ["PERSONA_JUDGE_DUMP_DIR"] = str(args.out_dir)
    os.environ["OPENROUTER_PROVIDER_ORDER"] = ""  # disable OpenRouter routing extras

    # Point the api client at endpoint[0]; the round-robin below overrides per-call.
    os.environ["OPENAI_API_BASE"] = endpoints[0]
    os.environ["OPENAI_API_KEY"] = "dummy"

    import aiohttp
    from training.grpo.reward import _score_pairwise_likert_with_info

    pairs = pd.read_parquet(args.pairs)
    if args.max_pairs is not None:
        pairs = pairs.head(args.max_pairs)
    print(f"[sweep] cell={args.model} mode={args.thinking_mode} pairs={len(pairs)} "
          f"endpoints={len(endpoints)} conc/ep={args.concurrency_per_endpoint}", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "reward").mkdir(exist_ok=True)
    (args.out_dir / "http").mkdir(exist_ok=True)
    (args.out_dir / "vllm_server").mkdir(exist_ok=True)

    total_concurrency = args.concurrency_per_endpoint * len(endpoints)
    sem = asyncio.Semaphore(total_concurrency)
    endpoint_counter = {"i": 0}
    counter_lock = asyncio.Lock()

    connector = aiohttp.TCPConnector(limit=total_concurrency + 8)
    session_timeout = aiohttp.ClientTimeout(total=1200)
    async with aiohttp.ClientSession(connector=connector, timeout=session_timeout) as session:
        async def do_one(idx: int, row) -> None:
            async with sem:
                async with counter_lock:
                    ep_idx = endpoint_counter["i"] % len(endpoints)
                    endpoint_counter["i"] += 1
                ep = endpoints[ep_idx]
                # Override per-call by setting env var on this task? No -
                # api_client reads env at call time. Simpler: patch by env var each call.
                os.environ["OPENAI_API_BASE"] = ep
                try:
                    await _score_pairwise_likert_with_info(
                        session=session,
                        api_key="dummy",
                        response=row["generated"],
                        ground_truth=row["human"],
                        user_history=row["user_history"],
                        context=row["context"],
                        prompt_template=TURING_PROMPT,
                        persona=row.get("persona", "") or "",
                        user_id=row["user_id"],
                        post_id=row["post_id"],
                        target_idx=int(row["target_idx"]),
                        randomization_seed_material=f"{args.seed}::{row['pair_id']}",
                    )
                except Exception as exc:
                    print(f"[sweep] pair {idx} failed: {type(exc).__name__}: {exc}", flush=True)

        t0 = time.time()
        tasks = [do_one(i, r) for i, r in enumerate(pairs.to_dict(orient="records"))]
        await asyncio.gather(*tasks)
        dt = time.time() - t0

    meta = {
        "model": args.model,
        "thinking_mode": args.thinking_mode,
        "endpoints": endpoints,
        "concurrency_per_endpoint": args.concurrency_per_endpoint,
        "sampling": sampling_params_for_mode(args.thinking_mode),
        "sampling_note": "Sampling not sent on wire (upstream server defaults apply). "
                         "Sweep spec locks server-side defaults via generation_config.json.",
        "persona_judge_json_schema": "1",
        "max_completion_tokens": 8192,
        "n_pairs": int(len(pairs)),
        "n_calls_expected": int(len(pairs)) * 2,
        "wall_seconds": dt,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "pair_source": str(args.pairs),
    }
    (args.out_dir / "run_metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"[sweep] cell done in {dt:.1f}s -> {args.out_dir}", flush=True)


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Syntax check**

Run: `/home/lancewicki/miniconda3/envs/turing-rl-train/bin/python -c "import ast; ast.parse(open('/storage/home/lancewicki/projects/turing-rl/scripts/run_judge_sweep_cell.py').read())"`
Expected: no output.

- [ ] **Step 3: Commit**

```bash
git add scripts/run_judge_sweep_cell.py
git commit -m "add judge-sweep single-cell client (round-robin, dumps)"
```

---

## Task 16: Per-cell sbatch template + launcher for the sweep

**Files:**
- Create: `scripts/slurm/judge_sweep_cell.sh`
- Create: `scripts/launch_judge_sweep.sh`

- [ ] **Step 1: Write the per-cell sbatch template**

```bash
#!/bin/bash
#SBATCH --job-name=sweep_cell
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --time=12:00:00
#SBATCH --output=/home/lancewicki/projects/turing-rl/logs/sweep_cell-%j.out
#SBATCH --partition=a100
#SBATCH --account=rfai

# One (judge, thinking_mode) cell of the sweep.
# Required env vars: MODEL, TP, REPLICAS, THINKING_MODE, CELL_NAME.
# Optional: PORT_BASE (default 8130), MAX_PAIRS (default all 880).

set -uo pipefail
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY

for var in MODEL TP REPLICAS THINKING_MODE CELL_NAME; do
    if [ -z "${!var:-}" ]; then echo "ERROR: env var $var must be set" >&2; exit 2; fi
done
PORT_BASE=${PORT_BASE:-8130}
MAX_PAIRS=${MAX_PAIRS:-}

REPO=/home/lancewicki/projects/turing-rl
PY_CLIENT=/home/lancewicki/miniconda3/envs/turing-rl-train/bin/python
PY_SERVER=/home/lancewicki/miniconda3/envs/judge-vllm/bin/python
PAIRS=$REPO/results/2026-07-08-judge-sweep/raw/pairs/prism_heldout_880.parquet
OUT_DIR=$REPO/results/2026-07-08-judge-sweep/raw/sweep/$CELL_NAME/$THINKING_MODE
export OUT_DIR  # judge_serve_sweep.sh looks at this for replica logs
mkdir -p "$OUT_DIR/vllm_server"

echo "=== cell: model=$MODEL mode=$THINKING_MODE tp=$TP replicas=$REPLICAS ==="
date; hostname; nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

if [ $((REPLICAS * TP)) -gt 8 ]; then
    echo "ERROR: REPLICAS ($REPLICAS) * TP ($TP) > 8" >&2; exit 2
fi

# Boot N replicas of the server on this node.
PARSER_ARGS=""
if [ "$THINKING_MODE" = "on" ]; then PARSER_ARGS="--reasoning-parser qwen3"; fi

export HF_HOME=/home/lancewicki/data/hf_cache
export HF_HUB_CACHE=/home/lancewicki/data/hf_cache
export HF_HUB_DISABLE_XET=1
export PYTHONUNBUFFERED=1

PIDS=()
URLS=()
for i in $(seq 0 $((REPLICAS - 1))); do
    gpu_start=$((i * TP))
    gpu_end=$((gpu_start + TP - 1))
    gpus=$(seq -s, $gpu_start $gpu_end)
    port=$((PORT_BASE + i))
    log=$OUT_DIR/vllm_server/replica_${i}.log
    echo "start replica $i GPUs=[$gpus] port=$port"
    CUDA_VISIBLE_DEVICES=$gpus $PY_SERVER -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" \
        --download-dir /home/lancewicki/data/hf_cache \
        --tensor-parallel-size "$TP" \
        --max-model-len 32768 \
        --gpu-memory-utilization 0.85 \
        --dtype bfloat16 \
        $PARSER_ARGS \
        --host 0.0.0.0 --port $port \
        > "$log" 2>&1 &
    PIDS+=($!)
    URLS+=("http://localhost:${port}/v1")
done

cleanup() {
    echo "[cell] cleanup ${#PIDS[@]} replicas"
    for pid in "${PIDS[@]}"; do kill $pid 2>/dev/null || true; done
}
trap cleanup EXIT

# Wait for all replicas.
for i in $(seq 0 $((REPLICAS - 1))); do
    port=$((PORT_BASE + i))
    ok=0
    for tries in $(seq 1 900); do
        if curl -s -m 2 http://localhost:$port/v1/models >/dev/null 2>&1; then ok=1; break; fi
        sleep 2
    done
    [ $ok -eq 1 ] || { echo "TIMEOUT replica $i"; exit 3; }
    echo "[cell] replica $i ready"
done

# Fire the client.
ENDPOINTS=$(IFS=,; echo "${URLS[*]}")
EXTRA=""
if [ -n "$MAX_PAIRS" ]; then EXTRA="--max_pairs $MAX_PAIRS"; fi
cd "$REPO"
$PY_CLIENT scripts/run_judge_sweep_cell.py \
    --pairs "$PAIRS" \
    --endpoints "$ENDPOINTS" \
    --model "$MODEL" \
    --thinking_mode "$THINKING_MODE" \
    --out_dir "$OUT_DIR" \
    --concurrency_per_endpoint 16 \
    $EXTRA
RC=$?
echo "=== client exit: $RC ==="
exit $RC
```

- [ ] **Step 2: Write the orchestrator that launches all 10 cells**

```bash
#!/bin/bash
# Orchestrate the 10-cell judge sweep across nodes.
#
# Usage:
#   FAMILY=qwen35 bash scripts/launch_judge_sweep.sh          # full sweep
#   FAMILY=qwen35 CALIBRATION=1 bash scripts/launch_judge_sweep.sh   # 50-pair calibration
#
# FAMILY must match the winner from the family-selection smoke.
# CALIBRATION=1 exports MAX_PAIRS=50 into each cell.

set -uo pipefail
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY

REPO=/home/lancewicki/projects/turing-rl
FAMILY=${FAMILY:?FAMILY must be set to qwen3 or qwen35}
CALIBRATION=${CALIBRATION:-0}
LAUNCHER=$REPO/scripts/slurm/judge_sweep_cell.sh

# Sizes per family, from spec Section 1.
declare -a MODELS_qwen3=(
    "qwen3-4b|Qwen/Qwen3-4B|1|8"
    "qwen3-8b|Qwen/Qwen3-8B|1|8"
    "qwen3-14b|Qwen/Qwen3-14B|1|8"
    "qwen3-32b|Qwen/Qwen3-32B|2|4"
)
declare -a MODELS_qwen35=(
    "qwen35-4b|Qwen/Qwen3.5-4B|1|8"
    "qwen35-9b|Qwen/Qwen3.5-9B|1|8"
    "qwen35-27b|Qwen/Qwen3.5-27B|2|4"
    "qwen35-35b-a3b|Qwen/Qwen3.5-35B-A3B-GPTQ-Int4|1|8"
)
declare -a ANCHOR=(
    "qwen35-397b|Qwen/Qwen3.5-397B-A17B-GPTQ-Int4|8|1"
)

case "$FAMILY" in
    qwen3)  MODELS=("${MODELS_qwen3[@]}") ;;
    qwen35) MODELS=("${MODELS_qwen35[@]}") ;;
    *) echo "unknown FAMILY: $FAMILY"; exit 2 ;;
esac

MODELS+=("${ANCHOR[@]}")

EXTRA_EXPORT=""
if [ "$CALIBRATION" = "1" ]; then EXTRA_EXPORT="MAX_PAIRS=50"; fi

for entry in "${MODELS[@]}"; do
    IFS='|' read -r cell_name model tp replicas <<< "$entry"
    for mode in off on; do
        exports="MODEL=$model,TP=$tp,REPLICAS=$replicas,THINKING_MODE=$mode,CELL_NAME=$cell_name"
        if [ -n "$EXTRA_EXPORT" ]; then exports="$exports,$EXTRA_EXPORT"; fi
        echo "sbatch cell=$cell_name mode=$mode tp=$tp replicas=$replicas"
        sbatch --parsable \
            --gres=gpu:$((tp * replicas)) \
            --job-name="sw_${cell_name}_${mode}" \
            --export=ALL,$exports \
            "$LAUNCHER"
    done
done
```

- [ ] **Step 3: Make executable, commit**

```bash
chmod +x /storage/home/lancewicki/projects/turing-rl/scripts/slurm/judge_sweep_cell.sh
chmod +x /storage/home/lancewicki/projects/turing-rl/scripts/launch_judge_sweep.sh
git add scripts/slurm/judge_sweep_cell.sh scripts/launch_judge_sweep.sh
git commit -m "add sweep-cell sbatch and multi-cell launcher"
```

---

## Task 17: Run calibration (50 pairs per cell)

**Files:** (no new files; uses existing launcher)

- [ ] **Step 1: Launch calibration for all 10 cells**

Substitute `FAMILY` with your Task 13 decision:

```bash
FAMILY=<qwen3 or qwen35 per family-decision.md> CALIBRATION=1 bash /storage/home/lancewicki/projects/turing-rl/scripts/launch_judge_sweep.sh
```

Expected: 10 sbatch jobs submitted, prints job IDs. Wait for them all (`squeue --me`).

- [ ] **Step 2: Copy calibration dumps to raw/calibration/ and aggregate**

```bash
REPO=/home/lancewicki/projects/turing-rl
CAL_DIR=$REPO/results/2026-07-08-judge-sweep/raw/calibration
mkdir -p "$CAL_DIR"
# Each cell wrote to raw/sweep/<cell>/<mode>/run_metadata.json + reward/*.jsonl.
# We aggregate a single JSONL of (cell, mode, wall_seconds, n_pairs).
/home/lancewicki/miniconda3/envs/turing-rl-train/bin/python - <<'PYEOF'
import json, os, glob
from pathlib import Path
root = Path("/storage/home/lancewicki/projects/turing-rl/results/2026-07-08-judge-sweep/raw/sweep")
out = Path("/storage/home/lancewicki/projects/turing-rl/results/2026-07-08-judge-sweep/raw/calibration/calibration_metadata.json")
rows = []
for meta_path in root.glob("*/*/run_metadata.json"):
    meta = json.loads(meta_path.read_text())
    cell = meta_path.parent.parent.name
    mode = meta_path.parent.name
    n = meta.get("n_pairs", 0)
    wall = meta.get("wall_seconds", 0)
    calls = n * 2
    req_s = calls / wall if wall > 0 else 0.0
    projected_1760 = 1760 / req_s if req_s > 0 else None
    rows.append({
        "cell": cell, "mode": mode,
        "n_pairs": n, "wall_seconds": wall,
        "req_per_s": req_s,
        "projected_1760_call_wall_s": projected_1760,
    })
out.write_text(json.dumps(rows, indent=2))
print(json.dumps(rows, indent=2))
PYEOF
```

Expected: prints one row per cell with `req_per_s` and `projected_1760_call_wall_s`.

- [ ] **Step 3: Write derived calibration_report.md**

```bash
REPO=/home/lancewicki/projects/turing-rl
/home/lancewicki/miniconda3/envs/turing-rl-train/bin/python - <<'PYEOF'
import json
from pathlib import Path
root = Path("/storage/home/lancewicki/projects/turing-rl/results/2026-07-08-judge-sweep")
rows = json.loads((root/"raw"/"calibration"/"calibration_metadata.json").read_text())
out = root/"derived"/"calibration_report.md"
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w") as f:
    f.write("# Per-cell throughput calibration\n\n")
    f.write("| Cell | Mode | Pairs | Wall (s) | Req/s | Projected 1760-call wall |\n")
    f.write("|---|---|---|---|---|---|\n")
    for r in sorted(rows, key=lambda x: (x["cell"], x["mode"])):
        proj = r["projected_1760_call_wall_s"]
        proj_str = f"{proj:.0f}s ({proj/60:.1f}m)" if proj else "-"
        f.write(f"| {r['cell']} | {r['mode']} | {r['n_pairs']} | {r['wall_seconds']:.1f} "
                f"| {r['req_per_s']:.2f} | {proj_str} |\n")
    f.write("\n## Gate\n\n")
    f.write("Any cell with projected wall > 4h (14400s) triggers a spec Section 5.0 decision.\n")
print("wrote", out)
PYEOF
```

- [ ] **Step 4: Purge calibration dumps so the full sweep has clean output dirs**

```bash
find /storage/home/lancewicki/projects/turing-rl/results/2026-07-08-judge-sweep/raw/sweep -type d -mindepth 2 -maxdepth 2 -exec rm -rf {} +
```

- [ ] **Step 5: Commit report only (raw dumps stay out of git; they're ~large)**

```bash
git add results/2026-07-08-judge-sweep/derived/calibration_report.md
git commit -m "add calibration report from 50-pair per-cell smoke"
```

---

## Task 18: Run full sweep (10 cells, 1760 calls each)

**Files:** (no new files)

- [ ] **Step 1: Launch the full sweep**

```bash
FAMILY=<qwen3 or qwen35 per family-decision.md> bash /storage/home/lancewicki/projects/turing-rl/scripts/launch_judge_sweep.sh
```

Expected: 10 sbatch jobs, all writing to `results/2026-07-08-judge-sweep/raw/sweep/<cell>/<mode>/`.

- [ ] **Step 2: Wait for all cells; verify counts**

```bash
squeue --me
# When empty:
find /storage/home/lancewicki/projects/turing-rl/results/2026-07-08-judge-sweep/raw/sweep -name 'reward-*.jsonl' -exec wc -l {} +
```
Expected: each cell's reward dumps total ≥ 1760 lines (could be more if multiple `pid` files per replica). If any cell is short, re-submit that specific cell (edit `launch_judge_sweep.sh` or invoke `sbatch` manually with the same env exports).

- [ ] **Step 3: Copy slurm logs to raw/logs/**

```bash
mkdir -p /storage/home/lancewicki/projects/turing-rl/results/2026-07-08-judge-sweep/raw/logs/slurm
cp /home/lancewicki/projects/turing-rl/logs/sw_*.out \
   /storage/home/lancewicki/projects/turing-rl/results/2026-07-08-judge-sweep/raw/logs/slurm/
```

- [ ] **Step 4: Nothing to commit (raw dumps are data, not code)**

Just verify the output tree looks like the spec's expected shape:

Run: `find /storage/home/lancewicki/projects/turing-rl/results/2026-07-08-judge-sweep/raw -type f | sort | head -60`
Expected: `pairs/prism_heldout_880.parquet`, `generator/heldout_inference.pkl`, `sweep/<cell>/<mode>/reward/*.jsonl`, `sweep/<cell>/<mode>/http/*.jsonl`, `sweep/<cell>/<mode>/run_metadata.json`, `sweep/<cell>/<mode>/vllm_server/replica_*.log`.

---

## Task 19: Offline batched bonus cell

**Files:**
- Create: `scripts/run_offline_sweep_cell.py`
- Create: `scripts/slurm/offline_sweep_cell.sh`

- [ ] **Step 1: Write the offline client**

```python
"""Offline batched vLLM version of the sweep cell for Qwen3-8B thinking-off.

Runs in-process LLM.generate on all 1760 judge prompts at once, then writes
one JSONL row per pair per ordering to raw/sweep/<cell>/<mode>_offline/reward/.
Compares to the corresponding server cell in the analyzer.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from shared.judge_prompts import TURING_PROMPT
from shared.judge_utils import build_source_copy_warning, format_source_copy_watchlist


MODEL_ID = "Qwen/Qwen3-8B"
SAMPLING_OFF = dict(temperature=0.7, top_p=0.8, top_k=20, min_p=0.0, max_tokens=8192)


def build_prompt(row: dict, generated_is_b: bool) -> str:
    if generated_is_b:
        resp_a, resp_b = row["human"], row["generated"]
    else:
        resp_a, resp_b = row["generated"], row["human"]
    warn_a = build_source_copy_warning(resp_a, user_history=row["user_history"], thread_context=row["context"])
    warn_b = build_source_copy_warning(resp_b, user_history=row["user_history"], thread_context=row["context"])
    return TURING_PROMPT.format(
        persona=row.get("persona", ""),
        user_history=row["user_history"],
        context=row["context"],
        response_a=resp_a,
        response_b=resp_b,
        source_copy_watchlist=format_source_copy_watchlist(
            [warn_a, warn_b], item_label="Response", labels=["Response A", "Response B"],
        ),
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pairs", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True,
                   help="e.g. .../raw/sweep/qwen3-8b/off_offline/")
    p.add_argument("--tensor_parallel_size", type=int, default=8)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import GuidedDecodingParams

    pairs = pd.read_parquet(args.pairs).to_dict(orient="records")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "reward").mkdir(exist_ok=True)

    llm = LLM(
        model=MODEL_ID,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=0.85,
        max_model_len=32768,
        dtype="bfloat16",
    )
    tok = llm.get_tokenizer()

    # Same schema as PERSONA_JUDGE_JSON_SCHEMA in training/grpo/reward.py:363-374.
    guided = GuidedDecodingParams(json={
        "type": "object",
        "properties": {"rating": {"type": "integer", "minimum": 1, "maximum": 7}},
        "required": ["rating"],
        "additionalProperties": True,
    })
    sp = SamplingParams(guided_decoding=guided, **SAMPLING_OFF)

    prompts: list[str] = []
    metadata: list[dict] = []
    for row in pairs:
        for generated_is_b in (True, False):
            p_text = build_prompt(row, generated_is_b)
            chat = tok.apply_chat_template(
                [{"role": "user", "content": p_text}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
            prompts.append(chat)
            metadata.append({
                "pair_id": row["pair_id"],
                "user_id": row["user_id"],
                "post_id": row["post_id"],
                "target_idx": row["target_idx"],
                "generated_is_b": generated_is_b,
            })

    print(f"[offline] {len(prompts)} prompts total, TP={args.tensor_parallel_size}", flush=True)
    t0 = time.time()
    outputs = llm.generate(prompts, sp)
    dt = time.time() - t0
    print(f"[offline] done in {dt:.1f}s ({len(prompts)/dt:.2f} req/s)", flush=True)

    dump_path = args.out_dir / "reward" / f"reward-offline-{os.getpid()}.jsonl"
    with dump_path.open("w") as fh:
        for meta, out in zip(metadata, outputs):
            first = out.outputs[0]
            row = dict(meta)
            row["judge_raw_content"] = first.text
            row["judge_finish_reason"] = first.finish_reason
            row["judge_model"] = MODEL_ID
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    meta_out = {
        "model": MODEL_ID,
        "thinking_mode": "off",
        "backend": "offline",
        "tensor_parallel_size": args.tensor_parallel_size,
        "sampling": SAMPLING_OFF,
        "guided_decoding_schema": "required rating only (same as PERSONA_JUDGE_JSON_SCHEMA)",
        "n_pairs": len(pairs),
        "n_calls": len(prompts),
        "wall_seconds": dt,
        "req_per_s": len(prompts) / dt if dt > 0 else 0.0,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    (args.out_dir / "run_metadata.json").write_text(json.dumps(meta_out, indent=2))
    print(f"[offline] wrote {dump_path} and metadata", flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write the slurm launcher**

```bash
#!/bin/bash
#SBATCH --job-name=sweep_offline
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH --mem=512G
#SBATCH --time=04:00:00
#SBATCH --output=/home/lancewicki/projects/turing-rl/logs/sweep_offline-%j.out
#SBATCH --partition=a100
#SBATCH --account=rfai

set -uo pipefail
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY
export HF_HOME=/home/lancewicki/data/hf_cache
export HF_HUB_CACHE=/home/lancewicki/data/hf_cache
export PYTHONUNBUFFERED=1
export VLLM_LOGGING_LEVEL=INFO

REPO=/home/lancewicki/projects/turing-rl
PY=/home/lancewicki/miniconda3/envs/judge-vllm/bin/python
PAIRS=$REPO/results/2026-07-08-judge-sweep/raw/pairs/prism_heldout_880.parquet
OUT=$REPO/results/2026-07-08-judge-sweep/raw/sweep/qwen3-8b/off_offline
mkdir -p "$OUT"

cd "$REPO"
$PY scripts/run_offline_sweep_cell.py \
    --pairs "$PAIRS" \
    --out_dir "$OUT" \
    --tensor_parallel_size 8
RC=$?
echo "=== exit: $RC ==="
exit $RC
```

- [ ] **Step 3: Submit**

```bash
chmod +x /storage/home/lancewicki/projects/turing-rl/scripts/slurm/offline_sweep_cell.sh
JOB=$(sbatch --parsable /storage/home/lancewicki/projects/turing-rl/scripts/slurm/offline_sweep_cell.sh)
tail -f /home/lancewicki/projects/turing-rl/logs/sweep_offline-$JOB.out
```

Expected: 1760 outputs written to `raw/sweep/qwen3-8b/off_offline/reward/reward-offline-<pid>.jsonl`, metadata with `req_per_s`. Should be 2-4× faster than the corresponding server cell.

- [ ] **Step 4: Commit scripts**

```bash
git add scripts/run_offline_sweep_cell.py scripts/slurm/offline_sweep_cell.sh
git commit -m "add offline-batched bonus sweep cell"
```

---

## Task 20: Analyzer — parse dumps to per-pair metrics parquet

**Files:**
- Create: `scripts/analyze_judge_sweep.py`

- [ ] **Step 1: Write the analyzer (part 1: parsing)**

```python
"""Compute derived metrics from raw judge-sweep dumps.

Reads results/2026-07-08-judge-sweep/raw/sweep/<cell>/<mode>/reward/*.jsonl,
produces results/2026-07-08-judge-sweep/derived/{summary.md, summary.parquet,
per_pair_metrics.parquet, plots/*.png}.

Idempotent: safe to delete derived/ and re-run at any time.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

# Full Turing rubric fields we look for in each judge response.
RUBRIC_FIELDS = [
    "immediate_target_score_a", "immediate_target_score_b",
    "human_goal_score_a", "human_goal_score_b",
    "communication_style_score_a", "communication_style_score_b",
    "response_a_source_copy", "response_b_source_copy",
    "source_copy_penalty_a", "source_copy_penalty_b",
    "response_a_wrong_target_or_role", "response_b_wrong_target_or_role",
    "wrong_target_or_role_penalty_a", "wrong_target_or_role_penalty_b",
    "response_a_unsupported_adversarial_reframing", "response_b_unsupported_adversarial_reframing",
    "unsupported_adversarial_reframing_penalty_a", "unsupported_adversarial_reframing_penalty_b",
    "response_a_assistant_like", "response_b_assistant_like",
    "assistant_like_penalty_a", "assistant_like_penalty_b",
    "base_score_a", "base_score_b",
    "response_a_score", "response_b_score",
    "score_gap", "reasoning", "rating",
]

RATING_RE = re.compile(r'"rating"\s*:\s*(\d+)')


def try_parse_json(text: str) -> dict | None:
    if not text:
        return None
    # Strip anything before first { and after last matching }.
    s = text.strip()
    if not s.startswith("{"):
        i = s.find("{")
        if i < 0: return None
        s = s[i:]
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    # Try trimming trailing garbage.
    for end in range(len(s), 0, -1):
        try:
            return json.loads(s[:end])
        except json.JSONDecodeError:
            continue
    return None


def recover_rating_from_text(text: str) -> int | None:
    if not text:
        return None
    m = RATING_RE.search(text)
    if not m: return None
    try:
        r = int(m.group(1))
        return r if 1 <= r <= 7 else None
    except ValueError:
        return None


def load_cell_rows(cell_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for jsonl in sorted((cell_dir / "reward").glob("*.jsonl")):
        with jsonl.open() as fh:
            for line in fh:
                line = line.strip()
                if not line: continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def per_call_features(row: dict) -> dict:
    text = row.get("judge_raw_content") or ""
    parsed = try_parse_json(text)
    rating = None
    parsed_rating = None
    if parsed is not None and isinstance(parsed.get("rating"), int):
        parsed_rating = int(parsed["rating"])
    recovered = recover_rating_from_text(text) if parsed_rating is None else None
    rating = parsed_rating if parsed_rating is not None else recovered
    field_presence = {f: (parsed is not None and f in parsed) for f in RUBRIC_FIELDS}
    length_chars = len(text)
    finish = row.get("judge_finish_reason") or ""
    usage = row.get("judge_usage") or {}
    completion_tokens = None
    if isinstance(usage, dict):
        completion_tokens = usage.get("completion_tokens")
    return {
        "pair_id": row.get("pair_id") or row.get("post_id"),
        "user_id": row.get("user_id"),
        "generated_is_b": bool(row.get("generated_is_b", row.get("randomized_order", False))),
        "rating": rating,
        "rating_parsed_from_json": parsed_rating is not None,
        "rating_recovered_from_text": recovered is not None and parsed_rating is None,
        "format_ok": parsed is not None and parsed_rating is not None,
        "budget_hit": finish == "length",
        "length_chars": length_chars,
        "completion_tokens": completion_tokens,
        "raw_text": text,
        **{f"has_{f}": field_presence[f] for f in RUBRIC_FIELDS},
    }
```

(Analyzer continues in Task 21 — file gets split across two tasks to keep each task 2–5 minutes.)

- [ ] **Step 2: Syntax check**

Run: `/home/lancewicki/miniconda3/envs/turing-rl-train/bin/python -c "import ast; ast.parse(open('/storage/home/lancewicki/projects/turing-rl/scripts/analyze_judge_sweep.py').read())"`
Expected: no output.

- [ ] **Step 3: Commit partial**

```bash
git add scripts/analyze_judge_sweep.py
git commit -m "add sweep analyzer: parsing + per-call features"
```

---

## Task 21: Analyzer — per-pair aggregation, cell summary, plots

**Files:**
- Modify: `scripts/analyze_judge_sweep.py` (append)

- [ ] **Step 1: Append the aggregator and plotting**

Append the following to `scripts/analyze_judge_sweep.py`:

```python


def aggregate_cell(cell: str, mode: str, calls: list[dict]) -> dict:
    df = pd.DataFrame(calls)
    if df.empty:
        return {"cell": cell, "mode": mode, "n_calls": 0}

    # Rebuild per-pair with both orderings.
    per_pair = defaultdict(dict)
    for _, r in df.iterrows():
        pid = r["pair_id"]
        side = "b" if r["generated_is_b"] else "a"
        per_pair[pid][f"rating_{side}"] = r["rating"]
        per_pair[pid][f"format_ok_{side}"] = r["format_ok"]
        per_pair[pid][f"budget_hit_{side}"] = r["budget_hit"]

    pair_df_rows = []
    for pid, d in per_pair.items():
        r_a = d.get("rating_a"); r_b = d.get("rating_b")
        # Ordering 1 (generated=B): pick human if rating <= 3.
        # Ordering 2 (generated=A): pick human if rating >= 5.
        pick_ord1 = (r_b is not None) and (r_b <= 3)
        pick_ord2 = (r_a is not None) and (r_a >= 5)
        abstain_ord1 = r_b == 4
        abstain_ord2 = r_a == 4
        both_parsed = (d.get("format_ok_a") and d.get("format_ok_b"))
        pair_df_rows.append({
            "pair_id": pid, "cell": cell, "mode": mode,
            "rating_gen_b": r_b, "rating_gen_a": r_a,
            "picks_human_gen_b": pick_ord1,
            "picks_human_gen_a": pick_ord2,
            "abstain_gen_b": abstain_ord1,
            "abstain_gen_a": abstain_ord2,
            "both_orderings_parsed": both_parsed,
        })
    pair_df = pd.DataFrame(pair_df_rows)

    total_calls = len(df)
    format_ok_rate = df["format_ok"].mean()
    recovered_rate = df["rating_recovered_from_text"].mean()
    budget_hit_rate = df["budget_hit"].mean()
    field_presence = {
        f.replace("has_", ""): df[f].mean()
        for f in df.columns if f.startswith("has_")
    }

    # Accuracy over both orderings (denominator = pairs with a rating in that ordering).
    n_ord1_rated = int(pair_df["rating_gen_b"].notna().sum())
    n_ord2_rated = int(pair_df["rating_gen_a"].notna().sum())
    acc_ord1 = pair_df["picks_human_gen_b"].sum() / max(n_ord1_rated, 1)
    acc_ord2 = pair_df["picks_human_gen_a"].sum() / max(n_ord2_rated, 1)
    n_both = int(pair_df["both_orderings_parsed"].sum())
    acc_both = 0.0
    if n_both > 0:
        m = pair_df["both_orderings_parsed"]
        acc_both = ((pair_df.loc[m, "picks_human_gen_b"].sum()
                    + pair_df.loc[m, "picks_human_gen_a"].sum()) / (2 * n_both))
    position_bias = abs(acc_ord1 - acc_ord2)

    # Rating distribution.
    all_ratings = df["rating"].dropna().astype(int).tolist()
    hist = {str(k): all_ratings.count(k) for k in range(1, 8)}
    rating_mean = float(np.mean(all_ratings)) if all_ratings else 0.0
    rating_mode = int(max(hist, key=lambda k: hist[k])) if all_ratings else 0

    return {
        "cell": cell, "mode": mode,
        "n_calls": total_calls,
        "n_pairs_ord1_rated": n_ord1_rated,
        "n_pairs_ord2_rated": n_ord2_rated,
        "n_pairs_both_parsed": n_both,
        "format_ok_rate": format_ok_rate,
        "rating_recovery_rate": recovered_rate,
        "budget_hit_rate": budget_hit_rate,
        "accuracy_gen_b": acc_ord1,
        "accuracy_gen_a": acc_ord2,
        "accuracy_both_orderings": acc_both,
        "position_bias_delta": position_bias,
        "rating_mean": rating_mean,
        "rating_mode": rating_mode,
        "field_presence": field_presence,
        "rating_histogram": hist,
    }, pair_df


def compute_kappa_vs_anchor(pair_dfs: dict[tuple[str, str], pd.DataFrame], anchor_cell: str) -> dict[tuple[str, str], float]:
    from sklearn.metrics import cohen_kappa_score
    kappas: dict[tuple[str, str], float] = {}
    for mode in ("off", "on"):
        anchor_df = pair_dfs.get((anchor_cell, mode))
        if anchor_df is None: continue
        anchor_map = {}
        for _, r in anchor_df.iterrows():
            if r["both_orderings_parsed"]:
                anchor_map[r["pair_id"]] = int(r["picks_human_gen_b"]) + int(r["picks_human_gen_a"])
        for (cell, m), df in pair_dfs.items():
            if m != mode or cell == anchor_cell: continue
            xs, ys = [], []
            for _, r in df.iterrows():
                pid = r["pair_id"]
                if pid not in anchor_map: continue
                if not r["both_orderings_parsed"]: continue
                xs.append(int(r["picks_human_gen_b"]) + int(r["picks_human_gen_a"]))
                ys.append(anchor_map[pid])
            if len(xs) >= 30:
                kappas[(cell, mode)] = cohen_kappa_score(xs, ys)
            else:
                kappas[(cell, mode)] = float("nan")
    return kappas


def write_summary(rows: list[dict], kappas: dict, out_md: Path, out_parquet: Path):
    summary = []
    for r in rows:
        summary.append({
            "cell": r["cell"], "mode": r["mode"],
            "n_calls": r["n_calls"],
            "format_ok": r["format_ok_rate"],
            "rating_recovery": r["rating_recovery_rate"],
            "budget_hit": r["budget_hit_rate"],
            "accuracy": r["accuracy_both_orderings"],
            "pos_bias_delta": r["position_bias_delta"],
            "rating_mean": r["rating_mean"],
            "rating_mode": r["rating_mode"],
            "kappa_vs_anchor": kappas.get((r["cell"], r["mode"]), float("nan")),
        })
    df = pd.DataFrame(summary).sort_values(["mode", "cell"])
    df.to_parquet(out_parquet, index=False)
    with out_md.open("w") as fh:
        fh.write("# Judge sweep summary\n\n")
        fh.write(df.to_markdown(index=False))
        fh.write("\n")


def write_plots(rows: list[dict], size_map: dict[str, int], out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out_dir.mkdir(parents=True, exist_ok=True)

    for metric in ("accuracy_both_orderings", "format_ok_rate", "budget_hit_rate", "position_bias_delta"):
        fig, ax = plt.subplots(figsize=(6, 4))
        for mode, marker in (("off", "o"), ("on", "s")):
            xs, ys, labels = [], [], []
            for r in rows:
                if r["mode"] != mode: continue
                size = size_map.get(r["cell"])
                if size is None: continue
                xs.append(size); ys.append(r[metric]); labels.append(r["cell"])
            if xs:
                order = sorted(range(len(xs)), key=lambda i: xs[i])
                xs = [xs[i] for i in order]
                ys = [ys[i] for i in order]
                ax.plot(xs, ys, marker=marker, label=f"thinking={mode}")
        ax.set_xscale("log")
        ax.set_xlabel("judge size (params or active-params for MoE)")
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} vs judge size")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / f"{metric}.png")
        plt.close(fig)


# Params-per-model (billions). MoE uses active params for the x-axis.
SIZE_MAP = {
    "qwen3-4b": 4, "qwen3-8b": 8, "qwen3-14b": 14, "qwen3-32b": 32,
    "qwen35-4b": 4, "qwen35-9b": 9, "qwen35-27b": 27,
    "qwen35-35b-a3b": 3,   # active params (MoE)
    "qwen35-397b": 17,     # active params (MoE)
}
ANCHOR_CELL = "qwen35-397b"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_root", type=Path,
                    default=Path("/storage/home/lancewicki/projects/turing-rl/results/2026-07-08-judge-sweep/raw"))
    ap.add_argument("--derived_root", type=Path,
                    default=Path("/storage/home/lancewicki/projects/turing-rl/results/2026-07-08-judge-sweep/derived"))
    args = ap.parse_args()
    args.derived_root.mkdir(parents=True, exist_ok=True)
    plots_dir = args.derived_root / "plots"

    all_pair_dfs: dict[tuple[str, str], pd.DataFrame] = {}
    all_calls: list[dict] = []
    summaries: list[dict] = []

    for cell_dir in sorted((args.raw_root / "sweep").iterdir()):
        if not cell_dir.is_dir(): continue
        cell = cell_dir.name
        for mode_dir in sorted(cell_dir.iterdir()):
            if not mode_dir.is_dir(): continue
            mode = mode_dir.name
            rows = load_cell_rows(mode_dir)
            calls = [per_call_features(r) for r in rows]
            all_calls.extend([dict(c, cell=cell, mode=mode) for c in calls])
            summ, pair_df = aggregate_cell(cell, mode, calls)
            summ["cell"] = cell; summ["mode"] = mode
            summaries.append(summ)
            all_pair_dfs[(cell, mode)] = pair_df
            print(f"[analyzer] {cell}/{mode}: n_calls={summ['n_calls']} acc={summ.get('accuracy_both_orderings', 0):.3f}", flush=True)

    kappas = compute_kappa_vs_anchor(all_pair_dfs, ANCHOR_CELL)

    write_summary(
        summaries, kappas,
        args.derived_root / "summary.md",
        args.derived_root / "summary.parquet",
    )

    per_pair_all = pd.concat(list(all_pair_dfs.values()), ignore_index=True) if all_pair_dfs else pd.DataFrame()
    per_pair_all.to_parquet(args.derived_root / "per_pair_metrics.parquet", index=False)

    write_plots(summaries, SIZE_MAP, plots_dir)
    print(f"[analyzer] done. summary={args.derived_root/'summary.md'}", flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Syntax check**

Run: `/home/lancewicki/miniconda3/envs/turing-rl-train/bin/python -c "import ast; ast.parse(open('/storage/home/lancewicki/projects/turing-rl/scripts/analyze_judge_sweep.py').read())"`
Expected: no output.

- [ ] **Step 3: Install matplotlib + scikit-learn if not present**

Run: `/home/lancewicki/miniconda3/envs/turing-rl-train/bin/python -c "import matplotlib, sklearn; print(matplotlib.__version__, sklearn.__version__)"`
Expected: prints versions. If ImportError, run: `TMPDIR=~/tmp/build PIP_CACHE_DIR=~/tmp/pip-cache /home/lancewicki/miniconda3/envs/turing-rl-train/bin/pip install matplotlib scikit-learn`

- [ ] **Step 4: Run against the sweep outputs**

Run: `/home/lancewicki/miniconda3/envs/turing-rl-train/bin/python /storage/home/lancewicki/projects/turing-rl/scripts/analyze_judge_sweep.py`
Expected: prints `[analyzer] done` and writes `derived/summary.md`, `derived/summary.parquet`, `derived/per_pair_metrics.parquet`, `derived/plots/*.png`.

- [ ] **Step 5: Commit**

```bash
git add scripts/analyze_judge_sweep.py
git commit -m "add sweep analyzer: aggregate, kappa, plots"
```

---

## Task 22: Results-tree README

**Files:**
- Create: `results/2026-07-08-judge-sweep/README.md`

- [ ] **Step 1: Write the README**

```markdown
# 2026-07-08 Judge Sweep — results tree

Spec: `docs/superpowers/specs/2026-07-08-judge-sweep-design.md`
Plan: `docs/superpowers/plans/2026-07-08-judge-sweep-implementation.md`

## Layout

```
raw/           # Immutable. Never re-generated. Captured once.
  pairs/                  880-row (human, generated) pair set (input to sweep)
  generator/              Heldout inference pickle + metadata
  sweep/<cell>/<mode>/    Per-cell reward-layer dumps, HTTP dumps, run_metadata, vllm server logs
  calibration/            Per-cell 50-pair calibration outputs + aggregated metadata
  family_smoke/           Qwen3 vs Qwen3.5 4B smoke reports
  logs/                   Slurm/vLLM stdout copies

derived/       # Regeneratable. Delete + re-run `scripts/analyze_judge_sweep.py`.
  summary.md              10-row metrics table
  summary.parquet         Same, machine-readable
  per_pair_metrics.parquet
  plots/                  accuracy_vs_size.png etc.
  family_decision.md      Written by hand from family_smoke reports
  calibration_report.md   From raw/calibration/
  split_verification.md   From scripts/verify_prism_split.sh
```

## Regenerating derived/

```bash
rm -rf results/2026-07-08-judge-sweep/derived
python scripts/analyze_judge_sweep.py
```

Fully idempotent. No judge calls re-run.
```

- [ ] **Step 2: Commit**

```bash
git add results/2026-07-08-judge-sweep/README.md
git commit -m "add results-tree README"
```

---

## Task 23: Append Results section to spec doc

**Files:**
- Modify: `docs/superpowers/specs/2026-07-08-judge-sweep-design.md`

- [ ] **Step 1: Replace the Results placeholder with the actual results**

Find the `## Results\n\n_(To be appended after execution.)_` at the bottom of the spec and replace with:

```markdown
## Results

Executed: <fill in date range>

### Sweep summary
See `results/2026-07-08-judge-sweep/derived/summary.md`.

### Plots
- `results/2026-07-08-judge-sweep/derived/plots/accuracy_both_orderings.png`
- `results/2026-07-08-judge-sweep/derived/plots/format_ok_rate.png`
- `results/2026-07-08-judge-sweep/derived/plots/budget_hit_rate.png`
- `results/2026-07-08-judge-sweep/derived/plots/position_bias_delta.png`

### Family decision
See `docs/superpowers/decisions/2026-07-08-family-decision.md`.

### Calibration
See `results/2026-07-08-judge-sweep/derived/calibration_report.md`.

### Split verification
See `results/2026-07-08-judge-sweep/derived/split_verification.md`.

### Offline-vs-server comparison (Step 5.2 bonus)
Server-mode Qwen3-8B thinking-off vs offline batched:
- Server cell: `results/2026-07-08-judge-sweep/raw/sweep/qwen3-8b/off/run_metadata.json` (req/s)
- Offline cell: `results/2026-07-08-judge-sweep/raw/sweep/qwen3-8b/off_offline/run_metadata.json` (req/s)
- Accuracy delta: computed from `per_pair_metrics.parquet` (should be within noise).

### Deviations from paper (record here as they emerge)
- Anchor served as GPTQ-Int4 (paper uses un-quantized weights).
- CoT model self-hosted thinking-off; paper's stated directive matches, code default diverged (see Section 4.1 Deviations block for the resolution).
- Judge sampling: model-card defaults, thinking-mode-dependent (paper does not specify judge sampling).

### Deliverables
- [ ] Full 3272-row CoT parquet: `data/sft/prism_full_s42_sft_cot.parquet`
- [ ] SFT adapter: `checkpoints/sft/qwen3_8b_prism_full_s42/final/`
- [ ] Heldout inference pickle: `results/2026-07-08-judge-sweep/raw/generator/heldout_inference.pkl`
- [ ] 880-pair parquet: `results/2026-07-08-judge-sweep/raw/pairs/prism_heldout_880.parquet`
- [ ] Sweep dumps: `results/2026-07-08-judge-sweep/raw/sweep/` (10 cells + 1 offline)
- [ ] Derived metrics + plots: `results/2026-07-08-judge-sweep/derived/`
```

- [ ] **Step 2: Fill in `<fill in date range>` with actual dates from your run**

- [ ] **Step 3: Commit**

```bash
git add docs/superpowers/specs/2026-07-08-judge-sweep-design.md
git commit -m "append results section to judge-sweep spec"
```

---

## Self-review (before executor starts)

Checked against spec sections:
- Section 1 (goal/anchor/scope, sampling, JSON schema, non-goals, deliverables) → Tasks 14–18 (sweep infra), 20-21 (metrics).
- Section 2 (PRISM split verification, 7 checks, byte-diff) → Tasks 1, 2, 3.
- Section 3 (data foundation, row schema, pair construction, position bias) → Tasks 11, 12.
- Section 4.1 (batched CoT) → Task 5, 8. Deviations block → committed already in the spec.
- Section 4.1a (batched-vs-served parity) → Tasks 6, 7.
- Section 4.2 (build_sft_jsonl) → Task 9.
- Section 4.3 (LoRA SFT + auto-resume patch) → Tasks 4, 10.
- Section 4.4 (heldout inference) → Task 11.
- Section 4.5 (pair-set construction) → Task 12.
- Section 5.0 (calibration) → Task 17.
- Section 5.1 (sweep serving plan + client + orchestrator) → Tasks 14, 15, 16, 18.
- Section 5.2 (offline batched bonus) → Task 19.
- Section 5 (metrics + output artifacts + results tree) → Tasks 20, 21, 22.
- Section 6.1 (analysis pipeline) → Tasks 20, 21.
- Section 6.2 (ordering) → Reflected in task numbering.
- Section 6.3 (risks) → Handled inline (calibration gate, resume patch, parity gate).
- Section 6.4 (guardrails) → No task violates them.
- Results section → Task 23.
- Family selection → Task 13.

Type/name consistency check: `PORT_BASE`, `MODEL`, `TP`, `REPLICAS`, `THINKING_MODE`, `CELL_NAME` used consistently across Tasks 14/16/17; `pair_id` schema `<user>::<post>::<idx>` in Task 12 matches `pair_id` field consumed by Task 15 and Task 20; `SIZE_MAP` in Task 21 keys match cell names in Task 16's launcher arrays.

Placeholders scanned: no "TBD", "TODO", "similar to Task N", or "handle edge cases" left. Every code step shows the actual code. Every command has expected output.

One known gap: Task 15's `run_judge_sweep_cell.py` sets `OPENAI_API_BASE` via env var for each request, which relies on `shared/api_client.py` reading it fresh per call. If it caches at import time, replicas will all hit endpoint[0]. If Task 15's step-3 sanity run shows only one endpoint getting traffic, patch by threading the URL through explicitly (e.g., subclass or monkey-patch `shared.api_client.chat_url`).

---

Plan complete and saved to `docs/superpowers/plans/2026-07-08-judge-sweep-implementation.md`. Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
