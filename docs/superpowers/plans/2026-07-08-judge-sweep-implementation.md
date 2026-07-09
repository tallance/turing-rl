# Judge-Model Comparison Experiment — Implementation Plan (FINAL, merged)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **This is the single canonical plan.** It merges an earlier unaudited draft (complete runnable script bodies) with an independent TDD-structured draft, and adds the three patches the spec audit introduced, OpenRouter-probe sampling, a config SSOT, a round-robin race fix, and the Mac→cluster agent-comms workflow. It is fully self-contained — every referenced script and test is inlined below. ("the earlier draft" in rationale notes below refers to those two now-removed drafts; no external file is needed.)

**Goal:** Measure how well smaller Qwen judges approximate the paper's 397B training-reward judge on the pairwise Turing task, across thinking-on/off modes, on the PRISM heldout split — producing a reproducible `results/2026-07-08-judge-sweep/` tree.

**Architecture:** Build/unit-test the net-new *software* (env-gated judge-path patches, reward-layer dumps, served CoT client, config-driven sweep-cell client, offline analyzer) with local TDD on the Mac; run the GPU/data steps on the cluster via `docs/agent-comms/`. Every judge call goes through the real GRPO reward path (`training/grpo/reward.py`) so the sweep is apples-to-apples with training-time reward. CoT generation uses the upstream **served** path (faithful to `data/sft/generate_cot.py`), self-hosted. Family choice (Qwen3 vs Qwen3.5) is a runtime config input, not a code fork.

**Tech Stack:** Python 3, `aiohttp`, vLLM (OpenAI-compatible server; `turing-rl-train` env vLLM 0.18 for CoT/SFT/inference/smaller judges, `judge-vllm` env for the 397B anchor), TRL `SFTTrainer` + PEFT LoRA, `pandas`/`pyarrow`, `pytest`, `matplotlib`/`scikit-learn`, Slurm (3× A100-40GB nodes).

---

## Global Constraints

Copied verbatim from the spec + the four decisions taken during planning. Every task's requirements implicitly include this section.

- **Maximum fidelity to the paper/upstream code.** Use upstream code as-is; revert patches for the SFT path unless strictly necessary. Any deviation is a documented caveat, never silently absorbed.
- **Self-hosted only in production.** No OpenRouter/Morph on the cluster. The **only** OpenRouter use is two one-off Mac-side validation tasks (Task 1 sampling probe, Task 9 CoT fidelity check) against the user's personal $10 account.
- **Sampling policy = probe-then-match.** The paper's code passes *no* sampling params; OpenRouter/Morph applied provider defaults. Task 1 measures those defaults; whatever it finds becomes the frozen sampling for every cell (anchor included), recorded in `derived/sampling_fidelity.md`. Do NOT hard-code the spec's model-card table until Task 1 confirms it.
- **CoT generation = served + self-hosted**, matching upstream `generate_cot.py`'s HTTP path (not batched offline). Thinking-off via `chat_template_kwargs={"enable_thinking": False}`.
- **Reasoning parser = `--reasoning-parser qwen3`** on every thinking-on vLLM server (NOT `deepseek_r1`). Thinking-off servers run no reasoning parser. Re-validate the existing `_extract_chat_content` empty-return patch (written for `deepseek_r1`) under `qwen3` (Task 6 smoke).
- **Thinking-mode toggle = on the wire** via `chat_template_kwargs={"enable_thinking": <bool>}` (self-hosted vLLM has no OpenRouter `reasoning` field).
- **JSON schema mode locked ON:** `PERSONA_JUDGE_JSON_SCHEMA=1` for every judge cell.
- **`max_completion_tokens=8192`** for every judge call.
- **Dumps on:** `PERSONA_JUDGE_DUMP_RATE=1.0` for every sweep cell — both HTTP-layer (existing `api_client.py` hook) and reward-layer (new, Task 6) dumps.
- **Frozen inputs:** the 880-row pair-set (Task 13) and `TURING_PROMPT`/rubric schema are never modified after freeze. Heldout split only — never sweep on SFT/GRPO train splits.
- **Anchor fixed:** `Qwen/Qwen3.5-397B-A17B-GPTQ-Int4`, TP=8, regardless of family. Re-baselined under Task-1 sampling.
- **Results layout:** `results/2026-07-08-judge-sweep/` split into immutable `raw/` and regeneratable `derived/`.
- **Cluster paths:** repo `/storage/home/lancewicki/projects/turing-rl` (= `$REPO`); HF cache `/home/lancewicki/data/hf_cache`; envs `/home/lancewicki/miniconda3/envs/turing-rl-train/bin/python` (vLLM 0.18, pandas/torch) and `/home/lancewicki/miniconda3/envs/judge-vllm/bin/python` (vLLM 0.23, anchor serving).
- **Commit style:** lower-case, short, imperative; **no** `feat:`/`scope:` prefixes (matches recent git log).
- **Sync/exec workflow:** edit → `git commit` → push to `mine/lancewicki/main`. Cluster execution is requested by writing `docs/agent-comms/2026-07-08-judge-sweep-implementation/mac-to-cluster.md` (referencing the task #), committing, pushing; the cluster agent runs it and replies in `cluster-to-mac.md` (never edit that file).

---

## Where each task runs

| Runs on Mac (local TDD) | Runs on cluster (via agent-comms) |
|---|---|
| Tasks 1, 4, 5, 6, 7, 8, 13, 14, 15, 18, 21 — code + unit tests on synthetic fixtures | Tasks 2, 3, 9, 10, 11, 12, 16, 17, 19, 20, 22 execution; and the *execution* half of 6/7/8/15/18/21 |

Each code task is written + unit-tested locally; GPU/data execution is a separate handoff. "Commit" uses `git`; push happens at each cluster handoff.

---

## File Structure

**New files (ours; under `scripts/`, `tests/`, `configs/`, `results/` — not tracked in `our_patches.md`):**
- `scripts/openrouter_sampling_probe.py` — Task 1.
- `tests/prism_verification_helpers.py`, `tests/test_prism_split_verification.py` — Task 2.
- `scripts/verify_prism_split.sh` — Task 3 (re-split-only hash compare).
- `scripts/generate_cot_served.py` — Task 8 served CoT client.
- `scripts/cot_fidelity_check.py` — Task 9 self-hosted-vs-OpenRouter check.
- `scripts/slurm/{cot_serve.sh, sft_full.sh, heldout_inference.sh, judge_serve_cell.sh, judge_sweep_cell.sh, offline_sweep_cell.sh, family_smoke.sh}` — serving/run sbatches.
- `scripts/launch_judge_sweep.sh` — multi-cell orchestrator (reads the config module).
- `scripts/build_judge_pairs.py` — Task 13.
- `scripts/run_judge_sweep_cell.py` — Task 15 sweep-cell client (also calibration via `--max_pairs`).
- `scripts/run_offline_sweep_cell.py` — Task 20 offline bonus cell.
- `configs/judge_sweep_cells.py` — Task 14 cell config SSOT.
- `scripts/analyze_judge_sweep.py` — Task 21 analyzer.
- `tests/{test_judge_payload.py, test_reward_dump_row.py, test_lora_sft_config.py, test_build_judge_pairs.py, test_sweep_cell_config.py, test_analyze_judge_sweep.py}` — unit tests.

**Modified upstream files (must be logged in `our_patches.md` — Task 22):**
- `shared/api_client.py` — `build_chat_payload` gains `sampling`/`chat_template_kwargs`; `PERSONA_DISABLE_OPENROUTER_EXTRAS` gate (Task 4).
- `training/grpo/reward.py` — `_openai_chat` reads sampling/thinking env + `_resolve_response_format` (Tasks 4, 5); new `_build_reward_dump_row`/`_dump_reward_call` wired into `_score_pairwise_likert_with_info` (Task 6).
- `training/sft/lora_sft.py` + `training/sft/configs/qwen3_8b_lora.yaml` — save-cadence yaml-configurable + `--resume_from_checkpoint auto` (Task 7).

---

## Interfaces (pinned names used across tasks)

- `build_chat_payload(*, model, messages, max_completion_tokens, response_format=None, reasoning, sampling: dict|None=None, chat_template_kwargs: dict|None=None) -> dict` (Task 4).
- Env read by `reward.py._openai_chat`: `PERSONA_JUDGE_SAMPLING` (JSON dict), `PERSONA_JUDGE_ENABLE_THINKING` (`"1"`/`"0"`/unset), `PERSONA_JUDGE_JSON_SCHEMA` (`"1"`), `PERSONA_REWARD_DUMP_DIR` (path), `PERSONA_DISABLE_OPENROUTER_EXTRAS` (`"1"`) — plus existing `PERSONA_JUDGE_DUMP_RATE`/`PERSONA_JUDGE_DUMP_DIR`, `JUDGE_MODEL`, `PERSONA_JUDGE_MAX_COMPLETION_TOKENS`, `OPENAI_API_BASE`.
- `_resolve_response_format() -> dict` (Task 5).
- `_build_reward_dump_row(**fields) -> dict`; `_dump_reward_call(row: dict) -> None` (Task 6). Output keys match `scripts/dump_viewer.py`'s reward-row contract exactly (detection key: `generated_is_b` present).
- `resolve_resume_checkpoint(arg, output_dir) -> str|None`; `save_kwargs_from_config(cfg) -> dict` (Task 7).
- `build_pairs(inference_pkl_path, test_parquet_path) -> (pandas.DataFrame, dict)` — cols `pair_id,user_id,post_id,target_idx,user_history,context,persona,human,generated` (Task 13).
- `tp_for_size(size_b: int, is_moe: bool) -> (tp, replicas)`; `cell_list(family: str) -> list[dict]` where each cell = `{cell_name, model_id, tp, replicas, size_b, is_moe}` (Task 14).
- Analyzer helpers `per_call_features(row)`, `aggregate_cell(cell, mode, calls) -> (dict, DataFrame)`, `compute_kappa_vs_anchor(...)`, `accuracy`/`budget_hit_rate`/`position_bias` (Task 21).

---

## Task 1: OpenRouter sampling-fidelity probe (Mac-only)

Resolves the sampling policy empirically before any server config is frozen.

**Files:** Create `scripts/openrouter_sampling_probe.py`; writes `results/2026-07-08-judge-sweep/derived/sampling_fidelity.md`.

**Interfaces:** Produces frozen thinking-on/off sampling dicts recorded in `sampling_fidelity.md`, consumed by Tasks 8/14/16.

- [ ] **Step 1: Write the probe** (sends the repo's exact OpenRouter payload — `reasoning={"enabled": bool}`, no sampling — N=20 per mode; records completion-token/length distributions + any echoed params).

```python
"""One-off: probe what sampling OpenRouter/Morph applies to Qwen3 by default.
Run from the Mac only. Requires OPENROUTER_API_KEY (personal $10 account).
NOT part of any cluster path."""
import argparse, json, os, statistics, urllib.request
from pathlib import Path

URL = "https://openrouter.ai/api/v1/chat/completions"
PROMPT = "Reply with one short sentence about the weather."

def call(reasoning_enabled: bool) -> dict:
    payload = {"model": "qwen/qwen3-8b",
               "messages": [{"role": "user", "content": PROMPT}],
               "max_completion_tokens": 512,
               "provider": {"order": ["Morph"], "allow_fallbacks": True}}
    if reasoning_enabled:
        payload["reasoning"] = {"enabled": True}
    req = urllib.request.Request(URL, data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
                 "Content-Type": "application/json"}, method="POST")
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
            rows[mode].append({"completion_tokens": u.get("completion_tokens"),
                               "content_len": len((d["choices"][0]["message"].get("content") or "")),
                               "params_echo": d.get("provider") or d.get("generation_config")})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        f.write("# OpenRouter Qwen3 sampling fidelity probe\n\n")
        for mode in ("on", "off"):
            lens = [r["completion_tokens"] for r in rows[mode] if r["completion_tokens"]]
            f.write(f"## thinking-{mode} (n={len(rows[mode])})\n")
            if lens:
                f.write(f"- completion_tokens: mean={statistics.mean(lens):.0f} min={min(lens)} max={max(lens)}\n")
            f.write(f"- sample params echo: {json.dumps(rows[mode][0]['params_echo'])}\n\n")
        f.write("## DECISION\n\nFrozen sampling to replicate server-side (fill after review):\n"
                "- thinking-on: T=?, top_p=?, top_k=?, min_p=?\n"
                "- thinking-off: T=?, top_p=?, top_k=?, min_p=?\n")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run** — `OPENROUTER_API_KEY=... python scripts/openrouter_sampling_probe.py --n 20`. Cost ≪ $1. Expected: `sampling_fidelity.md` written.

- [ ] **Step 3: Fill the DECISION block.** If OpenRouter echoes explicit sampling, replicate it server-side. If not (common), the honest match is vLLM's `generation_config.json` defaults for the model — record that with the caveat "OpenRouter defaults not observable; matched to model generation_config." Sharp length differences on/off ⇒ provider varied sampling; record accordingly.

- [ ] **Step 4: Commit** — `git add scripts/openrouter_sampling_probe.py results/2026-07-08-judge-sweep/derived/sampling_fidelity.md && git commit -m "add openrouter sampling-fidelity probe + decision"`

---

## Task 2: PRISM split verification suite (7 checks)

**Files:** Create `tests/prism_verification_helpers.py`, `tests/test_prism_split_verification.py`.

**Interfaces:** Consumes `data/prism/full_s42_history_sft40_grpo60_test10/*` + raw `HannahRoseKirk/prism-alignment`. Produces a passing test; unblocks downstream data use.

- [ ] **Step 1: Write the raw-PRISM loader helper**

```python
# tests/prism_verification_helpers.py
"""Loads the raw HF PRISM dataset (cached) for split-verification test 6."""
from __future__ import annotations
import os
from typing import Any
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HOME", "/home/lancewicki/data/hf_cache")
os.environ.setdefault("HF_DATASETS_CACHE", "/home/lancewicki/data/hf_cache/datasets")

def load_raw_prism_replies() -> dict[tuple[str, str, int], str]:
    """Return {(user_id, conversation_id, turn_idx): reply} for every [HUMAN] turn.
    NOTE: confirm the raw field name for turns during execution — this uses
    'conversation_turns'/role=='user'; if that KeyErrors, inspect one raw row."""
    from datasets import load_dataset
    ds = load_dataset("HannahRoseKirk/prism-alignment", "conversations", split="train")
    out: dict[tuple[str, str, int], str] = {}
    for row in ds:
        uid, cid = str(row["user_id"]), str(row["conversation_id"])
        t = 0
        for turn in row.get("conversation_turns", []) or []:
            if str(turn.get("role") or "").lower() != "user":
                continue
            out[(uid, cid, t)] = str(turn.get("content") or "")
            t += 1
    return out

def extra_info_key(extra_info: dict[str, Any]) -> tuple[str, str, int]:
    """Build the raw-PRISM lookup key from a row's extra_info.
    NOTE: confirm 'raw_user_id' vs 'user_id' and 'post_id' vs 'conversation_id'."""
    return (str(extra_info.get("raw_user_id", extra_info["user_id"])),
            str(extra_info["post_id"]), int(extra_info["target_idx"]))
```

- [ ] **Step 2: Write the 7-check pytest**

```python
# tests/test_prism_split_verification.py
"""Row-level verification of the paper-faithful PRISM split.
Complements tests/test_prism_split_determinism.py (counts + byte-determinism only)."""
from __future__ import annotations
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
}  # CONFIRM against split_metadata.json on first cluster run (agent-comms Request 1).

def _load(rel: str) -> pd.DataFrame:
    p = SPLIT_DIR / rel
    if not p.exists():
        pytest.skip(f"missing {p}; rebuild the split first")
    return pd.read_parquet(p)

def _users(df: pd.DataFrame) -> set[str]:
    return {str(e["user_id"]) for e in df["extra_info"]}

def test_1_files_exist():
    for name in list(EXPECTED_COUNTS) + ["split_metadata.json"]:
        assert (SPLIT_DIR / name).exists(), f"missing {name}"

@pytest.mark.parametrize("rel,exp", list(EXPECTED_COUNTS.items()))
def test_2_row_counts(rel, exp):
    assert len(_load(rel)) == exp["rows"], rel

@pytest.mark.parametrize("rel,exp", list(EXPECTED_COUNTS.items()))
def test_3_user_counts(rel, exp):
    assert len(_users(_load(rel))) == exp["users"], rel

def test_4_user_disjointness():
    sft, grpo, held = _users(_load("sft/train.parquet")), _users(_load("grpo/train.parquet")), _users(_load("test.parquet"))
    assert sft & grpo == set() and sft & held == set() and grpo & held == set()

@pytest.mark.parametrize("rel", list(EXPECTED_COUNTS))
def test_5_prompt_schema(rel):
    df = _load(rel); rng = random.Random(42)
    for i in rng.sample(range(len(df)), min(50, len(df))):
        row = df.iloc[i]
        prompt = list(row["prompt"])
        assert prompt and all(("role" in m and "content" in m) for m in prompt), f"{rel}[{i}] bad prompt"
        assert str(row["reward_model"]["ground_truth"]).strip(), f"{rel}[{i}] empty gt"
        for k in ("user_id", "post_id", "target_idx", "user_history", "context"):
            assert k in row["extra_info"], f"{rel}[{i}] extra_info missing {k}"
        assert row["data_source"] == "prism_alignment_user_sim", f"{rel}[{i}] data_source"

def test_6_heldout_gt_matches_raw():
    df = _load("test.parquet"); raw = load_raw_prism_replies(); rng = random.Random(42)
    bad = []
    for i in rng.sample(range(len(df)), 20):
        row = df.iloc[i]
        try:
            key = extra_info_key(dict(row["extra_info"]))
        except (KeyError, TypeError) as e:
            bad.append(f"row {i}: bad extra_info ({e})"); continue
        r = raw.get(key); got = str(row["reward_model"]["ground_truth"])
        if r is None:
            bad.append(f"row {i} key {key} not in raw PRISM")
        elif r.strip() != got.strip():
            bad.append(f"row {i} key {key}: raw != split (raw[:80]={r[:80]!r})")
    assert not bad, "heldout gt mismatches:\n" + "\n".join(bad)

def test_7_no_text_leak_heldout_from_sft_targets():
    sft, held = _load("sft/train.parquet"), _load("test.parquet")
    long_targets = {str(rm["ground_truth"]) for rm in sft["reward_model"] if len(str(rm["ground_truth"])) >= 60}
    rng = random.Random(42); leaks = []
    for i in rng.sample(range(len(held)), 20):
        gt = str(held.iloc[i]["reward_model"]["ground_truth"])
        if any(t in gt for t in long_targets):
            leaks.append(f"heldout[{i}] contains an SFT target verbatim")
    assert not leaks, "\n".join(leaks)
```
(`EXPECTED_COUNTS` and the `extra_info`/raw-turns field names are confirmed against agent-comms Request 1's report before this is expected to pass.)

- [ ] **Step 3: Hand off to cluster** (needs data + HF cache). Write `docs/agent-comms/2026-07-08-judge-sweep-implementation/mac-to-cluster.md`: "Run Task 2 — `cd $REPO && /home/lancewicki/miniconda3/envs/turing-rl-train/bin/python -m pytest tests/test_prism_split_verification.py -v`. If `split_metadata.json` keys, the raw turns field, or `extra_info` key names differ, report actual schema." Commit + push.

- [ ] **Step 4: On green report, commit** — `git add tests/prism_verification_helpers.py tests/test_prism_split_verification.py && git commit -m "add prism split row-level verification"`

---

## Task 3: PRISM re-split hash-compare orchestrator (re-split only)

Catches post-hoc parquet tampering. **Corrected vs v1:** re-run ONLY `split_data.py` (spec §2 explicitly says skip the expensive `build.py` rerun; the determinism test already covers pipeline determinism).

**Files:** Create `scripts/verify_prism_split.sh`; writes `derived/split_verification.md`.

- [ ] **Step 1: Write the orchestrator**

```bash
#!/bin/bash
# Re-run ONLY data/prism/split_data.py on the cached PRISM raw + current build,
# SHA-256 each output parquet, compare to the current split. NO build.py rerun.
set -uo pipefail
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY
REPO=/home/lancewicki/projects/turing-rl
PY=/home/lancewicki/miniconda3/envs/turing-rl-train/bin/python
TMP=$(mktemp -d /tmp/prism-verify.XXXXXX)
CUR_SPLIT=$REPO/data/prism/full_s42_history_sft40_grpo60_test10
FRESH=$TMP/full_s42_history_sft40_grpo60_test10
OUT=$REPO/results/2026-07-08-judge-sweep/derived/split_verification.md
mkdir -p "$FRESH" "$(dirname "$OUT")"
export HF_HOME=/home/lancewicki/data/hf_cache HF_HUB_CACHE=/home/lancewicki/data/hf_cache
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
cd "$REPO"
echo "=== re-split only (upstream data.prism.split_data) ==="
# CONFIRMED flags (data/prism/split_data.py:225-236): --input-dir, --output-dir,
# --heldout-user-frac, --grpo-frac, --seed. To reproduce the CURRENT split
# byte-identically, pass the SAME args scripts/slurm/split_prism_full_s42.sh used
# (mirror that script rather than relying on defaults).
$PY -u -m data.prism.split_data --input-dir "$REPO/data/prism/full_s42_history" \
    --output-dir "$FRESH" --seed 42 || { echo "split failed"; exit 2; }
STATUS=0
{ echo "# PRISM split verification (re-split hash compare)"; echo;
  echo "Date: $(date -Iseconds)"; echo;
  echo '| File | current | fresh | match |'; echo '|---|---|---|---|'; } > "$OUT"
for f in sft/train.parquet grpo/train.parquet grpo/val.parquet test.parquet; do
    cur=$CUR_SPLIT/$f; fresh=$FRESH/$f
    if [ ! -f "$cur" ] || [ ! -f "$fresh" ]; then
        echo "| $f | MISSING | MISSING | FAIL |" >> "$OUT"; STATUS=1; continue; fi
    cs=$(sha256sum "$cur" | cut -d' ' -f1); fs=$(sha256sum "$fresh" | cut -d' ' -f1)
    [ "$cs" = "$fs" ] && echo "| $f | ${cs:0:12} | ${fs:0:12} | OK |" >> "$OUT" \
        || { echo "| $f | ${cs:0:12} | ${fs:0:12} | FAIL |" >> "$OUT"; STATUS=1; }
done
echo >> "$OUT"; echo "## pytest 7-check suite" >> "$OUT"
$PY -m pytest tests/test_prism_split_verification.py -v > "$(dirname "$OUT")/split_verification_pytest.out" 2>&1 \
    && echo "all 7 checks passed" >> "$OUT" || { echo "FAILURES (see .out)" >> "$OUT"; STATUS=1; }
cat "$OUT"; rm -rf "$TMP"; exit $STATUS
```

- [ ] **Step 2: Hand off to cluster** — run `bash scripts/verify_prism_split.sh`; expect exit 0, all `OK`. If hash mismatch → surface diff, rebuild from fresh before proceeding (spec R4).

- [ ] **Step 3: Commit** — `git add scripts/verify_prism_split.sh && git commit -m "add prism re-split hash-compare orchestrator"`

---

## Task 4: Extend the judge payload for sampling + thinking-mode (env-gated)

Makes the real reward path able to set per-cell sampling and `enable_thinking` without diverging from GRPO. Defaults are no-ops.

**Files:** Modify `shared/api_client.py` (`build_chat_payload` ~72-89, `openrouter_request_extras` ~53-69); Modify `training/grpo/reward.py` (`_openai_chat` ~330-352); Test `tests/test_judge_payload.py`.

- [ ] **Step 1: Failing test**

```python
# tests/test_judge_payload.py
import os
from shared.api_client import build_chat_payload

def test_sampling_merged():
    p = build_chat_payload(model="m", messages=[], max_completion_tokens=8192,
                           reasoning=False, sampling={"temperature": 0.6, "top_k": 20})
    assert p["temperature"] == 0.6 and p["top_k"] == 20

def test_chat_template_kwargs():
    p = build_chat_payload(model="m", messages=[], max_completion_tokens=8192,
                           reasoning=False, chat_template_kwargs={"enable_thinking": False})
    assert p["chat_template_kwargs"] == {"enable_thinking": False}

def test_defaults_noop():
    p = build_chat_payload(model="m", messages=[], max_completion_tokens=8192, reasoning=False)
    assert "temperature" not in p and "chat_template_kwargs" not in p

def test_disable_openrouter_extras(monkeypatch):
    monkeypatch.setenv("PERSONA_DISABLE_OPENROUTER_EXTRAS", "1")
    p = build_chat_payload(model="m", messages=[], max_completion_tokens=8192, reasoning=False)
    assert "provider" not in p and "reasoning" not in p
```

- [ ] **Step 2: Run — expect FAIL** (`unexpected keyword argument 'sampling'`).

- [ ] **Step 3: Extend `build_chat_payload` + gate openrouter extras**

```python
def build_chat_payload(*, model: str, messages: list[dict], max_completion_tokens: int,
                       response_format: dict | None = None, reasoning: bool,
                       sampling: dict | None = None,
                       chat_template_kwargs: dict | None = None) -> dict[str, Any]:
    """Build a chat-completions payload."""
    payload: dict[str, Any] = {"model": model, "messages": messages,
                               "max_completion_tokens": int(max_completion_tokens)}
    if response_format:
        payload["response_format"] = response_format
    if sampling:
        payload.update(sampling)                      # T/top_p/top_k/min_p top-level (OpenAI-compat)
    if chat_template_kwargs:
        payload["chat_template_kwargs"] = chat_template_kwargs
    if os.environ.get("PERSONA_DISABLE_OPENROUTER_EXTRAS") != "1":
        payload.update(openrouter_request_extras(reasoning=reasoning))
    return payload
```

- [ ] **Step 4: Wire env into `reward.py._openai_chat`** (before the `build_chat_payload(...)` call):

```python
import json as _json  # ensure present at top of reward.py
_s = os.environ.get("PERSONA_JUDGE_SAMPLING")
_sampling = _json.loads(_s) if _s else None
_te = os.environ.get("PERSONA_JUDGE_ENABLE_THINKING")
_ctk = {"enable_thinking": _te == "1"} if _te in ("0", "1") else None
payload = build_chat_payload(model=..., messages=messages, max_completion_tokens=...,
                            response_format=response_format, reasoning=False,
                            sampling=_sampling, chat_template_kwargs=_ctk)
```

- [ ] **Step 5: Run — expect PASS.** `python -m pytest tests/test_judge_payload.py -v`

- [ ] **Step 6: Commit** — `git add shared/api_client.py training/grpo/reward.py tests/test_judge_payload.py && git commit -m "add env-gated sampling + enable_thinking to judge payload"`

---

## Task 5: `PERSONA_JUDGE_JSON_SCHEMA=1` strict-schema patch

**(Missing entirely from v1 — the sweep set the env but nothing read it.)**

**Files:** Modify `training/grpo/reward.py` (response_format at the `_openai_chat` call, ~line 504); Test `tests/test_judge_payload.py` (append).

- [ ] **Step 1: Append failing tests**

```python
from training.grpo.reward import _resolve_response_format

def test_json_schema_on(monkeypatch):
    monkeypatch.setenv("PERSONA_JUDGE_JSON_SCHEMA", "1")
    rf = _resolve_response_format()
    assert rf["type"] == "json_schema"
    assert "rating" in rf["json_schema"]["schema"]["required"]

def test_json_schema_off(monkeypatch):
    monkeypatch.delenv("PERSONA_JUDGE_JSON_SCHEMA", raising=False)
    assert _resolve_response_format() == {"type": "json_object"}
```

- [ ] **Step 2: Run — expect FAIL** (`_resolve_response_format` undefined).

- [ ] **Step 3: Implement + use**

```python
# training/grpo/reward.py
def _resolve_response_format() -> dict:
    """json_object by default; strict json_schema (rating required) when PERSONA_JUDGE_JSON_SCHEMA=1."""
    if os.environ.get("PERSONA_JUDGE_JSON_SCHEMA") == "1":
        return {"type": "json_schema", "json_schema": {"name": "turing_rating", "schema": {
            "type": "object",
            "properties": {"rating": {"type": "integer", "minimum": 1, "maximum": 7}},
            "required": ["rating"], "additionalProperties": True}}}
    return {"type": "json_object"}
```
Replace the literal `response_format={"type": "json_object"}` at the `_openai_chat` call with `response_format=_resolve_response_format()`.

- [ ] **Step 4: Run — expect PASS.**
- [ ] **Step 5: Commit** — `git add training/grpo/reward.py tests/test_judge_payload.py && git commit -m "add PERSONA_JUDGE_JSON_SCHEMA strict rating schema"`

---

## Task 6: Reward-layer dump helpers (viewer-compatible)

**(Missing entirely from v1 — its sweep client imported `_build_reward_dump_row`/`_dump_reward_call` that never existed.)** The HTTP-wire dump already exists (`our_patches.md`). This adds the reward-layer dump the GUI viewer renders (detection key `generated_is_b`).

**Files:** Modify `training/grpo/reward.py` (new helpers + `_openai_chat` returns latency/finish_reason/usage; wire into `_score_pairwise_likert_with_info` ~686-708); Test `tests/test_reward_dump_row.py`.

- [ ] **Step 1: Failing test pinning the full viewer contract** (from the dump_viewer parser: 30+ top-level keys; `rating` is NOT stored — the viewer derives it from `rating_gt_first`/`rating_gen_first`; `generated_is_b` must be present as a key even when False).

```python
# tests/test_reward_dump_row.py
from training.grpo.reward import _build_reward_dump_row
REQUIRED = {"generated_is_b","human_side","rating_gt_first","rating_gen_first","randomized_order",
    "response","ground_truth","context","user_history","judge_response","judge_prompt",
    "judge_raw_content","judge_reasoning","judge_latency_ms","judge_finish_reason","judge_model",
    "judge_usage","final_reward","turing_judge_score_raw","turing_judge_score_clipped",
    "source_copy_penalty","assistant_like_penalty","wrong_target_or_role_penalty",
    "unsupported_adversarial_reframing_penalty","call_id","user_id","post_id","target_idx",
    "persona","ts","worker_pid"}
KW = dict(response="g", ground_truth="h", context="c", user_history="hist", human_side="A",
    generated_is_b=True, randomized_order="gt_first", rating_gt_first=3, rating_gen_first=None,
    judge_response={"rating": 3, "reasoning": "..."}, judge_prompt="P", judge_raw_content="{...}",
    judge_reasoning="<think></think>", judge_latency_ms=1, judge_finish_reason="stop",
    judge_model="qwen3-8b", judge_usage={"completion_tokens": 9}, final_reward=0.3,
    turing_judge_score_raw=3.0, turing_judge_score_clipped=3.0, source_copy_penalty=0.0,
    assistant_like_penalty=0.0, wrong_target_or_role_penalty=0.0,
    unsupported_adversarial_reframing_penalty=0.0, call_id="c1", user_id="u", post_id="p",
    target_idx=0, persona="", ts=1.0, worker_pid=42)

def test_has_all_viewer_keys():
    assert REQUIRED <= set(_build_reward_dump_row(**KW))

def test_generated_is_b_present_when_false():
    assert "generated_is_b" in _build_reward_dump_row(**{**KW, "generated_is_b": False})

def test_rating_not_stored():
    assert "rating" not in _build_reward_dump_row(**KW)  # viewer derives it
```

- [ ] **Step 2: Run — expect FAIL.**

- [ ] **Step 3: Implement helpers**

```python
# training/grpo/reward.py
import time
_REWARD_DUMP_KEYS = ("generated_is_b","human_side","rating_gt_first","rating_gen_first",
    "randomized_order","response","ground_truth","context","user_history","judge_response",
    "judge_prompt","judge_raw_content","judge_reasoning","judge_latency_ms","judge_finish_reason",
    "judge_model","judge_usage","final_reward","turing_judge_score_raw","turing_judge_score_clipped",
    "source_copy_penalty","assistant_like_penalty","wrong_target_or_role_penalty",
    "unsupported_adversarial_reframing_penalty","call_id","user_id","post_id","target_idx",
    "persona","ts","worker_pid")

def _build_reward_dump_row(**f) -> dict:
    """Reward-layer dump row matching scripts/dump_viewer.py. `rating` intentionally
    omitted — the viewer derives it from rating_gt_first/rating_gen_first."""
    return {k: f.get(k) for k in _REWARD_DUMP_KEYS}

def _dump_reward_call(row: dict) -> None:
    if float(os.environ.get("PERSONA_JUDGE_DUMP_RATE", "0")) <= 0:
        return
    d = os.environ.get("PERSONA_REWARD_DUMP_DIR")
    if not d:
        return
    os.makedirs(d, exist_ok=True)
    job = os.environ.get("SLURM_JOB_ID", "local")
    with open(os.path.join(d, f"reward-{job}-{os.getpid()}.jsonl"), "a") as fh:
        fh.write(json.dumps(row, default=str) + "\n")
```

- [ ] **Step 4: Thread call-meta out of `_openai_chat` + wire the dump.** Make `_openai_chat` return (or stash on a contextvar) `judge_latency_ms` (time around the `post_chat_async` await), `judge_finish_reason`, `judge_usage`, and the raw content. In `_score_pairwise_likert_with_info`, at the final aggregation (~686-708), build the row from already-computed values (`human_side`, `generated_is_b`, `rating_gt_first`/`rating_gen_first`, the 4 penalties, `score`→`turing_judge_score_raw`, clipped, `final_reward`, the formatted prompt, raw text, parsed reasoning, `judge_response`=parsed dict incl. a `reasoning` key, ids/persona, `ts=time.time()`, `worker_pid=os.getpid()`) and call `_dump_reward_call(row)`.

- [ ] **Step 5: Run — expect PASS.**

- [ ] **Step 6: (Cluster) viewer smoke + qwen3-parser re-validation.** Handoff: 2-pair judge call with `PERSONA_JUDGE_DUMP_RATE=1.0 PERSONA_REWARD_DUMP_DIR=/tmp/rewtest` against a live Qwen3-8B served with `--reasoning-parser qwen3`, then `python scripts/dump_viewer.py --dumps /tmp/rewtest --port 8090`; confirm rows render as reward rows (all 10 tabs). Also confirm the existing `_extract_chat_content` empty-return patch still behaves under `qwen3` (no crash on length-truncated `<think>`). Report back.

- [ ] **Step 7: Commit** — `git add training/grpo/reward.py tests/test_reward_dump_row.py && git commit -m "add reward-layer dump rows (dump_viewer-compatible)"`

---

## Task 7: SFT checkpointing patches + full-run sbatch

**(Supersedes v1 Task 4, which only added resume. Spec patches #1+#2 require BOTH save-cadence config-ization AND resume.)**

**Files:** Modify `training/sft/lora_sft.py` (SFTConfig ~376, argparse ~240, `trainer.train()` ~413); Modify `training/sft/configs/qwen3_8b_lora.yaml`; Create `scripts/slurm/sft_full.sh`; Test `tests/test_lora_sft_config.py`.

- [ ] **Step 1: Failing tests**

```python
# tests/test_lora_sft_config.py
from training.sft.lora_sft import resolve_resume_checkpoint, save_kwargs_from_config

def test_resolve_auto_highest(tmp_path):
    (tmp_path/"checkpoint-10").mkdir(); (tmp_path/"checkpoint-70").mkdir()
    assert resolve_resume_checkpoint("auto", str(tmp_path)).endswith("checkpoint-70")

def test_resolve_auto_empty(tmp_path):
    assert resolve_resume_checkpoint("auto", str(tmp_path)) is None

def test_save_kwargs_steps():
    assert save_kwargs_from_config({"save_strategy": "steps", "save_steps": 10, "save_total_limit": 2}) == \
        {"save_strategy": "steps", "save_steps": 10, "save_total_limit": 2}

def test_save_kwargs_default_epoch():
    assert save_kwargs_from_config({}) == {"save_strategy": "epoch"}
```

- [ ] **Step 2: Run — expect FAIL.**

- [ ] **Step 3: Implement helpers + apply**

```python
# training/sft/lora_sft.py
import glob, os
def resolve_resume_checkpoint(arg: str | None, output_dir: str) -> str | None:
    if arg != "auto":
        return arg or None
    ck = glob.glob(os.path.join(output_dir, "checkpoint-*"))
    return max(ck, key=lambda p: int(p.rsplit("-", 1)[-1])) if ck else None
def save_kwargs_from_config(config: dict) -> dict:
    ss = config.get("save_strategy", "epoch")
    out = {"save_strategy": ss}
    if ss == "steps":
        out["save_steps"] = config.get("save_steps", 10)
        out["save_total_limit"] = config.get("save_total_limit", 2)
    return out
```
- In `SFTConfig(...)` replace hard-coded `save_strategy="epoch"` with `**save_kwargs_from_config(config)`.
- Add argparse `--resume_from_checkpoint` (default None) after `--exit_after_trainer_build`.
- Replace bare `trainer.train()` with `trainer.train(resume_from_checkpoint=resolve_resume_checkpoint(args.resume_from_checkpoint, output_dir))`.
- Note: yaml is selected by `--model` (no `--config` flag exists); `sft_full.sh` must NOT pass `--config`.

- [ ] **Step 4: yaml** — append to `training/sft/configs/qwen3_8b_lora.yaml`:
```yaml
save_strategy: steps
save_steps: 10
save_total_limit: 2
```

- [ ] **Step 5: `scripts/slurm/sft_full.sh`** — `--gres=gpu:8 --time=20:00:00`, `report_to: wandb` sed+trap (as in `sft_smoke.sh`), invoking:
```bash
$PY -u -m training.sft.lora_sft --model qwen3-8b \
    --data_path $REPO/data/sft/prism_full_s42_sft_cot.jsonl \
    --output_dir $REPO/checkpoints/sft/qwen3_8b_prism_full_s42 \
    --max_seq_length 8192 --resume_from_checkpoint auto
```
(CONFIRMED: `lora_sft.py --model` takes a `MODEL_MAP` **key** — `qwen3-8b` (→ `Qwen/Qwen3-8B`); passing the full HF id would fail `choices`. There is **no** `--config` flag; the yaml is selected from `--model`.)

- [ ] **Step 6: Run — expect PASS.** `python -m pytest tests/test_lora_sft_config.py -v`
- [ ] **Step 7: Commit** — `git add training/sft/lora_sft.py training/sft/configs/qwen3_8b_lora.yaml scripts/slurm/sft_full.sh tests/test_lora_sft_config.py && git commit -m "add sft step-checkpointing + auto-resume"`

---

## Task 8: Served CoT client (thinking-off, self-hosted)

Faithful to upstream `generate_cot.py`'s served/HTTP path. Reuses its business logic; swaps transport to self-hosted async round-robin with `enable_thinking=False`.

**Files:** Create `scripts/generate_cot_served.py`; Create `scripts/slurm/cot_serve.sh` (8-replica launcher). Test `tests/test_generate_cot_served.py`.

**Interfaces:** Produces `data/sft/prism_full_s42_sft_cot.parquet` (3272 rows) + `.cot_metadata.json`.

- [ ] **Step 1: Reuse the confirmed upstream symbols.** `data/sft/generate_cot.py` exports (verified locally): `RATIONALIZE_SYSTEM_PROMPT`, `RATIONALIZE_USER_TEMPLATE`, `REGEN_NUDGE`, `_row_context`, `_as_text`, `reasoning_leaks_reply`, `THINKING_TRACE_SOURCE`, and the per-row keys it writes into `extra_info` (`ground_truth_reasoning`, `thinking_trace_source`, `thinking_trace_model`, `thinking_trace_num_regen_attempts`, `thinking_trace_failed_leakage_guard`). The served client imports all of these. Do NOT call `generate_reasoning_for_row` (it is sync `post_chat_sync` + hard-coded `reasoning=True`); build the payload directly for async round-robin.

- [ ] **Step 2: Failing unit test** (payload shape + round-robin; no live HTTP)

```python
# tests/test_generate_cot_served.py
from scripts.generate_cot_served import build_cot_payload, pick_endpoint

def test_thinking_off_payload():
    p = build_cot_payload("Qwen/Qwen3-8B", [{"role": "user", "content": "hi"}],
                          sampling={"temperature": 0.7}, max_completion_tokens=4096)
    assert p["chat_template_kwargs"] == {"enable_thinking": False}
    assert "reasoning" not in p and "provider" not in p
    assert p["temperature"] == 0.7 and p["max_completion_tokens"] == 4096

def test_round_robin():
    eps = ["a", "b", "c"]
    assert [pick_endpoint(eps, i) for i in range(4)] == ["a", "b", "c", "a"]
```

- [ ] **Step 3: Implement** — async `aiohttp`, `asyncio.Semaphore(16*n_endpoints)`, round-robin by request index. `build_cot_payload` sets `max_completion_tokens=4096`, `chat_template_kwargs={"enable_thinking": False}`, sampling from `sampling_fidelity.md` (thinking-off), NO `reasoning`, NO `provider` (build the dict directly — do not route through OpenRouter extras). Per-request `api_base` = the chosen endpoint. After generation: `reasoning_leaks_reply` per row → collect leaked → regen up to 10 tries. Write parquet (writing `ground_truth_reasoning` into `extra_info` like upstream) + `.cot_metadata.json` (`{n_rows, endpoints, sampling, thinking:"off", wall_s, leak_regen_counts}`).

- [ ] **Step 4: `scripts/slurm/cot_serve.sh`** — 8 replicas of `Qwen/Qwen3-8B` TP=1 on ports 8000-8007 (no `--reasoning-parser` — thinking-off), env `turing-rl-train`; writes `endpoints.txt`; waits `/health`; then runs `generate_cot_served.py --endpoints endpoints.txt`.

- [ ] **Step 5: Run unit test — expect PASS.**

- [ ] **Step 6: (Cluster) full run.** Handoff: `sbatch scripts/slurm/cot_serve.sh`. Gate: 3272 rows, 0 residual leaks, spot-check 5 rows thinking-off style (no `<think>`). **Discard the 138-row smoke parquet.** Report row count + leak stats.

- [ ] **Step 7: Commit** — `git add scripts/generate_cot_served.py scripts/slurm/cot_serve.sh tests/test_generate_cot_served.py && git commit -m "add served thinking-off cot client"`

---

## Task 9: CoT fidelity check — self-hosted vs OpenRouter (Mac + cluster)

Validates the self-hosting substitution against the original's actual reference (OpenRouter Qwen3-8B).

**Files:** Create `scripts/cot_fidelity_check.py`; writes `derived/cot_fidelity.md`.

- [ ] **Step 1: Implement** — take 20 fixed SFT rows (seed 42). Generate CoT for each via (a) self-hosted served Qwen3-8B (reuse `generate_cot_served.build_cot_payload`, thinking-off) and (b) OpenRouter Qwen3-8B (the upstream `generate_cot.py` payload — `reasoning` off, Morph provider). Compare distributions: completion length, first-person vs third-person perspective, leakage rate; token-level match where a shared seed is honorable. Write PASS/FAIL + side-by-side of 5 rows to `cot_fidelity.md`. This is a *fidelity report*, not a hard gate — large divergence prompts a sampling/template re-check.

- [ ] **Step 2: Run** — self-hosted half on the cluster (needs a served Qwen3-8B); OpenRouter half from the Mac ($10 account). Coordinate via agent-comms: cluster writes its 20 outputs to a json; Mac runs the OpenRouter half + comparison. Report distributions.

- [ ] **Step 3: Commit** — `git add scripts/cot_fidelity_check.py results/2026-07-08-judge-sweep/derived/cot_fidelity.md && git commit -m "add self-hosted-vs-openrouter cot fidelity check"`

---

## Task 10: Build SFT JSONL (cluster orchestration)

**Files:** none new (upstream `data/sft/build_sft_jsonl.py`).

- [ ] **Step 1: (Cluster)** `python -m data.sft.build_sft_jsonl --input_parquet data/sft/prism_full_s42_sft_cot.parquet --output_jsonl data/sft/prism_full_s42_sft_cot.jsonl` (flags CONFIRMED: `--input_parquet`, `--output_jsonl`, `--max_rows`). It **requires** `ground_truth_reasoning` in each row's `extra_info` (raises otherwise) — Task 8's served client writes it, so this is satisfied. Output rows are `{"messages": [...]}` with the assistant turn from `format_sft_assistant_content(ground_truth, ground_truth_reasoning)`. Gate: `wc -l` == 3272; one line's assistant target contains the reasoning envelope + `[HUMAN]:`. Report back. No commit (data).

---

## Task 11: SFT training + heldout inference (cluster orchestration)

**Files:** Create `scripts/slurm/heldout_inference.sh`:

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
set -uo pipefail
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY
export HF_HOME=/home/lancewicki/data/hf_cache HF_HUB_CACHE=/home/lancewicki/data/hf_cache PYTHONUNBUFFERED=1
PY=/home/lancewicki/miniconda3/envs/turing-rl-train/bin/python
REPO=/home/lancewicki/projects/turing-rl
CKPT=$REPO/checkpoints/sft/qwen3_8b_prism_full_s42
TEST=$REPO/data/prism/full_s42_history_sft40_grpo60_test10/test.parquet
OUT_DIR=$REPO/results/2026-07-08-judge-sweep/raw/generator
OUT=$OUT_DIR/heldout_inference.pkl
mkdir -p "$OUT_DIR"; cd "$REPO"
$PY -u -m eval.generate_trained --checkpoint_dir "$CKPT" --test_parquet "$TEST" \
    --model_id Qwen/Qwen3-8B --gen_num 1 --output "$OUT" --conditioning_mode history \
    --vllm_tensor_parallel_size 1 --vllm_gpu_memory_utilization 0.6 --vllm_max_num_seqs 32
RC=$?
$PY -c "import json,os; json.dump({'checkpoint_dir':'$CKPT','test_parquet':'$TEST',\
'base_model':'Qwen/Qwen3-8B','gen_num':1,'sampling':'paper Table 4 PRISM defaults (domain-inferred)',\
'output':'$OUT','slurm_job_id':os.environ.get('SLURM_JOB_ID')}, open('$OUT_DIR/heldout_inference_metadata.json','w'), indent=2)"
echo "=== exit: $RC ==="; exit $RC
```

- [ ] **Step 1: (Cluster) SFT** — `sbatch scripts/slurm/sft_full.sh`; ~78 steps, ~8 checkpoints; on crash resubmit (auto-resume). Gate: `checkpoints/sft/qwen3_8b_prism_full_s42/final/` has adapter `.safetensors`; wandb loss curve.

- [ ] **Step 2: (Cluster) heldout inference** — `heldout_inference.sh` runs:
```bash
$PY -u -m eval.generate_trained --checkpoint_dir $CKPT \
    --test_parquet data/prism/full_s42_history_sft40_grpo60_test10/test.parquet \
    --model_id Qwen/Qwen3-8B --gen_num 1 --conditioning_mode history \
    --vllm_tensor_parallel_size 1 --vllm_gpu_memory_utilization 0.6 --vllm_max_num_seqs 32 \
    --output results/2026-07-08-judge-sweep/raw/generator/heldout_inference.pkl
```
Sampling is domain-inferred (prism → T=0.6, top_p=1.0, top_k=-1, pres_pen=0.5, max_tokens=2048; paper Table 4). Gate: 880 generations / 128 users; write `heldout_inference_metadata.json`. Report the pickle's actual nesting (per-user dict → `test_targets`/`test_results` → `generations`) so Task 13 aligns to it. Commit the launcher.

---

## Task 12: (reserved — folded into Task 11)

_No task. Heldout inference is Task 11 Step 2._

---

## Task 13: Pair-set builder (TDD)

**Files:** Create `scripts/build_judge_pairs.py`; Test `tests/test_build_judge_pairs.py`.

**Interfaces:** `build_pairs(inference_pkl, test_parquet) -> (df, meta)`; writes `raw/pairs/prism_heldout_880.parquet`.

- [ ] **Step 1: Failing test on synthetic fixtures**

```python
# tests/test_build_judge_pairs.py
import pickle, pandas as pd
from scripts.build_judge_pairs import build_pairs

def _make(tmp_path):
    infer = {"u1": {"test_results": [{"target_idx": 0, "user_id": "u1", "post_id": "p1",
              "generations": [{"text": "<reasoning>r</reasoning>[HUMAN]: hi there"}]}]}}
    (tmp_path/"inf.pkl").write_bytes(pickle.dumps(infer))
    pd.DataFrame([{"reward_model": {"ground_truth": "hello"},
        "extra_info": {"user_id": "u1", "post_id": "p1", "target_idx": 0,
                       "user_history": "h", "context": "c", "persona": ""}}]
        ).to_parquet(tmp_path/"test.parquet")
    return str(tmp_path/"inf.pkl"), str(tmp_path/"test.parquet")

def test_cols_and_strip(tmp_path):
    df, meta = build_pairs(*_make(tmp_path))
    assert list(df.columns) == ["pair_id","user_id","post_id","target_idx",
        "user_history","context","persona","human","generated"]
    assert df.iloc[0]["human"] == "hello" and df.iloc[0]["generated"] == "hi there"
    assert "<reasoning>" not in df.iloc[0]["generated"]

def test_flags_exact_matches(tmp_path):
    _, meta = build_pairs(*_make(tmp_path))
    assert "exact_match_count" in meta and "exact_match_frac" in meta
```

- [ ] **Step 2: Run — expect FAIL.**

- [ ] **Step 3: Implement** — flatten the pickle by user → per-target records (handle `test_results`/`test_targets` + `generations`/`outputs`, dict-or-str text — confirm from Task 11's report). Align to `test.parquet` by `(user_id, post_id, target_idx)`. `human = reward_model.ground_truth`; `generated = parse_reasoning_and_response(raw)[1].strip()` (returns a **tuple**, take index 1). Assert no residual `<reasoning>`; assert every row present. **Count** (don't drop/assert) `human == generated`; WARN if `exact_match_frac > 0.01`. `pair_id = f"{user_id}::{post_id}::{target_idx}"`. Return `(df, meta)`; `__main__` writes parquet + sidecar meta.

- [ ] **Step 4: Run — expect PASS.**
- [ ] **Step 5: (Cluster) run on real inference** — confirm 880 rows; report exact-match count; freeze the parquet.
- [ ] **Step 6: Commit** — `git add scripts/build_judge_pairs.py tests/test_build_judge_pairs.py && git commit -m "add judge-pair builder + sanity checks"`

---

## Task 14: Cell config SSOT (TDD)

Single source of truth for the 10-cell matrix + TP/replica lookup — imported by the launcher (Task 16) and analyzer (Task 21), replacing v1's three-place duplication.

**Files:** Create `configs/judge_sweep_cells.py`; Test `tests/test_sweep_cell_config.py`.

- [ ] **Step 1: Failing test**

```python
# tests/test_sweep_cell_config.py
from configs.judge_sweep_cells import tp_for_size, cell_list, SIZE_MAP, ANCHOR_CELL

def test_tp_lookup():
    assert tp_for_size(4, False) == (1, 8)
    assert tp_for_size(14, False) == (1, 8)
    assert tp_for_size(27, False) == (2, 4)
    assert tp_for_size(32, False) == (2, 4)
    assert tp_for_size(35, True) == (1, 8)     # MoE-Int4

def test_cell_list_qwen35():
    cells = cell_list("qwen3.5")
    assert len(cells) == 5                      # 4 judges + anchor (modes handled by launcher)
    assert any(c["model_id"].endswith("397B-A17B-GPTQ-Int4") for c in cells)

def test_size_map_covers_all_cells():
    for c in cell_list("qwen3.5") + cell_list("qwen3"):
        assert c["cell_name"] in SIZE_MAP
```

- [ ] **Step 2: Run — expect FAIL.**

- [ ] **Step 3: Implement**

```python
# configs/judge_sweep_cells.py
ANCHOR = {"cell_name": "qwen35-397b", "model_id": "Qwen/Qwen3.5-397B-A17B-GPTQ-Int4",
          "tp": 8, "replicas": 1, "size_b": 17, "is_moe": True}
_FAMILIES = {
    "qwen3":   [("qwen3-4b","Qwen/Qwen3-4B",4,False),("qwen3-8b","Qwen/Qwen3-8B",8,False),
                ("qwen3-14b","Qwen/Qwen3-14B",14,False),("qwen3-32b","Qwen/Qwen3-32B",32,False)],
    "qwen3.5": [("qwen35-4b","Qwen/Qwen3.5-4B",4,False),("qwen35-9b","Qwen/Qwen3.5-9B",9,False),
                ("qwen35-27b","Qwen/Qwen3.5-27B",27,False),
                ("qwen35-35b-a3b","Qwen/Qwen3.5-35B-A3B-GPTQ-Int4",35,True)],
}
def tp_for_size(size_b: int, is_moe: bool) -> tuple[int, int]:
    """Dense >=20B -> TP2/4 replicas; else TP1/8 replicas (MoE-Int4 counts as small)."""
    if not is_moe and size_b >= 20:
        return (2, 4)
    return (1, 8)
def cell_list(family: str) -> list[dict]:
    out = []
    for name, mid, size_b, is_moe in _FAMILIES[family]:
        tp, rep = tp_for_size(size_b, is_moe)
        out.append({"cell_name": name, "model_id": mid, "tp": tp, "replicas": rep,
                    "size_b": size_b, "is_moe": is_moe})
    out.append(dict(ANCHOR))
    return out
# x-axis sizes (active-params for MoE); anchor plotted at 17B active.
SIZE_MAP = {"qwen3-4b":4,"qwen3-8b":8,"qwen3-14b":14,"qwen3-32b":32,
            "qwen35-4b":4,"qwen35-9b":9,"qwen35-27b":27,"qwen35-35b-a3b":3,"qwen35-397b":17}
ANCHOR_CELL = "qwen35-397b"
```

- [ ] **Step 4: Run — expect PASS.**
- [ ] **Step 5: Commit** — `git add configs/judge_sweep_cells.py tests/test_sweep_cell_config.py && git commit -m "add config-driven cell list ssot"`

---

## Task 15: Sweep-cell client (process-sharded round-robin, reward path)

**Corrected vs v1:** v1 mutated `os.environ["OPENAI_API_BASE"]` per async task — a concurrency race. Here each endpoint gets its **own client process** (pairs sharded across endpoints), so `OPENAI_API_BASE` is set once per process. Calls the real reward path (`_score_pairwise_likert_with_info`), reuses the Task-6 dump helpers.

**Files:** Create `scripts/run_judge_sweep_cell.py`; Test `tests/test_run_judge_sweep_cell.py`.

**Interfaces:** Consumes pair-set parquet, an endpoints list, a cell dict. Produces reward + HTTP dumps under `raw/sweep/{cell}/{mode}/{reward,http}/` + `run_metadata.json`. `--max_pairs N` for calibration; `--endpoint_index k --num_endpoints n` for sharding.

- [ ] **Step 1: Failing test for env setup + shard math (no HTTP)**

```python
# tests/test_run_judge_sweep_cell.py
from scripts.run_judge_sweep_cell import cell_env, shard_indices, cell_output_dirs

def test_cell_env_locks_config():
    env = cell_env(model_id="Qwen/Qwen3-8B", mode="off", sampling={"temperature": 0.7},
                   out_dir="/tmp/x")
    assert env["PERSONA_JUDGE_JSON_SCHEMA"] == "1"
    assert env["PERSONA_JUDGE_DUMP_RATE"] == "1.0"
    assert env["PERSONA_JUDGE_ENABLE_THINKING"] == "0"
    assert env["PERSONA_DISABLE_OPENROUTER_EXTRAS"] == "1"
    assert env["JUDGE_MODEL"] == "Qwen/Qwen3-8B"
    assert env["PERSONA_JUDGE_MAX_COMPLETION_TOKENS"] == "8192"

def test_shard():
    assert shard_indices(list(range(10)), endpoint_index=0, num_endpoints=2) == [0,2,4,6,8]
    assert shard_indices(list(range(10)), endpoint_index=1, num_endpoints=2) == [1,3,5,7,9]

def test_output_dirs(tmp_path):
    d = cell_output_dirs(str(tmp_path), "qwen3-8b", "off")
    assert d["reward"].endswith("qwen3-8b/off/reward") and d["http"].endswith("qwen3-8b/off/http")
```

- [ ] **Step 2: Run — expect FAIL.**

- [ ] **Step 3: Implement** — `cell_env(...)` sets the locked env (JSON_SCHEMA=1, DUMP_RATE=1.0, ENABLE_THINKING by mode, PERSONA_JUDGE_SAMPLING=json, DISABLE_OPENROUTER_EXTRAS=1, JUDGE_MODEL, MAX_COMPLETION_TOKENS=8192, PERSONA_JUDGE_DUMP_DIR=http dir, PERSONA_REWARD_DUMP_DIR=reward dir, OPENAI_API_BASE=this shard's endpoint). Apply to `os.environ` **before** importing `training.grpo.reward`. `shard_indices` selects this process's pairs. For each pair, `await score_turing_with_info(...)` (drives `_score_pairwise_likert_with_info`, both orderings) under `asyncio.Semaphore(16)`. The Task-6 dump wiring emits reward+http rows automatically. Rank-0 process (endpoint_index 0) writes `run_metadata.json` (model, mode, endpoints, concurrency, sampling, schema state, slurm id, timestamps, pair source).

- [ ] **Step 4: Run — expect PASS.**
- [ ] **Step 5: (Cluster) 10-pair smoke** against a live 8B endpoint; confirm reward+http dumps land + viewer renders. Report back.
- [ ] **Step 6: Commit** — `git add scripts/run_judge_sweep_cell.py tests/test_run_judge_sweep_cell.py && git commit -m "add sweep-cell client (process-sharded, reward path)"`

---

## Task 16: Serving sbatch + multi-cell launcher (reads config SSOT)

**Files:** Create `scripts/slurm/judge_sweep_cell.sh` (all-in-one: boots N replicas + launches N process-sharded clients) and `scripts/launch_judge_sweep.sh` (reads `configs/judge_sweep_cells.py`). One script handles every cell including the anchor (REPLICAS=1, TP=8).

- [ ] **Step 1: `scripts/slurm/judge_sweep_cell.sh`**

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
# Required env: MODEL TP REPLICAS THINKING_MODE CELL_NAME. Optional: PORT_BASE(8130), MAX_PAIRS.
set -uo pipefail
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY
for v in MODEL TP REPLICAS THINKING_MODE CELL_NAME; do [ -z "${!v:-}" ] && { echo "ERROR: $v unset"; exit 2; }; done
PORT_BASE=${PORT_BASE:-8130}; MAX_PAIRS=${MAX_PAIRS:-}
[ $((REPLICAS*TP)) -gt 8 ] && { echo "ERROR: REPLICAS*TP>8"; exit 2; }
export HF_HOME=/home/lancewicki/data/hf_cache HF_HUB_CACHE=/home/lancewicki/data/hf_cache
export HF_HUB_DISABLE_XET=1 PYTHONUNBUFFERED=1 VLLM_LOGGING_LEVEL=INFO
REPO=/home/lancewicki/projects/turing-rl
# 397B anchor serves from judge-vllm; smaller judges + the client run from turing-rl-train.
case "$MODEL" in *397B*) PY_SERVER=/home/lancewicki/miniconda3/envs/judge-vllm/bin/python ;;
                 *) PY_SERVER=/home/lancewicki/miniconda3/envs/turing-rl-train/bin/python ;; esac
PY_CLIENT=/home/lancewicki/miniconda3/envs/turing-rl-train/bin/python
PAIRS=$REPO/results/2026-07-08-judge-sweep/raw/pairs/prism_heldout_880.parquet
OUT_DIR=$REPO/results/2026-07-08-judge-sweep/raw/sweep/$CELL_NAME/$THINKING_MODE
mkdir -p "$OUT_DIR/vllm_server" "$OUT_DIR/reward" "$OUT_DIR/http"
RP=(); [ "$THINKING_MODE" = "on" ] && RP=(--reasoning-parser qwen3)
PIDS=(); URLS=()
for i in $(seq 0 $((REPLICAS-1))); do
  gpus=$(seq -s, $((i*TP)) $((i*TP+TP-1))); port=$((PORT_BASE+i))
  CUDA_VISIBLE_DEVICES=$gpus $PY_SERVER -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --download-dir "$HF_HOME" --tensor-parallel-size "$TP" \
    --max-model-len 32768 --gpu-memory-utilization 0.85 --dtype bfloat16 \
    "${RP[@]}" --host 0.0.0.0 --port $port > "$OUT_DIR/vllm_server/replica_$i.log" 2>&1 &
  PIDS+=($!); URLS+=("http://localhost:$port/v1")
done
cleanup() { for p in "${PIDS[@]}"; do kill $p 2>/dev/null || true; done; }
trap cleanup EXIT
for i in $(seq 0 $((REPLICAS-1))); do port=$((PORT_BASE+i)); ok=0
  for t in $(seq 1 900); do curl -sf -m 2 http://localhost:$port/v1/models >/dev/null 2>&1 && { ok=1; break; }; sleep 2; done
  [ $ok -eq 1 ] || { echo "TIMEOUT replica $i"; exit 3; }; echo "replica $i ready"; done
ENDPOINTS=$(IFS=,; echo "${URLS[*]}")
EXTRA=""; [ -n "$MAX_PAIRS" ] && EXTRA="--max_pairs $MAX_PAIRS"
cd "$REPO"; CLIENT_PIDS=()
for i in $(seq 0 $((REPLICAS-1))); do
  $PY_CLIENT scripts/run_judge_sweep_cell.py --pairs "$PAIRS" --endpoints "$ENDPOINTS" \
    --model "$MODEL" --thinking_mode "$THINKING_MODE" --out_dir "$OUT_DIR" \
    --concurrency_per_endpoint 16 --endpoint_index $i --num_endpoints $REPLICAS $EXTRA &
  CLIENT_PIDS+=($!)
done
RC=0; for p in "${CLIENT_PIDS[@]}"; do wait $p || RC=1; done
echo "=== clients exit: $RC ==="; exit $RC
```

- [ ] **Step 2: `scripts/launch_judge_sweep.sh`** (reads the config SSOT; no bash arrays)

```bash
#!/bin/bash
# FAMILY=qwen3.5 bash scripts/launch_judge_sweep.sh            # full sweep
# FAMILY=qwen3.5 CALIBRATION=1 bash scripts/launch_judge_sweep.sh   # 50-pair calibration
set -uo pipefail
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY
REPO=/home/lancewicki/projects/turing-rl
PY=/home/lancewicki/miniconda3/envs/turing-rl-train/bin/python
FAMILY=${FAMILY:?set FAMILY (qwen3 or qwen3.5)}; CALIBRATION=${CALIBRATION:-0}
cd "$REPO"
CELLS=$($PY -c "import json; from configs.judge_sweep_cells import cell_list; print(json.dumps(cell_list('$FAMILY')))")
echo "$CELLS" | $PY -c '
import json,sys,os,subprocess
cells=json.load(sys.stdin); calib=os.environ.get("CALIBRATION")=="1"
for c in cells:
  for mode in ("off","on"):
    exp=f"MODEL={c[\"model_id\"]},TP={c[\"tp\"]},REPLICAS={c[\"replicas\"]},THINKING_MODE={mode},CELL_NAME={c[\"cell_name\"]}"
    if calib: exp+=",MAX_PAIRS=50"
    subprocess.run(["sbatch","--parsable",f"--gres=gpu:{c[\"tp\"]*c[\"replicas\"]}",
      f"--job-name=sw_{c[\"cell_name\"]}_{mode}","--export=ALL,"+exp,
      "scripts/slurm/judge_sweep_cell.sh"],check=True)
    print("submitted",c["cell_name"],mode)
'
```
`FAMILY` comes from the Task-17 decision; `CALIBRATION=1` caps each cell at 50 pairs.

- [ ] **Step 3: `bash -n` both scripts; commit** — `git add scripts/slurm/judge_sweep_cell.sh scripts/launch_judge_sweep.sh && git commit -m "add all-in-one sweep-cell sbatch + config-driven launcher"`

---

## Task 17: Family gate — 4B smoke (cluster)

**Files:** none new — reuse `scripts/slurm/judge_sweep_cell.sh` (Task 16) for two candidate 4B cells; writes `derived/family_decision.md`.

- [ ] **Step 1: (Cluster)** run the sweep-cell script for each 4B candidate on 50 pairs, both modes (TP=1, REPLICAS=1 so they fit anywhere), e.g.:
```bash
for M in "qwen3-4b|Qwen/Qwen3-4B" "qwen35-4b|Qwen/Qwen3.5-4B"; do
  IFS='|' read -r name mid <<<"$M"
  for mode in off on; do
    sbatch --gres=gpu:1 --job-name=fam_${name}_${mode} \
      --export=ALL,MODEL=$mid,TP=1,REPLICAS=1,THINKING_MODE=$mode,CELL_NAME=fam_$name,MAX_PAIRS=50 \
      scripts/slurm/judge_sweep_cell.sh
  done
done
```
Compare tokens/sec (from each `run_metadata.json`'s `wall_seconds`) at parse-rate parity (from the reward dumps); tiebreak on 397B-agreement over the 50 pairs; inconclusive → default `qwen3.5` (spec R6).
- [ ] **Step 2:** write `derived/family_decision.md` with the chosen `FAMILY`, size list, and rationale. This gate sets `FAMILY` for Tasks 16/18/19.
- [ ] **Step 3: Commit** — `git add results/2026-07-08-judge-sweep/derived/family_decision.md && git commit -m "add family-selection 4b decision"`

---

## Task 18: Per-cell calibration + >4h gate

Uses the Task-15 client (`--max_pairs 50`) so calibration exercises the **real** judge path (v2 improvement over v1, which reused the divergent `benchmark_judge_throughput.py`).

**Files:** `scripts/calibration_report.py` (aggregator) → `raw/calibration/calibration_metadata.json`, `derived/calibration_report.md`.

- [ ] **Step 1: (Cluster)** `FAMILY=<decision> CALIBRATION=1 bash scripts/launch_judge_sweep.sh` (all 10 cells, ~1h parallel).
- [ ] **Step 2: aggregate** — `scripts/calibration_report.py` reads each cell's `run_metadata.json`, extrapolates, writes the report:

```python
"""Aggregate 50-pair calibration cells into a per-cell throughput report."""
import json
from pathlib import Path
ROOT = Path("/storage/home/lancewicki/projects/turing-rl/results/2026-07-08-judge-sweep")

def extrapolate_wall_hours(n_calls, wall_s, total_calls=1760):
    return (total_calls / (n_calls / wall_s)) / 3600 if wall_s > 0 and n_calls else float("inf")

def main():
    rows = []
    for meta_path in (ROOT / "raw" / "sweep").glob("*/*/run_metadata.json"):
        m = json.loads(meta_path.read_text())
        cell, mode = meta_path.parent.parent.name, meta_path.parent.name
        n = m.get("n_pairs", 0); wall = m.get("wall_seconds", 0.0); calls = n * 2
        req_s = calls / wall if wall > 0 else 0.0
        rows.append({"cell": cell, "mode": mode, "n_pairs": n, "wall_s": wall,
                     "req_per_s": req_s, "proj_1760_h": extrapolate_wall_hours(calls, wall)})
    (ROOT / "raw" / "calibration").mkdir(parents=True, exist_ok=True)
    (ROOT / "raw" / "calibration" / "calibration_metadata.json").write_text(json.dumps(rows, indent=2))
    out = ROOT / "derived" / "calibration_report.md"; out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        f.write("# Per-cell throughput calibration\n\n")
        f.write("_Precision caveat: 100 calls/cell → extrapolations are ±30%; use only for the >4h gate._\n\n")
        f.write("| Cell | Mode | Pairs | Wall(s) | Req/s | Proj 1760-call | >4h? |\n|---|---|---|---|---|---|---|\n")
        for r in sorted(rows, key=lambda x: (x["cell"], x["mode"])):
            gate = "**YES**" if r["proj_1760_h"] > 4 else "no"
            f.write(f"| {r['cell']} | {r['mode']} | {r['n_pairs']} | {r['wall_s']:.0f} "
                    f"| {r['req_per_s']:.2f} | {r['proj_1760_h']:.1f}h | {gate} |\n")
    print("wrote", out)

if __name__ == "__main__":
    main()
```
- [ ] **Step 3: Unit-test the extrapolation** (`tests/test_calibration_report.py`), run on the Mac:
```python
from scripts.calibration_report import extrapolate_wall_hours
def test_extrapolate(): assert round(extrapolate_wall_hours(100, 300, 1760), 2) == 1.47
def test_gate_over_4h(): assert extrapolate_wall_hours(100, 3600, 1760) > 4
```
- [ ] **Step 4: Gate:** any cell projecting >4h (<0.12 req/s) → surface, decide reduce-concurrency / drop-cell / accept (spec §5.0, R2: anchor may drop `max_completion_tokens` to 4096). Purge calibration dumps so the full sweep starts clean.
- [ ] **Step 5: Commit** — `git add scripts/calibration_report.py tests/test_calibration_report.py results/2026-07-08-judge-sweep/derived/calibration_report.md && git commit -m "add per-cell calibration report + 4h gate"`

---

## Task 19: Full judge sweep (cluster)

**Files:** none new.

- [ ] **Step 1: (Cluster)** `FAMILY=<decision> bash scripts/launch_judge_sweep.sh`. Anchor (Node 1) runs 2 cells sequentially (on/off, TP=8); smaller judges parallel across Nodes 2&3 (one cell per node, node filled with replicas). Copy vLLM logs + slurm out into `raw/logs/`.
- [ ] **Step 2: Gate per cell:** reward-dump rows total ≥ 1760; `run_metadata.json` present. Re-submit any short cell. Report per-cell completion + anomalies. No commit (data).

---

## Task 20: Offline batched bonus cell

**Corrected vs v1:** reuse `_build_reward_dump_row` (Task 6) instead of hand-duplicating the 30-key row schema (DRY / drift-proof). Keep v1's GuidedDecodingParams + prompt-build.

**Files:** Create `scripts/run_offline_sweep_cell.py`, `scripts/slurm/offline_sweep_cell.sh`.

- [ ] **Step 1: Implement the offline cell** — in-process `LLM.generate` (TP=8) over all 1760 prompts (880 pairs × 2 orderings), Qwen3-8B thinking-off, `GuidedDecodingParams(json={required rating})`, `apply_chat_template(enable_thinking=False)`. Build each dump row via the shared helper (DRY — do not re-list the 30 keys):

```python
"""Offline batched vLLM sweep cell for Qwen3-8B thinking-off. Reuses the reward-dump
schema helper so the GUI viewer renders these identically to server cells."""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO_ROOT))
import pandas as pd
from shared.judge_prompts import TURING_PROMPT
from shared.judge_utils import build_source_copy_warning, format_source_copy_watchlist
from training.grpo.reward import _build_reward_dump_row   # DRY: shared viewer contract

MODEL_ID = "Qwen/Qwen3-8B"
SAMPLING_OFF = dict(temperature=0.7, top_p=0.8, top_k=20, min_p=0.0, max_tokens=8192)  # replace with Task-1 frozen values

def build_prompt(row: dict, generated_is_b: bool) -> str:
    resp_a, resp_b = (row["human"], row["generated"]) if generated_is_b else (row["generated"], row["human"])
    wa = build_source_copy_warning(resp_a, thread_context=row["context"])
    wb = build_source_copy_warning(resp_b, thread_context=row["context"])
    return TURING_PROMPT.format(persona=row.get("persona", ""), user_history=row["user_history"],
        context=row["context"], response_a=resp_a, response_b=resp_b,
        source_copy_watchlist=format_source_copy_watchlist([wa, wb], item_label="Response",
            labels=["Response A", "Response B"]))

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)   # .../raw/sweep/qwen3-8b/off_offline
    ap.add_argument("--tensor_parallel_size", type=int, default=8)
    args = ap.parse_args()
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import GuidedDecodingParams
    pairs = pd.read_parquet(args.pairs).to_dict(orient="records")
    (args.out_dir / "reward").mkdir(parents=True, exist_ok=True)
    llm = LLM(model=MODEL_ID, tensor_parallel_size=args.tensor_parallel_size,
              gpu_memory_utilization=0.85, max_model_len=32768, dtype="bfloat16")
    tok = llm.get_tokenizer()
    guided = GuidedDecodingParams(json={"type": "object",
        "properties": {"rating": {"type": "integer", "minimum": 1, "maximum": 7}},
        "required": ["rating"], "additionalProperties": True})
    sp = SamplingParams(guided_decoding=guided, **SAMPLING_OFF)
    prompts, meta = [], []
    for row in pairs:
        for gib in (True, False):
            chat = tok.apply_chat_template([{"role": "user", "content": build_prompt(row, gib)}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False)
            prompts.append(chat); meta.append((row, gib))
    t0 = time.time(); outs = llm.generate(prompts, sp); dt = time.time() - t0
    path = args.out_dir / "reward" / f"reward-offline-{os.getpid()}.jsonl"
    with path.open("w") as fh:
        for i, ((row, gib), o) in enumerate(zip(meta, outs)):
            first = o.outputs[0]
            fh.write(json.dumps(_build_reward_dump_row(
                generated_is_b=gib, human_side=("A" if gib else "B"),
                randomized_order=("gt_first" if not gib else "gen_first"),
                rating_gt_first=None, rating_gen_first=None,
                response=row["generated"], ground_truth=row["human"], context=row["context"],
                user_history=row["user_history"], judge_response={}, judge_prompt="",
                judge_raw_content=first.text, judge_reasoning="", judge_latency_ms=None,
                judge_finish_reason=first.finish_reason, judge_model=MODEL_ID, judge_usage={},
                final_reward=0.0, turing_judge_score_raw=0.0, turing_judge_score_clipped=0.0,
                source_copy_penalty=0.0, assistant_like_penalty=0.0,
                wrong_target_or_role_penalty=0.0, unsupported_adversarial_reframing_penalty=0.0,
                call_id=i, user_id=row["user_id"], post_id=row["post_id"],
                target_idx=row["target_idx"], persona="", ts=time.time(), worker_pid=os.getpid()),
                default=str) + "\n")
    (args.out_dir / "run_metadata.json").write_text(json.dumps({"model": MODEL_ID,
        "thinking_mode": "off", "backend": "offline", "tensor_parallel_size": args.tensor_parallel_size,
        "sampling": SAMPLING_OFF, "n_pairs": len(pairs), "n_calls": len(prompts),
        "wall_seconds": dt, "req_per_s": len(prompts) / dt if dt else 0.0,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID")}, indent=2))
    print(f"[offline] {len(prompts)} calls in {dt:.1f}s -> {path}", flush=True)

if __name__ == "__main__":
    main()
```
`scripts/slurm/offline_sweep_cell.sh`: `--gres=gpu:8 --time=04:00:00`, `judge-vllm` python, `--pairs <880 parquet> --out_dir .../raw/sweep/qwen3-8b/off_offline --tensor_parallel_size 8`.
- [ ] **Step 2: (Cluster) run** post-primary-sweep on a free node (~30-60 min). Gate: 1760 rows; accuracy matches the server cell within noise; higher throughput. Report the server-vs-offline ratio → `derived/offline_vs_server_comparison.md`.
- [ ] **Step 3: Commit** — `git add scripts/run_offline_sweep_cell.py scripts/slurm/offline_sweep_cell.sh && git commit -m "add offline batched bonus cell"`

---

## Task 21: Analyzer (raw → derived) with TDD metric tests

**Files:** Create `scripts/analyze_judge_sweep.py` (parsing → `aggregate_cell` → `compute_kappa_vs_anchor` (sklearn) → `write_summary`/`write_plots`), importing `SIZE_MAP`/`ANCHOR_CELL` from `configs.judge_sweep_cells`. Test `tests/test_analyze_judge_sweep.py`.

- [ ] **Step 1: Failing tests for the metric functions** (on synthetic reward rows)

```python
# tests/test_analyze_judge_sweep.py
from scripts.analyze_judge_sweep import per_call_features, aggregate_cell

def _row(rating, gen_b=True, finish="stop"):
    return {"pair_id":"p1","user_id":"u","generated_is_b":gen_b,
            "judge_finish_reason":finish,
            "judge_raw_content": f'{{"rating": {rating}}}'}

def test_per_call_parses_rating():
    f = per_call_features(_row(3))
    assert f["rating"] == 3 and f["format_ok"] and not f["budget_hit"]

def test_budget_hit():
    assert per_call_features(_row(3, finish="length"))["budget_hit"]

def test_aggregate_accuracy_tie_excluded():
    # both orderings rating 4 -> abstain -> excluded from accuracy
    calls = [per_call_features(_row(4, gen_b=True)), per_call_features(_row(4, gen_b=False))]
    summ, _ = aggregate_cell("c","off", calls)
    assert summ["n_calls"] == 2
```

- [ ] **Step 2: Run — expect FAIL.**
- [ ] **Step 3: Implement `scripts/analyze_judge_sweep.py`**

```python
"""Compute derived metrics from raw reward dumps. Idempotent: delete derived/ and re-run.
Reads raw/sweep/<cell>/<mode>/reward/*.jsonl -> derived/{summary.md,summary.parquet,per_pair_metrics.parquet,plots/}."""
from __future__ import annotations
import argparse, json, re, sys
from collections import defaultdict
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO_ROOT))
import numpy as np, pandas as pd
from configs.judge_sweep_cells import SIZE_MAP, ANCHOR_CELL   # single source of truth

RUBRIC_FIELDS = ["immediate_target_score_a","immediate_target_score_b","human_goal_score_a",
    "human_goal_score_b","communication_style_score_a","communication_style_score_b",
    "response_a_source_copy","response_b_source_copy","source_copy_penalty_a","source_copy_penalty_b",
    "response_a_wrong_target_or_role","response_b_wrong_target_or_role","wrong_target_or_role_penalty_a",
    "wrong_target_or_role_penalty_b","response_a_unsupported_adversarial_reframing",
    "response_b_unsupported_adversarial_reframing","unsupported_adversarial_reframing_penalty_a",
    "unsupported_adversarial_reframing_penalty_b","response_a_assistant_like","response_b_assistant_like",
    "assistant_like_penalty_a","assistant_like_penalty_b","base_score_a","base_score_b",
    "response_a_score","response_b_score","score_gap","reasoning","rating"]
RATING_RE = re.compile(r'"rating"\s*:\s*(\d+)')

def try_parse_json(text: str) -> dict | None:
    if not text: return None
    s = text.strip()
    if not s.startswith("{"):
        i = s.find("{");  s = s[i:] if i >= 0 else s
        if i < 0: return None
    try: return json.loads(s)
    except json.JSONDecodeError: pass
    for end in range(len(s), 0, -1):
        try: return json.loads(s[:end])
        except json.JSONDecodeError: continue
    return None

def recover_rating_from_text(text: str) -> int | None:
    if not text: return None
    m = RATING_RE.search(text)
    if not m: return None
    r = int(m.group(1)); return r if 1 <= r <= 7 else None

def load_cell_rows(mode_dir: Path) -> list[dict]:
    rows = []
    for jl in sorted((mode_dir / "reward").glob("*.jsonl")):
        for line in jl.read_text().splitlines():
            line = line.strip()
            if line:
                try: rows.append(json.loads(line))
                except json.JSONDecodeError: pass
    return rows

def per_call_features(row: dict) -> dict:
    text = row.get("judge_raw_content") or ""
    parsed = try_parse_json(text)
    parsed_rating = int(parsed["rating"]) if parsed and isinstance(parsed.get("rating"), int) else None
    recovered = recover_rating_from_text(text) if parsed_rating is None else None
    rating = parsed_rating if parsed_rating is not None else recovered
    usage = row.get("judge_usage") or {}
    return {"pair_id": row.get("pair_id") or f'{row.get("user_id")}::{row.get("post_id")}::{row.get("target_idx")}',
        "user_id": row.get("user_id"), "generated_is_b": bool(row.get("generated_is_b")),
        "rating": rating, "format_ok": parsed is not None and parsed_rating is not None,
        "rating_recovered_from_text": recovered is not None and parsed_rating is None,
        "budget_hit": (row.get("judge_finish_reason") or "") == "length",
        "length_chars": len(text),
        "completion_tokens": usage.get("completion_tokens") if isinstance(usage, dict) else None,
        **{f"has_{f}": (parsed is not None and f in parsed) for f in RUBRIC_FIELDS}}

def accuracy(calls: list[dict]) -> float | None:
    """rating<4 -> judge picks A; >4 -> picks B; ==4 -> tie (excluded). Compare to human side."""
    num = den = 0
    for c in calls:
        r = c["rating"]
        if r is None or r == 4: continue
        human_is_b = c["generated_is_b"] is False  # human=A when generated=B, else human=B
        judge_picks_b = r > 4
        # human_side: if generated_is_b, human is A -> correct when judge picks A (r<4)
        correct = (judge_picks_b == (not c["generated_is_b"]))  # picks human?
        num += int(correct); den += 1
    return num / den if den else None

def budget_hit_rate(calls): return float(np.mean([c["budget_hit"] for c in calls])) if calls else 0.0

def aggregate_cell(cell: str, mode: str, calls: list[dict]):
    df = pd.DataFrame(calls)
    if df.empty: return {"cell": cell, "mode": mode, "n_calls": 0}, pd.DataFrame()
    per_pair = defaultdict(dict)
    for _, r in df.iterrows():
        side = "b" if r["generated_is_b"] else "a"
        per_pair[r["pair_id"]][f"rating_{side}"] = r["rating"]
        per_pair[r["pair_id"]][f"fmt_{side}"] = r["format_ok"]
    pair_rows = []
    for pid, d in per_pair.items():
        ra, rb = d.get("rating_a"), d.get("rating_b")
        pair_rows.append({"pair_id": pid, "cell": cell, "mode": mode, "rating_a": ra, "rating_b": rb,
            "picks_human_gen_b": (rb is not None and rb < 4),   # generated=B -> human=A -> pick A
            "picks_human_gen_a": (ra is not None and ra > 4),   # generated=A -> human=B -> pick B
            "both_parsed": bool(d.get("fmt_a") and d.get("fmt_b"))})
    pdf = pd.DataFrame(pair_rows)
    n1 = int(pdf["rating_b"].notna().sum()); n2 = int(pdf["rating_a"].notna().sum())
    acc1 = pdf["picks_human_gen_b"].sum() / max(n1, 1); acc2 = pdf["picks_human_gen_a"].sum() / max(n2, 1)
    both = pdf["both_parsed"]; nb = int(both.sum())
    acc_both = ((pdf.loc[both, "picks_human_gen_b"].sum() + pdf.loc[both, "picks_human_gen_a"].sum()) / (2 * nb)) if nb else 0.0
    ratings = df["rating"].dropna().astype(int).tolist()
    hist = {str(k): ratings.count(k) for k in range(1, 8)}
    summ = {"cell": cell, "mode": mode, "n_calls": len(df),
        "format_ok_rate": float(df["format_ok"].mean()),
        "rating_recovery_rate": float(df["rating_recovered_from_text"].mean()),
        "budget_hit_rate": float(df["budget_hit"].mean()),
        "accuracy_both_orderings": acc_both, "position_bias_delta": abs(acc1 - acc2),
        "rating_mean": float(np.mean(ratings)) if ratings else 0.0,
        "rating_mode": int(max(hist, key=lambda k: hist[k])) if ratings else 0,
        "field_presence": {f.replace("has_", ""): float(df[f].mean()) for f in df.columns if f.startswith("has_")},
        "rating_histogram": hist}
    return summ, pdf

def compute_kappa_vs_anchor(pair_dfs, anchor_cell):
    from sklearn.metrics import cohen_kappa_score
    out = {}
    for mode in ("off", "on"):
        adf = pair_dfs.get((anchor_cell, mode))
        if adf is None: continue
        amap = {r["pair_id"]: int(r["picks_human_gen_b"]) + int(r["picks_human_gen_a"])
                for _, r in adf.iterrows() if r["both_parsed"]}
        for (cell, m), df in pair_dfs.items():
            if m != mode or cell == anchor_cell: continue
            xs, ys = [], []
            for _, r in df.iterrows():
                if r["both_parsed"] and r["pair_id"] in amap:
                    xs.append(int(r["picks_human_gen_b"]) + int(r["picks_human_gen_a"])); ys.append(amap[r["pair_id"]])
            out[(cell, mode)] = cohen_kappa_score(xs, ys) if len(xs) >= 30 else float("nan")
    return out

def write_summary(rows, kappas, out_md, out_pq):
    s = [{"cell": r["cell"], "mode": r["mode"], "n_calls": r["n_calls"],
          "format_ok": r.get("format_ok_rate"), "rating_recovery": r.get("rating_recovery_rate"),
          "budget_hit": r.get("budget_hit_rate"), "accuracy": r.get("accuracy_both_orderings"),
          "pos_bias": r.get("position_bias_delta"), "rating_mean": r.get("rating_mean"),
          "kappa_vs_anchor": kappas.get((r["cell"], r["mode"]), float("nan"))} for r in rows if r["n_calls"]]
    df = pd.DataFrame(s).sort_values(["mode", "cell"]); df.to_parquet(out_pq, index=False)
    with out_md.open("w") as f:
        f.write("# Judge sweep summary\n\n")
        f.write("_Caveat: accuracies are vs ONE stochastic generator draw (1 sample/row, T=0.6). "
                "Judge-vs-judge gaps <5pp are within generator sampling noise._\n\n")
        f.write(df.to_markdown(index=False)); f.write("\n")

def write_plots(rows, out_dir):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    out_dir.mkdir(parents=True, exist_ok=True)
    for metric in ("accuracy_both_orderings", "format_ok_rate", "budget_hit_rate", "position_bias_delta"):
        fig, ax = plt.subplots(figsize=(6, 4))
        for mode, mk in (("off", "o"), ("on", "s")):
            pts = sorted(((SIZE_MAP[r["cell"]], r[metric]) for r in rows
                          if r["mode"] == mode and r["n_calls"] and r["cell"] in SIZE_MAP), key=lambda t: t[0])
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts], marker=mk, label=f"thinking={mode}")
        ax.set_xscale("log"); ax.set_xlabel("judge size (B; active-params for MoE)")
        ax.set_ylabel(metric); ax.set_title(f"{metric} vs size"); ax.legend(); ax.grid(True, alpha=0.3)
        fig.tight_layout(); fig.savefig(out_dir / f"{metric}.png"); plt.close(fig)

def main() -> None:
    ap = argparse.ArgumentParser()
    base = Path("/storage/home/lancewicki/projects/turing-rl/results/2026-07-08-judge-sweep")
    ap.add_argument("--raw_root", type=Path, default=base / "raw")
    ap.add_argument("--derived_root", type=Path, default=base / "derived")
    args = ap.parse_args(); args.derived_root.mkdir(parents=True, exist_ok=True)
    pair_dfs, summaries = {}, []
    for cell_dir in sorted((args.raw_root / "sweep").iterdir()):
        if not cell_dir.is_dir(): continue
        for mode_dir in sorted(cell_dir.iterdir()):
            if not mode_dir.is_dir(): continue
            calls = [per_call_features(r) for r in load_cell_rows(mode_dir)]
            summ, pdf = aggregate_cell(cell_dir.name, mode_dir.name, calls)
            summaries.append(summ); pair_dfs[(cell_dir.name, mode_dir.name)] = pdf
            print(f"[analyzer] {cell_dir.name}/{mode_dir.name}: n={summ['n_calls']} "
                  f"acc={summ.get('accuracy_both_orderings', 0):.3f}", flush=True)
    kappas = compute_kappa_vs_anchor(pair_dfs, ANCHOR_CELL)
    write_summary(summaries, kappas, args.derived_root / "summary.md", args.derived_root / "summary.parquet")
    nonempty = [d for d in pair_dfs.values() if not d.empty]
    (pd.concat(nonempty, ignore_index=True) if nonempty else pd.DataFrame()).to_parquet(
        args.derived_root / "per_pair_metrics.parquet", index=False)
    write_plots(summaries, args.derived_root / "plots")
    print("[analyzer] done", flush=True)

if __name__ == "__main__":
    main()
```
(Accuracy rule = spec's rating<4→A / >4→B / ==4→tie-excluded. `per_field_presence`, `rating_recovery_rate`, and the generator-sampling caveat are emitted; add per-confidence-bucket accuracy — bucket by anchor rating {1,7}/{2,3,5,6}/{4} — as a follow-on column if wanted.)
- [ ] **Step 4: Run — expect PASS.**
- [ ] **Step 5: (Cluster) run** `python scripts/analyze_judge_sweep.py`; ensure `matplotlib`/`scikit-learn` present. Report summary table.
- [ ] **Step 6: Commit** — `git add scripts/analyze_judge_sweep.py tests/test_analyze_judge_sweep.py && git commit -m "add sweep analyzer with metric unit tests"`

---

## Task 22: GUI viewer verification + docs + patches log

**Files:** Modify `our_patches.md`; Create `results/2026-07-08-judge-sweep/README.md` + `README.txt`; Modify the spec's Results section.

- [ ] **Step 1: (Cluster) viewer verification** — on the login pod, `python scripts/dump_viewer.py --dumps results/2026-07-08-judge-sweep/raw/sweep/{cell}/{mode} --port 8082`; `ssh -L 8082:localhost:8082 <host>` from the Mac; confirm a reward row renders all 10 tabs (context/history/response/ground_truth/prompt/raw/reasoning/judge/reward/metadata) with a Human A/B badge + rating, and an HTTP row renders its 5 tabs. No `dump_viewer.py` edits.
- [ ] **Step 2: Update `our_patches.md`** with all FOUR patches: (Task 4) `api_client.py` sampling/chat_template_kwargs + OpenRouter-extras gate; (Tasks 4+5) `reward.py` sampling/thinking env + `_resolve_response_format`; (Task 6) `reward.py` reward-dump helpers; (Task 7) `lora_sft.py` save-cadence config + resume. Mark all PERSISTENT with rationale. Add a line that the existing `_extract_chat_content` empty-return patch was re-validated under `--reasoning-parser qwen3` (Task 6 smoke).
- [ ] **Step 3: `README.md`** (tour of `raw/`+`derived/`, "regenerate derived via `python scripts/analyze_judge_sweep.py`") + `README.txt` (exact commands, input provenance, upstream steps — per the user's global repro rule).
- [ ] **Step 4: Append Results** to `docs/superpowers/specs/2026-07-08-judge-sweep-design.md` pointing into `derived/` (summary, plots, family_decision, calibration_report, split_verification, cot_fidelity, offline_vs_server_comparison, sampling_fidelity). Record deviations: Int4 anchor; probe-derived sampling; qwen3 reasoning-parser; env-gated payload extension.
- [ ] **Step 5: Commit** — `git add our_patches.md results/2026-07-08-judge-sweep/README.md results/2026-07-08-judge-sweep/README.txt docs/superpowers/specs/2026-07-08-judge-sweep-design.md && git commit -m "add patches log, repro readme, results write-up"`

---

## Self-Review

**Spec coverage:** §1 goal/anchor/family/sampling → Tasks 1,14,16,17,19; §2 PRISM verification → Tasks 2,3; §3 data foundation → Tasks 2,13; §4 generator (CoT/SFT/inference/pairs) → Tasks 8,9,10,11,13; §5 sweep (fixed inputs, matrix, calibration, serving, offline, viewer, metrics, artifacts) → Tasks 6,15,16,18,19,20,21,22; §6 analysis/ordering/risks → Tasks 18(R2 gate),19,21; patches summary → Tasks 4,5,6,7,22; deliverables 1-5 → 21/22 (table+plots), 17 (family doc), 11 (checkpoint+inference), 13 (pairs), 2/3 (split verification). No gaps.

**Three v1 correctness gaps fixed:** the missing `PERSONA_JUDGE_JSON_SCHEMA` reader (Task 5), the missing `_build_reward_dump_row`/`_dump_reward_call` helpers (Task 6), and the missing payload sampling/`enable_thinking` support (Task 4) — v1's sweep client imported/relied on all three but no v1 task created them.

**Other corrections vs v1:** round-robin race → process-sharding (Task 15); PRISM verify re-runs only `split_data.py`, not `build.py` (Task 3, per spec §2); SFT patch now includes save-cadence config-ization, not just resume (Task 7); offline cell reuses the dump helper instead of duplicating the schema (Task 20); size/TP config centralized in one tested module (Task 14).

**Deliberate deviations (documented):** sampling is probe-derived not spec-table (Task 1); calibration uses the real reward path not `benchmark_judge_throughput.py` (Task 18); `--reasoning-parser qwen3` not `deepseek_r1`; env-gated payload extension (the spec's "unmodified reward path" is unachievable for thinking-off + per-cell sampling on self-hosted vLLM). CoT stays **served** (faithful to upstream `generate_cot.py`), with a self-hosted-vs-OpenRouter fidelity check (Task 9) replacing v1's dropped batched-vs-served parity test.

**Placeholder scan:** none. Code steps show code; cluster steps give commands + gates.

**Code-level items resolved from the local checkout (no longer open):** `generate_cot.py` reusable symbols (all confirmed present), `split_data.py` flags (`--input-dir/--output-dir/--heldout-user-frac/--grpo-frac/--seed`), `build_sft_jsonl.py` flags (`--input_parquet/--output_jsonl/--max_rows`, requires `ground_truth_reasoning`), `lora_sft.py --model` = `MODEL_MAP` key `qwen3-8b` (no `--config` flag).

**Genuinely data-dependent confirmations (deferred to the cluster agent's first report, not downloadable to the Mac):** PRISM raw turns field name + `extra_info` key names + `split_metadata.json` keys (Task 2 helper/`EXPECTED_COUNTS`), and the heldout inference pickle's nesting (Task 13, produced by Task 11). The agent-comms first handoff asks the cluster to paste these back.

**Type consistency:** `build_pairs → (df, meta)`; `cell_list`/`tp_for_size`/`SIZE_MAP`/`ANCHOR_CELL` shared by Tasks 14/16/21; `_build_reward_dump_row` key set matches the viewer contract and the Task-21 readers (`judge_finish_reason`, `rating_gt_first`, `randomized_order`, `judge_response`); `pair_id = user::post::idx` consistent across Tasks 13/15/21.

---

Plan complete and saved to `docs/superpowers/plans/2026-07-08-judge-sweep-implementation.md`. Two execution options:

**1. Subagent-Driven (recommended)** — dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — execute tasks in this session with executing-plans, batched with checkpoints.

Which approach?
