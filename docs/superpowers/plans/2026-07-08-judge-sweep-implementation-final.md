# Judge-Model Comparison Experiment — Implementation Plan (FINAL, merged)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **This plan supersedes both** `2026-07-08-judge-sweep-implementation.md` (v1, unaudited) and `2026-07-08-judge-sweep-implementation-v2.md` (independent). It is the merge of the two: v1's complete runnable script bodies + v2's TDD structure, the three patches the audit added, OpenRouter-probe sampling, a config SSOT, the round-robin race fix, and the Mac→cluster agent-comms workflow.

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
- **Sync/exec workflow:** edit → `git commit` → push to `mine/lancewicki/main`. Cluster execution is requested by writing `docs/agent-comms/2026-07-08-judge-sweep-implementation-final/mac-to-cluster.md` (referencing the task #), committing, pushing; the cluster agent runs it and replies in `cluster-to-mac.md` (never edit that file).

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

**Files:** Create `tests/prism_verification_helpers.py`, `tests/test_prism_split_verification.py`. (Lifted from v1 Tasks 1-2; solid.)

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

- [ ] **Step 2: Write the 7-check pytest** (lift v1 Task 2 verbatim: `test_1_files_exist` … `test_7_no_text_leak_heldout_from_sft_targets`, with `EXPECTED_COUNTS = {sft:3272/464, grpo/train:4174/696, grpo/val:705/696, test:880/128}`). Full body is in v1 plan Task 2, lines 105-237 — reproduce it into `tests/test_prism_split_verification.py`.

- [ ] **Step 3: Hand off to cluster** (needs data + HF cache). Write `docs/agent-comms/2026-07-08-judge-sweep-implementation-final/mac-to-cluster.md`: "Run Task 2 — `cd $REPO && /home/lancewicki/miniconda3/envs/turing-rl-train/bin/python -m pytest tests/test_prism_split_verification.py -v`. If `split_metadata.json` keys, the raw turns field, or `extra_info` key names differ, report actual schema." Commit + push.

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
# NOTE: confirm split_data.py's actual flags during execution; adjust here.
$PY -u -m data.prism.split_data --input-dir "$REPO/data/prism/full_s42_history" \
    --output-dir "$FRESH" || { echo "split failed"; exit 2; }
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
(Confirm `--model` accepts `qwen3-8b` — argparse `choices` includes it.)

- [ ] **Step 6: Run — expect PASS.** `python -m pytest tests/test_lora_sft_config.py -v`
- [ ] **Step 7: Commit** — `git add training/sft/lora_sft.py training/sft/configs/qwen3_8b_lora.yaml scripts/slurm/sft_full.sh tests/test_lora_sft_config.py && git commit -m "add sft step-checkpointing + auto-resume"`

---

## Task 8: Served CoT client (thinking-off, self-hosted)

Faithful to upstream `generate_cot.py`'s served/HTTP path. Reuses its business logic; swaps transport to self-hosted async round-robin with `enable_thinking=False`.

**Files:** Create `scripts/generate_cot_served.py`; Create `scripts/slurm/cot_serve.sh` (8-replica launcher). Test `tests/test_generate_cot_served.py`.

**Interfaces:** Produces `data/sft/prism_full_s42_sft_cot.parquet` (3272 rows) + `.cot_metadata.json`.

- [ ] **Step 1: Inspect `generate_cot.py`** to confirm reusable symbols (`reasoning_leaks_reply` confirmed; the prompt-building template + `_row_context`/`_as_text` need confirming). Reuse what imports cleanly; inline the prompt build otherwise (document which).

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

- [ ] **Step 1: (Cluster)** `python -m data.sft.build_sft_jsonl --input_parquet data/sft/prism_full_s42_sft_cot.parquet --output_jsonl data/sft/prism_full_s42_sft_cot.jsonl` (confirm exact flag names). Gate: `wc -l` == 3272; one line has assistant target `<reasoning>...</reasoning>\n[HUMAN]: ...`. Report back. No commit (data).

---

## Task 11: SFT training + heldout inference (cluster orchestration)

**Files:** Create `scripts/slurm/heldout_inference.sh` (lift v1 Task 11 body).

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

**Files:** Create `scripts/build_judge_pairs.py`; Test `tests/test_build_judge_pairs.py`. (v1 Task 12 code + TDD wrapper.)

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

**Files:** Create `scripts/slurm/judge_serve_cell.sh` (parameterized N-replica server; lift v1 Task 14 body), `scripts/slurm/judge_sweep_cell.sh` (serve + run client; lift v1 Task 16 body but launch **one client per endpoint** with `--endpoint_index/--num_endpoints`), `scripts/launch_judge_sweep.sh` (reads `configs/judge_sweep_cells.py`).

- [ ] **Step 1: `judge_sweep_cell.sh`** — boot `REPLICAS` vLLM servers (TP each) on ports `PORT_BASE+i`, `--reasoning-parser qwen3` iff `THINKING_MODE=on`; wait `/v1/models`; then launch `REPLICAS` client processes, process `i` pinned to endpoint `i` with `--endpoint_index i --num_endpoints REPLICAS` (fixes the race). Use `judge-vllm` python for the 397B server, `turing-rl-train` for smaller. Guard `REPLICAS*TP<=8`.

- [ ] **Step 2: `launch_judge_sweep.sh`** — emit the cell list from the config module instead of bash arrays:
```bash
CELLS=$($PY -c "import json,sys; from configs.judge_sweep_cells import cell_list; \
  print(json.dumps(cell_list('$FAMILY')))")
# iterate cells x modes(off,on); sbatch judge_sweep_cell.sh with --gres=gpu:$((tp*replicas))
# and --export=ALL,MODEL=..,TP=..,REPLICAS=..,THINKING_MODE=..,CELL_NAME=..[,MAX_PAIRS=50]
```
`FAMILY` comes from the Task-17 decision; `CALIBRATION=1` exports `MAX_PAIRS=50`.

- [ ] **Step 3: `bash -n` each script; commit** — `git add scripts/slurm/judge_serve_cell.sh scripts/slurm/judge_sweep_cell.sh scripts/launch_judge_sweep.sh && git commit -m "add serving sbatch + config-driven sweep launcher"`

---

## Task 17: Family gate — 4B smoke (cluster)

**Files:** Create `scripts/slurm/family_smoke.sh` (lift v1 Task 13 body); writes `derived/family_decision.md`.

- [ ] **Step 1: (Cluster)** serve Qwen3-4B + Qwen3.5-4B (TP=1), run the sweep client with `--max_pairs 50` on each (100 calls), both modes. Compare tokens/sec at parse-rate parity; tiebreak on 397B-agreement over 50 pairs; inconclusive → default `qwen3.5` (spec R6).
- [ ] **Step 2:** write `family_decision.md` with the chosen `FAMILY`, sizes, rationale. This gate sets `FAMILY` for Tasks 16/18/19.
- [ ] **Step 3: Commit** — `git add scripts/slurm/family_smoke.sh docs/... && git commit -m "add family-selection 4b smoke + decision"`

---

## Task 18: Per-cell calibration + >4h gate

Uses the Task-15 client (`--max_pairs 50`) so calibration exercises the **real** judge path (v2 improvement over v1, which reused the divergent `benchmark_judge_throughput.py`).

**Files:** aggregation inline (lift v1 Task 17 python) → `raw/calibration/calibration_metadata.json`, `derived/calibration_report.md`.

- [ ] **Step 1: (Cluster)** `FAMILY=<decision> CALIBRATION=1 bash scripts/launch_judge_sweep.sh` (all 10 cells, ~1h parallel).
- [ ] **Step 2:** aggregate per-cell `wall_seconds`/`n_pairs` → req/s → extrapolated 1760-call wall; write `calibration_report.md` with the ±30% precision caveat inline.
- [ ] **Step 3: Gate:** any cell projecting >4h (<0.12 req/s) → surface, decide reduce-concurrency / drop-cell / accept (spec §5.0, R2: anchor may drop `max_completion_tokens` to 4096). Purge calibration dumps so the full sweep starts clean.
- [ ] **Step 4: Commit** — `git add results/2026-07-08-judge-sweep/derived/calibration_report.md && git commit -m "add per-cell calibration report + 4h gate"`

---

## Task 19: Full judge sweep (cluster)

**Files:** none new.

- [ ] **Step 1: (Cluster)** `FAMILY=<decision> bash scripts/launch_judge_sweep.sh`. Anchor (Node 1) runs 2 cells sequentially (on/off, TP=8); smaller judges parallel across Nodes 2&3 (one cell per node, node filled with replicas). Copy vLLM logs + slurm out into `raw/logs/`.
- [ ] **Step 2: Gate per cell:** reward-dump rows total ≥ 1760; `run_metadata.json` present. Re-submit any short cell. Report per-cell completion + anomalies. No commit (data).

---

## Task 20: Offline batched bonus cell

**Corrected vs v1:** reuse `_build_reward_dump_row` (Task 6) instead of hand-duplicating the 30-key row schema (DRY / drift-proof). Keep v1's GuidedDecodingParams + prompt-build.

**Files:** Create `scripts/run_offline_sweep_cell.py` (v1 Task 19 body, with the row built via `_build_reward_dump_row`), `scripts/slurm/offline_sweep_cell.sh`.

- [ ] **Step 1: Implement** — in-process `LLM.generate` (TP=8) over all 1760 prompts for Qwen3-8B thinking-off, `GuidedDecodingParams(json={required rating})`, `enable_thinking=False`. For each output build the dump row via `from training.grpo.reward import _build_reward_dump_row` and write to `raw/sweep/qwen3-8b/off_offline/reward/`. Same pair-set/sampling/prompt as the server cell.
- [ ] **Step 2: (Cluster) run** post-primary-sweep on a free node (~30-60 min). Gate: 1760 rows; accuracy matches the server cell within noise; higher throughput. Report the server-vs-offline ratio → `derived/offline_vs_server_comparison.md`.
- [ ] **Step 3: Commit** — `git add scripts/run_offline_sweep_cell.py scripts/slurm/offline_sweep_cell.sh && git commit -m "add offline batched bonus cell"`

---

## Task 21: Analyzer (raw → derived) with TDD metric tests

**Files:** Create `scripts/analyze_judge_sweep.py` (lift v1 Tasks 20-21 body — parsing → `aggregate_cell` → `compute_kappa_vs_anchor` (sklearn) → `write_summary`/`write_plots`); import `SIZE_MAP`/`ANCHOR_CELL`/`cell_list` from `configs.judge_sweep_cells` instead of re-hard-coding. Test `tests/test_analyze_judge_sweep.py`.

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
- [ ] **Step 3: Implement** — port v1's analyzer verbatim, EXCEPT: (a) import `SIZE_MAP`/`ANCHOR_CELL` from the config module; (b) confirm `pair_id` key (v1 uses `pair_id`); (c) keep accuracy rule rating<4→A / >4→B / ==4→tie-excluded (v1's ≤3 / ≥5 is equivalent); add per-field presence over the ~28 rubric fields, rating-recovery-rate, per-confidence-bucket accuracy, and the generator-sampling caveat inline in `summary.md`.
- [ ] **Step 4: Run — expect PASS.**
- [ ] **Step 5: (Cluster) run** `python scripts/analyze_judge_sweep.py`; ensure `matplotlib`/`scikit-learn` present. Report summary table.
- [ ] **Step 6: Commit** — `git add scripts/analyze_judge_sweep.py tests/test_analyze_judge_sweep.py && git commit -m "add sweep analyzer with metric unit tests"`

---

## Task 22: GUI viewer verification + docs + patches log

**Files:** Modify `our_patches.md`; Create `results/2026-07-08-judge-sweep/README.md` + `README.txt`; Modify the spec's Results section.

- [ ] **Step 1: (Cluster) viewer verification** (lift v1 Task 22): point `scripts/dump_viewer.py --dumps raw/sweep/{cell}/{mode}` on the login pod, tunnel from Mac, confirm all reward tabs + HTTP tabs render. No `dump_viewer.py` edits.
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

**Placeholder scan:** none. Code steps show code; cluster steps give commands + gates. Runtime confirmations are explicitly flagged (PRISM raw field names/metadata keys, `split_data.py` flags, `build_sft_jsonl` flags, the heldout pickle nesting, `generate_cot.py` reusable symbols) because they can't be verified from the Mac.

**Type consistency:** `build_pairs → (df, meta)`; `cell_list`/`tp_for_size`/`SIZE_MAP`/`ANCHOR_CELL` shared by Tasks 14/16/21; `_build_reward_dump_row` key set matches the viewer contract and the Task-21 readers (`judge_finish_reason`, `rating_gt_first`, `randomized_order`, `judge_response`); `pair_id = user::post::idx` consistent across Tasks 13/15/21.

---

Plan complete and saved to `docs/superpowers/plans/2026-07-08-judge-sweep-implementation-final.md`. Two execution options:

**1. Subagent-Driven (recommended)** — dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — execute tasks in this session with executing-plans, batched with checkpoints.

Which approach?
