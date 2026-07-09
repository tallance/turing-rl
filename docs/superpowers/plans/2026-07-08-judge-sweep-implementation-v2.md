# Judge-Model Comparison Experiment — Implementation Plan (v2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure how well smaller Qwen judges approximate the paper's 397B training-reward judge on the pairwise Turing task, across thinking-on/off modes, on the PRISM heldout split — producing a reproducible `results/2026-07-08-judge-sweep/` tree.

**Architecture:** Build the net-new *software* (env-gated judge-path patches, reward-layer dumps, a served round-robin CoT client, a config-driven sweep-cell client, an offline analyzer) with local TDD on the Mac, then run the GPU/data steps on the cluster via `docs/agent-comms/`. All judge calls go through the real GRPO reward path (`training/grpo/reward.py`) so the comparison is apples-to-apples with training-time reward. Family choice (Qwen3 vs Qwen3.5) is a runtime config input, not a code fork.

**Tech Stack:** Python 3, `aiohttp`, vLLM (OpenAI-compatible server), `pandas`/`pyarrow`, `pytest`, TRL `SFTTrainer`, Slurm.

---

## Global Constraints

Copied verbatim from the spec; every task's requirements implicitly include these.

- **Maximum fidelity to the paper's SFT checkpoint.** Use upstream code as-is; revert patches for the SFT path unless strictly necessary. Any deviation is a documented caveat, never silently absorbed.
- **Self-hosted only.** No OpenRouter/Morph in production paths (no cluster egress). The single exception is the one-off fidelity probe in Task 1, run from the Mac against the user's personal $10 OpenRouter account.
- **Sampling policy:** determined empirically by the Task 1 OpenRouter probe, then matched server-side. Do NOT hard-code the spec's model-card sampling table until Task 1 confirms it; whatever Task 1 finds is the frozen sampling for every cell (anchor included) and is recorded in `derived/sampling_fidelity.md`.
- **Reasoning parser:** `--reasoning-parser qwen3` on every thinking-on vLLM server (NOT `deepseek_r1`). Thinking-off servers run no reasoning parser.
- **Thinking-mode toggle:** on the wire via `chat_template_kwargs={"enable_thinking": <bool>}` (self-hosted vLLM has no OpenRouter `reasoning` field).
- **JSON schema mode locked ON:** `PERSONA_JUDGE_JSON_SCHEMA=1` for every cell.
- **`max_completion_tokens=8192`** for every judge call.
- **Dumps on:** `PERSONA_JUDGE_DUMP_RATE=1.0` for every sweep cell (both HTTP-layer and reward-layer dumps).
- **Frozen inputs:** the 880-row pair-set (Task 11) and `TURING_PROMPT`/rubric schema are never modified after freeze. Heldout split only — never sweep on SFT/GRPO train splits.
- **Anchor stays fixed:** `Qwen/Qwen3.5-397B-A17B-GPTQ-Int4`, TP=8, regardless of family choice. Re-baselined under the Task 1 sampling.
- **Results layout:** `results/2026-07-08-judge-sweep/` split into immutable `raw/` and regeneratable `derived/`.
- **Cluster paths:** repo `/storage/home/lancewicki/projects/turing-rl`; HF cache `/home/lancewicki/data/hf_cache`; conda envs `turing-rl-train` (vLLM 0.18, cu130 — CoT/SFT/inference/serving smaller judges) and `judge-vllm` (serves the 397B anchor).
- **Sync workflow:** edit → `git commit` → push to `mine/lancewicki/main`; cluster agent pulls. Cluster execution is requested via `docs/agent-comms/2026-07-08-judge-sweep-implementation-v2/mac-to-cluster.md`; reports come back in `cluster-to-mac.md` (do not edit that file).

---

## Where each task runs

| Runs on Mac (local TDD) | Runs on cluster (via agent-comms) |
|---|---|
| Tasks 1, 3, 4, 5, 7, 8, 11, 12, 13, 16, 17 code + unit tests on synthetic data | Tasks 2 (data), 6 (serving), 9, 10, 14, 15, 18, 19 execution; and the *execution* half of 7/8/12/16 |

Every code task is written and unit-tested locally with synthetic fixtures; the GPU/data execution is a separate handoff. "Commit" steps use `git`; push happens at each cluster handoff.

---

## File Structure

**New files (ours, untracked in `our_patches.md` — under `scripts/`, `tests/`, `configs/`, `results/`):**
- `scripts/openrouter_sampling_probe.py` — Task 1 fidelity probe (Mac-only, OpenRouter).
- `tests/test_prism_split_verification.py` — Task 2 (7-check suite + hash compare).
- `tests/prism_verification_helpers.py` — PRISM raw-loader helper for Task 2.
- `scripts/generate_cot_served.py` — Task 7 served async round-robin CoT client.
- `scripts/vllm_serve_replicas.sh` — Task 6 multi-replica launcher (shared CoT + sweep).
- `scripts/slurm/cot_serve_8rep.sh`, `scripts/slurm/sft_full.sh`, `scripts/slurm/judge_serve_cell.sh`, `scripts/slurm/judge_serve_anchor.sh` — Task 6/8/15 sbatch wrappers.
- `scripts/build_judge_pairs.py` — Task 11 pair-set builder.
- `scripts/run_judge_sweep_cell.py` — Task 12 sweep-cell client (also does calibration via `--limit`).
- `configs/judge_sweep_cells.py` — Task 13 cell-config generation + TP/replica lookup.
- `scripts/calibration_report.py` — Task 14 calibration report.
- `scripts/analyze_judge_sweep.py` — Task 17 analyzer (raw → derived).
- `scripts/plot_judge_sweep.py` — Task 18 plots.
- `tests/test_reward_dump_row.py`, `tests/test_judge_payload.py`, `tests/test_build_judge_pairs.py`, `tests/test_sweep_cell_config.py`, `tests/test_analyze_judge_sweep.py` — unit tests.

**Modified upstream files (must be added to `our_patches.md`):**
- `shared/api_client.py` — `build_chat_payload` gains optional `sampling`/`chat_template_kwargs` passthrough (Task 3).
- `training/grpo/reward.py` — `_openai_chat` reads sampling/thinking/json-schema env (Tasks 3, 4); new `_build_reward_dump_row` + `_dump_reward_call` wired into `_score_pairwise_likert_with_info` (Task 5).
- `training/sft/lora_sft.py` — `save_strategy`/`save_steps`/`save_total_limit` yaml-configurable + `--resume_from_checkpoint auto` (Task 8).
- `training/sft/configs/qwen3_8b_lora.yaml` — add checkpoint-cadence keys (Task 8).

---

## Interfaces (pinned names used across tasks)

- `build_chat_payload(*, model, messages, max_completion_tokens, response_format=None, reasoning, sampling: dict | None = None, chat_template_kwargs: dict | None = None) -> dict` (Task 3).
- Env vars read by `reward.py._openai_chat`: `PERSONA_JUDGE_SAMPLING` (JSON dict), `PERSONA_JUDGE_ENABLE_THINKING` (`"1"`/`"0"`/unset), `PERSONA_JUDGE_JSON_SCHEMA` (`"1"`), `PERSONA_REWARD_DUMP_DIR` (path) — plus existing `PERSONA_JUDGE_DUMP_RATE`, `PERSONA_JUDGE_DUMP_DIR`.
- `_build_reward_dump_row(*, response, ground_truth, context, user_history, human_side, generated_is_b, randomized_order, rating_gt_first, rating_gen_first, judge_response, judge_prompt, judge_raw_content, judge_reasoning, judge_latency_ms, judge_finish_reason, judge_model, judge_usage, final_reward, turing_judge_score_raw, turing_judge_score_clipped, source_copy_penalty, assistant_like_penalty, wrong_target_or_role_penalty, unsupported_adversarial_reframing_penalty, call_id, user_id, post_id, target_idx, persona, ts, worker_pid) -> dict` (Task 5) — output keys match `scripts/dump_viewer.py` reward-row contract exactly.
- `build_pairs(inference_pkl_path: str, test_parquet_path: str) -> (pandas.DataFrame, dict)` returns `(pairs_df, sanity_metadata)` (Task 11). Columns: `pair_id, user_id, post_id, target_idx, user_history, context, persona, human, generated`.
- `cell_list(family: str) -> list[dict]` returns cells `{judge_id, model_id, tp, replicas, size_b, is_moe}`; `tp_for_size(size_b, is_moe) -> (tp, replicas)` (Task 13).
- `analyze(raw_dir: str, derived_dir: str) -> None` (Task 17); metric helpers `cell_metrics(reward_rows: list[dict], anchor_rows_by_pair: dict) -> dict`.

---

## Task 1: OpenRouter sampling-fidelity probe (Mac-only)

Resolves the sampling policy empirically before any server config is frozen. The paper's code passes **no** sampling params, so OpenRouter/Morph applied provider defaults; we measure them and match server-side.

**Files:**
- Create: `scripts/openrouter_sampling_probe.py`
- Create: `derived/` note written to `results/2026-07-08-judge-sweep/derived/sampling_fidelity.md` (created by the run)

**Interfaces:**
- Produces: frozen sampling dicts for thinking-on and thinking-off, recorded in `sampling_fidelity.md`, consumed by Tasks 6/13/15 serving configs.

- [ ] **Step 1: Write the probe script**

Sends the *same* small prompt to OpenRouter Qwen3-8B (a) with the repo's exact payload (`reasoning={"enabled": True}`, no sampling) and (b) `reasoning` disabled, N=20 each, and records the response `generation_config`/effective params OpenRouter echoes plus output-length and token-usage distributions. It also queries the OpenRouter models endpoint for the provider's default sampling if exposed.

```python
"""One-off: probe what sampling OpenRouter/Morph applies to Qwen3 by default.

Run from the Mac only. Requires OPENROUTER_API_KEY in env (personal $10 account).
NOT part of any cluster path.
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
        URL, data=json.dumps(payload).encode(),
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
            usage = d.get("usage", {})
            rows[mode].append({
                "completion_tokens": usage.get("completion_tokens"),
                "content_len": len((d["choices"][0]["message"].get("content") or "")),
                "params_echo": d.get("provider") or d.get("generation_config"),
            })
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        f.write("# OpenRouter Qwen3 sampling fidelity probe\n\n")
        for mode in ("on", "off"):
            lens = [r["completion_tokens"] for r in rows[mode] if r["completion_tokens"]]
            f.write(f"## thinking-{mode} (n={len(rows[mode])})\n")
            f.write(f"- completion_tokens: mean={statistics.mean(lens):.0f} "
                    f"min={min(lens)} max={max(lens)}\n")
            f.write(f"- sample params echo: {json.dumps(rows[mode][0]['params_echo'])}\n\n")
        f.write("## DECISION\n\nFrozen sampling to replicate server-side (fill after review):\n"
                "- thinking-on: T=?, top_p=?, top_k=?, min_p=?\n"
                "- thinking-off: T=?, top_p=?, top_k=?, min_p=?\n")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the probe (Mac, needs the personal key)**

Run: `OPENROUTER_API_KEY=... python scripts/openrouter_sampling_probe.py --n 20`
Expected: `results/2026-07-08-judge-sweep/derived/sampling_fidelity.md` written with token-length distributions and any echoed params. Cost ≪ $1.

- [ ] **Step 3: Fill the DECISION block**

Interpret the probe: if OpenRouter echoes explicit sampling, replicate it. If it does not (common), the honest server-side match is vLLM's default sampling (`generation_config.json`) for the model — record that as the decision and note "OpenRouter defaults not observable; matched to model generation_config" as a caveat. If output-length distributions differ sharply thinking-on vs off, that is evidence the provider *did* vary sampling; record accordingly.

- [ ] **Step 4: Commit**

```bash
git add scripts/openrouter_sampling_probe.py results/2026-07-08-judge-sweep/derived/sampling_fidelity.md
git commit -m "feat(judge-sweep): OpenRouter sampling-fidelity probe + frozen decision"
```

---

## Task 2: PRISM split verification suite (cluster execution)

**Files:**
- Create: `tests/prism_verification_helpers.py`
- Create: `tests/test_prism_split_verification.py`
- Output: `results/2026-07-08-judge-sweep/derived/split_verification.md`

**Interfaces:**
- Consumes: `data/prism/full_s42_history_sft40_grpo60_test10/{sft/train,grpo/train,grpo/val,test}.parquet`, `split_metadata.json`; raw `HannahRoseKirk/prism-alignment`.
- Produces: a passing test + `split_verification.md`; unblocks all downstream data use.

- [ ] **Step 1: Write the PRISM raw-loader helper**

```python
# tests/prism_verification_helpers.py
"""Helpers for PRISM split verification (Section 2 of the spec)."""
from functools import lru_cache
import pandas as pd
from datasets import load_dataset

@lru_cache(maxsize=1)
def load_prism_raw():
    """Return the raw PRISM conversations dataset (cached HF)."""
    return load_dataset("HannahRoseKirk/prism-alignment", "conversations", split="train")

def raw_human_reply(ds, user_id: str, conversation_id: str, turn_idx: int) -> str | None:
    """Look up the [HUMAN] reply for (user, thread, turn) in raw PRISM."""
    for row in ds:
        if row.get("user_id") == user_id and row.get("conversation_id") == conversation_id:
            turns = row.get("conversation_history") or []
            if 0 <= turn_idx < len(turns):
                return turns[turn_idx].get("content")
    return None

def users_of(df: pd.DataFrame) -> set[str]:
    return set(df["extra_info"].map(lambda e: e["user_id"]))
```

- [ ] **Step 2: Write the failing verification test (7 checks + hash compare)**

```python
# tests/test_prism_split_verification.py
import hashlib, json, subprocess, sys
from pathlib import Path
import pandas as pd
import pytest
from tests.prism_verification_helpers import load_prism_raw, raw_human_reply, users_of

SPLIT = Path("data/prism/full_s42_history_sft40_grpo60_test10")
PARQUETS = {"sft": SPLIT/"sft/train.parquet", "grpo_train": SPLIT/"grpo/train.parquet",
            "grpo_val": SPLIT/"grpo/val.parquet", "test": SPLIT/"test.parquet"}

@pytest.fixture(scope="module")
def meta():
    return json.loads((SPLIT/"split_metadata.json").read_text())

def test_files_exist():
    for p in PARQUETS.values():
        assert p.exists(), f"missing {p}"
    assert (SPLIT/"split_metadata.json").exists()

def test_row_counts_match_metadata(meta):
    for name, p in PARQUETS.items():
        assert len(pd.read_parquet(p)) == meta["row_counts"][name], name

def test_user_counts_match_metadata(meta):
    for name, p in PARQUETS.items():
        assert len(users_of(pd.read_parquet(p))) == meta["user_counts"][name], name

def test_user_disjointness():
    sft = users_of(pd.read_parquet(PARQUETS["sft"]))
    gtr = users_of(pd.read_parquet(PARQUETS["grpo_train"]))
    hel = users_of(pd.read_parquet(PARQUETS["test"]))
    assert sft & gtr == set()
    assert gtr & hel == set()
    assert sft & hel == set()

def test_prompt_schema_wellformed():
    for p in PARQUETS.values():
        df = pd.read_parquet(p).sample(min(50, len(pd.read_parquet(p))), random_state=42)
        for _, row in df.iterrows():
            assert isinstance(row["prompt"], (list,)) and row["prompt"]
            assert all({"role", "content"} <= set(m) for m in row["prompt"])
            assert isinstance(row["reward_model"]["ground_truth"], str)
            assert row["reward_model"]["ground_truth"].strip()
            assert {"user_id", "target_idx"} <= set(row["extra_info"])

def test_heldout_ground_truth_matches_raw():
    ds = load_prism_raw()
    df = pd.read_parquet(PARQUETS["test"]).sample(20, random_state=42)
    for _, row in df.iterrows():
        e = row["extra_info"]
        raw = raw_human_reply(ds, e["user_id"], e.get("post_id") or e.get("conversation_id"),
                              e["target_idx"])
        if raw is not None:
            assert row["reward_model"]["ground_truth"].strip() == raw.strip()

def test_resplit_hash_matches():
    subprocess.run([sys.executable, "data/prism/split_data.py",
                    "--config", "data/prism/config_history_persona_s42.json",
                    "--out", "/tmp/prism_resplit"], check=True)
    for name, p in PARQUETS.items():
        fresh = Path("/tmp/prism_resplit") / p.relative_to(SPLIT)
        assert hashlib.sha256(fresh.read_bytes()).hexdigest() == \
               hashlib.sha256(p.read_bytes()).hexdigest(), f"hash mismatch {name}"
```

> Note: the exact `split_metadata.json` key names (`row_counts`/`user_counts`) and `split_data.py` CLI flags must be confirmed against the actual files during execution; adjust the fixture accessors to match. The re-split test writes to `/tmp` and never overwrites the real split.

- [ ] **Step 3: Hand off to cluster (needs data + HF cache)**

Write to `docs/agent-comms/2026-07-08-judge-sweep-implementation-v2/mac-to-cluster.md`: "Run Task 2 — `cd /storage/home/lancewicki/projects/turing-rl && python -m pytest tests/test_prism_split_verification.py -v`. If `split_metadata.json` keys or `split_data.py` flags differ from the test's assumptions, report the actual schema back. On pass, write `results/2026-07-08-judge-sweep/derived/split_verification.md` summarizing counts, disjointness, and the hash-compare result." Commit + push.

- [ ] **Step 4: On green report, commit the derived doc pointer and proceed**

```bash
git add tests/prism_verification_helpers.py tests/test_prism_split_verification.py
git commit -m "test(judge-sweep): PRISM split content-verification suite"
```

---

## Task 3: Extend the judge payload for sampling + thinking-mode (env-gated)

Makes the real reward path able to set per-cell sampling and `enable_thinking` without diverging from GRPO's code path. Defaults are no-ops → upstream behavior unchanged when env unset.

**Files:**
- Modify: `shared/api_client.py` (`build_chat_payload`, ~lines 72-89)
- Modify: `training/grpo/reward.py` (`_openai_chat`, ~lines 330-352)
- Test: `tests/test_judge_payload.py`

**Interfaces:**
- Produces: `build_chat_payload(..., sampling=None, chat_template_kwargs=None)`; `_openai_chat` reads `PERSONA_JUDGE_SAMPLING`, `PERSONA_JUDGE_ENABLE_THINKING`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_judge_payload.py
from shared.api_client import build_chat_payload

def test_sampling_merged_into_payload():
    p = build_chat_payload(model="m", messages=[], max_completion_tokens=8192,
                           reasoning=False, sampling={"temperature": 0.6, "top_k": 20})
    assert p["temperature"] == 0.6 and p["top_k"] == 20

def test_chat_template_kwargs_passthrough():
    p = build_chat_payload(model="m", messages=[], max_completion_tokens=8192,
                           reasoning=False, chat_template_kwargs={"enable_thinking": False})
    assert p["chat_template_kwargs"] == {"enable_thinking": False}

def test_defaults_are_noop():
    p = build_chat_payload(model="m", messages=[], max_completion_tokens=8192, reasoning=False)
    assert "temperature" not in p and "chat_template_kwargs" not in p
```

- [ ] **Step 2: Run it — expect FAIL**

Run: `python -m pytest tests/test_judge_payload.py -v`
Expected: FAIL (`build_chat_payload() got an unexpected keyword argument 'sampling'`).

- [ ] **Step 3: Extend `build_chat_payload`**

```python
def build_chat_payload(
    *, model: str, messages: list[dict], max_completion_tokens: int,
    response_format: dict | None = None, reasoning: bool,
    sampling: dict | None = None, chat_template_kwargs: dict | None = None,
) -> dict[str, Any]:
    """Build a chat-completions payload."""
    payload: dict[str, Any] = {
        "model": model, "messages": messages,
        "max_completion_tokens": int(max_completion_tokens),
    }
    if response_format:
        payload["response_format"] = response_format
    if sampling:
        payload.update(sampling)  # temperature/top_p/top_k/min_p go top-level (OpenAI-compat)
    if chat_template_kwargs:
        payload["chat_template_kwargs"] = chat_template_kwargs
    payload.update(openrouter_request_extras(reasoning=reasoning))
    return payload
```

- [ ] **Step 4: Wire env into `reward.py._openai_chat`**

In `_openai_chat` (after the existing `response_format` handling, before the `build_chat_payload(...)` call), add:

```python
import json as _json  # top of reward.py if not present
...
_sampling_env = os.environ.get("PERSONA_JUDGE_SAMPLING")
_sampling = _json.loads(_sampling_env) if _sampling_env else None
_think_env = os.environ.get("PERSONA_JUDGE_ENABLE_THINKING")
_ctk = {"enable_thinking": _think_env == "1"} if _think_env in ("0", "1") else None
payload = build_chat_payload(
    model=..., messages=messages, max_completion_tokens=...,
    response_format=response_format, reasoning=False,
    sampling=_sampling, chat_template_kwargs=_ctk,
)
```

- [ ] **Step 5: Run tests — expect PASS**

Run: `python -m pytest tests/test_judge_payload.py -v` → PASS.

- [ ] **Step 6: Commit**

```bash
git add shared/api_client.py training/grpo/reward.py tests/test_judge_payload.py
git commit -m "feat(judge-sweep): env-gated sampling + enable_thinking on judge payload"
```

---

## Task 4: `PERSONA_JUDGE_JSON_SCHEMA=1` strict-schema patch

**Files:**
- Modify: `training/grpo/reward.py` (`_openai_chat`, response_format selection near line 504 call site)
- Test: `tests/test_judge_payload.py` (extend)

**Interfaces:**
- Produces: when `PERSONA_JUDGE_JSON_SCHEMA=1`, judge calls send a strict `json_schema` response_format with `required: ["rating"]` instead of `{"type": "json_object"}`.

- [ ] **Step 1: Add failing test**

```python
# append to tests/test_judge_payload.py
import os
from training.grpo.reward import _resolve_response_format  # new helper

def test_json_schema_env_on(monkeypatch):
    monkeypatch.setenv("PERSONA_JUDGE_JSON_SCHEMA", "1")
    rf = _resolve_response_format()
    assert rf["type"] == "json_schema"
    assert "rating" in rf["json_schema"]["schema"]["required"]

def test_json_schema_env_off(monkeypatch):
    monkeypatch.delenv("PERSONA_JUDGE_JSON_SCHEMA", raising=False)
    assert _resolve_response_format() == {"type": "json_object"}
```

- [ ] **Step 2: Run — expect FAIL** (`_resolve_response_format` undefined).

- [ ] **Step 3: Implement helper + use it**

```python
# training/grpo/reward.py
def _resolve_response_format() -> dict:
    """json_object by default; strict json_schema (rating required) when PERSONA_JUDGE_JSON_SCHEMA=1."""
    if os.environ.get("PERSONA_JUDGE_JSON_SCHEMA") == "1":
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "turing_rating",
                "schema": {
                    "type": "object",
                    "properties": {"rating": {"type": "integer", "minimum": 1, "maximum": 7}},
                    "required": ["rating"],
                },
            },
        }
    return {"type": "json_object"}
```

Replace the literal `response_format={"type": "json_object"}` at the `_openai_chat` call (line ~504) with `response_format=_resolve_response_format()`.

- [ ] **Step 4: Run — expect PASS.**

- [ ] **Step 5: Commit**

```bash
git add training/grpo/reward.py tests/test_judge_payload.py
git commit -m "feat(judge-sweep): PERSONA_JUDGE_JSON_SCHEMA strict rating schema"
```

---

## Task 5: Reward-layer dump helpers (viewer-compatible)

The HTTP-wire dump already exists (`our_patches.md` — `_dump_judge_response`). This adds the *reward-layer* dump that `scripts/dump_viewer.py` renders (detection key `generated_is_b`).

**Files:**
- Modify: `training/grpo/reward.py` (`_build_reward_dump_row`, `_dump_reward_call`; wire into `_score_pairwise_likert_with_info` return, ~line 686-708)
- Test: `tests/test_reward_dump_row.py`

**Interfaces:**
- Produces: rows written to `${PERSONA_REWARD_DUMP_DIR}/reward-{SLURM_JOB_ID}-{pid}.jsonl`, one per judge call, whose top-level keys exactly match the viewer contract.

- [ ] **Step 1: Write the failing test (pin the viewer contract)**

```python
# tests/test_reward_dump_row.py
from training.grpo.reward import _build_reward_dump_row

REQUIRED = {
    "generated_is_b", "human_side", "rating_gt_first", "rating_gen_first", "randomized_order",
    "response", "ground_truth", "context", "user_history", "judge_response", "judge_prompt",
    "judge_raw_content", "judge_reasoning", "judge_latency_ms", "judge_finish_reason",
    "judge_model", "judge_usage", "final_reward", "turing_judge_score_raw",
    "turing_judge_score_clipped", "source_copy_penalty", "assistant_like_penalty",
    "wrong_target_or_role_penalty", "unsupported_adversarial_reframing_penalty",
    "call_id", "user_id", "post_id", "target_idx", "persona", "ts", "worker_pid",
}

def _row():
    return _build_reward_dump_row(
        response="gen", ground_truth="human", context="ctx", user_history="hist",
        human_side="A", generated_is_b=True, randomized_order="gt_first",
        rating_gt_first=3, rating_gen_first=None,
        judge_response={"rating": 3, "reasoning": "..."}, judge_prompt="P",
        judge_raw_content="{...}", judge_reasoning="<think>..</think>",
        judge_latency_ms=1234, judge_finish_reason="stop", judge_model="qwen3-8b",
        judge_usage={"completion_tokens": 100}, final_reward=0.33,
        turing_judge_score_raw=3.0, turing_judge_score_clipped=3.0,
        source_copy_penalty=0.0, assistant_like_penalty=0.0,
        wrong_target_or_role_penalty=0.0, unsupported_adversarial_reframing_penalty=0.0,
        call_id="c1", user_id="u1", post_id="p1", target_idx=0, persona="",
        ts=1000.0, worker_pid=42)

def test_reward_row_has_all_viewer_keys():
    assert REQUIRED <= set(_row())

def test_generated_is_b_present_even_if_false():
    row = _build_reward_dump_row(**{**_kwargs_with(generated_is_b=False)})
    assert "generated_is_b" in row  # membership, not truthiness (viewer line 73)
```

(Provide `_kwargs_with` as a small helper in the test mirroring `_row()`'s kwargs.)

- [ ] **Step 2: Run — expect FAIL.**

- [ ] **Step 3: Implement the two helpers**

```python
# training/grpo/reward.py
import time, os

def _build_reward_dump_row(**f) -> dict:
    """Assemble a reward-layer dump row matching scripts/dump_viewer.py's contract.
    `rating` is intentionally NOT stored — the viewer derives it from rating_gt/gen_first."""
    return {k: f.get(k) for k in (
        "generated_is_b", "human_side", "rating_gt_first", "rating_gen_first", "randomized_order",
        "response", "ground_truth", "context", "user_history", "judge_response", "judge_prompt",
        "judge_raw_content", "judge_reasoning", "judge_latency_ms", "judge_finish_reason",
        "judge_model", "judge_usage", "final_reward", "turing_judge_score_raw",
        "turing_judge_score_clipped", "source_copy_penalty", "assistant_like_penalty",
        "wrong_target_or_role_penalty", "unsupported_adversarial_reframing_penalty",
        "call_id", "user_id", "post_id", "target_idx", "persona", "ts", "worker_pid")}

def _dump_reward_call(row: dict) -> None:
    """Append one reward-layer row to the per-worker JSONL if dumping is enabled."""
    if float(os.environ.get("PERSONA_JUDGE_DUMP_RATE", "0")) <= 0:
        return
    d = os.environ.get("PERSONA_REWARD_DUMP_DIR")
    if not d:
        return
    os.makedirs(d, exist_ok=True)
    job = os.environ.get("SLURM_JOB_ID", "local")
    path = os.path.join(d, f"reward-{job}-{os.getpid()}.jsonl")
    with open(path, "a") as fh:
        fh.write(json.dumps(row, default=str) + "\n")
```

- [ ] **Step 4: Wire into `_score_pairwise_likert_with_info`**

At the point where the final aggregated dict is assembled (before `return`, ~line 686-708), populate the fields already computed (`human_side`, `generated_is_b`, `rating_gt_first`/`rating_gen_first`, penalties, `score`, latency, `finish_reason`, `usage`, the formatted `prompt`, raw text, parsed `reasoning`) into `_build_reward_dump_row(...)` and call `_dump_reward_call(row)`. Capture `judge_latency_ms` around the `_openai_chat` await, and thread `judge_finish_reason`/`judge_usage` out of `_openai_chat` (extend it to return them, or read from the last response). `ts=time.time()`, `worker_pid=os.getpid()`.

- [ ] **Step 5: Run — expect PASS.**

- [ ] **Step 6: (Cluster) smoke against the real viewer**

Handoff: run a 2-pair judge call with `PERSONA_JUDGE_DUMP_RATE=1.0 PERSONA_REWARD_DUMP_DIR=/tmp/rewtest`, then `python scripts/dump_viewer.py --dumps /tmp/rewtest --port 8090` and confirm rows render as reward rows (all 10 tabs populated). Report back.

- [ ] **Step 7: Commit**

```bash
git add training/grpo/reward.py tests/test_reward_dump_row.py
git commit -m "feat(judge-sweep): reward-layer dump rows (dump_viewer-compatible)"
```

---

## Task 6: Multi-replica vLLM launcher + serving sbatches

**Files:**
- Create: `scripts/vllm_serve_replicas.sh`
- Create: `scripts/slurm/cot_serve_8rep.sh`, `scripts/slurm/judge_serve_cell.sh`, `scripts/slurm/judge_serve_anchor.sh`

**Interfaces:**
- Produces: N vLLM OpenAI servers on one node at right-sized TP (ports `BASE..BASE+N-1`), plus a `endpoints.txt` listing URLs; consumed by Tasks 7, 12, 14, 15.

- [ ] **Step 1: Write `vllm_serve_replicas.sh`**

Params (env): `MODEL`, `TP`, `REPLICAS`, `BASE_PORT` (default 8000), `THINKING` (`on`/`off`), `PY`, `LOG_DIR`, `EXTRA_ARGS`. For `i in 0..REPLICAS-1`, bind GPUs `i*TP..i*TP+TP-1` via `CUDA_VISIBLE_DEVICES`, launch `python -m vllm.entrypoints.openai.api_server --model $MODEL --tensor-parallel-size $TP --port $((BASE_PORT+i))`, adding `--reasoning-parser qwen3` iff `THINKING=on`. Write each URL to `$LOG_DIR/endpoints.txt`, tail-wait on `/health` for all, then `wait`.

```bash
#!/bin/bash
set -uo pipefail
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY
export HF_HOME=/home/lancewicki/data/hf_cache HF_HUB_CACHE=/home/lancewicki/data/hf_cache
export HF_HUB_DISABLE_XET=1 PYTHONUNBUFFERED=1
: "${BASE_PORT:=8000}"; : "${THINKING:=off}"; : "${LOG_DIR:=/tmp/vllm_logs}"
mkdir -p "$LOG_DIR"; : > "$LOG_DIR/endpoints.txt"
RP=(); [ "$THINKING" = "on" ] && RP=(--reasoning-parser qwen3)
for i in $(seq 0 $((REPLICAS-1))); do
  start=$((i*TP)); gpus=$(seq -s, $start $((start+TP-1))); port=$((BASE_PORT+i))
  CUDA_VISIBLE_DEVICES=$gpus "$PY" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --download-dir "$HF_HOME" --tensor-parallel-size "$TP" \
    --max-model-len 32768 --gpu-memory-utilization 0.85 --dtype bfloat16 \
    "${RP[@]}" ${EXTRA_ARGS:-} --host 0.0.0.0 --port "$port" \
    > "$LOG_DIR/server-$port.log" 2>&1 &
  echo "http://$(hostname):$port/v1" >> "$LOG_DIR/endpoints.txt"
done
# wait for all /health then block
for port in $(seq $BASE_PORT $((BASE_PORT+REPLICAS-1))); do
  until curl -sf "http://localhost:$port/health" >/dev/null; do sleep 5; done
done
echo "all $REPLICAS replicas healthy"; wait
```

- [ ] **Step 2: Write the three sbatch wrappers** (`--gres=gpu:8`, `--partition=a100`, `--account=rfai`). `cot_serve_8rep.sh`: `MODEL=Qwen/Qwen3-8B TP=1 REPLICAS=8 THINKING=off PY=<turing-rl-train>`. `judge_serve_cell.sh`: takes `MODEL/TP/REPLICAS/THINKING` from env (set per cell). `judge_serve_anchor.sh`: `MODEL=Qwen/Qwen3.5-397B-A17B-GPTQ-Int4 TP=8 REPLICAS=1 PY=<judge-vllm>`, `THINKING` per cell.

- [ ] **Step 3: (Cluster) validate a 2-replica bring-up** and confirm `endpoints.txt` + `/health`. Report back. No local test (GPU-only).

- [ ] **Step 4: Commit**

```bash
git add scripts/vllm_serve_replicas.sh scripts/slurm/cot_serve_8rep.sh scripts/slurm/judge_serve_cell.sh scripts/slurm/judge_serve_anchor.sh
git commit -m "feat(judge-sweep): multi-replica vLLM launcher + serving sbatches"
```

---

## Task 7: Served async round-robin CoT client (thinking-off)

Replaces the dead OpenRouter path. Produces the full 3272-row CoT parquet with thinking-off.

**Files:**
- Create: `scripts/generate_cot_served.py` (imports `reasoning_leaks_reply` from `data/sft/generate_cot.py`)
- Test: `tests/test_generate_cot_served.py`

**Interfaces:**
- Consumes: `endpoints.txt` (Task 6), the SFT source rows.
- Produces: `data/sft/prism_full_s42_sft_cot.parquet` (3272 rows) + `.cot_metadata.json`.

- [ ] **Step 1: Write failing unit test for round-robin + payload shape**

```python
# tests/test_generate_cot_served.py
from scripts.generate_cot_served import build_cot_payload, pick_endpoint

def test_thinking_off_payload():
    p = build_cot_payload("Qwen/Qwen3-8B", [{"role": "user", "content": "hi"}],
                          sampling={"temperature": 0.7})
    assert p["chat_template_kwargs"] == {"enable_thinking": False}
    assert "reasoning" not in p and p["temperature"] == 0.7
    assert p["max_completion_tokens"] == 4096

def test_round_robin():
    eps = ["a", "b", "c"]
    assert [pick_endpoint(eps, i) for i in range(4)] == ["a", "b", "c", "a"]
```

- [ ] **Step 2: Run — expect FAIL.**

- [ ] **Step 3: Implement the client**

Async `aiohttp`, `asyncio.Semaphore(16 * n_endpoints)`, round-robin by request index, `build_cot_payload` sets `max_completion_tokens=4096`, `chat_template_kwargs={"enable_thinking": False}`, sampling from `sampling_fidelity.md` decision (thinking-off), **no** `reasoning` field, **no** provider extras. After generation: run `reasoning_leaks_reply(reasoning, ground_truth)` per row; collect leaked indices; re-generate up to 10 tries. Write parquet + metadata (`{n_rows, endpoints, sampling, thinking: "off", wall_s, leak_regen_counts}`).

- [ ] **Step 4: Run — expect PASS.**

- [ ] **Step 5: (Cluster) full run**

Handoff: launch `cot_serve_8rep.sh`, then `python scripts/generate_cot_served.py --endpoints <log>/endpoints.txt --out data/sft/prism_full_s42_sft_cot.parquet`. Gate: assert 3272 rows, 0 residual leaks, spot-check 5 rows are thinking-off style (start "The user…", no `<think>`). **Discard the 138-row smoke parquet.** Report row count + leak stats.

- [ ] **Step 6: Commit**

```bash
git add scripts/generate_cot_served.py tests/test_generate_cot_served.py
git commit -m "feat(judge-sweep): served round-robin thinking-off CoT client"
```

---

## Task 8: SFT checkpointing patches + full-run sbatch

**Files:**
- Modify: `training/sft/lora_sft.py` (SFTConfig ~line 376; argparse ~line 240; `trainer.train()` ~line 413)
- Modify: `training/sft/configs/qwen3_8b_lora.yaml`
- Create: `scripts/slurm/sft_full.sh`
- Test: `tests/test_lora_sft_config.py`

**Interfaces:**
- Produces: yaml-configurable `save_strategy`/`save_steps`/`save_total_limit`; `--resume_from_checkpoint auto`; checkpoint `checkpoints/sft/qwen3_8b_prism_full_s42/`.

- [ ] **Step 1: Write failing test for the resume-resolver + config plumbing**

```python
# tests/test_lora_sft_config.py
from training.sft.lora_sft import resolve_resume_checkpoint, save_kwargs_from_config

def test_resolve_auto_picks_highest_step(tmp_path):
    (tmp_path/"checkpoint-10").mkdir(); (tmp_path/"checkpoint-70").mkdir()
    assert resolve_resume_checkpoint("auto", str(tmp_path)).endswith("checkpoint-70")

def test_resolve_auto_empty_returns_none(tmp_path):
    assert resolve_resume_checkpoint("auto", str(tmp_path)) is None

def test_save_kwargs_from_yaml():
    cfg = {"save_strategy": "steps", "save_steps": 10, "save_total_limit": 2}
    assert save_kwargs_from_config(cfg) == {
        "save_strategy": "steps", "save_steps": 10, "save_total_limit": 2}

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
    ckpts = glob.glob(os.path.join(output_dir, "checkpoint-*"))
    if not ckpts:
        return None
    return max(ckpts, key=lambda p: int(p.rsplit("-", 1)[-1]))

def save_kwargs_from_config(config: dict) -> dict:
    ss = config.get("save_strategy", "epoch")
    out = {"save_strategy": ss}
    if ss == "steps":
        out["save_steps"] = config.get("save_steps", 10)
        out["save_total_limit"] = config.get("save_total_limit", 2)
    return out
```

In `SFTConfig(...)` replace the hard-coded `save_strategy="epoch"` with `**save_kwargs_from_config(config)`. Add argparse `--resume_from_checkpoint` (default `None`). Replace bare `trainer.train()` with `trainer.train(resume_from_checkpoint=resolve_resume_checkpoint(args.resume_from_checkpoint, output_dir))`.

- [ ] **Step 4: Update the yaml**

Append to `training/sft/configs/qwen3_8b_lora.yaml`:
```yaml
save_strategy: steps
save_steps: 10
save_total_limit: 2
```

- [ ] **Step 5: Write `sft_full.sh`** (`--gres=gpu:8`, `--time=20:00:00`), always passing `--model qwen3-8b --data_path data/sft/prism_full_s42_sft_cot.jsonl --output_dir checkpoints/sft/qwen3_8b_prism_full_s42 --max_seq_length 8192 --resume_from_checkpoint auto`, plus the existing `report_to: wandb` sed-trap trick.

- [ ] **Step 6: Run — expect PASS.** `python -m pytest tests/test_lora_sft_config.py -v`.

- [ ] **Step 7: Commit**

```bash
git add training/sft/lora_sft.py training/sft/configs/qwen3_8b_lora.yaml scripts/slurm/sft_full.sh tests/test_lora_sft_config.py
git commit -m "feat(judge-sweep): SFT step-checkpointing + auto-resume"
```

---

## Task 9: Build SFT JSONL (cluster orchestration)

**Files:** none new (upstream `data/sft/build_sft_jsonl.py`).

- [ ] **Step 1: (Cluster) run** `python data/sft/build_sft_jsonl.py --in data/sft/prism_full_s42_sft_cot.parquet --out data/sft/prism_full_s42_sft_cot.jsonl`. Gate: line count == 3272; spot-check one line has `<reasoning>...</reasoning>\n[HUMAN]: ...` in the assistant target. Report back.

---

## Task 10: SFT training + heldout inference (cluster orchestration)

**Files:** none new (upstream `training/sft/lora_sft.py` via `sft_full.sh`, `eval/generate_trained.py`).

- [ ] **Step 1: (Cluster) launch SFT** via `sbatch scripts/slurm/sft_full.sh`. Monitor wandb; expect ~78 steps, ~8 checkpoints. On failure, resubmit (auto-resume picks up). Gate: `checkpoints/sft/qwen3_8b_prism_full_s42/final/` exists with adapter `.safetensors`.

- [ ] **Step 2: (Cluster) heldout inference**

```bash
python eval/generate_trained.py --metric turing \
  --checkpoint_dir checkpoints/sft/qwen3_8b_prism_full_s42 \
  --test_parquet data/prism/full_s42_history_sft40_grpo60_test10/test.parquet \
  --gen_num 1 --vllm_tensor_parallel_size 1 \
  --output results/2026-07-08-judge-sweep/raw/generator/heldout_inference.pkl
```
Sampling is domain-inferred (prism → T=0.6, top_p=1.0, top_k=-1, pres_pen=0.5, max_tokens=2048) — matches paper Table 4. Gate: 880 generations across 128 users. Write `heldout_inference_metadata.json` (adapter path, base model, sampling, wall time). Report back.

---

## Task 11: Pair-set builder

**Files:**
- Create: `scripts/build_judge_pairs.py`
- Test: `tests/test_build_judge_pairs.py`

**Interfaces:**
- Consumes: `heldout_inference.pkl` (nested `{user_id: {..., test_targets: [{generations: [{response, reasoning, ...}], ...}]}}`), `test.parquet`.
- Produces: `results/2026-07-08-judge-sweep/raw/pairs/prism_heldout_880.parquet` + returns `(df, sanity_metadata)`.

- [ ] **Step 1: Write failing test on synthetic fixtures**

```python
# tests/test_build_judge_pairs.py
import pickle, pandas as pd
from scripts.build_judge_pairs import build_pairs

def _make(tmp_path):
    infer = {"u1": {"test_targets": [
        {"generations": [{"response": "hi there", "reasoning": "r"}]}]}}
    (tmp_path/"inf.pkl").write_bytes(pickle.dumps(infer))
    df = pd.DataFrame([{"reward_model": {"ground_truth": "hello"},
                        "extra_info": {"user_id": "u1", "post_id": "p1", "target_idx": 0,
                                       "user_history": "h", "context": "c", "persona": ""}}])
    df.to_parquet(tmp_path/"test.parquet")
    return str(tmp_path/"inf.pkl"), str(tmp_path/"test.parquet")

def test_pairs_columns_and_strip(tmp_path):
    df, meta = build_pairs(*_make(tmp_path))
    assert list(df.columns) == ["pair_id", "user_id", "post_id", "target_idx",
                                "user_history", "context", "persona", "human", "generated"]
    assert df.iloc[0]["human"] == "hello" and df.iloc[0]["generated"] == "hi there"
    assert "<reasoning>" not in df.iloc[0]["generated"]

def test_flags_exact_matches(tmp_path):
    # generated == human should be counted, not asserted
    _, meta = build_pairs(*_make(tmp_path))
    assert "exact_match_count" in meta and "exact_match_frac" in meta
```

- [ ] **Step 2: Run — expect FAIL.**

- [ ] **Step 3: Implement**

Walk the pickle by `user_id`, align each target with its `test.parquet` row (by `user_id` + `target_idx`), take `human = reward_model.ground_truth`, `generated = generations[0]["response"]` (already reasoning-stripped by `parse_reasoning_and_response`). Assert no residual `<reasoning>` tag; assert every row present; **count** (don't assert) `human == generated`; if `exact_match_frac > 0.01`, print a WARN. Assign `pair_id = f"{user_id}:{target_idx}"`. Return `(df, meta)`; when run as `__main__`, write the parquet + a sidecar metadata json.

- [ ] **Step 4: Run — expect PASS.**

- [ ] **Step 5: (Cluster) run on real inference**, confirm 880 rows, report exact-match count. Freeze the parquet.

- [ ] **Step 6: Commit**

```bash
git add scripts/build_judge_pairs.py tests/test_build_judge_pairs.py
git commit -m "feat(judge-sweep): 880-pair-set builder + sanity checks"
```

---

## Task 12: Sweep-cell client (round-robin, calls the real reward path)

Core of the sweep. Runs all 880 pairs (both orderings handled inside `_score_pairwise_likert_with_info`) through the endpoints for one cell, sets the env locking sampling/thinking/schema/dumps.

**Files:**
- Create: `scripts/run_judge_sweep_cell.py`
- Test: `tests/test_run_judge_sweep_cell.py`

**Interfaces:**
- Consumes: pair-set parquet, `endpoints.txt`, a cell dict `{judge_id, model_id, thinking_mode, sampling}`.
- Produces: reward + HTTP dumps under `raw/sweep/{judge}/{thinking_mode}/{reward,http}/`, `run_metadata.json`. Supports `--limit N` (used by calibration).

- [ ] **Step 1: Write failing test for env setup + output pathing (no real HTTP)**

```python
# tests/test_run_judge_sweep_cell.py
from scripts.run_judge_sweep_cell import cell_env, cell_output_dirs

def test_cell_env_locks_config():
    env = cell_env({"thinking_mode": "off", "sampling": {"temperature": 0.7}}, model_id="Qwen/Qwen3-8B")
    assert env["PERSONA_JUDGE_JSON_SCHEMA"] == "1"
    assert env["PERSONA_JUDGE_DUMP_RATE"] == "1.0"
    assert env["PERSONA_JUDGE_ENABLE_THINKING"] == "0"
    assert env["JUDGE_MODEL"] == "Qwen/Qwen3-8B"
    assert env["PERSONA_JUDGE_MAX_COMPLETION_TOKENS"] == "8192"

def test_output_dirs(tmp_path):
    d = cell_output_dirs(str(tmp_path), "qwen3-8b", "off")
    assert d["reward"].endswith("qwen3-8b/off/reward")
    assert d["http"].endswith("qwen3-8b/off/http")
```

- [ ] **Step 2: Run — expect FAIL.**

- [ ] **Step 3: Implement**

`cell_env(cell, model_id)` returns the env dict: `PERSONA_JUDGE_JSON_SCHEMA=1`, `PERSONA_JUDGE_DUMP_RATE=1.0`, `PERSONA_JUDGE_ENABLE_THINKING` (`1`/`0` by mode), `PERSONA_JUDGE_SAMPLING=json.dumps(cell["sampling"])`, `JUDGE_MODEL=model_id`, `PERSONA_JUDGE_MAX_COMPLETION_TOKENS=8192`, `PERSONA_JUDGE_DUMP_DIR`/`PERSONA_REWARD_DUMP_DIR` (the cell's http/reward dirs). The client applies these to `os.environ`, then for each pair calls `score_turing_with_info(...)` (which drives `_score_pairwise_likert_with_info`, both orderings) under an `asyncio.Semaphore(16*n_endpoints)`, round-robin selecting the endpoint by setting `OPENAI_API_BASE`/passing `api_base` per call. Write `run_metadata.json` (judge id, model, sampling, thinking, schema state, endpoints, concurrency, slurm id, start/end ts). `--limit N` truncates the pair list.

> Note: round-robin across endpoints via the reward path requires per-call `api_base`. If `_score_pairwise_likert_with_info` doesn't accept `api_base`, thread it through (small extension) OR run one client process per endpoint each pinned to `OPENAI_API_BASE` and shard the pairs — pick the process-sharding approach if threading is invasive; document the choice.

- [ ] **Step 4: Run — expect PASS.**

- [ ] **Step 5: (Cluster) 10-pair synthetic smoke** against a live 8B endpoint; confirm reward+http dumps land and the viewer renders them. Report back.

- [ ] **Step 6: Commit**

```bash
git add scripts/run_judge_sweep_cell.py tests/test_run_judge_sweep_cell.py
git commit -m "feat(judge-sweep): sweep-cell client over the real reward path"
```

---

## Task 13: Cell config + family gate

**Files:**
- Create: `configs/judge_sweep_cells.py`
- Test: `tests/test_sweep_cell_config.py`

**Interfaces:**
- Produces: `cell_list(family)` (10 cells: 4 sizes × 2 modes + anchor × 2), `tp_for_size(size_b, is_moe)`.

- [ ] **Step 1: Write failing test**

```python
# tests/test_sweep_cell_config.py
from configs.judge_sweep_cells import tp_for_size, cell_list

def test_tp_lookup():
    assert tp_for_size(4, False) == (1, 8)
    assert tp_for_size(14, False) == (1, 8)
    assert tp_for_size(27, False) == (2, 4)
    assert tp_for_size(32, False) == (2, 4)
    assert tp_for_size(35, True) == (1, 8)   # MoE-Int4

def test_cell_list_qwen35_has_10_cells():
    cells = cell_list("qwen3.5")
    assert len(cells) == 10  # (4 judges + anchor) x 2 modes
    assert any(c["model_id"].endswith("397B-A17B-GPTQ-Int4") for c in cells)
```

- [ ] **Step 2: Run — expect FAIL.**

- [ ] **Step 3: Implement**

`tp_for_size`: TP=2/replicas=4 for dense ≥ ~20B (27B/32B), else TP=1/replicas=8. `cell_list(family)`: pick the 4 model IDs by family (`qwen3` → 4B/8B/14B/32B; `qwen3.5` → 4B/9B/27B/35B-A3B-Int4), add the 397B anchor (TP=8, replicas=1), cross with `["on", "off"]`, attach `sampling` per mode from `sampling_fidelity.md`. Anchor `is_moe=True` (Int4 MoE) but TP fixed at 8.

- [ ] **Step 4: Run — expect PASS.**

- [ ] **Step 5: Family gate — 4B smoke (cluster)**

Handoff: serve Qwen3-4B and Qwen3.5-4B (TP=1), run `run_judge_sweep_cell.py --limit 50` on each (100 calls), compare tokens/sec at parse-rate parity; tiebreak on 397B-agreement over 50 pairs; if inconclusive default `qwen3.5`. Write `derived/family_decision.md` and set the chosen `family`. This is the gate that populates `cell_list`.

- [ ] **Step 6: Commit**

```bash
git add configs/judge_sweep_cells.py tests/test_sweep_cell_config.py
git commit -m "feat(judge-sweep): config-driven cell list + family gate"
```

---

## Task 14: Per-cell throughput calibration + >4h gate

Reuses Task 12's client (`--limit 50`) so calibration exercises the *real* judge path (more faithful than `benchmark_judge_throughput.py`, whose payload differs — documented deviation from spec Step 5.0).

**Files:**
- Create: `scripts/calibration_report.py`
- Test: `tests/test_calibration_report.py`

**Interfaces:**
- Consumes: `run_judge_sweep_cell.py --limit 50` outputs (latencies) for all 10 cells.
- Produces: `raw/calibration/{calibration_results.jsonl, calibration_metadata.json}`, `derived/calibration_report.md`.

- [ ] **Step 1: Write failing test for extrapolation math**

```python
# tests/test_calibration_report.py
from scripts.calibration_report import extrapolate_wall_hours

def test_extrapolate():
    # 100 calls in 300s -> 0.333 req/s -> 1760 calls
    assert round(extrapolate_wall_hours(n_calls=100, wall_s=300, total_calls=1760), 2) == 1.47

def test_gate_flags_over_4h():
    assert extrapolate_wall_hours(100, 3600, 1760) > 4
```

- [ ] **Step 2: Run — expect FAIL.** Then implement `extrapolate_wall_hours` and a report writer that emits per-cell req/s, p50/p95, extrapolated hours, and a `>4h` flag with the ±30% precision caveat inline. Run — expect PASS.

- [ ] **Step 3: (Cluster) run calibration** for all 10 cells (~1h, parallel), write the report. **Gate:** any cell projecting >4h → surface and decide (reduce concurrency / drop cell / accept) before Task 15. Report back.

- [ ] **Step 4: Commit**

```bash
git add scripts/calibration_report.py tests/test_calibration_report.py
git commit -m "feat(judge-sweep): per-cell throughput calibration + 4h gate"
```

---

## Task 15: Full judge sweep (cluster orchestration)

**Files:** none new (uses Tasks 6, 12, 13 artifacts).

- [ ] **Step 1: (Cluster) launch** — Node 1: anchor (`judge_serve_anchor.sh`, 2 cells sequential: on/off). Nodes 2 & 3: one smaller-judge cell per node, each filling the node with replicas per `tp_for_size`; submit 8 independent sbatches (4 sizes × 2 modes) so Slurm parallelizes. Each cell = serve (Task 6) then `run_judge_sweep_cell.py` (Task 12) over 880 pairs. Copy vLLM logs + slurm out into `raw/logs/`.

- [ ] **Step 2: Gate per cell:** reward-dump row count == 1760; `run_metadata.json` present. If R2 (anchor throughput) triggers, drop anchor `max_completion_tokens` to 4096 and note as deviation. Report per-cell completion + any anomalies.

---

## Task 16: Bonus offline batched cell

**Files:**
- Create: `scripts/run_offline_batched_cell.py`
- Test: reuse `tests/test_reward_dump_row.py` contract.

- [ ] **Step 1: Implement** an in-process `LLM.generate` (TP=8) over all 1760 prompts for Qwen3-8B thinking-off, computing the same rating logic and emitting reward-layer dumps via `_build_reward_dump_row` into `raw/sweep/qwen3-8b/off_offline/reward/`. Same pair-set/sampling/prompt as the server cell.

- [ ] **Step 2: (Cluster) run** post-primary-sweep on a free node (~30-60 min). Gate: 1760 rows; accuracy matches the server cell within noise; throughput higher. Report the server-vs-offline ratio.

- [ ] **Step 3: Commit**

```bash
git add scripts/run_offline_batched_cell.py
git commit -m "feat(judge-sweep): offline batched bonus cell"
```

---

## Task 17: Analyzer (raw → derived)

**Files:**
- Create: `scripts/analyze_judge_sweep.py`
- Test: `tests/test_analyze_judge_sweep.py`

**Interfaces:**
- Consumes: `raw/sweep/**/reward/*.jsonl`.
- Produces: `derived/{summary.md, summary.parquet, per_pair_metrics.parquet}`. Idempotent.

- [ ] **Step 1: Write failing tests for the metric functions on synthetic rows**

```python
# tests/test_analyze_judge_sweep.py
from scripts.analyze_judge_sweep import accuracy, position_bias, kappa_vs_anchor, budget_hit_rate

def _row(rating, human_side="A", finish="stop", order="gt_first"):
    return {"rating_gt_first": rating, "rating_gen_first": None, "human_side": human_side,
            "judge_finish_reason": finish, "randomized_order": order,
            "judge_response": {"rating": rating}}

def test_accuracy_picks_human():
    # human_side=A, rating<4 -> picks A -> correct
    rows = [_row(2), _row(6, human_side="B")]
    assert accuracy(rows) == 1.0

def test_tie_excluded():
    assert accuracy([_row(4)]) is None  # only ties -> undefined, excluded from denom

def test_budget_hit_rate():
    assert budget_hit_rate([_row(3, finish="length"), _row(3)]) == 0.5

def test_kappa_perfect_agreement():
    rows = [_row(2), _row(6, human_side="B")]
    assert kappa_vs_anchor(rows, rows) == 1.0
```

- [ ] **Step 2: Run — expect FAIL.**

- [ ] **Step 3: Implement metrics + driver**

Functions: `accuracy` (rating<4→A, >4→B, ==4→tie excluded; compare to `human_side`), `accuracy_format_correct`, `position_bias` (|acc_A_first − acc_B_first|), `kappa_vs_anchor` (Cohen's kappa on binary human-pick vs anchor by `pair_id`), `budget_hit_rate` (`finish_reason=="length"`), `format_correct_rate`, `rating_recovery_rate` (via `_extract_turing_rating` on rows lacking parsed rating), `per_field_presence` (over the ~28 rubric fields in `judge_response`), `rating_distribution`, `per_confidence_bucket_accuracy` (bucketed by anchor rating {1,7}/{2,3,5,6}/{4}), `length_distribution`. `analyze(raw_dir, derived_dir)` loads all cells, builds `per_pair_metrics.parquet` (one row per pair×judge×mode), aggregates to `summary.parquet`/`summary.md` (10 rows), and writes the generator-sampling caveat inline in `summary.md`.

- [ ] **Step 4: Run — expect PASS.**

- [ ] **Step 5: (Cluster) run** `python scripts/analyze_judge_sweep.py --raw results/2026-07-08-judge-sweep/raw --derived results/2026-07-08-judge-sweep/derived`. Report summary table.

- [ ] **Step 6: Commit**

```bash
git add scripts/analyze_judge_sweep.py tests/test_analyze_judge_sweep.py
git commit -m "feat(judge-sweep): offline analyzer (raw -> derived)"
```

---

## Task 18: Plots + derived docs

**Files:**
- Create: `scripts/plot_judge_sweep.py`

- [ ] **Step 1: Implement** plots from `summary.parquet`/`per_pair_metrics.parquet`: `accuracy_vs_size.png` (on/off overlaid; MoE at active-param w/ note), `kappa_vs_size.png`, `per_field_presence_vs_size.png`, `budget_hit_vs_size.png`, `throughput_vs_size.png`, and one `rating_distribution_{judge}_{mode}.png` per cell.

- [ ] **Step 2: (Cluster) run**, produce PNGs into `derived/plots/`. Report back.

- [ ] **Step 3: Commit**

```bash
git add scripts/plot_judge_sweep.py
git commit -m "feat(judge-sweep): plots"
```

---

## Task 19: Docs, `our_patches.md`, and Results write-up

**Files:**
- Modify: `our_patches.md`
- Create: `results/2026-07-08-judge-sweep/README.md`, `results/2026-07-08-judge-sweep/README.txt` (repro per user global rule)
- Modify: `docs/superpowers/specs/2026-07-08-judge-sweep-design.md` (append Results)

- [ ] **Step 1: Update `our_patches.md`** with the four new patches (Tasks 3+4 on `reward.py`/`api_client.py`, Task 5 reward dumps, Task 8 SFT), each marked PERSISTENT with rationale. Note the `_extract_chat_content` empty-return patch (existing) was written for `deepseek_r1`; add a line that it was re-validated under `--reasoning-parser qwen3` (Task 6 smoke).

- [ ] **Step 2: Write `README.md`** (one-page tour of `raw/`+`derived/`, how to regenerate `derived/`) and `README.txt` (exact commands, input provenance, upstream steps).

- [ ] **Step 3: Append the Results section** to the spec, pointing into `derived/` tables/plots; finalize `family_decision.md`, `calibration_report.md`, `split_verification.md`, `offline_vs_server_comparison.md`.

- [ ] **Step 4: Commit**

```bash
git add our_patches.md results/2026-07-08-judge-sweep/README.md results/2026-07-08-judge-sweep/README.txt docs/superpowers/specs/2026-07-08-judge-sweep-design.md
git commit -m "docs(judge-sweep): patches log, repro README, Results write-up"
```

---

## Self-Review

**Spec coverage:** §1 goal/anchor/family/sampling → Tasks 1, 13, 15; §2 PRISM verification → Task 2; §3 data foundation → Tasks 2, 11; §4 generator (CoT/SFT/inference/pairs) → Tasks 7, 8, 9, 10, 11; §5 sweep (fixed inputs, matrix, calibration, serving, offline, viewer, metrics, artifacts) → Tasks 6, 12, 13, 14, 15, 16, 17, 18; §6 analysis/ordering/risks → Tasks 14 (R2 gate), 15, 17; patches summary → Tasks 3, 4, 5, 8, 19. Deliverables 1-5 → Tasks 17/18 (table), 13 (family doc), 8/10 (checkpoint), 10/11 (inference+pairs), 2 (split verification). No gaps.

**Deviations from spec (deliberate, all documented in-plan):**
1. Sampling is probe-derived (Task 1), not the spec's model-card table — per the fidelity decision.
2. Calibration reuses the real reward-path client (Task 14) instead of `benchmark_judge_throughput.py`, whose payload differs — improves fidelity of the ETA.
3. `--reasoning-parser qwen3` (not `deepseek_r1`) per the user's fidelity call; requires re-validating the existing `_extract_chat_content` patch.
4. Judge payload gains env-gated sampling/`chat_template_kwargs` (Task 3) — unavoidable; the spec's "unmodified reward path" is not achievable for thinking-off + per-cell sampling on self-hosted vLLM.

**Placeholder scan:** none — every code step shows code; cluster steps give exact commands + gates. Two explicitly-flagged confirmations remain (Task 2 metadata key names; Task 12 round-robin threading) because they depend on runtime schema/signatures not verifiable from the Mac.

**Type consistency:** `build_pairs → (df, meta)`; `cell_list`/`tp_for_size` names match across Tasks 13/15; `_build_reward_dump_row` key set matches the viewer contract and the Task 17 metric readers (`rating_gt_first`, `judge_finish_reason`, `randomized_order`, `judge_response`).
