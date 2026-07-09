# Judge-Model Comparison Experiment: Design

Date: 2026-07-08
Repo: `/storage/home/lancewicki/projects/turing-rl`
Paper: "Learning User Simulators with Turing Rewards" (arXiv:2606.19336)

## Guiding Principle

**Maximum fidelity to the paper's SFT checkpoint.** Where our repo already contains upstream Turing-RL code, we use it as-is. Where we've patched it (documented in `our_patches.md`), we revert to upstream for the SFT path unless the patch is strictly necessary. Any deviation from paper procedure gets called out as a caveat, not silently absorbed.

---

## 1. Goal, Anchor, and Scope

### Goal
Measure how well smaller Qwen judges approximate the paper's self-hosted training reward judge (Qwen3.5-397B-A17B-GPTQ-Int4) on the pairwise Turing task, across thinking-on and thinking-off modes, using the paper's PRISM heldout split.

### Anchor
**Qwen3.5-397B-A17B-GPTQ-Int4**, served locally via `/storage/home/lancewicki/projects/turing-rl/scripts/slurm/judge_serve.sh` (TP=8, port 8000). This is what our GRPO training uses as reward.

**Caveat**: the paper uses un-quantized Qwen3.5-397B-A17B; we serve GPTQ-Int4 because paper's un-quantized weights need 8× 80GB A100s and we have 40GB. The Int4 anchor is treated as "the paper's judge as we can actually run it," not a paper-perfect reproduction.

### Family Selection
Day-1 4B throughput smoke: Qwen3-4B vs Qwen3.5-4B, same prompt / same concurrency / same thinking-mode config. Winner picked on tokens/sec at parse-rate parity. Tiebreak: higher agreement with the 397B anchor on a 50-pair accuracy spot-check. If inconclusive, default to Qwen3.5 (matches anchor family and quantization scheme).

- **If Qwen3 wins** → sweep sizes: 4B / 8B / 14B / 32B (all dense).
- **If Qwen3.5 wins** → sweep sizes: 4B (dense) / 9B (dense) / 27B (dense) / 35B-A3B-GPTQ-Int4 (MoE).

The Qwen3.5-397B-A17B-Int4 anchor stays fixed regardless.

### Sampling (Frozen)
Model-card defaults, thinking-mode-dependent:
- **Thinking-on**: `T=0.6, top_p=0.95, top_k=20, min_p=0`
- **Thinking-off**: `T=0.7, top_p=0.8, top_k=20, min_p=0`
- **`max_completion_tokens=8192`** for both.

The 397B anchor gets the same treatment. Previously it ran with vLLM defaults; the sweep run re-baselines it under model-card defaults so all comparisons are apples-to-apples.

### JSON schema mode (locked)
**`PERSONA_JUDGE_JSON_SCHEMA=1` for every cell.** This is our patch, documented in `our_patches.md`. It forces the judge's JSON output to contain a `rating` field before the model can end its turn.

Rationale: this is how a small judge would actually be deployed for training-time reward use. The `rating` format-correctness metric becomes ~100% *when the model has budget to reach the rating field* — but budget-exhausted responses (`finish_reason=="length"`) can still truncate mid-JSON before `rating` is emitted, dropping presence below 100%. Full-rubric completeness (per-field presence for the other 27 fields) still varies and remains the meaningful signal. Budget hit-rate is reported separately so we can see when rating-presence-drop is caused by truncation.

### Non-goals
- No Sonnet API calls (paper uses Sonnet 4.6 for evaluation, but we're anchoring on the training-time reward judge).
- No comparison to Sim/Logprob rewards.
- No GRPO retraining with a smaller judge.
- No LoRA-tuning or distillation of judges.
- No trainable-judge feasibility columns in the results table.
- No tuning of judge sampling parameters beyond the model-card defaults.
- No modifications to `TURING_PROMPT` or the rubric schema.

### Deliverables
1. **Results table** indexed by (judge_size × thinking_mode) with all metrics from Section 5.
2. **Family-decision doc** recording the 4B smoke result and the chosen family.
3. **Reusable SFT'd generator checkpoint** trained on the paper's 464-user SFT split.
4. **Heldout inference artifact**: 880 pairs of `(context, human_reply, generated_reply, metadata)` with `<reasoning>…</reasoning>` stripped from generator output.
5. **PRISM-split verification report** confirming Table-3 semantics beyond just counts.

---

## 2. PRISM Split Verification

### Motivation
Existing `/storage/home/lancewicki/projects/turing-rl/tests/test_prism_split_determinism.py` proves counts match paper Table 3 and two runs are byte-identical, but doesn't verify contents.

### Verification suite
New file `tests/test_prism_split_verification.py`, standalone (no slurm), runs against `data/prism/full_s42_history_sft40_grpo60_test10/`:

1. **Files exist**: `sft/train.parquet`, `grpo/train.parquet`, `grpo/val.parquet`, `test.parquet`, `split_metadata.json`.
2. **Row counts match metadata** for each parquet.
3. **User counts match metadata** (nunique on `extra_info.user_id`).
4. **User disjointness at row level** — `set(sft_users) ∩ set(grpo_train_users) == ∅`, same for `grpo_train ∩ heldout` and `sft ∩ heldout`. If metadata says 0/0/0 but parquets have overlap, we surface that.
5. **Prompt schema well-formed** — for a stratified sample of 50 rows per parquet, assert `row["prompt"]` is a list of `{role, content}` messages, `row["reward_model"]["ground_truth"]` is a non-empty string, `row["extra_info"]` has expected keys.
6. **Heldout `ground_truth` matches PRISM raw text** — for a sample of 20 heldout rows, look up (user_id, thread, turn_idx) in raw HuggingFace `HannahRoseKirk/prism-alignment` and assert equality.
7. **No cross-contamination via prompt content** — for 20 heldout rows, no substring of an SFT user's training-time target appears verbatim in the heldout prompt/ground_truth.

### Additional check: hash compare against fresh re-split
Re-run upstream `data/prism/split_data.py` on the existing PRISM raw cache (no re-download, no rebuild). SHA-256 each output parquet and compare against the current split parquets. If hashes match, no tampering happened between the split-agent's run and now. If not, we surface the diff and rebuild.

We skip a fresh `data/prism/build.py` re-run: it's the expensive step (network + processing), and the determinism test already covers "the pipeline is deterministic." What we're catching here is post-hoc parquet tampering, which the split re-run alone suffices for.

### Failure semantics
Any assertion failure fails the whole test with a diagnostic including offending row indices. If tests 1–5 fail, stop and rebuild. If 6–7 fail on a sample, escalate to full-scan.

### Runtime
~2 minutes for the 7-check suite plus ~5 minutes for the re-split + hash compare. Adds one test file (~150 lines), one PRISM-loader helper (~50 lines), no new dependencies.

---

## 3. Data Foundation

### Sources
- **PRISM raw**: `/home/lancewicki/data/hf_cache/datasets--HannahRoseKirk--prism-alignment` (cached).
- **Split parquets**: `/storage/home/lancewicki/projects/turing-rl/data/prism/full_s42_history_sft40_grpo60_test10/`, produced by upstream-unmodified `data/prism/build.py` + `data/prism/split_data.py`.

### Verification
Per Section 2: the 7-check suite plus a re-run of `data/prism/split_data.py` against the cached PRISM raw data with SHA-256 hash compare against current parquets.

### Row schema
Each row (verified against `data/sft/build_sft_jsonl.py:23`):
- `prompt`: list of `{role, content}` chat messages (paper Figure 10 template).
- `reward_model.ground_truth`: raw PRISM `[HUMAN]` reply string.
- `extra_info`: dict with `user_id`, `post_id`, `target_idx`, `user_history`, `context`, `persona`, `prompt_mode`.
- `data_source`: `"prism_alignment_user_sim"`.

### Judge-side use
`TURING_PROMPT` at `/storage/home/lancewicki/projects/turing-rl/shared/judge_prompts.py:337` takes `user_history`, `context`, `response_a`, `response_b`, `source_copy_watchlist` — sourced from heldout row's `extra_info` plus two candidate responses.

### Pair construction
For each of 880 heldout rows:
- `human = row["reward_model"]["ground_truth"]`
- `generated = strip_reasoning_envelope(sft_generator(row["prompt"]))`
- Metadata: `user_id`, `post_id`, `target_idx`, `user_history`, `context`, `persona`.

### Position bias
Every pair judged in both orderings:
- Ordering 1: `(A=human, B=generated)` → judge picks human when rating ≤ 3.
- Ordering 2: `(A=generated, B=human)` → judge picks human when rating ≥ 5.

**1760 judge calls per (judge_size × thinking_mode) cell.**

### Sample-size sanity
880 pairs → 80% accuracy Wilson 95% CI is ±2.6pp. Kappa agreement CIs ±0.05.

**Generator-sampling caveat**: the CI above bounds judge-side sampling noise only. Every accuracy figure is measured against **one stochastic draw** of the generator (Step 4.4 uses 1 sample/row at T=0.6). A different generator seed would produce a different set of 880 responses and different judge accuracies. Practical implication: judge-vs-judge gaps <5pp should be treated as within generator sampling noise, not real quality differences. Multi-sampling the generator (k>1) was considered and dropped — it would ~3× the sweep cost for an incremental honesty improvement. This caveat is stated inline in `summary.md`.

---

## 4. Generator Pipeline

### Step 4.1 — Full CoT generation (served 8-replica)

**Server config** (8× A100 40GB, single node):
- 8 vLLM server processes on the same node, one per GPU, each TP=1, ports 8000–8007.
- Model: `Qwen/Qwen3-8B` (already cached at `/home/lancewicki/data/hf_cache/models--Qwen--Qwen3-8B`).
- **No `--reasoning-parser qwen3`** — dropped since we're explicit thinking-off.
- Same "N replicas on one node + round-robin client" pattern used by the judge sweep (Section 5.1), so the launcher and client code are shared between CoT gen and the sweep.

**Client config** (existing `data/sft/generate_cot.py`, HTTP path):
- `extra_body={"chat_template_kwargs": {"enable_thinking": False}}` forces thinking-off at chat-template level.
- Drop `reasoning=True` from `build_chat_payload`.
- Sampling: `T=0.7, top_p=0.8, top_k=20, min_p=0` (Qwen3 thinking-off model-card defaults).
- `max_completion_tokens=4096` (matches `DEFAULT_MAX_COMPLETION_TOKENS` at `generate_cot.py:22`).
- Round-robins requests across the 8 endpoints; concurrency 16 per endpoint.

**Pipeline**: (a) generate all 3272 via the HTTP client; (b) leakage check via `reasoning_leaks_reply` at `data/sft/generate_cot.py:60`; (c) re-generate leaked subset up to 10 tries.

**Wall time**: ~15-20 min.
**Output**: `data/sft/prism_full_s42_sft_cot.parquet` (3272 rows), plus `.cot_metadata.json`.
**Discard the smoke's 138 rows.**

### Deviations & ambiguities block (paper-vs-code CoT thinking mode)

**Paper** (line 1313 of the arXiv PDF): "we run Qwen3-8B with thinking mode disabled" for CoT generation.

**Repo default** (`data/sft/generate_cot.py:17`): `DEFAULT_COT_MODEL = "qwen/qwen3-8b"` (OpenRouter slug); line 105 calls `build_chat_payload(..., reasoning=True)`.

**Smoke that produced `data/sft/qwen3-8b_prism_smoke_sft_cot.parquet`** (138 rows): re-implemented self-hosted vLLM with `--reasoning-parser qwen3` and `reasoning=True` on the wire. Server probe at `/storage/home/lancewicki/projects/turing-rl/logs/cot_generate_smoke-8739.out` confirms Qwen3-8B was in thinking mode ON — `.message.reasoning` contained genuine first-person thinking, `.message.content` had the third-person rationalization that got stored. The `.reasoning` field was silently discarded by `_extract_chat_content`.

**Inspection of the 138-row parquet**: 0/138 rows contain `<think>`; 136/138 start with "The user…" (third-person); reasoning length 448–1005 chars (avg 710). The stored text is thinking-off-style regardless of whether the server was internally in thinking mode.

**Our resolution**: follow the paper's stated directive. Full 3272-row run uses explicit thinking-off via `chat_template_kwargs.enable_thinking=False`, no `--reasoning-parser qwen3` on the server. Sampling from Qwen3 model card thinking-off defaults.

**Why**: the fidelity principle prefers eliminating the ambiguity entirely over a code-inherited default whose semantics we can't audit at OpenRouter's provider layer. The observed 138-row stored content already looks thinking-off-style, so this is the least-divergent option under uncertainty.

### Step 4.1a — (dropped)

Previous drafts of this spec proposed a batched-offline vLLM path alongside the served path, with a 20-row "batched-vs-served parity test" as a gate. Since Step 4.1 now uses served exclusively (matching the sweep's serving pattern), the parity test is no longer needed and this step is dropped.

### Step 4.2 — Build SFT JSONL
`data/sft/build_sft_jsonl.py`, upstream-untouched. Output: `data/sft/prism_full_s42_sft_cot.jsonl` (chat-formatted with `<reasoning>t</reasoning>\n[HUMAN]: y⋆` in assistant target).

### Step 4.3 — LoRA SFT training

**Config** — `training/sft/configs/qwen3_8b_lora.yaml`, unchanged from paper Table 5:
- LoRA r=64, α=128, dropout=0.05
- LR 2e-4 cosine, warmup 0.05, weight decay 0.01
- 3 epochs, per-device BS=1, grad_accum=16, 8 GPUs → effective BS=128
- `max_seq_length=8192` (paper Table 5; smoke validated at this value)

**Base model**: Qwen3-8B (thinking-disabled per paper, matches CoT model).
**Wall time**: paper says ~12h; realistic on 8× A100 40GB is 12-16h.
**Wandb**: enabled via existing patch trick in `our_patches.md`.
**Output**: `checkpoints/sft/qwen3_8b_prism_full_s42/`.

**Checkpointing patch (NEW in our_patches.md)**:
- Add `--resume_from_checkpoint auto` CLI arg to `training/sft/lora_sft.py`.
  - If set to `auto`, glob `${output_dir}/checkpoint-*`, pick highest-step directory, pass to `trainer.train(resume_from_checkpoint=path)`.
  - If glob empty, start fresh.
  - ~20 lines patched, documented as PERSISTENT in `our_patches.md`.
- **Override upstream `save_strategy="epoch"`** with `save_strategy="steps", save_steps=10, save_total_limit=2` in the yaml. At ~26 steps/epoch (3272 rows / effective BS 128, 3 epochs → ~78 steps total), this yields ~8 checkpoints spaced ~110 min apart, bounded to 2 on disk (~32GB). Rationale: `save_strategy` is not a load-bearing hyperparameter — model weights, gradients, and optimizer state are byte-identical regardless of save frequency; the only cost is checkpoint I/O (~30s each, ~4 min total). Max wall-loss on failure drops from ~5h (epoch) to ~110 min.
- New sbatch `scripts/slurm/sft_full.sh` always passes `--resume_from_checkpoint auto`.

### Step 4.4 — Heldout inference

`eval/generate_trained.py`, upstream-untouched.
- Input: `data/prism/full_s42_history_sft40_grpo60_test10/test.parquet` (880 rows / 128 heldout users).
- **LoRA adapter from Step 4.3** — this is the artifact; it carries everything SFT learned.
- Base weights: Qwen3-8B (must match the base used during Step 4.3's SFT — the adapter's low-rank deltas are calibrated to these specific weights). Loaded via vLLM `enable_lora=True`; adapter attached per-request with `LoRARequest`.
- Sampling (already the code default at `eval/generate_trained.py:44-46`): `T=0.6, top_p=1.0, top_k=-1, min_p=None, pres_pen=0.5, max_tokens=2048` — matches paper Table 4 for PRISM.
- 1 sample per row → 880 generations.
- vLLM backend, TP=1 on 1 GPU, ~1 GPU-hour.
- Output: pickle/parquet with per-row `{prompt_text, response, reasoning, parsed_response, extra_info}`. Uses `parse_reasoning_and_response` at `/storage/home/lancewicki/projects/turing-rl/shared/prompt_utils.py` to split `<reasoning>...</reasoning>` from `[HUMAN]: reply`.

### Step 4.5 — Pair-set construction

New thin script `scripts/build_judge_pairs.py`:
- Input: heldout inference output + `test.parquet`.
- For each of 880 rows: `human = row.reward_model.ground_truth`; `generated = parsed_response.reply` (post-`[HUMAN]:` text; reasoning envelope stripped).
- Sanity checks:
  - No residual `<reasoning>` tag in `generated`.
  - Every row present.
  - **Count-and-log** `human == generated` rows (do not assert). Exact matches on very short replies ("yes", "lol", "same") are possible and not necessarily degenerate. Flag investigation if the count exceeds 1% of rows; otherwise record in metadata and continue.
- Output: `results/2026-07-08-judge-sweep/raw/pairs/prism_heldout_880.parquet` with columns `pair_id, user_id, post_id, target_idx, user_history, context, persona, human, generated`.
- **Frozen artifact** all 5 judges × 2 thinking modes consume.

### Deliverables from Section 4
- Full 3272-row CoT parquet.
- LoRA adapter checkpoint (`.safetensors`).
- Heldout inference pickle (880 rows).
- Pair-set parquet (880 rows).

---

## 5. Judge Sweep

### Fixed inputs to every judge run
- 880-row pair-set from Section 4.5.
- Turing prompt: `TURING_PROMPT` at `/storage/home/lancewicki/projects/turing-rl/shared/judge_prompts.py:337`, unmodified.
- Judge call path: `_score_pairwise_likert_with_info` at `/storage/home/lancewicki/projects/turing-rl/training/grpo/reward.py:491`, unmodified.
- Response format: `PERSONA_JUDGE_JSON_SCHEMA=1` (grammar-forced rating).
- `max_completion_tokens=8192`.
- Dumps: `PERSONA_JUDGE_DUMP_RATE=1.0` for every sweep run. Two dump types land per cell:
  - HTTP dumps: automatic via `shared/api_client.py:post_chat_async` → `results/2026-07-08-judge-sweep/raw/sweep/{judge}/{thinking_mode}/http/judge-<slurm>-<pid>.jsonl` (wire-level payload + response).
  - Reward-layer dumps: manually emitted by the sweep client after each `_score_pairwise_likert_with_info` call, using upstream helpers `_build_reward_dump_row` + `_dump_reward_call` from `training/grpo/reward.py`. Land at `results/2026-07-08-judge-sweep/raw/sweep/{judge}/{thinking_mode}/reward/reward-<slurm>-<pid>.jsonl`. Row shape matches what `compute_score` produces in GRPO training, so the existing GUI viewer at `scripts/dump_viewer.py` renders sweep dumps without modification (Human A/B badge, ground-truth vs generator tabs, parsed rubric, penalty breakdown).

### Sweep matrix
**5 sizes (4 smaller judges + 1 anchor) × 2 thinking modes × 1760 calls = 17,600 total judge calls.** 10 cells total.

Thinking-mode config:
- **Thinking-on**: `T=0.6, top_p=0.95, top_k=20, min_p=0`, `enable_thinking=True`, `--reasoning-parser qwen3` on the vLLM server.
- **Thinking-off**: `T=0.7, top_p=0.8, top_k=20, min_p=0`, `enable_thinking=False`, no reasoning-parser.

### Step 5.0 — Per-cell throughput calibration (NEW, upfront)

Before committing to full sweep:
- For every (size, thinking-mode) pair — 10 cells total.
- Each calibration: 50 pairs (100 judge calls with both orderings), same right-sized TP + replica serving config we plan to use for the full cell, real prompts sampled from the 880-pair set, concurrency=16 per endpoint.
- Reuses `scripts/benchmark_judge_throughput.py` with the pair-set path.
- Output: `results/2026-07-08-judge-sweep/raw/calibration/calibration_results.jsonl` (per-call rows) + `calibration_metadata.json` (per-cell serving config + measured req/s). Derived report at `derived/calibration_report.md` — per-cell measured req/s + p50/p95 latency + extrapolated full-cell wall time (1760 calls).
- **Precision caveat**: at 100 calls per cell (~6-7 waves at concurrency 16, with warmup noise in the first 2-3), extrapolated wall-time projections are rough (±30%). Sufficient for the >4h gate decision below, but do not treat as a precise ETA. This caveat is stated inline in `calibration_report.md`.
- Wall time: **~1h upfront** (10 cells × ~5 min each, parallel across nodes).
- **Gate**: if a cell's extrapolated wall time is >4h (i.e., <0.12 req/s), surface it and decide whether to reduce concurrency, drop the cell, or accept it before launching the real 1760-call cell.

### Step 5.1 — Full sweep serving plan

**Node 1** — Qwen3.5-397B-A17B-Int4 anchor, TP=8 (fills the node — no choice). 2 cells sequential (thinking-on, thinking-off).

**Nodes 2 & 3** — smaller judges. **One cell per node**, but **each cell fills the node with replicas of the same model at right-sized TP**:

| Model class | Right-sized TP | Replicas/node |
|---|---|---|
| 4B / 8B / 9B / 14B / 35B-A3B-Int4 | TP=1 | 8 |
| 27B / 32B | TP=2 | 4 |

**Scheduling**: launch each smaller-judge cell as an independent sbatch (8 cells: 4 sizes × 2 modes). Slurm schedules across nodes 2 & 3 in parallel.

**Client**: `scripts/run_judge_sweep_cell.py` round-robins pair requests across the endpoint URLs for that cell. Uses the existing async client pattern in `benchmark_judge_throughput.py`.

**Client concurrency**: 16 per endpoint. 8-replica cells drive 128 concurrent in-flight requests; 4-replica cells drive 64.

**Wall time end-to-end**: per calibration, realistically 3-8h dominated by 32B and 397B cells.

### Step 5.2 — Bonus offline batched cell

Qwen3-8B thinking-off, TP=8, in-process `LLM.generate` on all 1760 pairs at once. Post-primary sweep on whichever node opens up.
- Compares to the corresponding server-mode cell.
- Same pair-set, same sampling, same prompt.
- Accuracy should be identical (same model + sampling + prompt); throughput should be higher.
- Quantifies the server-vs-offline gap for downstream "training-time judge selection" question.
- **Emits reward-layer dumps in the same shape as the server cells** (call `_build_reward_dump_row` per generation and write to `raw/sweep/qwen3-8b/off_offline/reward/`) so the GUI viewer works on it identically.

### Step 5.3 — GUI dump-viewer workflow

The existing `scripts/dump_viewer.py` (FastAPI + Jinja) auto-detects reward vs HTTP dump schemas and recurses into subdirs. Sweep produces both. To browse a cell:

```bash
# On login pod:
/home/lancewicki/miniconda3/envs/turing-rl-train/bin/python \
    /storage/home/lancewicki/projects/turing-rl/scripts/dump_viewer.py \
    --dumps /storage/home/lancewicki/projects/turing-rl/results/2026-07-08-judge-sweep/raw/sweep/qwen3-8b/off \
    --port 8082

# From Mac: ssh -L 8082:localhost:8082 <cluster-host>, then http://localhost:8082/
```

The viewer renders per-row: Human A/B badge, final rating, tabs for context / history / response / ground_truth / prompt / raw / reasoning / judge (parsed rubric) / metadata. HTTP tab set (prompt / response / parsed / reasoning / metadata) for wire-debugging rows.

**Cross-cell comparison workflow**: the current viewer has no per-cell filter chip, so browse one cell at a time by pointing `--dumps` at `raw/sweep/{judge}/{thinking_mode}/`. Cross-cell comparison happens in `summary.md`, not in the viewer.

**No modifications to `dump_viewer.py`** — the emitted rows match the shape it already consumes.

### Metrics (per judge, per thinking mode)

Computed offline from dump JSONLs by `scripts/analyze_judge_sweep.py`. Reproducible from `raw/` alone.

**Format & budget metrics**:
- **Format correctness (rating)**: fraction of pairs with valid JSON and a rating field. Grammar-forced to ~100% when the model completes within budget; falls below 100% when `finish_reason=="length"` truncates mid-JSON before `rating` is emitted. Reported alongside budget-hit-rate so we can distinguish "model can't produce rating" from "model over-thinks and runs out of tokens."
- **Rating recovery rate (via fallback parser)**: fraction where `_extract_turing_rating` at `shared/judge_utils.py:652` recovers a rating from malformed/truncated text. Measures the safety net's effectiveness for the training-loop deployment scenario.
- **Per-field presence probability**: for each of ~28 rubric fields, fraction of responses that include it (this is the meaningful format-correctness signal under the schema patch).
- **Response length**: distribution of `judge_raw_content` lengths (chars + tokens if `usage.completion_tokens` populated).
- **Budget hit-rate**: fraction of responses with `finish_reason == "length"`.

**Accuracy metrics**:
- **Accuracy**: fraction of pairs where the judge picks the human. Rating < 4 → picks A; rating > 4 → picks B; rating == 4 → tie (excluded from accuracy denominator, counted in abstain rate).
- **Accuracy | format-correct**: computed only over pairs where both orderings parsed successfully.
- **Position bias Δ**: `|accuracy_A_first − accuracy_B_first|` at pair level; also per-rating-distribution mean shift.

**Alignment metrics**:
- **Judge ⇔ 397B kappa agreement**: Cohen's kappa on binary human-picking decision, judge vs 397B anchor.
- **Rating distribution per judge**: histogram of 1–7 ratings + summary stats (mean, mode, entropy). Reveals judges collapsing to 4/4.
- **Per-confidence-bucket accuracy**: for pairs where 397B rated {1,7} (confident) vs {2,3,5,6} (moderate) vs {4} (abstain), report smaller-judge accuracy separately.

**Ops metrics** (from throughput calibration):
- **Wall-clock throughput** (req/s at concurrency=16 per endpoint).
- **Latency p50 / p95** per call.
- **GPU-time per pair**: throughput ÷ GPU count.

### Output artifacts

All results for this experiment live under **`results/2026-07-08-judge-sweep/`**, structured as `raw/` (immutable — captured once, never regenerated) and `derived/` (computed from `raw/`; safe to delete and recompute).

```
results/2026-07-08-judge-sweep/
├── raw/
│   ├── pairs/
│   │   └── prism_heldout_880.parquet         # 880-row pair-set from Step 4.5 (frozen input)
│   ├── generator/
│   │   ├── heldout_inference.pkl             # per-row {prompt_text, response, reasoning, parsed_response, extra_info}
│   │   └── heldout_inference_metadata.json   # adapter path, base model, sampling params, wall time
│   ├── sweep/
│   │   └── {judge}/{thinking_mode}/
│   │       ├── reward/reward-<slurm>-<pid>.jsonl   # 1760 rows: full rubric + wire metadata
│   │       ├── http/judge-<slurm>-<pid>.jsonl      # HTTP wire dump (redundant safety net)
│   │       └── run_metadata.json             # cell config: judge id, thinking mode, sampling, server args, slurm job id, start/end timestamps, endpoint URLs, concurrency
│   ├── calibration/
│   │   ├── calibration_results.jsonl         # per-call rows from Step 5.0 (50-pair micro-cell each)
│   │   └── calibration_metadata.json         # per-cell serving config + measured req/s
│   ├── family_smoke/
│   │   └── qwen3_vs_qwen35_4b/               # Step 6 raw benchmark output
│   │       ├── results.jsonl
│   │       └── metadata.json
│   └── logs/
│       ├── slurm/*.out                       # copies of all slurm stdout for judge/CoT/SFT jobs
│       ├── vllm_server/{judge}/{thinking_mode}/server-<pid>.log   # per-endpoint vLLM startup + serving logs
│       └── sft/                              # SFT training logs (wandb export + slurm out)
├── derived/
│   ├── summary.md                            # 10 rows (5 sizes × 2 thinking modes), all metrics as columns
│   ├── summary.parquet                       # same data, machine-readable for downstream plots
│   ├── per_pair_metrics.parquet              # per (pair_id, judge, thinking_mode) rows: rating, format_ok, budget_hit, etc.
│   ├── plots/
│   │   ├── accuracy_vs_size.png              # thinking-on/off overlaid; MoE plotted at active-param with a note
│   │   ├── kappa_vs_size.png
│   │   ├── per_field_presence_vs_size.png
│   │   ├── budget_hit_vs_size.png
│   │   ├── throughput_vs_size.png
│   │   └── rating_distribution_{judge}_{thinking_mode}.png    # one per cell
│   ├── family_decision.md                    # writeup of Step 6 result
│   ├── calibration_report.md                 # per-cell measured req/s + p50/p95 + extrapolated wall time
│   ├── split_verification.md                 # Section 2 report
│   └── offline_vs_server_comparison.md       # Step 5.2 bonus cell writeup
└── README.md                                 # one-page tour of what's here and how to regenerate derived/
```

**Design principles**:

1. **Raw is immutable.** Once a `reward/*.jsonl` dump lands, it's never touched. All metric changes happen in `derived/` by re-running the analyzer.
2. **Everything in `derived/` is regeneratable.** `scripts/analyze_judge_sweep.py` takes `raw/` as input and produces the entire `derived/` tree. If we want a new metric next month, we add it to the analyzer, delete `derived/`, and re-run — no judge calls repeated.
3. **Raw rows contain enough to re-derive anything reasonable.** Every reward-dump row contains: full judge prompt, raw content, parsed reasoning, wire metadata, latency, finish_reason, usage tokens, per-ordering ratings, ground truth, generated response, context, user history. Any future metric we might want — e.g., correlation of rating with response length, or per-user accuracy — should be computable without new inference.
4. **`run_metadata.json` per cell** captures the exact serving config used, so a cell can be exactly reproduced later if needed. Includes: judge id, sampling params, `PERSONA_JUDGE_JSON_SCHEMA` state, endpoint replicas, slurm job id, timestamps.
5. **Logs kept as-is** in `raw/logs/` alongside the parquet/jsonl dumps. If a cell later looks anomalous, the vLLM server log and slurm stdout are right there.

**Design-doc Results section**: after execution, appended inline to this spec file. Points into `results/2026-07-08-judge-sweep/derived/` for tables and plots.

**GUI viewer compatibility**: `scripts/dump_viewer.py` renders `raw/sweep/{judge}/{thinking_mode}/` directly. Both reward-layer and HTTP dumps produced by the sweep client match the shape the viewer already consumes. See Step 5.3 for the workflow.

---

## 6. Analysis, Ordering, and Risks

### 6.1 Analysis pipeline
`scripts/analyze_judge_sweep.py` consumes `results/2026-07-08-judge-sweep/raw/` and produces the entire `derived/` tree. Reproducible from `raw/` alone — no judge calls repeated. Idempotent; safe to delete `derived/` and re-run at any time.

### 6.2 Ordering (max-parallel across 3 nodes)

Step-by-step table:

| # | Step | GPUs | Wall time | Blocks | Notes |
|---|---|---|---|---|---|
| 1 | PRISM split verification (7-check suite + re-split hash compare) | 0 | 10 min | 3 | Runs on login pod |
| 2 | _(dropped — batched-vs-served parity test no longer needed; Step 4.1 uses served exclusively)_ | — | — | — | — |
| 3 | Full CoT served generation (3272 rows, 8-replica served, thinking-off) | 8 | 15-20 min | 4 | + leakage-guard regen pass |
| 4 | Build SFT JSONL | 0 | 2 min | 5 | |
| 5 | **SFT LoRA training** (3 epochs, effective BS=128, `max_seq_length=8192`) | 8 | **12-16h** | 11 | Paper Table 5 config; the long pole. Auto-resume patched, `save_steps=10 save_total_limit=2` (~110 min max loss on failure). |
| 6 | Qwen3 vs Qwen3.5 4B family-selection smoke | 1-2 | 30 min | 7, 8, 13 | Parallel with SFT |
| 7 | Step 5.0 per-cell throughput calibration (10 cells, 50 pairs each) | up to 24 | 1h | 13 | Parallel with SFT |
| 8 | Family-decision doc written | 0 | 10 min | 13 | |
| 9 | Judge-sweep tooling (pair builder, run-cell client, analyzer, smoke on synthetic 10-pair) | 0 | 4-8h | 12 | Parallel with SFT |
| 10 | Judge servers spin-up (warmup) | 24 | 5-10 min | 13 | Buffered inside sweep sbatch |
| 11 | Heldout inference (880 rows, 1 sample each, paper Table 4 sampling) | 1 | 1h | 12 | Sequential after step 5 |
| 12 | Pair-set construction + sanity checks | 0 | 5 min | 13 | |
| 13 | **Full judge sweep** (10 cells, 1760 calls each, server-mode, right-sized TP + replicas) | up to 24 | **3-8h** | 14 | Dominated by 32B + 397B cells |
| 14 | Bonus offline batched cell (Qwen3-8B thinking-off, TP=8, in-process LLM.generate) | 8 | 30 min - 1h | 15 | Post primary sweep |
| 15 | Analysis pipeline run: aggregate metrics, plots, summary.md | 0 | 2-4h | 16 | May iterate |
| 16 | Design-doc Results section appended + family-decision finalized + report writeup | 0 | 1-2h | — | |

**Critical path**: 1 → 3 → 4 → 5 → 11 → 12 → 13 → 15 → 16 ≈ **20-33h of active wall time** (SFT + sweep dominate).

Non-critical work (steps 6, 7, 8, 9, 10, 14) fits in parallel windows during steps 5 and 13.

**Realistic end-to-end**: **2-3 days nominal.** The critical-path arithmetic (~20-33h) only holds with zero idle time between handoffs — no Slurm queue waits, no rerun of any step, active supervision at every transition. Realistically, expect:
- Slurm queue latency between jobs (minutes to hours per handoff).
- Overnight gaps between steps that need human decisions (family-decision doc, cell-drop gate, analysis iteration).
- One SFT restart (~110 min lost, per the checkpointing patch) is likely enough that we should plan for it.

A 1-day best case is only achievable with continuous supervision; more common is 2-3 days including a full night's sleep or two.

### 6.3 Risks and mitigations

- **R1 — Judge parse failures at high rate on small judges (even with schema patch)**. With `PERSONA_JUDGE_JSON_SCHEMA=1`, `rating` is grammar-forced *when the model has budget to reach it*. Failure modes:
  - Model emits `{"rating": 4}` and stops — accuracy valid, per-field presence for other 27 fields drops to near zero. Both metrics report honestly.
  - Model runs out of tokens before emitting `rating` (long `<think>` block + long verbose keys eating the 8192 budget) — `finish_reason=="length"`, JSON truncated, no rating. Rating-presence drops below 100%. Fallback parser `_extract_turing_rating` recovers some cases from raw text.
  - The interaction of {format correctness, rating recovery rate, budget hit-rate, per-field presence} tells the story per cell. No mitigation needed — the analysis distinguishes them.

- **R2 — 397B anchor throughput dips under real Turing prompts vs the smoke measurement**. Calibration in Step 5.0 will surface this. If a 397B cell projects to >2h, drop `max_completion_tokens` for the anchor to 4096 and note as Deviation. This is our patch territory anyway (the anchor already deviates from paper via GPTQ-Int4).

- **R3 — Server-vs-batched offline throughput gap is huge and changes the story**. Flag in the report; bonus offline cell (Step 5.2) shows the raw ratio. Downstream question not addressed in this project.

- **R4 — Split verification fails**. The 7-check suite plus `split_data.py` re-run + hash compare catches post-hoc parquet tampering. If it fails: re-run `data/prism/split_data.py` from the current codebase against the cached PRISM raw data, replace the tampered parquets, redo any downstream artifact that consumed them.

- **R5 — (dropped)** — The batched-vs-served parity risk no longer applies since Step 4.1 uses served exclusively.

- **R6 — Qwen3 vs Qwen3.5 4B smoke inconclusive**. Default to Qwen3.5 (matches anchor family + quantization scheme).

### 6.4 Guardrails (explicitly NOT doing)

- Not tuning judge sampling.
- Not touching `TURING_PROMPT` or the rubric schema.
- Not retraining the 397B anchor or fine-tuning any judge.
- Not comparing to Sim/Logprob rewards.
- Not calling Sonnet API.
- Not running the sweep on the SFT or GRPO training splits — heldout only.
- Not measuring generator quality beyond what the judge sweep incidentally produces.

---

## Patches summary (updates to `our_patches.md`)

New patches introduced by this project:
1. **`training/sft/lora_sft.py`** — add `--resume_from_checkpoint auto` CLI arg. ~20 lines. PERSISTENT.
2. **`training/sft/configs/qwen3_8b_lora.yaml`** — override `save_strategy` from `epoch` to `steps` with `save_steps=10, save_total_limit=2`. PERSISTENT. Yaml-only, no code change. Not a semantic model change — reduces max failure loss from ~5h to ~110 min at ~4 min total I/O cost.

Existing patches leveraged (no new modifications):
- **`training/grpo/reward.py`** — `PERSONA_JUDGE_JSON_SCHEMA=1` env flag (already in `our_patches.md`).
- **`training/sft/configs/qwen3_8b_lora.yaml`** — `report_to: wandb` via sed-patch in launcher (already in `our_patches.md`).
- **`shared/api_client.py`** — judge response dump hook (already in `our_patches.md`).
- **`training/grpo/reward.py`** — semantic reward-layer dump (already in `our_patches.md`).

---

## Results

_(To be appended after execution.)_
