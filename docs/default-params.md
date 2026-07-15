# Default parameters — judge / generator / SFT

Stable defaults for the turing-rl pipeline, so future sessions/plans start from one
agreed baseline. Values below are the ones actually in use per the judge-sweep plan
(`docs/superpowers/plans/2026-07-08-judge-sweep-implementation.md`) and post-plans
(`docs/superpowers/post-plans/2026-07-08-judge-sweep/`). Where a config file is the
source of truth (SSOT), it's named — edit there, not here.

> **New default (2026-07-14):** judge sampling now uses **`repetition_penalty=1.1`**.
> The diagnostic runs showed thinking-on parse failures are runaway repetition loops
> (judge echoing a degenerate candidate until it hits the token cap); `repetition_penalty=1.1`
> cut 397B parse-error 0.11→0.03 and raised penalized accuracy 0.686→0.720. See
> `post-plans/2026-07-08-judge-sweep/2026-07-14-cot-failure-diagnostic.md`.

## Judge (reward model)
SSOT: `configs/judge_sweep_cells.py` (model matrix), `scripts/run_judge_sweep_cell.py`
(`cell_env`), `training/grpo/reward.py` (reward math), `scripts/slurm/judge_sweep_cell.sh` (serving).

**These defaults apply to ALL judges** (any Qwen3.5 size), not just the 397B anchor.
Only two things vary per judge — the **model_id** and the **serving shape** (TP/replicas,
quantization), chosen by memory footprint via `tp_for_size` in `configs/judge_sweep_cells.py`.
Sampling, reasoning parser, output schema, and reward math are **shared across judges**.

Per-judge serving (footprint-based): fits one 40GB GPU (≤30GB) → **TP=1, 8 replicas**;
else whole node → **TP=8, 1 replica**.

| Judge | model_id | serving |
|---|---|---|
| 397B anchor (training judge) | `Qwen/Qwen3.5-397B-A17B-GPTQ-Int4` | TP=8/1rep, Int4, env `judge-vllm` (hybrid-Mamba MoE) |
| 122B | `Qwen/Qwen3.5-122B-A10B-GPTQ-Int4` | TP=8/1rep, Int4 |
| 4B / 9B / 27B / 35B-A3B | `Qwen/Qwen3.5-{4B,9B,27B,35B-A3B}` | TP=1/8rep (bf16), env `turing-rl-train` |

Shared serving/sampling defaults (all judges):

| Param | Default | Notes |
|---|---|---|
| Serving common | `--dtype bfloat16`, `--max-model-len 32768`, `--gpu-memory-utilization 0.85`, `--disable-custom-all-reduce` (TP>1) | |
| Thinking mode | `on` → `--reasoning-parser qwen3`, `PERSONA_JUDGE_ENABLE_THINKING=1` | off = no reasoning parser. **`qwen3` is the correct parser for Qwen — NOT `deepseek_r1`** (see Flags) |
| **Sampling** | **`repetition_penalty=1.1`**; temperature/top_p = model `generation_config.json` defaults (~0.6) | inject via `PERSONA_JUDGE_SAMPLING='{"repetition_penalty":1.1}'` |
| Output schema | `PERSONA_JUDGE_JSON_SCHEMA=1` (strict json_schema; `rating` required) | |
| Max completion tokens | `PERSONA_JUDGE_MAX_COMPLETION_TOKENS=8192` | |
| Client timeout | `PERSONA_OPENAI_TIMEOUT_SECONDS=1800` (thinking-on 397B) | reward.py fallback is 400 |
| Retries | `PERSONA_OPENAI_MAX_RETRIES=3` | |
| Concurrency | judge sweep: 8 per endpoint; GRPO: `TURING_JUDGE_MAX_CONCURRENCY=4` | 40GB KV pressure |
| Reward math | clip judge score at **5.0**, `(clip−1)/6`, ×**0.9**; rating re-derived from 6 dims + penalties (mean×3) | `TURING_JUDGE_SCORE_CLIP_MAX`, `TURING_RAW_REWARD_SCALE` |

## Generator
SSOT: `training/sft/configs/qwen3_8b_lora.yaml` (base), `bash_scripts/grpo/train_grpo.sh` (GRPO, upstream = paper Table 6).

| Param | Default | Notes |
|---|---|---|
| Base model | `qwen3-8b` (SFT LoRA adapter → GRPO actor) | |
| **GRPO rollout temperature** | **1.0 (verl default, not overridden)** | training uses high temp for exploration; validation/eval rollout = 0 (greedy). verl `trainer/config/rollout/rollout.yaml` |
| Heldout-inference sampling | **T=0.6, 1 sample/pair** | eval only (not training); per judge-sweep `derived/README.txt` |

> **GRPO training hyperparameters (batch, `rollout.n`, lengths, epochs) follow upstream
> `bash_scripts/grpo/train_grpo.sh` = paper Table 6** (batch 128, G=4, 3 epochs,
> max-response 1024, LR 1e-5, KL β 1e-3, PPO clip 0.2). We'll confirm/adjust after our own
> full run. The old `train_grpo_smoke.sh` (138-row, 40GB-shrunk) is **deprecated/misleading
> — do not use as a reference.**

## SFT
SSOT: `training/sft/configs/qwen3_8b_lora.yaml`; launcher `scripts/slurm/sft_variant.sh` (torchrun, 8-GPU).

| Param | Default | Notes |
|---|---|---|
| Base model | `qwen3-8b`, `max_seq_length=8192`, packing=True | `NOPACK=1` disables packing |
| LoRA | r=**64**, alpha=**128**, dropout=0.05, **no QLoRA (bf16)** | paper Table 5 |
| Optim | epochs=3, batch_size=1, grad_accum=16, lr=2e-4, cosine, warmup 0.05, wd 0.01 | |
| Misc | gradient_checkpointing=true, save_steps=10, save_total_limit=2 | |
| Data | `data/sft/prism_full_s42_sft_cot.jsonl` (PRISM full CoT slice, ~3272 rows, seed 42) | |
| Launch variant | `bf16_fsdp` (full_shard, wrap `Qwen3DecoderLayer`) — verified on 40GB | also `qlora_r64`, `bf16_fa2` |

## Flags / things to fix
- **Judge parser bug — `deepseek_r1` is wrong for Qwen.** The correct parser is `qwen3`
  (source-verified in `scripts/slurm/judge_serve_8b.sh:20`; used by the judge sweep). But the
  **training-side judge servers still use `deepseek_r1`**: `scripts/slurm/judge_serve.sh:53`,
  `scripts/slurm/grpo_smoke.sh`, `scripts/slurm/cot_server.sh:53` (all also `--max-model-len 16384`).
  These should be reconciled to `qwen3` + `32768`. *(Not in `train_grpo_smoke.sh` itself — it
  inherits `JUDGE_MODEL` and the external judge server sets the parser.)*
- `repetition_penalty=1.1` overrides the Task-1 "no wire sampling override" policy for this one
  judge param (intended, per the cot-failure result).
