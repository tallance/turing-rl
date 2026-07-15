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

| Param | Default | Notes |
|---|---|---|
| Anchor / training judge | `Qwen/Qwen3.5-397B-A17B-GPTQ-Int4` | Int4 (bf16 397B won't fit 40GB); hybrid-Mamba MoE |
| Serving | TP=8, 1 replica, `--dtype bfloat16`, `--max-model-len 32768`, `--gpu-memory-utilization 0.85` | whole 8-GPU node; env `judge-vllm` |
| Thinking mode | `on` → `--reasoning-parser qwen3`, `PERSONA_JUDGE_ENABLE_THINKING=1` | off = no reasoning parser |
| **Sampling** | **`repetition_penalty=1.1`**; temperature/top_p = model `generation_config.json` defaults (~0.6) | inject via `PERSONA_JUDGE_SAMPLING='{"repetition_penalty":1.1}'` |
| Output schema | `PERSONA_JUDGE_JSON_SCHEMA=1` (strict json_schema; `rating` required) | |
| Max completion tokens | `PERSONA_JUDGE_MAX_COMPLETION_TOKENS=8192` | |
| Client timeout | `PERSONA_OPENAI_TIMEOUT_SECONDS=1800` (thinking-on 397B) | reward.py fallback is 400 |
| Retries | `PERSONA_OPENAI_MAX_RETRIES=3` | |
| Concurrency | judge sweep: 8 per endpoint; GRPO: `TURING_JUDGE_MAX_CONCURRENCY=4` | 40GB KV pressure |
| Reward math | clip judge score at **5.0**, `(clip−1)/6`, ×**0.9**; rating re-derived from 6 dims + penalties (mean×3) | `TURING_JUDGE_SCORE_CLIP_MAX`, `TURING_RAW_REWARD_SCALE` |

## Generator
SSOT: `training/sft/configs/qwen3_8b_lora.yaml` (base), `bash_scripts/grpo/train_grpo_smoke.sh` (GRPO).

| Param | Default | Notes |
|---|---|---|
| Base model | `qwen3-8b` (SFT LoRA adapter → GRPO actor) | |
| GRPO rollout group size | `rollout.n=2` | reduced from 4 for 40GB |
| GRPO batch | `train_batch_size=32`, `ppo_mini_batch_size=32`, `micro_batch=1` | smoke slice = 138 rows |
| GRPO lengths | `max_prompt_length=6144`, `rollout.max_model_len=7168`, `max_num_batched_tokens=8192` | 40GB safety |
| GRPO rollout mem | `gpu_memory_utilization=0.35`; `use_remove_padding=false` (no flash_attn) | |
| GRPO epochs | `total_epochs=1` | |
| Heldout-inference sampling | **T=0.6, 1 sample/pair** | per judge-sweep `derived/README.txt` |

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

## Flags / things to confirm
- **GRPO rollout temperature/top_p** are not pinned in `train_grpo_smoke.sh` → verl/vLLM default (likely T=1.0). If GRPO generation should match the T=0.6 heldout policy, set it explicitly. *(unconfirmed)*
- The GRPO values above are the **smoke** config (138-row slice, 40GB-shrunk: batch 32, n=2, mem 0.35). A **full-scale** GRPO run will likely raise batch / `rollout.n` / lengths.
- `train_grpo_smoke.sh` judge wiring uses `--reasoning-parser deepseek_r1` + `--max-model-len 16384`; the current judge-sweep serving uses `qwen3` + 32768. Treat the **judge-sweep config as canonical**; reconcile the GRPO judge launch to match.
- `repetition_penalty=1.1` overrides the Task-1 "no wire sampling override" policy for this one param (intended, per the cot-failure result).
