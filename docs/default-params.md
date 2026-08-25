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

> **CORRECTION (2026-08-04): GRPO judge concurrency was far too low, and the stated
> reason ("40GB KV pressure") is not supported by measurement.** GRPO job 13634 spent
> **41.5 of its 44.1 h (93.9%) waiting on the judge** — 9952 calls at ~101 s mean latency
> with an effective concurrency of only 6.3. Generation + backprop + checkpointing was
> just 2.7 h.
>
> Probe 13999 swept client concurrency against the trainer's own DP-8 topology using real
> ~22k-char judge prompts (`scripts/slurm/judge_concurrency_probe.sh`):
>
> | concurrency | req/s | p50 | p95 |
> |---|---|---|---|
> | 8 | 0.072 | 116 s | 131 s |
> | 32 | 0.238 | 124 s | 140 s |
> | **64** | **0.458** | 124 s | 140 s |
> | 128 | 0.460 | 130 s | 139 s |
>
> Throughput scales **6.4× up to 64** and saturates there, while **latency is flat**
> (p50 116→124 s, p95 ~140 s). There is no KV-cache collapse: at concurrency 8 the DP-8
> server was starved (one in-flight request per rank, so vLLM never batched), not
> protected. Concurrency 8 reproduced training's rate (0.072 vs 0.063 measured), which
> validates the probe.
>
> The cap traces to the job-13628 timeout cascade (concurrency 128 at a **400 s** timeout:
> queue wait exceeded the timeout, so every request failed). The effective fix is the
> **timeout**, not the concurrency — at concurrency 64 the measured p95 is 140 s, well
> inside even the old 400 s limit.
>
> **New defaults: `TURING_JUDGE_MAX_CONCURRENCY=64` with
> `PERSONA_OPENAI_TIMEOUT_SECONDS=1800`.** Projected effect on a 13634-shaped run:
> ~6 h of judging instead of 41.5 h, i.e. **~9 h end-to-end instead of 44 h**.
> Caveat: measured on **DP-8** (one server, 8 data-parallel ranks). An older non-DP sweep
> (`report-20260715-162423.md`) did collapse at 64+, so this applies to the DP topology.

## Judge (reward model)
SSOT: `configs/judge_sweep_cells.py` (model matrix), `scripts/run_judge_sweep_cell.py`
(`cell_env`), `training/grpo/reward.py` (reward math), `scripts/slurm/judge_sweep_cell.sh` (serving).

These defaults apply to the Qwen and Gemma judges used by the full evaluation. Model ID,
serving shape, reasoning parser, runtime environment, and client concurrency vary by model;
sampling, output schema, completion budget, and reward math remain shared.

Per-judge serving (footprint-based): fits one 40GB GPU (≤30GB) → **TP=1, 8 replicas**;
else whole node → **TP=8, 1 replica**.

| Judge | model_id | serving |
|---|---|---|
| 397B anchor (training judge) | `Qwen/Qwen3.5-397B-A17B-GPTQ-Int4` | TP=8/1rep, Int4, env `judge-vllm` (hybrid-Mamba MoE) |
| 122B | `Qwen/Qwen3.5-122B-A10B-GPTQ-Int4` | TP=8/1rep, Int4 |
| 4B / 9B | `Qwen/Qwen3.5-{4B,9B}` | TP=1/8rep (bf16), env `turing-rl-train` |
| 27B / 35B-A3B | `Qwen/Qwen3.5-{27B,35B-A3B}` | TP=8/1rep (bf16), env `turing-rl-train` |
| Gemma 4 12B | `google/gemma-4-12B-it` | TP=1/8rep, concurrency 4, pinned nightly env/snapshot |
| Gemma 4 31B | `google/gemma-4-31B-it` | TP=8/1rep, concurrency 4, pinned nightly env/snapshot |

Shared serving/sampling defaults (all judges):

| Param | Default | Notes |
|---|---|---|
| Serving common | `--dtype bfloat16`, `--max-model-len 32768`, `--disable-custom-all-reduce` (TP>1) | GPU utilization 0.85 Qwen, 0.90 Gemma |
| Thinking mode | `PERSONA_JUDGE_ENABLE_THINKING=1`; parser `qwen3` for Qwen, `gemma4` for Gemma | off = no reasoning parser |
| **Sampling** | **`repetition_penalty=1.1`, `temperature=0.6`** (pin explicitly) | inject via `PERSONA_JUDGE_SAMPLING='{"repetition_penalty":1.1,"temperature":0.6}'`. Pin temp so it's **uniform across judges** regardless of each model's `generation_config.json` (see Flags) |
| Output schema | `PERSONA_JUDGE_JSON_SCHEMA=1` (ordered 37-field schema; all fields required, no extras, `rating` last) | See `docs/judge-response-schema.md` |
| Max completion tokens | `PERSONA_JUDGE_MAX_COMPLETION_TOKENS=8192` | GRPO judge training matches this via `data.max_response_length: 8192` |
| **In-training validation sampling** | **`temperature=0.6`, `top_p=0.8`, `top_k=20`, `n=1`** | SSOT `training/grpo/configs/qwen35_judge_grpo.yaml` → `rollout.val_kwargs`. This is a **separate mechanism** from `PERSONA_JUDGE_SAMPLING` above, which only reaches the judge served over HTTP — it does not reach the trainer's own validation rollouts. Keep the temperatures equal so a checkpoint is validated the way it is scored. No `repetition_penalty` here yet (2026-08-25) |
| Client timeout | `PERSONA_OPENAI_TIMEOUT_SECONDS=1800` (thinking-on 397B) | reward.py fallback is 400 |
| Retries | `PERSONA_OPENAI_MAX_RETRIES=3` | |
| Concurrency | full eval: Qwen 32 per endpoint, Gemma 4 per endpoint; GRPO **`TURING_JUDGE_MAX_CONCURRENCY=64`** on DP-8 | **Corrected 2026-08-04** — see the correction note above. The old GRPO value starved the server; pair 64 with `PERSONA_OPENAI_TIMEOUT_SECONDS=1800` |
| Reward math | **no clip** (`TURING_JUDGE_SCORE_CLIP_MAX=7`, a no-op on the 1–7 Likert), then `(score−1)/6`, ×**0.9**; rating re-derived from 6 dimensions + penalties (mean×3) | `TURING_JUDGE_SCORE_CLIP_MAX`, `TURING_RAW_REWARD_SCALE` |

## Generator
SSOT: `training/sft/configs/qwen3_8b_lora.yaml` (base), `bash_scripts/grpo/train_grpo.sh` (GRPO, upstream = paper Table 6).

| Param | Default | Notes |
|---|---|---|
| Base model | `qwen3-8b` (SFT LoRA adapter → GRPO actor) | |
| **GRPO rollout temperature** | **1.0 (verl default, not overridden)** | training uses high temp for exploration. verl `trainer/config/rollout/rollout.yaml` |
| Heldout-inference sampling | **T=0.6, 1 sample/pair** | eval only (not training); per judge-sweep `derived/README.txt` |

> **GRPO training hyperparameters (batch, `rollout.n`, lengths, epochs) follow upstream
> `bash_scripts/grpo/train_grpo.sh` = paper Table 6** (batch 128, G=4, 3 epochs,
> max-response 1024, LR 1e-5, KL β 1e-3, PPO clip 0.2). We'll confirm/adjust after our own
> full run.

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
- **Judge score clip: NO clip, and that is now the code default too (changed 2026-08-04).**
  `training/grpo/reward.py` sets `TURING_JUDGE_SCORE_CLIP_MAX = 7.0`, a no-op on the 1–7 Likert.
  Upstream shipped **5.0** (commit 6aaecfb — an extra beyond the paper's own `(min{s,5}−1)/6`);
  we do not want it, because clipping at 5 flattens the advantage across ratings 5/6/7 and kills
  exactly the gradient that pushes a generator past ~50% (see
  `specs/2026-07-15-rl-generator-vs-fixed-judge-design.md:74`). Every launcher already exported
  `=7` explicitly (`scripts/slurm/rl_generator_run{,_9b,_9b_1node}.sh`, `scripts/launch_test_eval.sh`,
  guarded by `tests/test_rl_9b_launcher.py:68`), so **no completed run's numbers change** — this
  only fixes the unset-env path, which previously clipped silently. Those explicit exports are now
  redundant but harmless; left in place as documentation. Set the env to 5 to reproduce upstream.
- **Judge parser** — the correct parser for Qwen is `qwen3` (source-verified in
  `scripts/slurm/judge_serve_8b.sh:20`; used by the judge sweep). The 397B training judge
  `scripts/slurm/judge_serve.sh` is now fixed to `qwen3` (`--max-model-len 32768` was already
  correct). `scripts/slurm/cot_server.sh` still uses `deepseek_r1` + `16384`, but that server is
  the **thinking-OFF CoT teacher** (Qwen3-8B, emits `<reasoning>` not `<think>`), so its parser
  is cosmetic — left as-is.
- `repetition_penalty=1.1` overrides the Task-1 "no wire sampling override" policy for this one
  judge param (intended, per the cot-failure result).
- **Zero-shot sweep temperature was NOT uniform (post-hoc finding).** The completed judge sweep
  used the Task-1 "no wire override" policy → each judge ran at whatever its shipped
  `generation_config.json` sets. Actual temps: **0.6** for 27B / 122B / 397B / qwen3-8B; **~1.0**
  for **4B & 9B** (ship no `generation_config.json` → vLLM server default); **1.0** for
  **35B-A3B** (its config sets 1.0). So 4B/9B/35B-A3B ran hotter than 0.6, which can inflate their
  variance / repetition / parse-failure rates vs a 0.6 run — cross-judge zero-shot numbers for
  those cells are **not strictly comparable** to the 0.6 cells. **Accepted, not re-run.** The 397B
  anchor + all `repetition_penalty` / cot-failure conclusions are unaffected (anchor was 0.6).
  **All future zero-shot runs pin `temperature=0.6`** (see Sampling row) for a clean comparison.
