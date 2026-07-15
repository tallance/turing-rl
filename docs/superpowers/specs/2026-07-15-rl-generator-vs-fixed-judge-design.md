# RL Generator vs. Fixed Judge — Reward-Hacking Probe (Turing-RL "First Experiment")

**Date:** 2026-07-15 · **Status:** design (pre-plan) · **Author:** Tal Lancewicki (Mac agent)

## 1. Objective & hypothesis

Train **only the generator** with GRPO, starting from the SFT LoRA checkpoint, to maximize a
**frozen** LLM judge's belief that its turn is the human one — with the paper's reward **cap
removed**. This is the "First Experiment" of `Adversarial-User-Simulation.md`.

**Hypothesis:** with the cap lifted, the generator drives the judge's win-rate against the *real*
human turn **above 50%** ("more human than human") — i.e. a frozen judge is gameable. Turing-RL
caps its reward at `min{s,5}` precisely to suppress this; removing the cap is the one deliberate
deviation and is the whole point of the probe.

Success is *demonstrating the hack*, not producing a good simulator. It motivates the adversarial
(trainable-discriminator) extension.

## 2. Relation to Turing-RL (fidelity ledger)

Maximal fidelity to Turing-RL (arXiv 2606.19336) is a hard requirement everywhere except the one
intended deviation. Established during design:

| Item | Decision | Fidelity status |
|---|---|---|
| Reward cap `min{s,5}` | **Lift to 7** (env-configurable, see §4) | **Only deliberate deviation** — the experiment |
| Reward "extras" (0.9 scale, format bonuses, meaningful-thinking hard-zero) | **Keep** | Faithful — in the authors' released code `6aaecfb` (not the written formula, but their real runs) |
| Judge-rubric penalties (source-copy, wrong-target/role, assistant-like) + length penalty | **Keep** | Faithful — App E judge prompt + App C.3 |
| Judge thinking mode | **ON** for both judges | Faithful — OpenRouter probe proved the paper's judge thinks by default (~1783 reasoning tokens even with `reasoning=False`); the paper never disables it |
| GRPO hyperparameters | Upstream code yaml `6aaecfb` (= paper Table 6 on every row except one) | Faithful to the authors' released config |
| `train_batch_size` | **64** (keep upstream) | **code≠paper:** paper Table 6 says 128, the authors' released yaml trains at 64. We match the code they actually ran (same "trust the code" precedent as the reward extras). `ppo_mini_batch_size=64` (paper silent). |
| KL penalty | `use_kl_loss=true`, `β=1e-3`, `πref`=SFT ckpt | Faithful — Table 6; SFT auxiliary loss dropped (paper line 290), KL kept |
| Generator base | Qwen3-8B, thinking-disabled, SFT init | Faithful |
| Judge sampling | **`repetition_penalty=1.1`, `temperature=0.6`** (via `PERSONA_JUDGE_SAMPLING`) | Current post-sweep judge default (`docs/default-params.md`); fixes the long-thinking loop (parse-error 0.111→0.032 on 397B-on, job 9825). Curbs the ~6400-token CoT that caused truncation. |
| Judge model | Small = **Qwen3.5-9B**; anchor = **Qwen3.5-397B-A17B-GPTQ-Int4** | 9B is a deliberate small-judge variable (paper used only 397B); 397B matches paper |

All deviations are logged in the post-plan decisions doc as we implement.

## 3. Judges

Two **frozen** judges, both **thinking-on**, scored through the real GRPO reward path
(`training/grpo/reward.py`, `metric="turing"`, `TURING_PROMPT`):

- **Qwen3.5-9B** — the small trainable-discriminator candidate (exercised frozen here). Served as
  **8×1-GPU replicas** on one node (data-parallel), env `turing-rl-train`, `--reasoning-parser qwen3`.
- **Qwen3.5-397B-A17B-GPTQ-Int4** — the paper's anchor. Served **TP=8** (whole node), env
  `judge-vllm`, `--reasoning-parser`, `--disable-custom-all-reduce`.

Measured throughput (probes 9827/9892, real 34k-char Turing prompts, thinking-on):

| Judge | deploy | per-GPU sat. | ~tokens/call | 8192-cap parse-fail |
|---|---|---|---|---|
| 9B | 1 GPU | ~0.13 calls/s (~900 tok/s) | ~6400 | ~40% |
| 9B | 8 replicas (1 node) | **~1.0 calls/s** | — | — |
| 397B | TP=8 (1 node) | **~0.044 calls/s** (~296 tok/s, saturated at any concurrency) | ~5000 | ~57% |

Consequence: **full GRPO training runs against 9B only** (feasible via 8-replica fan-out). 397B
cannot be replicated within a node (needs all 8 GPUs for one TP=8 replica) → ~13 days/full-run →
**397B gets overfit + a single-epoch plumbing run only**; the full 3-epoch 397B run is deferred
(pending a separate speculative-decoding evaluation).

**Caveat — these probe numbers are without `repetition_penalty`.** The probes used the raw
thinking-on CoT (~5000–6400 tokens, ~40–57% truncation at 8192). The RL reward judge runs with
`repetition_penalty=1.1` (§4), which shortens the CoT (parse-error 0.111→0.032 in the sweep) and
therefore raises calls/s and cuts truncation. So the throughput above is a **pessimistic lower
bound**; actual GRPO judge throughput will be higher. (A reppen throughput re-probe is optional
before the full runs.)

## 4. Reward configuration

- **Cap:** make the Likert score clip **env-configurable** — `TURING_JUDGE_SCORE_CLIP_MAX`
  (default **5.0**, preserving current behavior). These runs set **7.0** (no-op clip on the 1–7
  Likert) so ratings 6–7 ("more human than human") earn full reward and produce GRPO advantage
  (the 5.0 clip flattens advantage exactly in that band, killing the gradient that pushes past 50%).
  This is the sole code change to `reward.py` and the sole deliberate deviation.
- **Extras kept** (0.9 scale, format bonuses, hard-zero, rubric penalties, length penalty) — faithful
  to the authors' code.
- **Judge sampling: `PERSONA_JUDGE_SAMPLING={"repetition_penalty":1.1,"temperature":0.6}`** — the
  current post-sweep judge default (`docs/default-params.md`). reppen 1.1 shortens the thinking-on
  CoT that otherwise runs ~6400 tokens and truncates ~40–57% of calls at 8192; it drops parse-error
  to ~3% (sweep 397B-on: 0.111→0.032) *and* speeds the judge up. temp pinned to 0.6 for uniformity.
- **`max_completion_tokens = 8192`** (sweep-proven with reppen; do NOT raise to 16k — reppen fixes
  the overrun and 16k would only slow the judge).
- **`PERSONA_JUDGE_ENABLE_THINKING=1`** — make thinking-on explicit rather than relying on the
  served chat-template default. (Other sweep sampling — top_p 0.95 / top_k 20 — comes from each
  model's `generation_config`.)
- **Optional capped control:** the same runs with cap=5 (paper) to demonstrate the cap suppresses
  the hack. Kept optional (one extra run per judge).

## 5. Datasets (PRISM, `full_s42` lineage)

- **Overfit-10:** first 10 rows of `data/prism/full_s42_history_sft40_grpo60_test10/grpo/train.parquet`
  → a new `grpo/train_overfit10.parquet`.
- **Full train / val:** that split's `grpo/train.parquet` (4174 rows) / `grpo/val.parquet`.
- **Eval:** that split's `test.parquet` (880 held-out prompts) — the same set that became the
  sweep's `prism_heldout_880.parquet`, so the sweep's SFT-generator accuracy is a ready baseline.
- **SFT init/ref adapter:** `checkpoints/sft/qwen3_8b_prism_full_s42_bf16_fsdp_nopack/final`.

## 6. Training configuration (paper Table 6)

Base `qwen3_8b_grpo.yaml` already matches the paper on temperature 0.6 / top-p 1.0 / top-k -1,
LoRA r64/α32, lr 1e-5, KL β=1e-3, G=4, PPO epochs 1, total epochs 3, max_response_length 1024,
clip 0.2, token-mean. Overrides needed:

- `data.train_files` → PRISM grpo train (base default points at a non-existent convokit path).
- `data.train_batch_size` → **keep 64** (upstream code; matches the authors' released config — no
  override; paper Table 6 says 128, see §2). `ppo_mini_batch_size` stays 64.
- `lora_adapter_path` / `SFT_ADAPTER_PATH` → the SFT adapter (init **and** KL reference `πref`).
- `resume_mode: auto` (crash safety on long runs).
- Judge endpoint env + the §4 reward env.

New per-judge configs compose on `qwen3_8b_grpo_turing.yaml`.

## 7. Experiment structure

**Stage 0 — Overfit-10 gate (both judges).** `train_batch_size=10` (1 step/epoch), high epoch count,
no-cap reward, thinking-on judge. **Pass = training judge prefers the generated turn (Likert ≥5,
ties=4 excluded) on ≥8/10 training prompts.** Proves the judge is hackable and the plumbing works
before any full run. Cost: 9B ~30 min (8 replicas); 397B ~4–10 h (0.044 calls/s).

**Stage 1 — Full run vs 9B (headline).** Full grpo split, 3 epochs, no cap, thinking-on 9B
(8-replica judge node + trainer node). ~18–24 h, one 7-day job. wandb curves: reward, raw judge
score, win-rate proxy, parse-fail rate.

**Stage 2 — 397B overfit + single-epoch plumbing.** Overfit gate (Stage 0) + one full epoch
(~4–5 days, ~16,700 calls at 0.044/s) to confirm the pipeline scales against the anchor. Full
3-epoch 397B deferred pending speculative-decoding results (tracked separately).

**Optional — capped control** (§4) per judge.

## 8. Evaluation (headline metric)

Reuse the sweep pipeline with the RL adapter swapped in: `eval/generate_trained.py`
(RL-final checkpoint) on `test.parquet` → `scripts/build_judge_pairs.py` → score (real, RL-gen)
pairs with the **matching training judge**, **order matched to the sweep** (deterministic per-pair
hash, so numbers are directly comparable to the SFT baseline).

Report per judge: **directional accuracy** (judge picks true human), **generator win-rate**
(= 1 − accuracy on non-ties), **frac-4-ties** — **RL-final vs SFT baseline** (from the sweep). A
drop in accuracy / rise in win-rate above 50% = the hack. (Optional stretch: cross-eval with the
other judge as an independent scorer.)

## 9. Compute & sequencing

- 9B track: trainer node (8 GPU) + 9B judge node (8×1-GPU replicas) = 16 GPU, under the 24-GPU QOS
  cap. Overfit gate → full run.
- 397B track: trainer node + 397B judge node (TP=8) = 16 GPU. Overfit gate → single-epoch soak.
- `preflight-job-check` before every sbatch; ≤10 concurrent jobs; 7-day wall limit.

## 10. Components / interfaces

- **`reward.py`** — env-configurable `TURING_JUDGE_SCORE_CLIP_MAX` (default 5.0). TDD unit test on
  `clip_turing_judge_score`. Sole reward code change.
- **GRPO configs** — `qwen3_8b_grpo_turing_9bjudge.yaml` / `_397bjudge.yaml` (+ overfit variants):
  PRISM data (train_batch stays 64), SFT adapter, reward env (cap, reppen, thinking-on).
- **Judge serving** — 9B 8-replica launcher (reuse sweep pattern) + 397B `judge_serve.sh`; health +
  model-verify before trainer starts.
- **Launcher/orchestrator** — build **fresh** (the old `grpo_smoke*.sh` are broken/legacy: they
  called the deleted `train_grpo_smoke.sh`). Wrap the canonical `bash_scripts/grpo/train_grpo.sh`
  (authors' launcher; reads the YAML, takes `SFT_ADAPTER_PATH`, forwards `"$@"` Hydra overrides):
  serve judge (9B 8-replica / 397B TP=8) → wait `/v1/models` + model-verify → launch trainer with
  `JUDGE_HOST` + reward env → scancel judge on exit; parametrized over `{9b, 397b} × {overfit, full}`.
  Carry the 40GB overrides preserved in `our_patches.md`.
- **Overfit dataset builder** — write `grpo/train_overfit10.parquet`.
- **Eval** — reuse `eval/generate_trained.py` + `build_judge_pairs.py` + judge scoring; a small
  analyzer comparing RL-final vs SFT-baseline accuracy/win-rate/ties per judge.

## 11. Risks & open items

- **veRL LoRA wiring** — must confirm veRL loads the SFT adapter as *both* the RL init *and* the KL
  reference `πref`. Explicit early implementation step, not an assumption.
- **Parse-fail tail** — reppen 1.1 cuts truncation to ~3% (sweep); monitor parse-fail rate during
  training, and if material either bump reppen slightly or raise `max_completion_tokens`. A reppen
  throughput re-probe before the full runs is optional (would replace the pessimistic §3 numbers).
- **397B full 3-epoch run** — deferred; depends on speculative-decoding (evaluated separately) or
  accepting a multi-day/multi-job run.
- **Overfit not converging** — raise epochs / lower `kl_loss_coef` (already 1e-3).
- **Multi-agent repo** — additive commits only; `reward.py` is the one shared-file touch (other
  agent not expected to edit it now).

## 12. Success criteria

1. **Overfit gate:** ≥8/10 training-judge win on the 10 prompts (per judge) with cap lifted.
2. **Full 9B run:** on the 880 held-out set, generator win-rate vs the real human turn **exceeds
   the SFT baseline**, ideally **>50%** (accuracy <50%) — demonstrating the frozen 9B judge is
   gamed. Qualitative inspection of hacked turns (via the dump viewer).
3. **397B:** overfit gate passes + single-epoch run completes cleanly (plumbing validated).
4. Every setting traceable to the paper except the documented cap lift.
