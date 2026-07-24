# Reward-Hacking Probe, Repeated on a Proper SFT Checkpoint (+ Qwen3.5-9B generator arm)

**Date:** 2026-07-24 · **Status:** design (pre-plan) · **Author:** Tal Lancewicki (Mac agent)

## 1. Objective & hypothesis

Re-run the "RL generator vs. fixed judge" reward-hacking probe
(`docs/superpowers/specs/2026-07-15-rl-generator-vs-fixed-judge-design.md`), fixing the one
confound that muddied its verdict: the original probe seeded GRPO from an SFT checkpoint whose
completion mask **excluded the stop token** (`<|im_end|>`), so the generator never learned to
terminate (the "SFT stop-token masking bug"; fixed via `supervise_stop_token: true`). We now have a
**stop-token-supervised** SFT
checkpoint and repeat the experiment on it, for **two generators**:

- **Arm A — Qwen3-8B** (existing stack): the direct repeat.
- **Arm B — Qwen3.5-9B** (new stack, spike-gated): the new generator axis.

This experiment has **two purposes**:

**Purpose 1 — replicate (or refute) the hack on a clean checkpoint.** The headline finding of the
original probe was that the frozen 9B judge is gameable *given enough optimization pressure*: the
KL sweep (β∈{1e-3,1e-4,0}) stalled at ~0.57–0.60 win-rate (4–5/10 overfit gate), but a **10×
learning rate (lr=1e-4)** broke through to **8/10, win-rate 0.744** — showing the plateau was an
optimization artifact, not judge robustness. The confound: that entire conclusion rests on a
generator that never learned to stop. **H1: the same hack (lr=1e-4 clears the ≥8/10 overfit gate
and drives the overfit-set win-rate toward the buggy-checkpoint 0.744) replicates when the SFT init
is clean.**

**Purpose 2 — compare the 8B vs 9B generator.** With the frozen 9B judge and every other setting
held fixed, does generator **scale/architecture** change hackability? Qwen3.5-9B is both larger
and a different backbone (hybrid Gated-DeltaNet vs Qwen3-8B's full attention). **H2: a stronger/
different-arch generator reaches a *higher* win-rate and/or clears the overfit gate with *less*
optimization pressure (lower lr / stronger KL) than the 8B** — i.e. the frozen judge is more
easily gamed by a more capable simulator. The KL×LR grid, run identically on both generators
against the *same* judge, makes this a controlled comparison rather than an afterthought.

Success is *demonstrating (or refuting) H1 and H2 on a clean checkpoint*, not building a good
simulator. It de-risks the adversarial (trainable-discriminator) extension — including whether the
discriminator must scale with the generator.

## 2. Relation to the original probe (what changes, what doesn't)

Everything inherits the 2026-07-15 design unless listed here. Only the deltas below change.

| Item | Original (2026-07-15) | This experiment |
|---|---|---|
| SFT init + KL reference | `qwen3_8b_prism_full_s42_bf16_fsdp_nopack/final` — **stop-token masked (buggy)** | **checkpoint-78** of the stop-token-supervised trajectory run (10715/8B, 10716/9B). ep3, 3-epoch recipe. |
| KL reference wiring | B4 bug then fix: merge SFT adapter → standalone backbone, `lora_adapter_path=null`, fresh RL LoRA, disable-adapter recovers the **merged SFT** policy | **Keep the corrected wiring** (`scripts/merge_sft_adapter.py`, `_merged_sft_ref` tag). Reference = merged SFT, not base. |
| Swept parameters | KL sweep (inconclusive) + a single lr=1e-4 probe | **Full KL×LR grid**: KL∈{1e-3,1e-4,0} × LR∈{1e-5,1e-4}, all no-cap. Direct comparability to the buggy-checkpoint runs. |
| Generators | Qwen3-8B only | **Qwen3-8B (Arm A)** + **Qwen3.5-9B (Arm B, gated)** |
| Cap | Lifted to 7 (no-op on 1–7 Likert = "no cap") | **Same — no cap** (`TURING_JUDGE_SCORE_CLIP_MAX=7`). Unchanged. |
| Judge | Frozen Qwen3.5-9B (headline) + 397B (soak) | **Frozen Qwen3.5-9B** for both arms (comparability + feasibility). 397B optional, deferred. |
| Judge sampling / thinking | reppen 1.1, temp 0.6, thinking-on, 8192 cap | **Unchanged.** |
| GRPO HP (LoRA r64/α32, G=4, PPO ep 1, total 3, β/lr per-cell) | Paper Table 6 / upstream yaml | **Unchanged** except the swept KL/LR cells. |

**The sole scientific change is the SFT init (buggy → proper); the sole new axis is the 9B
generator.** Everything else is held fixed for comparability.

## 3. Arm A — Qwen3-8B (existing stack)

Runs on the current `turing-rl-train` stack with **no new dependencies** — identical plumbing to
the 2026-07-15 runs (`scripts/slurm/rl_generator_run.sh`, atomic 2-node judge+trainer, DP-8 9B
judge). Only the SFT checkpoint path and the KL/LR overrides change.

### 3.1 Sweep grid (overfit-10 gate)

All cells: **no cap** (`TURING_JUDGE_SCORE_CLIP_MAX=7`), frozen 9B judge, 50 overfit epochs
(no early-stop), corrected merged-SFT KL reference, proper checkpoint-78 init.

| Cell | KL β | LR | RUN_TAG |
|---|---|---|---|
| A1 (faithful) | 1e-3 | 1e-5 | `8b_proper_kl1e3_lr1e5` |
| A2 | 1e-4 | 1e-5 | `8b_proper_kl1e4_lr1e5` |
| A3 | 0 | 1e-5 | `8b_proper_kl0_lr1e5` |
| A4 (the hack) | 1e-3 | 1e-4 | `8b_proper_kl1e3_lr1e4` |
| A5 | 1e-4 | 1e-4 | `8b_proper_kl1e4_lr1e4` |
| A6 | 0 | 1e-4 | `8b_proper_kl0_lr1e4` |

Overrides via the existing `EXTRA_OVERRIDES` passthrough:
`actor_rollout_ref.actor.kl_loss_coef=<β> actor_rollout_ref.actor.optim.lr=<lr>`.

**Gate metric (unchanged):** `scripts/overfit_gate_check.py`, strict per-prompt majority
(`frac > 0.5`, ties excluded), pass = ≥8/10 on the final-epoch rollouts. Report last-K-epoch
average alongside the final snapshot (the ledger flagged single-snapshot noise, swings 5–9).

**Headline metric (overfit set).** The primary deliverable is the overfit-10 result itself — this
is where the original hack was measured: the buggy-checkpoint **0.744 / 8-of-10 was the overfit-set
win-rate**, so the clean-checkpoint overfit grid is *directly* comparable, same 10 turns, same
metric, no full run needed. Per cell report: final-epoch strict per-prompt win-rate + gate count
(≥8/10?), last-K-epoch average (snapshot-noise guard), and the per-example rating trajectory
(`plot_overfit_ratings.py`). **The head-to-head is clean-checkpoint vs buggy-checkpoint win-rate at
each (KL, LR) cell** — especially lr=1e-4 vs 0.744. Replicates → the hack is real and
checkpoint-independent; doesn't → the original hack was partly an artifact of the non-terminating
generator (a scientifically important correction).

### 3.2 Full run (deferred to post-plan)

Optional follow-on, **not in scope for this plan**. For any cell that clears the overfit gate, the
full run is straightforward reuse of the 2026-07-15 pipeline: full grpo split (4174 rows), 3 epochs,
no cap, frozen 9B judge → eval on the frozen 880-target heldout set (`eval/generate_trained.py` →
`build_judge_pairs.py` → 9B judge, order matched to the sweep) → directional accuracy / win-rate /
frac-4-ties vs the SFT baseline. Tracked as a post-plan continuation once the overfit results land.

## 4. Arm B — Qwen3.5-9B generator (new stack, spike-gated)

The 9B was previously only a **frozen judge** (served statically via a one-time LoRA→full splice).
As a GRPO-trained **generator** it needs vLLM to serve an *updating* Gated-DeltaNet policy in the
rollout loop. Feasibility (per the veRL/vLLM community investigation, 2026-07-24) is **works with
caveats**:

- The official Qwen3.5-9B checkpoint declares `Qwen3_5ForConditionalGeneration`, which **vLLM does
  support** (native GDN/MTP since vLLM 0.17) — even though literal `Qwen3_5ForCausalLM` is not
  registered. So the rollout engine *can* serve the arch.
- Requires a **dedicated upgraded env** (must not disturb `turing-rl-train`), built in veRL's
  Docker build order, not free-pip: **veRL 0.8.0+/main, vLLM ≥0.18 (0.20.2 recommended),
  transformers 5.4.0** (fixes the open 9B GDN actor crash, veRL #6549), **FLA 0.5.1**, causal-conv1d,
  flash-attn from the veRL image.
- **Rollout weight-sync: LoRA r64 + `lora.merge=True`** (merges the adapter into dense weights and
  syncs those to vLLM each step). This keeps **methodological parity with the 8B LoRA r64 recipe**
  *and* avoids native adapter-only DeltaNet LoRA (still broken/experimental). Full-param FSDP2 FT is
  the **fallback** only if merge=True misbehaves.
- Fits 8×A100-40GB with `enable_gradient_checkpointing`, param+optimizer offload,
  `rollout.free_cache_engine=True`, `enforce_eager=True`, `enable_prefix_caching=False`,
  mem-util ≈0.40, GEN_TP=4/FSDP_SIZE=8. ~18GB full-weight sync/step; offload costs throughput.
  Disable MTP/speculative decoding until stable (FSDP+MTP unresolved, veRL #6483). Always clear
  KV/GDN state after each weight update (vLLM #48312).
- No fully version-pinned public 9B run exists → **~1–2 eng-days to pin the happy path; 3–7 days if
  a real GDN loader bug bites**. This is why Arm B is gated.

### 4.1 Step B0 — env build + feasibility spike (GATE)

Build the dedicated env, then run **one clean multi-step (~5–10 step) GRPO** on Qwen3.5-9B init
from the already-spliced `merged_ep3` full checkpoint (from run 10716), LoRA r64 + `merge=True`,
frozen 9B judge, overfit-10 data. **Pass = steps complete cleanly, reward/judge scores logged,
generations terminate, no NaN/crash.** Fall back to full-param FT if merge=True fails; if both
fail within the time box, Arm B is deferred and reported as such.

### 4.2 Step B1 (only if B0 passes)

Mirror Arm A's **overfit grid** (§3.1, tags `9b_proper_*`) → overfit gate + per-cell win-rate,
judge held fixed to the **frozen 9B judge** so 8B and 9B generator results are directly comparable.
Full runs are the same post-plan follow-on as §3.2 (out of scope here).

## 5. Datasets, judges, reward, training config

Unchanged from the 2026-07-15 design (§§4–6, 8): PRISM `full_s42_history_sft40_grpo60_test10`
lineage (overfit-10, grpo train 4174 / val, heldout 880); reward extras kept; cap env=7;
`PERSONA_JUDGE_SAMPLING` reppen 1.1 / temp 0.6, thinking-on, 8192 cap; GRPO Table-6 HP with only
the per-cell KL/LR overridden. The frozen 9B judge is served DP-8 on one node (existing pattern).

## 6. Components / interfaces

- **Arm A launcher:** reuse `scripts/slurm/rl_generator_run.sh` unchanged; drive the grid via
  `RUN_TAG` + `EXTRA_OVERRIDES` (KL/LR) + the new `SFT_ADAPTER_PATH`/merged-ref pointing at
  checkpoint-78. No code change expected beyond config/path.
- **Merged-SFT reference:** `scripts/merge_sft_adapter.py` on checkpoint-78 (both models) → the
  standalone backbone used as `actor_rollout_ref.model.path`, `lora_adapter_path=null`.
- **Arm B env:** new `turing-rl-rl-qwen35` conda/Docker env (veRL main + vLLM 0.20.2 + transformers
  5.4.0 + FLA 0.5.1), documented like the SFT env recipe. **Do not touch `turing-rl-train`.**
- **Arm B launcher:** a 9B variant of `rl_generator_run.sh` starting from veRL's 27B FSDP2 GRPO
  recipe, adapted to 9B (GEN_TP=4, FSDP_SIZE=8, offload, merge=True, cache-clear). New file;
  additive.
- **Eval/analysis:** reuse `eval/generate_trained.py`, `build_judge_pairs.py`,
  `overfit_gate_check.py`, `plot_overfit_ratings.py`. A small analyzer tabulating
  clean-vs-buggy-checkpoint win-rate/accuracy per cell per arm.

## 7. Experiment structure & sequencing (user-approved)

1. **Arm A first, submit now.** Merge checkpoint-78 refs → overfit-10 grid (A1–A6) → gate +
   per-cell win-rate. Existing stack, no new risk. (Full runs = post-plan, §3.2.)
2. **Then Arm B.** Build the env → B0 feasibility spike (gate) → B1 overfit grid only if B0 passes.
   8B does **not** wait on the 9B env work.
3. **Post-plan (out of scope):** full-split runs + 880-heldout eval for gate-clearing cells.

`preflight-job-check` before every sbatch; ≤10 concurrent jobs; 7-day wall; additive commits only;
deploy via `sync_to_cluster.sh` (committed HEAD).

## 8. Tests (TDD, incremental over the 2026-07-15 suite)

The 2026-07-15 suite (cap env, overfit-gate metric, reward-env wiring, config integrity, overfit-10
builder, eval-vs-sweep parity, split integrity) still applies. New/changed:

| # | Test | Type | Asserts |
|---|---|---|---|
| 1 | Checkpoint-78 init resolves | unit | The stop-token-supervised checkpoint-78 path exists and is loadable; PEFT config r64/α32; distinct from the buggy `_final` path. |
| 2 | Grid config integrity | unit | Each of A1–A6 (and 9b_ tags) resolves with the intended (KL, LR) and no-cap; RUN_TAGs unique; merged-SFT ref set + `lora_adapter_path=null`. |
| 3 | Merged-SFT ref parity | regression | `base+SFT(ckpt-78) adapter` logits ≈ merged backbone; step-0 KL ≈ 0; only the fresh RL LoRA is trainable. |
| 4 | (Arm B) 9B rollout smoke | integration (spike) | B0: N GRPO steps complete, judge scores logged, gens terminate, no NaN. Gate for B1–B2. |

## 9. Risks & open items

- **Arm B env pinning** — the dominant risk; time-boxed (~1–2 days happy path). Fallback ladder:
  LoRA merge=True → full-param FSDP2 FT → defer Arm B (report as deferred). Loader-bug tail 3–7 days.
- **8B/9B methodological parity** — kept by using LoRA r64 on both; the only difference is the 9B's
  dense rollout weight-sync (merge=True). Note this deviation explicitly in results.
- **Hybrid cache/reload correctness** (vLLM #48312) — clear KV/GDN state after every weight update
  in the Arm B launcher; validate reward stability across steps in B0.
- **Overfit-gate snapshot noise** — report last-K-epoch average, not just the final snapshot.
- **397B anchor** — deferred this round (feasibility + comparability favor the 9B judge); revisit
  after the clean-checkpoint 9B-judge results land.
- **Multi-agent repo** — additive commits only; new Arm B launcher + configs are new files;
  `reward.py` already carries the cap env (no further shared-file touch expected).

## 10. Success criteria

1. **Arm A overfit gate (headline):** the lr=1e-4 cell (A4) clears ≥8/10 on the proper checkpoint,
   reported head-to-head against the buggy-checkpoint overfit win-rate (0.744); the KL-only cells
   behave as before (KL not the limiter). Either outcome (replicates / doesn't) is a clean result.
2. **Arm B:** B0 spike passes → the 9B generator reproduces (or not) the same overfit hackability
   pattern under the frozen 9B judge.
3. **8B vs 9B (H2):** with the frozen 9B judge and identical KL×LR overfit grid, compare the two
   generators head-to-head — peak overfit win-rate and the (KL, LR) cell at which each first clears
   the ≥8/10 gate. A clean statement of whether the more capable / different-arch generator games
   the frozen judge more readily.
4. Every setting traceable to the 2026-07-15 design + paper except the documented deltas (proper
   checkpoint, KL×LR grid, 9B arm).
