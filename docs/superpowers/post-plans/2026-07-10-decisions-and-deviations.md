# Judge-sweep execution — decisions & deviations (2026-07-10)

Log of decisions/discoveries made while executing
`docs/superpowers/plans/2026-07-08-judge-sweep-implementation.md`. Newest at top.
Links: plan task #, commit SHAs, Slurm job ids.

---

## D8 — SFT packing caused cross-document attention contamination → retrain unpacked
**Context.** SFT uses `trl` `packing=True` (the `SFTConfig` default, inherited from the original
repo, first commit `6aaecfb`) under `attn_implementation="sdpa"`. trl's padding-free packing
flattens several PRISM conversations into one 8192-token sequence and delegates cross-document
isolation to **FlashAttention varlen**. `sdpa` has no block-diagonal mask.

**Finding.** Tokens attend **across packed conversation boundaries** (cross-contamination). trl
printed the warning verbatim in both variant logs. Per-doc `position_ids`/RoPE reset and
`completion_only_loss` masking are correct; **only attention leaks**. Both packed checkpoints
(`bf16_fsdp`, `qlora_r64`) are affected.

**Repo-faithfulness check.** The original repo hardcodes `sdpa` everywhere (SFT/GRPO/baselines)
and, although `requirements.txt` pins `flash_attn==2.8.3`, the code never uses FA for attention;
`flash_attn` has no cu130 wheel and its source build fails on this cluster. So **upstream itself
trains sdpa+packing (contaminated)** — our packed checkpoints were faithful-but-buggy.

**Decision (user).** Choose **correctness over bit-faithfulness** for the generator: retrain with
`--no_packing` (clean per-conversation attention under sdpa). This is a **deliberate, documented
deviation** from upstream (which packs). Option B (flash-attn + packing) rejected — impractical on
cu130. Option C (keep contaminated, document) rejected in favor of a correct generator for the
downstream adversarial GRPO work.

**Impact.** Added a `NOPACK` toggle to `scripts/slurm/sft_variant.sh` (commit `014f158`):
`NOPACK=1` → `--no_packing`, writes to a distinct `..._bf16_fsdp_nopack/` dir. Smoke (job 9420)
clean (0 contamination warnings). Full retrain = job 9421 (~78 steps, ~1.5–2 h). The packed
checkpoints + their D3 inference (job 9419) are superseded. → **log in `our_patches.md` (Task 22).**

---

## D7 — Workflow: direct SSH-tunnel cluster access replaces the agent-comms relay
**Context.** Plan assumed a Mac↔cluster relay via `docs/agent-comms/*` (Mac writes instructions,
cluster agent runs them, replies in `cluster-to-mac.md`).

**Discovery.** The Mac can reach the cluster **directly** over the `localhost:2223` SSH tunnel
(squeue/sbatch/cat/logs). No relay agent needed.

**Decision.** Direct tunnel access is now the **primary** workflow; the git relay is a **fallback**
(tunnel down). `CLAUDE.md` rewritten accordingly (V3 gotchas preserved). The `preflight-job-check`
skill was slimmed to turing-rl (user_sim content archived in `user_sim-legacy.md`).

**Impact.** New `scripts/sync_to_cluster.sh` (commits `cb2f9a0`, `75d27d2`): ships the committed
tree via `git archive|tar`, stamps `DEPLOYED_SHA` on the cluster (every run maps to a SHA), and
verifies `.py`/`.sh` syntax remotely; **never touches** `checkpoints/ results/ logs/ wandb/`.
Modes: full (commit-gated, authoritative), `--wip` (whole working tree, no commit), partial
`<files>` (dirty debug). Deploy loop: edit → commit (or `--wip`) → `sync_to_cluster.sh` → run.

---

## D6 — SFT was launching single-GPU → OOM; fixed with real multi-GPU + 3-variant bake-off
**Context.** Plan/`sft_full.sh` launched SFT with plain `python -m training.sft.lora_sft`.

**Discovery.** Plain python = one process = **single GPU** (`WORLD_SIZE=1`): the full 16 GB bf16
8B model + 8192-seq activations landed on one 40 GB A100 → **OOM** (job 9408), wasting GPUs 1–7.
Plain DDP would not fix it (replicates the full model per GPU).

**Decision.** Launch distributed via `torch.distributed.run --nproc_per_node=8`, and — at the
user's call — run **three memory strategies in parallel and compare**: `qlora_r64` (4-bit),
`bf16_fsdp` (FSDP full-shard, full fidelity), `bf16_fa2` (bf16 + FlashAttention-2). New
parameterized launcher `scripts/slurm/sft_variant.sh` (commits `f0fcc96`, `51f0b2d`); CLI
overrides added to `lora_sft.py` (`--attn_implementation`, `--fsdp`, `--force_qlora`,
`--report_to`) so the committed yaml stays read-only Table-5 (no concurrent-`sed` race).

**Result.** Both ran on 40 GB with 8-GPU torchrun (~40 min each, not the paper's 12–16 h — that
gap is `packing`, see D8):
- `qlora_r64`: train_loss 0.935, token-acc 0.770, 35.7 min.
- `bf16_fsdp`: train_loss 0.908, token-acc 0.779, 44 min — **winner** (full Table 5 + better metrics).
- `bf16_fa2`: **skipped** — `flash-attn` not installed; torch 2.10/cu130 has no wheel and a
  source build isn't worth it (runs already ~40 min; FSDP already gives full fidelity).

**Impact.** FSDP full_shard + PEFT LoRA **works** in our stack (trl `SFTTrainer`,
`use_orig_params=True`) — this corrected a stale preflight warning that claimed the opposite.

---

## D5 — LoRA config corrected to paper Table 5 (upstream shipped divergent defaults)
**Context.** Plan Task 7 only added checkpointing; spec §4.3 *claimed* the yaml already matched
paper Table 5.

**Finding.** `training/sft/configs/qwen3_8b_lora.yaml` actually held upstream defaults
`lora_r=16, lora_alpha=32, use_qlora=true` — 4× lower rank, 4-bit — diverging from Table 5.

**Decision (user).** Fix the yaml to Table 5: `lora_r=64, lora_alpha=128, use_qlora=false` (bf16,
dropout 0.05 unchanged). `max_seq_length=8192` was already passed by the sbatch; LR/warmup/wd/
epochs/BS/grad_accum already matched. Spec §4.3 corrected.

**Impact.** Commit `c935dc5`. This is the config the variant runs used.

---

## D4 — Final whole-branch review of Batch 1: 2 fixes applied
**Context.** subagent-driven-development final review over `51e1a1d..7821660`.

**Findings (both CONFIRMED, fixed).**
1. Calibration metadata key mismatch — `run_judge_sweep_cell.py` wrote `n_pairs_total`/no wall
   time; `calibration_report.py` read `n_pairs`/`wall_seconds` → every server cell projected
   `inf`, breaking the >4h gate. Fixed: rank-0 re-writes `run_metadata.json` post-gather with
   `n_pairs`+`wall_seconds` (commit `b1cee3b`).
2. Reward dump never captured `judge_finish_reason`/`usage`/`latency` (Task 6 Step 4 unimplemented)
   → analyzer `budget_hit_rate` always 0. Fixed via a `ContextVar` side-channel in
   `shared/api_client.post_chat_async` (signature unchanged), read in `reward.py._call` (commit
   `22fc9d3`). Minor: removed dead unreachable `raise` in `_openai_chat`.

**Impact.** Re-review clean; all tests pass.

---

## D3 — Heldout inference pickle structure confirmed (validates Task 13 `build_pairs`)
Job 9419 (on the *packed* `bf16_fsdp`, now superseded) produced **880 gens / 128 users** and the
nesting: `dict[user_id]{test_targets:[{post_id, target_idx, ground_truth, context, user_history,
persona, …, generations:[{reasoning, response, raw_completion, finish_reason, stop_reason,
output_token_count}]}]}`. This matches Task 13's `build_pairs` flattener (align by
`(user_id, post_id, target_idx)`, parse `raw_completion`). Structure reusable; will rerun D3 on
the `_nopack` checkpoint (D8).

---

## D2 — CoT generation done (served, thinking-off)
Task 8: `data/sft/prism_full_s42_sft_cot.parquet` = 3272/3272 rows, 0 failures, **0 `<think>`
tags** (thinking-off enforced on the wire), 86 rows (2.6%) hit the leak-guard cap (kept with
flag). Job 9407, ~3.7 min. Metadata later patched with `resolved_sampling` (D1). PRISM split was
verified paper-faithful end-to-end first (Task 2 pytest 7-green + Task 3 byte-identical re-split).

---

## D1 — Sampling policy frozen to `generation_config` defaults (no wire override)
**Context.** Plan Task 1 said: probe OpenRouter's applied sampling, then match it.

**Finding.** OpenRouter's applied sampling is unobservable (the probe echoed only the provider
name). Both served models ship identical `generation_config.json` defaults
(`T=0.6, top_p=0.95, top_k=20, min_p=0`).

**Decision.** Send **no** sampling on the wire for any cell (anchor included); let vLLM apply each
model's `generation_config.json` defaults. `PERSONA_JUDGE_SAMPLING` stays unset. Recorded in
`results/2026-07-08-judge-sweep/derived/sampling_fidelity.md`. This governs the judge cells, the
CoT client, and the offline cell.
