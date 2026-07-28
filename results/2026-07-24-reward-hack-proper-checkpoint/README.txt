Reward-Hacking Probe, Repeated on a PROPER (stop-token) SFT Checkpoint — Arm A (Qwen3-8B)
=========================================================================================
Spec:  docs/superpowers/specs/2026-07-24-reward-hack-proper-checkpoint-design.md
Plan:  docs/superpowers/plans/2026-07-24-reward-hack-proper-checkpoint.md
Cluster DEPLOYED_SHA: b4a1c50 (code) ; runs used the merged proper checkpoint (below).
Branch: worktree-reward-hack-proper-ckpt

WHAT THIS IS
------------
Re-run of the 2026-07-15 "RL generator vs. fixed judge" reward-hack probe, fixing the confound that
the original seeded GRPO from a stop-token-MASKED SFT checkpoint (non-terminating generator). Here we
seed from the stop-token-SUPERVISED checkpoint-78 (trajectory run 10715), merged into a standalone
backbone (merge parity verified: argmax agreement 1.0, softmax-KL 6e-4). KL reference confirmed
correct: wandb step-1 actor/kl_loss = 0 (vs the buggy runs' ~0.63-0.86).

H1: does the lr=1e-4 hack (buggy checkpoint: 8/10, win-rate 0.744 on the 10-turn overfit set)
replicate when the SFT init actually terminates?

GRID (overfit-10, no cap [TURING_JUDGE_SCORE_CLIP_MAX=7], frozen Qwen3.5-9B judge, 50 epochs,
LoRA r64/alpha32 attn+MLP, merged-SFT KL ref). Jobs 11136-11141 (2026-07-25..26).

RESULTS (proper checkpoint) — won/10 = strict per-prompt majority (frac>0.5, ties excl) at final
epoch; final_win_rate = final-epoch fraction. Buggy-checkpoint numbers from the 2026-07-15 run.
--------------------------------------------------------------------------------------------------
  cell                 KL     LR      won/10   final_win_rate   buggy-ckpt
  8b_proper_kl1e3_lr1e5  1e-3  1e-5    4/10     0.525            5/10 ~0.60  (faithful baseline)
  8b_proper_kl1e4_lr1e5  1e-4  1e-5    5/10     0.500            4/10 0.590
  8b_proper_kl0_lr1e5    0     1e-5    3/10     0.450            4/10 0.575
  8b_proper_kl1e3_lr1e4  1e-3  1e-4    7/10     0.718            8/10 0.744  (THE HACK)
  8b_proper_kl1e4_lr1e4  1e-4  1e-4    6/10     0.641            (n/a)
  8b_proper_kl0_lr1e4    0     1e-4    6/10     0.684            (n/a)

EXTENDED RESULT (lr=1e-4 resumed 50->~99 epochs; jobs 11711/12/13, 2026-07-27..28)
----------------------------------------------------------------------------------
Resuming the three lr=1e-4 cells from global_step_50 for another ~50 epochs shows the win-rate
KEEPS CLIMBING (the 50-epoch 7/10 was under-optimization, not a ceiling):
  cell                 50ep            ~99ep
  kl1e3 (HACK)         7/10 @ 0.72  -> 9/10  @ 0.895
  kl1e4                6/10 @ 0.64  -> 10/10 @ 0.95
  kl0                  6/10 @ 0.68  -> 8/10  @ 0.75
=> With enough optimization pressure ALL three lr=1e-4 cells CLEAR the >=8/10 gate (8-10/10),
   win-rate 0.75-0.95 ("more than human than human" decisively). The frozen 9B judge is fully
   gameable per-turn on the clean checkpoint; the earlier 0.72 was simply mid-climb (see the
   0->99 winrate_over_time_proper.png: lr=1e-4 curves ramp monotonically, lr=1e-5 stay flat ~0.5).
   (3960 rows/cell = ~99 epochs; veRL resume off-by-one lands at 99 not 100. Jobs 11711/12 show
   FAILED = benign FSx teardown crash, data complete/identical shape to the COMPLETED 11713.)

VERDICT (H1): THE HACK REPLICATES on the clean checkpoint — and STRENGTHENS with more epochs.
  - At lr=1e-5 the frozen judge holds ~0.45-0.53 across ALL KL (1e-3/1e-4/0) -> KL is NOT the
    limiter, exactly as on the buggy checkpoint.
  - Raising to lr=1e-4 drives win-rate to 0.64-0.72 across all three KL values; the hack cell
    (kl=1e-3, lr=1e-4) reaches 7/10 @ 0.718, vs the buggy checkpoint's 8/10 @ 0.744.
  => The reward-hack is real and checkpoint-independent, NOT an artifact of the non-terminating
     (stop-token-masked) generator. It is marginally weaker on the clean init (7/10 vs 8/10, 0.718
     vs 0.744) — sensible: a properly-terminating SFT start is slightly harder to push to full
     overfit. The LR->win-rate trend (0.5 -> 0.72) is robust across 3 KL values.
  CAVEAT: final_win_rate is a single-final-epoch snapshot (noisy, swings observed 5-9 in the prior
  run); a last-K-epoch average would firm up the 7-vs-8 gate margin. See the per-cell scatter PNGs.

FILES
-----
  winrate_over_time_proper.png - 6-subplot overall win-rate vs epoch (per-epoch pooled, win=Likert>=5,
                               ties/parse-fails excluded; red = 3-epoch rolling mean; gray = 0.5).
                               HEADLINE: lr=1e-5 rows stay flat ~0.5 all 50 epochs (all 3 KL); lr=1e-4
                               rows ramp over epochs ~10-40 then plateau ~0.6-0.72 (genuine convergence,
                               not a lucky final epoch). Same script/style as the 2026-07-15 plot.
  <cell>_rating_scatter.png  - 10-subplot per-example judge Likert vs epoch (blue=per-rollout,
                               red=epoch mean; green line = win Likert>=5, gray = tie 4).

INPUT DATA (cluster, source of truth)
-------------------------------------
  Reward dumps: results/grpo/rl-generator/8b_proper_<cell>/reward_dump/reward-*.jsonl (2000 rows/cell
  = 50 epochs x 10 prompts x G=4). turing_judge_score_raw = oriented Likert.
  SFT init/KL-ref (merged): checkpoints/sft/qwen3_8b_prism_full_s42_bf16_fsdp_nopack_epochsave/merged_ep3
    (merged from checkpoint-78 via scripts/merge_sft_adapter.py).
  Overfit-10: data/prism/full_s42_history_sft40_grpo60_test10/grpo/train_overfit10.parquet
  wandb: https://meta.wandb.io/lancewicki/2026-07-15-rl-generator-vs-fixed-judge (run per cell,
    experiment_name qwen3-8b-grpo-turing-8b_proper_<cell>).

REPRODUCE (cluster; repo /home/lancewicki/projects/turing-rl)
-------------------------------------------------------------
  0. Deploy committed HEAD (git archive HEAD | tar; stamp DEPLOYED_SHA). NOTE: scripts/sync_to_cluster.sh
     operates on the MAIN checkout, not a worktree; for a worktree branch use the manual archive.
  1. Merge the proper checkpoint (CPU; set HF_HOME=/home/lancewicki/data/hf_cache):
       python scripts/merge_sft_adapter.py --base-model Qwen/Qwen3-8B \
         --adapter-dir checkpoints/sft/qwen3_8b_prism_full_s42_bf16_fsdp_nopack_epochsave/checkpoint-78 \
         --output-dir  checkpoints/sft/qwen3_8b_prism_full_s42_bf16_fsdp_nopack_epochsave/merged_ep3
     (merge parity: argmax agreement + softmax-KL, NOT raw max-logit — bf16 merge gives ~0.3 max-logit
      noise on logits of scale ~4; assert argmax_agree~1.0 & KL<1e-3.)
  2. Submit the grid:  bash scripts/slurm/submit_arm_a_grid.sh   (6 cells; serialize under 24-GPU QOS)
  3. Gate + plot per cell:
       python scripts/overfit_gate_check.py --dump_dir results/grpo/rl-generator/<cell>/reward_dump
       python scripts/plot_overfit_ratings.py --dump_dir .../reward_dump --out .../rating_scatter.png
Slurm jobs: 11136-11141 (11136 marked FAILED = benign FSx 'Stale file handle' at teardown AFTER 50
epochs; its 2000-row dump is complete).

OUT OF SCOPE (post-plan): full-split runs + 880-heldout eval; Arm B (Qwen3.5-9B generator) — blocked
on a veRL 0.9 patch-port (see memory: verl-09-refactor-breaks-grpo-patch).
