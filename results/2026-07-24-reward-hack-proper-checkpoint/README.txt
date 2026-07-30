Reward-hack proper-checkpoint experiment provenance
===================================================

References
----------
Spec: docs/superpowers/specs/2026-07-24-reward-hack-proper-checkpoint-design.md
Plan: docs/superpowers/plans/2026-07-24-reward-hack-proper-checkpoint.md
W&B project: https://meta.wandb.io/lancewicki/2026-07-15-rl-generator-vs-fixed-judge

Shared inputs
-------------
Dataset:
  data/prism/full_s42_history_sft40_grpo60_test10/grpo/train_overfit10.parquet

Frozen judge:
  Qwen/Qwen3.5-9B

Grid:
  KL in {1e-3, 1e-4, 0}
  LR in {1e-5, 1e-4}
  GRPO G=4
  TURING_JUDGE_SCORE_CLIP_MAX=7

Arm A: Qwen3-8B
----------------
Merged SFT/KL-reference checkpoint:
  checkpoints/sft/qwen3_8b_prism_full_s42_bf16_fsdp_nopack_epochsave/merged_ep3

Initial 50-update grid jobs (2026-07-25 through 2026-07-26):
  11136-11141

LR=1e-4 continuation jobs (2026-07-27 through 2026-07-28):
  11711-11713

Run tags:
  8b_proper_kl1e3_lr1e5
  8b_proper_kl1e4_lr1e5
  8b_proper_kl0_lr1e5
  8b_proper_kl1e3_lr1e4
  8b_proper_kl1e4_lr1e4
  8b_proper_kl0_lr1e4

Configuration:
  LoRA r64/alpha32
  target_modules=[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]
  10 prompts per update, 4 rollouts per prompt

Cluster source paths:
  /home/lancewicki/projects/turing-rl/results/grpo/rl-generator/8b_proper_*/reward_dump
  /home/lancewicki/projects/turing-rl/results/grpo/rl-generator/8b_proper_*/checkpoints
  /home/lancewicki/projects/turing-rl/logs/rl_gen-*.out

Arm B: Qwen3.5-9B
------------------
B0 rollout-sync job (2026-07-28):
  11910

Grid and continuation jobs (2026-07-28 through 2026-07-30):
  11915-11920
  12069 (9b_proper_kl1e4_lr1e4 continuation from global_step_50 to global_step_100)

Run tags represented by artifacts in this directory:
  9b_b0_calibrated_lr1e4
  9b_proper_kl1e3_lr1e4
  9b_proper_kl1e4_lr1e4

Configuration:
  Qwen/Qwen3.5-9B
  FSDP2 LoRA r64/alpha32
  lora.merge=True
  target_modules=[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]
  trainer GPUs 0-6; frozen judge GPU 7
  7 sampled prompts per update, 4 rollouts per prompt

Validated stack used by the B0 run:
  turing-rl SHA dfa0995e92bad4421be145b25d0b665adaacdb25
  verl 0.9.0.dev0, git c791da0bfcd7d7b560b1e461d2c188145b39c353
  vLLM 0.18.0+cu130
  transformers 5.4.0
  torch 2.10.0+cu130
  flash-linear-attention 0.5.1

Cluster source paths:
  /home/lancewicki/projects/turing-rl/results/grpo/rl-generator/9b_b0_calibrated_lr1e4
  /home/lancewicki/projects/turing-rl/results/grpo/rl-generator/9b_proper_*/reward_dump
  /home/lancewicki/projects/turing-rl/results/grpo/rl-generator/9b_proper_*/checkpoints
  /home/lancewicki/projects/turing-rl/logs/rl_gen_1node-*.out

Artifacts in this directory
---------------------------
  winrate_over_time_proper.png
  8b_proper_*_rating_scatter.png
  8b_proper_*_conversation_trajectory.txt
  9b_proper_*_rating_scatter.png
  9b_proper_kl1e4_lr1e4_conversation_trajectory_step50.txt
  winrate_over_time_9b_completed_lr1e4.png
  8b_vs_9b_kl1e4_lr1e4_winrate.png
  8b_vs_9b_kl1e4_lr1e4_winrate_vertical.png
  arm_b_b0_calibrated_rollout_sync.json
  arm_b_b0_calibrated_job_11910.log

Reproduction
------------
Cluster repo:
  /home/lancewicki/projects/turing-rl

Merge the Arm A SFT checkpoint:
  HF_HOME=/home/lancewicki/data/hf_cache \
    python scripts/merge_sft_adapter.py \
      --base-model Qwen/Qwen3-8B \
      --adapter-dir checkpoints/sft/qwen3_8b_prism_full_s42_bf16_fsdp_nopack_epochsave/checkpoint-78 \
      --output-dir checkpoints/sft/qwen3_8b_prism_full_s42_bf16_fsdp_nopack_epochsave/merged_ep3

Submit the Arm A grid:
  bash scripts/slurm/submit_arm_a_grid.sh

Analyze a run:
  python scripts/overfit_gate_check.py \
    --dump_dir results/grpo/rl-generator/<tag>/reward_dump

  python scripts/plot_overfit_ratings.py \
    --dump_dir results/grpo/rl-generator/<tag>/reward_dump \
    --out results/grpo/rl-generator/<tag>/rating_scatter.png

  python scripts/dump_conversation_trajectory.py \
    --dump_dir results/grpo/rl-generator/<tag>/reward_dump \
    --out results/grpo/rl-generator/<tag>/conversation_trajectory.txt \
    --group_size 4
