Arm B B0 rollout-sync provenance
================================

Date: 2026-07-28
Slurm job: 11910
Slurm state: COMPLETED, exit 0, elapsed 00:37:38
Node/GPU layout: 8x A100-SXM4-40GB; trainer GPUs 0-6, frozen judge GPU 7
Deployed turing-rl SHA: dfa0995e92bad4421be145b25d0b665adaacdb25

Stack
-----
verl: 0.9.0.dev0, git c791da0bfcd7d7b560b1e461d2c188145b39c353
vLLM: 0.18.0+cu130
transformers: 5.4.0
torch: 2.10.0+cu130
flash-linear-attention: 0.5.1
model: Qwen/Qwen3.5-9B
training: FSDP2 LoRA r64/alpha32, lora.merge=True, vLLM TP=1

Reproduction
------------
From /home/lancewicki/projects/turing-rl after deploying the SHA above:

  B0_ROLLOUT_SYNC=1 MODE=overfit OVERFIT_EPOCHS=2 \
    RUN_TAG=9b_b0_calibrated_lr1e4 \
    EXTRA_OVERRIDES='actor_rollout_ref.actor.optim.lr=1e-4' \
    sbatch --export=ALL scripts/slurm/rl_generator_run_9b_1node.sh

Cluster source paths
--------------------
/home/lancewicki/projects/turing-rl/results/grpo/rl-generator/9b_b0_calibrated_lr1e4/rollout_sync.json
/home/lancewicki/projects/turing-rl/logs/rl_gen_1node-11910.out

Verification command
--------------------
  /home/lancewicki/miniconda3/envs/turing-rl-rl-qwen35/bin/python -m pytest \
    tests/test_rl_grid.py \
    tests/test_proper_checkpoint.py \
    tests/test_submit_arm_a_grid.py \
    tests/test_rollout_sync_guard.py \
    tests/test_rl_9b_launcher.py \
    tests/test_verl_09_compat.py -q

Artifacts
---------
arm_b_b0_calibrated_rollout_sync.json
  sha256 8c1004dbc5788d962f1720a1eab9b8573276a129dbc0531fdf9b47724f505b99
arm_b_b0_calibrated_job_11910.log
  sha256 48815627bcd9b70152513ace05803093782f0f2bb43f4f81822bd6ad1717939f
