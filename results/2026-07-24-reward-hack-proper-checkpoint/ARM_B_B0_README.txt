Arm B B0 calibrated rollout-sync validation
============================================

Date: 2026-07-28
Slurm job: 11910 (COMPLETED, exit 0, elapsed 00:37:38)
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

Result
------
The calibrated guard passed: ok=true and synced=true. All seven rollout replicas
advanced from weight version 0 to 1. Actor and rollout mean logprob movement were
0.0296001 and 0.0298842 (ratio 1.00960), and their delta correlation was 0.848292.
HF/vLLM raw parity remained stable (correlation 0.999219 -> 0.999252; mean absolute
error 0.0114813 -> 0.0115702). This rejects a stale-rollout explanation and validates
merged-LoRA current-policy weight synchronization for this configuration.

The nested strict_delta.ok=false value is retained as a diagnostic only. Its fixed
absolute tolerance assumes identical HF and vLLM Gated-DeltaNet kernels; the measured
step-zero engine bias already violates that assumption. The top-level calibrated gate
uses the observed step-zero bias and requires stable raw parity plus matched movement.

Verification
------------
On the exact deployed SHA, the relevant six-module suite passed: 49 passed.

Artifacts
---------
arm_b_b0_calibrated_rollout_sync.json
  sha256 8c1004dbc5788d962f1720a1eab9b8573276a129dbc0531fdf9b47724f505b99
arm_b_b0_calibrated_job_11910.log
  sha256 48815627bcd9b70152513ace05803093782f0f2bb43f4f81822bd6ad1717939f
