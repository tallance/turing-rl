# Merged SFT KL Reference Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make veRL's LoRA-disabled reference policy equal the SFT policy by merging the SFT adapter into the base model before adding a fresh RL LoRA.

**Architecture:** A one-time utility loads Qwen3-8B and the existing SFT PEFT adapter, safely merges the adapter into a standalone Hugging Face checkpoint, and writes provenance metadata only after the artifact validates. The GRPO launcher uses that merged checkpoint as `actor_rollout_ref.model.path` and explicitly sets `lora_adapter_path=null`, causing veRL to create a fresh rank-64, alpha-32 RL LoRA whose disabled state is the merged SFT reference.

**Tech Stack:** Python, Transformers, PEFT, safetensors, veRL/Hydra, pytest, Bash/Slurm.

---

### Task 1: Lock the merged-artifact contract

**Files:**
- Create: `tests/test_merge_sft_adapter.py`
- Create: `scripts/merge_sft_adapter.py`

**Steps:**
1. Add failing tests for adapter/base validation, successful safe merge, tokenizer preservation, metadata, atomic output, and artifact validation.
2. Run `pytest -q tests/test_merge_sft_adapter.py` and confirm the missing module failure.
3. Implement the merge utility with lazy ML imports so metadata validation remains cheap.
4. Run `pytest -q tests/test_merge_sft_adapter.py` and confirm it passes.

### Task 2: Correct the GRPO model/reference wiring

**Files:**
- Modify: `scripts/slurm/rl_generator_run.sh`
- Modify: `scripts/slurm/rl_generator_train.sh`
- Modify: `tests/test_grpo_config.py`

**Steps:**
1. Add a failing launcher regression test requiring the merged model override and a null pretrained adapter.
2. Replace `SFT_ADAPTER_PATH` with `MERGED_SFT_MODEL_PATH` in the atomic driver.
3. Make the trainer fail early unless the merged checkpoint has config, weights, tokenizer, and merge metadata.
4. Override `actor_rollout_ref.model.path` and `actor_rollout_ref.model.lora_adapter_path=null`.
5. Run the targeted tests and shell syntax checks.

### Task 3: Record the corrected reference semantics

**Files:**
- Modify: `docs/superpowers/post-plans/2026-07-15-rl-generator-vs-fixed-judge/decisions.md`

**Steps:**
1. Document that veRL's colocated reference disables the active LoRA, so loading the SFT adapter as the active LoRA made the reference the base model.
2. Document that the old path also retained the SFT adapter's alpha 128 instead of creating the intended RL alpha-32 adapter.
3. Record the required cluster validation: logits parity, step-zero KL near zero, new-LoRA-only trainability, and fresh checkpoint directory.

### Task 4: Verify and commit

**Steps:**
1. Run `pytest -q tests/test_merge_sft_adapter.py tests/test_grpo_config.py tests/test_reward_cap.py tests/test_overfit_gate.py`.
2. Run `bash -n scripts/slurm/rl_generator_run.sh scripts/slurm/rl_generator_train.sh`.
3. Run `python -m py_compile scripts/merge_sft_adapter.py`.
4. Review `git diff --check`, `git diff`, and the current branch status.
5. Commit only the files in this plan with an additive commit.
