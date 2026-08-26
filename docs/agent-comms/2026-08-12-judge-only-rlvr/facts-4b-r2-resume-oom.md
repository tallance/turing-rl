# 4B judge-RLVR round 2 — observed facts

Facts only. Every number below was read from Slurm accounting, job logs, or the filesystem on
`rfai-research-aws-use2-1` on 2026-08-21. No hypotheses, no recommendations. Statements about what
was *not* found are marked as such and name the check that was run.

## 1. Scope

Two arms of the 4B judge discriminator (Qwen3.5-4B, GRPO, thinking-ON, 7,680-token response
budget), trained against the frozen pair set generated from the 9B SFT checkpoint `merged_ep3`.

## 2. Jobs

| job | arm | node | state | elapsed | exit | start → end (UTC) |
|---|---|---|---|---|---|---|
| 18701 | graded | a100-129-254 | FAILED | 17:31:02 | 1:0 | 2026-08-19 19:44:28 → 2026-08-20 13:15:30 |
| 18755 | directional | a100-208-085 | COMPLETED | 13:57:20 | 0:0 | 2026-08-20 18:13:51 → 2026-08-21 08:11:11 |
| 18780 | graded resume | a100-042-140 | CANCELLED | 01:28:46 | 0:0 | 2026-08-20 20:20:32 → 2026-08-20 21:49:18 |
| 18784 | graded resume | a100-042-140 | FAILED | 01:35:47 | 1:0 | 2026-08-20 21:55:37 → 2026-08-20 23:31:24 |

All four ran on different nodes except 18780/18784, which ran sequentially on the same node.
18780 was cancelled deliberately by the operator; it did not fail.

## 3. Configuration

Identical across 18701, 18755 and 18784 unless noted:

- `JUDGE_MODEL_PATH=Qwen/Qwen3.5-4B`, `MODE=full`, `HF_HUB_OFFLINE=1`
- `JUDGE_MAX_NUM_SEQS=24` — verified as the resolved `max_num_seqs` in all three job logs
- `JUDGE_FSDP_OFFLOAD` **unset in all runs** → the launcher default `True` applied
  (`scripts/slurm/judge_grpo_train.sh:100-104`). Note: the plan's table specifies offload *off*
  for the 4B; that was never applied to any run.
- `max_prompt_length=11264`, `max_response_length=7680`
- `trainer.total_epochs=1`, `trainer.test_freq=13`, `trainer.val_before_train=True`
- `save_freq`: 26 for 18701, 18755, 18780; **13** for 18784
- 18780 and 18784 additionally set `RL_CKPT_DIR` to 18701's checkpoint directory, so
  `resume_mode: auto` restored from `global_step_26`.

## 4. Source SHAs

| SHA | date | description |
|---|---|---|
| 88bae7f | 2026-08-19 15:22 | merge: reconcile thinking-mode guards with lancewicki/main |
| 6bb1b18 | 2026-08-20 09:20 | fix(judge-job): per-job TMPDIR |
| a00770f | 2026-08-20 16:15 | merge lancewicki/main into worktree branch |
| b91d21c | 2026-08-21 | comment-only correction in judge_grpo_train.sh |

`88bae7f` is an ancestor of `6bb1b18`, which is an ancestor of `a00770f`.

- 18701 ran under **88bae7f**; 18755 ran under **6bb1b18**; 18780/18784 ran under **a00770f**.
- `git diff 88bae7f 6bb1b18 -- training scripts` → **one file, `scripts/slurm/judge_grpo_train.sh`,
  +9/-1** (the TMPDIR change). So the graded and directional arms differ by that change only.
- `git diff 88bae7f a00770f` → 12 files, of which the only training-path file is the same
  `judge_grpo_train.sh` TMPDIR change. The other 11 are eval scripts, docs and tests not executed
  by the training job.

## 5. Checkpoints

Acceptance criteria used: file count, number of distinct shard sizes across
`actor/model_world_size_8_rank_*.pt`, and presence of `actor/lora_train_meta.json`.

| run root | checkpoints present | acceptance |
|---|---|---|
| `results/2026-08-19-judge-r2-4b-graded-thinkon-7680` | `global_step_26` | 32 files, 1 distinct shard size, `lora_train_meta.json` present |
| `results/2026-08-20-judge-r2-4b-directional-thinkon-7680-retry` | `global_step_26`, `global_step_52` | both: 32 files, 1 distinct shard size, `lora_train_meta.json` present |
| `results/2026-08-20-judge-r2-4b-graded-thinkon-7680-resume` | none (checkpoints directory empty) | — |

`latest_checkpointed_iteration.txt` reads 26 for the graded run root, 52 for the directional one.

18701 also produced a `global_step_52` directory containing 8 files across six different shard
sizes (643–757 MB, against `global_step_26`'s uniform 1157 MB), with no optimizer state and no
`lora_train_meta.json`. That directory was deleted by the operator on 2026-08-20 before 18780 was
submitted, so it can no longer be inspected. `scripts/merge_grpo_adapter.py` warns that a missing
`lora_train_meta.json` causes the LoRA delta to be silently dropped during merge.

## 6. Validation trajectories

Validation ran at steps 0, 13, 26, 39, 52 (`test_freq=13`). Values are
`val-aux/prism_judge/<metric>/mean@1` from the job stdout.

**18755, directional — five points (0, 13, 26, 39, 52):**

| metric | 0 | 13 | 26 | 39 | 52 |
|---|---|---|---|---|---|
| `judge_acc` | 0.3525 | 0.5138 | 0.5298 | 0.5752 | 0.6429 |
| `judge_tie` | 0.0142 | 0.2390 | 0.3447 | 0.2582 | 0.1554 |
| `judge_fmt_ordered_coverage` | 0.6974 | 0.9994 | 0.9938 | 0.9993 | 0.9993 |
| `judge_rung_unclosed_thinking` | 0.2028 | 0.0000 | 0.0057 | 0.0007 | 0.0007 |

**18701, graded — four points only (0, 13, 26, 39); no step-52 validation was logged:**

| metric | 0 | 13 | 26 | 39 |
|---|---|---|---|---|
| `judge_acc` | 0.3461 | 0.4887 | 0.5411 | 0.5585 |
| `judge_tie` | 0.0213 | 0.1716 | 0.1035 | 0.0333 |
| `judge_fmt_ordered_coverage` | 0.6911 | 0.9840 | 0.9838 | 0.9475 |
| `judge_rung_unclosed_thinking` | 0.2135 | 0.0142 | 0.0156 | 0.0511 |

18755 training rollouts at step 14: `response_length/mean` 3405.9, `max` 5611, `min` 1893,
`clip_ratio` 0.0.

These are training-time validation numbers on the judge's own validation split. They are **not**
the 880-pair evaluation set, and are not comparable to previously reported 880-pair figures
(e.g. 0.6869).

## 7. 18701 termination

- stderr progress bar last shows `51/52 [16:25:52<16:31]`; no `52/52` line.
- stdout logs `step:51` metrics, then a `local_global_step_folder:` line. `grep -o
  'global_step_[0-9]*'` over the stdout returns `global_step_26` (×1) and `global_step_52` (×1),
  so a step-52 save was initiated.
- The save line is timestamped ~12:59; the job ended 13:15:30, ~16 minutes later.
- Exit code `1:0` — nonzero exit, not signalled.
- `MaxRSS` 148128940K (≈141 GB) against `ReqMem` 512G.

Checks run that found nothing:
- `grep -inE 'quota|no space|ENOSPC|Disk'` over both logs → no matches.
- `grep -icE 'ActorDied|RayActorError|node.*died|Killed|SIGKILL|out of memory'` over stderr → 0.
- No traceback of any kind appears at the end of either stream; both simply stop.

The only tracebacks in 18701's stderr are two instances of
`OSError: [Errno 16] Device or resource busy: '/home/lancewicki/tmp/build/pymp-XXXX'`, at stderr
lines ~31–96, immediately followed by the startup dataset-filtering progress bars
(`Filtering prompts longer than 11264 tokens ... 3328` then `... 1410`). These occur at job
startup; the job then trained for 17 hours. The `sacct` `Reason` field reads `QOSMaxGRESPerUser`,
which is a scheduling reason recorded while the job was queued.

**The cause of 18701's termination is not established by any evidence gathered.**

## 8. 18784 termination

Genuine CUDA OOM. Verbatim from stderr:

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 MiB.
GPU 0 has a total capacity of 39.49 GiB
  this process has 38.87 GiB memory in use
  Process 146286 has 614.00 MiB memory in use
  Of the allocated memory 35.38 GiB ...
  and 252.57 MiB is reserved by PyTorch but unallocated
```

Raised through `ray::WorkerDict.actor_rollout_update_actor()` and
`ray::TaskRunner.run()`.

- Last logged training step: `step:28`. The resume had restored from step 26.
- 38.87 + 0.614 = 39.48 GiB of a 39.49 GiB card.
- Reserved-but-unallocated was 252.57 MiB.
- `PYTORCH_ALLOC_CONF=expandable_segments` is prohibited by project preflight check 38 and was
  not set.

18780, the earlier resume attempt with the same settings, ran 1h28m and reached the step-26
pre-train validation with `judge_acc` 0.5348, `judge_fmt_ordered_coverage` 0.9815,
`judge_rung_unclosed_thinking` 0.0184 before being cancelled. It did not OOM in that window.

Summary of the memory record: two continuous runs (18701 at 17h31m, 18755 at 13h57m) completed
their step sequences on this configuration with no OOM in either log; one resumed run (18784)
OOMed at step 28.

## 9. Provenance discrepancy

In `results/2026-08-19-judge-r2-4b-graded-thinkon-7680/provenance/`:

- `expected_runtime.json` records `source_sha` **88bae7f5** (correct for 18701).
- `launch.json` records `source_sha` **a00770fe**, with mtime 2026-08-20 20:16:32 UTC.

`a00770f` was authored 2026-08-20 16:15 EDT, after 18701 finished. The 20:16 mtime corresponds to
a submission attempt against this run root that was **rejected** by
`scripts/record_runtime_manifest.py:230` (`refusing to overwrite incompatible expected runtime`).
`launch.json` is written earlier in `scripts/cluster_launch.py` than that check runs, so the
rejected submission left the file overwritten. The run root now contains two provenance files
naming different source SHAs.

For comparison, `results/2026-08-20-judge-r2-4b-directional-thinkon-7680-retry/provenance/` has
`expected_runtime.json` and `launch.json` both reading `6bb1b189`.

## 10. Relevant enforcement code

- `scripts/cluster_launch.py:167` — a retained run's source must be a descendant of current
  `lancewicki/main` at submission time; `--debug --label` bypasses this but forces output below
  `results/debug/<label>/`.
- `scripts/record_runtime_manifest.py:225-230` — `--initialize` against an existing run root
  refuses when `source_sha`, `dependency_profile` or `enforced_fingerprint_sha256` differ.

## 11. Current cluster state (2026-08-21)

- No 4B judge training job is running or queued from this line of work.
- Job 18853 (`judge_grpo`, PENDING) exists under the same user account but was submitted by a
  different session.
- `df -h /home/lancewicki` → 34T available of 47T.

## 12. Paths

```
Run roots (cluster):
  /home/lancewicki/projects/turing-rl/results/2026-08-19-judge-r2-4b-graded-thinkon-7680
  /home/lancewicki/projects/turing-rl/results/2026-08-20-judge-r2-4b-directional-thinkon-7680-retry
  /home/lancewicki/projects/turing-rl/results/2026-08-20-judge-r2-4b-graded-thinkon-7680-resume

Logs:      <run root>/logs/slurm-judge_grpo-<jobid>.{out,err}
Provenance:<run root>/provenance/{launch.json,expected_runtime.json,jobs/<jobid>/}
Access:    ssh -p 2223 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null lancewicki@localhost "<cmd>"
```

## 13. Open items, stated without explanation

1. 18701's termination during the step-52 save is unexplained.
2. 18784's OOM at step 28 occurred on a configuration that two continuous runs completed without
   OOM. The mechanism is not established.
3. The 4B graded arm has no step-52 checkpoint. Its latest verified checkpoint is step 26.
4. `launch.json` for the graded run root is factually wrong and was corrupted by a rejected
   submission.
5. No 880-pair evaluation has been run on either arm's round-2 checkpoints.
