Corrected full-schema judge evaluation of the 9B full-5-epoch GRPO run
===========================================================================
Provenance and operational state only. Last updated 2026-08-12T19:21:29Z.

RUN
---
Cluster result root:
  /home/lancewicki/projects/turing-rl/results/2026-08-10-test-eval-9b-full5ep-full-schema
Submission source: /home/lancewicki/projects/turing-rl-runs/full-schema-a6b3990
Git commit: a6b3990f702f43b28ba0b233197c2885b01c5fff
launch_full_schema_eval.sh sha256:
  d36a2b395fd95506a71281349ec8886fb94b74d261e1673c06498be9f9dfee01

Held-out set: 880 rows / 128 users; split guard PASS; no user or row overlap with
SFT train or GRPO train/val. Test parquet sha256:
  c7b13e2d538630109e100b7f66e78b8fce4cb1b88c793db357dd1b97c8bf8e32

Steps: 0 32 64 96 128 160 192 224 256 288 320
Judge order: qwen35-9b, gemma4-12b, gemma4-31b, qwen35-4b, qwen35-27b.
All use thinking=on, temperature=0.6, repetition_penalty=1.1, max completion
tokens=8192, timeout=1800s, score clip max=7, and the full ordered response schema.
Pair generation is reused from the archived pre-fix run; all 11 old/new pair
parquets are byte-identical.

PAIR PARQUET SHA256 (gen_9b-full5ep-step<N>_880.parquet)
---------------------------------------------------------
0    95f48a9c52d85a6f6c49fd3387e60efe0e1ee5e436bd961f1884750ecfcf7783
32   70a454a091cfecab821a7a6bc2386f5966dd324fe0c91c86020e5b8b7cb60b7a
64   03e29c5b74829d75ff885142045443bd1f12b9841e582c54408378bf9569d59e
96   380ab4cf247ac1079c95b67b4e697aa3961f2486154129d2a9f272f7d0720476
128  91db5d626f12b2fa8d596d4509ce8b7254ea7c1f3632e4a9b293be1dc7dc0cbc
160  31df9b2dfbab069b0600e94f2cacb659838cc38d4fc89330a3857facd050a016
192  30f13c4eeeeca014d65a15471d8616e206d593b7570e1e3134e172e62d0fd3e2
224  af5861854b75cbe3f40fc5bd8fce20e14961e255c35e248b52fbfb22da3c0379
256  63ae7d36905e629c5aa6e1cd1ecb7f6e052b76913bffa895d0a7d50454128e0f
288  7213d715ceba519f8a5c6d30cb235640fd9772a2e19f1903c9ac046069ae30d0
320  768023f72a0933aabd619eb372a527329558a45b8edc8fd4a6150c804575d576

STATUS AT LAST UPDATE
---------------------
qwen35-9b: 11/11 complete. Jobs 15245-15251 and 15361-15364. Failed job
15280 (step 224) was superseded by completed job 15361.
gemma4-12b: 11/11 complete. Jobs 15365-15367, 15412-15418, and 15556.
gemma4-31b: steps 0/32/64 complete (15557-15559); step 96 running as
15560 (270/880 rows at the timestamp above); steps 128/160 queued as
15561/15562. Controller 15563 will automatically continue from OFFSET=28.
qwen35-4b and qwen35-27b: not started.

Every completed cell checked so far has exactly 880 unique rows, with no missing,
extra, or duplicate pair keys. Minimum valid ratings: qwen35-9b 870/880;
gemma4-12b 880/880; gemma4-31b 880/880.

MONITOR AND FINAL VERIFY
------------------------
  ssh -p 2223 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    lancewicki@localhost "squeue -u lancewicki"

After the chain has finished:
  PY=/home/lancewicki/miniconda3/envs/turing-rl-train/bin/python
  SRC=/home/lancewicki/projects/turing-rl-sources/30ad6975e188c46be05c43ae30e9fa4a536655fc
  EVAL=/home/lancewicki/projects/turing-rl/results/2026-08-10-test-eval-9b-full5ep-full-schema
  $PY $SRC/scripts/verify_judge_completeness.py --eval_root "$EVAL" \
    --max_missing_frac 0.02

ADD A NEW JUDGE CELL OVER THE EXISTING PAIRS
--------------------------------------------
Wait for the active chain to finish. Run preflight-job-check. The cell must exist in
configs/judge_sweep_cells.py. Use a NEW provenance run root but keep EVAL_ROOT pointed
at this result tree; never reuse this run root's runtime manifest. First plan, then
repeat without --plan-only:

  STATE=/home/lancewicki/projects/turing-rl
  EVAL=$STATE/results/2026-08-10-test-eval-9b-full5ep-full-schema
  CELL=<configured-cell-name>
  RUN=$STATE/results/2026-08-10-test-eval-9b-full5ep-full-schema-add-$CELL
  scripts/cluster_launch.sh --plan-only --dependency-profile eval \
    --run-root "$RUN" --env EVAL_ROOT="$EVAL" --env DO_GEN=0 \
    --env JUDGES="$CELL" --env "STEPS=0 32 64 96 128 160 192 224 256 288 320" \
    --env GEN_KEY_PREFIX=9b-full5ep-step scripts/launch_test_eval.sh

The launcher refuses an existing reward directory. Do not use FORCE_REJUDGE for an
addition; choose a new cell name or deliberately archive the old cell first.
