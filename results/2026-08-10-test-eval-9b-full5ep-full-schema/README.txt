Corrected full-schema judge evaluation of the 9B full-5-epoch GRPO run
===========================================================================
Provenance and mechanical validation only. Finalized 2026-08-17.

RUN
---
Cluster result root:
  /home/lancewicki/projects/turing-rl/results/2026-08-10-test-eval-9b-full5ep-full-schema
Submission source:
  /home/lancewicki/projects/turing-rl-runs/full-schema-a6b3990
Git commit: a6b3990f702f43b28ba0b233197c2885b01c5fff
Evaluation interval: 2026-08-11T00:54:56 through 2026-08-15T15:23:24.
Launcher: scripts/launch_full_schema_eval.sh
Launcher sha256:
  d36a2b395fd95506a71281349ec8886fb94b74d261e1673c06498be9f9dfee01

Generator training run: 9b_full5ep_kl1e4_lr1e4_temp1, job 14217
(COMPLETED). Checkpoint root:
  /home/lancewicki/projects/turing-rl/results/grpo/rl-generator/9b_full5ep_kl1e4_lr1e4_temp1/checkpoints
Evaluated steps: 0 32 64 96 128 160 192 224 256 288 320.
Step 0 is the shared pre-RL SFT initialization; nonzero steps are GRPO actor
checkpoints from the run above.

The pre-fix package remains separate at:
  results/2026-08-06-test-eval-9b-full5ep

RUNTIME AND MODELS
------------------
This grandfathered run predates SOURCE_MANIFEST/per-job runtime manifests, so
an immutable package inventory was not captured. Do not infer package versions
from the current cluster environments. Recorded execution paths were:
  Qwen/client Python: /home/lancewicki/miniconda3/envs/turing-rl-train/bin/python
  Gemma vLLM: /home/lancewicki/miniconda3/envs/turing-rl-gemma4-vllm-nightly/bin/vllm
  GPU allocation: eight NVIDIA A100-SXM4-40GB GPUs per judge cell.

Judge model IDs and serving shapes:
  qwen35-9b    Qwen/Qwen3.5-9B       TP=1, replicas=8
  gemma4-12b   google/gemma-4-12B-it TP=1, replicas=8
  gemma4-31b   google/gemma-4-31B-it TP=8, replicas=1
  qwen35-4b    Qwen/Qwen3.5-4B       TP=1, replicas=8
  qwen35-27b   Qwen/Qwen3.5-27B      TP=8, replicas=1
Gemma cache snapshots were pinned to 707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7
(12B) and 842da3794eaa0b77d5f08bae87a17459d91ff475 (31B). The Qwen launch
IDs were unpinned aliases; their resolved cache revisions were not recorded.

DATA AND EVALUATION CONFIGURATION
---------------------------------
Held-out parquet:
  /home/lancewicki/projects/turing-rl/data/prism/full_s42_history_sft40_grpo60_test10/test.parquet
Rows/users: 880 / 128
Parquet sha256:
  c7b13e2d538630109e100b7f66e78b8fce4cb1b88c793db357dd1b97c8bf8e32
split_guard.json: PASS; zero row and user overlap with SFT train, GRPO train,
and GRPO validation.

The 11 generation/pair parquets were reused from the archived pre-fix package
and verified byte-identical. Their sha256 values, in step order, are:
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

Judge order: qwen35-9b, gemma4-12b, gemma4-31b, qwen35-4b, qwen35-27b.
All cells used thinking=on, temperature=0.6, repetition_penalty=1.1,
max_completion_tokens=8192, timeout=1800 seconds, score_clip_max=7, and the
corrected full ordered response schema.

JOBS AND VALIDATION
-------------------
All listed jobs completed with exit code 0; judge_jobs.csv has timestamps and
elapsed times for each of the 55 retained judge cells.
  qwen35-9b:  15245,15246,15247,15248,15249,15250,15251,15361,15362,15363,15364
  gemma4-12b: 15365,15366,15367,15412,15413,15414,15415,15416,15417,15418,15556
  gemma4-31b: 15557,15558,15559,15560,15561,15562,15956,15957,15958,15959,15960
  qwen35-4b:  15961,15962,16278,16279,16280,16281,16282,16283,16284,17018,17020
  qwen35-27b: 17022,17024,17026,17028,17030,18298,18299,18300,18301,18302,18303
Historical failed qwen35-9b step-224 job 15280 was replaced by completed job
15361; only the replacement is included in judge_jobs.csv and the summaries.

Zero-tolerance verification: PASS. All 55 cells have exactly 880 expected unique
pair keys, with zero missing, extra, or duplicate keys. The minimum valid Likert
counts across checkpoints were qwen35-9b=870, gemma4-12b=880,
gemma4-31b=880, qwen35-4b=879, and qwen35-27b=876. See validation.txt and
the n_likert column in each summary CSV.

LOCAL ARTIFACTS
---------------
  test_eval_judges_full5ep_full_schema.png
  summary_<judge>.csv and summary_<judge>.md (five judges)
  judge_jobs.csv
  split_guard.json
  validation.txt
  plot/plot_test_eval_judges.py and plot/plotstyle.py
  SHA256SUMS (checksums for every artifact above)

REPRODUCTION
------------
On the cluster, regenerate the summaries and zero-tolerance validation from the
frozen raw result tree:

  PY=/home/lancewicki/miniconda3/envs/turing-rl-train/bin/python
  SRC=/home/lancewicki/projects/turing-rl-runs/full-schema-a6b3990
  EVAL=/home/lancewicki/projects/turing-rl/results/2026-08-10-test-eval-9b-full5ep-full-schema
  for CELL in qwen35-9b gemma4-12b gemma4-31b qwen35-4b qwen35-27b; do
    $PY $SRC/scripts/summarize_test_eval.py --eval_root "$EVAL" \
      --cell "$CELL" --mode on --expect_pairs 880 --max_missing_frac 0 \
      --out_csv "summary_${CELL}.csv" --out_md "summary_${CELL}.md"
  done
  $PY $SRC/scripts/verify_judge_completeness.py --eval_root "$EVAL" \
    --expect_pairs 880 --pairs_tag 880 --max_missing_frac 0

From the repository root, regenerate the local plot:

  D=results/2026-08-10-test-eval-9b-full5ep-full-schema
  python "$D/plot/plot_test_eval_judges.py" --eval_root "$D" \
    --stem test_eval_judges_full5ep_full_schema \
    --title "9B GRPO generator, full-dataset 5-epoch run, on the held-out test set" \
    --subtitle "{n} pairs per checkpoint, 128 unseen users. All five judges score the same generations with the corrected full ordered schema.  * = judge the run was trained against. Run tag 9b_full5ep_kl1e4_lr1e4_temp1; step 0 is the shared SFT init."

Verify the local artifact manifest from this directory with:
  shasum -a 256 -c SHA256SUMS
