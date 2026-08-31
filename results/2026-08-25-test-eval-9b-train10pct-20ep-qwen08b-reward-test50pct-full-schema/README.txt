QWEN 0.8B-REWARD GENERATOR EVALUATION
=====================================

Run identity
------------

Cluster result root:
  /home/lancewicki/projects/turing-rl/results/2026-08-25-test-eval-9b-train10pct-20ep-qwen08b-reward-test50pct-full-schema

Generator training tag:
  9b_frac10_20ep_qwen08b_nothink_kl1e4_lr1e4_temp1

Evaluation dates:
  2026-08-25 through 2026-08-28 UTC

Configuration
-------------

Generator checkpoints:
  GRPO steps 0, 12, 24, 36, 48, 60, 72, 84, 96, 108, and 120.
  Step 0 is the shared SFT initialization. Steps 12-120 were merged from the
  generator training run before inference.

Evaluation set:
  /home/lancewicki/projects/turing-rl/data/prism/full_s42_history_sft40_grpo60_test10/eval_subsets/test_seed42_n440.parquet
  440 held-out pairs from 123 users.
  SHA-256: 0c727d910484aeed0bc59c7f88f3dedbef51bd6b6babe1e3cd053a854e45e1d2

Generator sampling:
  temperature=0.7, top_p=0.8, top_k=20, max_tokens=1024,
  truncate_prompt_tokens=12500, max_model_len=13524.

Judge cells, in launch order:
  qwen35-0.8b  Qwen/Qwen3.5-0.8B       thinking OFF  TP=1, replicas=8, concurrency=32
  gemma4-12b   google/gemma-4-12B-it  thinking ON   TP=1, replicas=8, concurrency=4
  gemma4-31b   google/gemma-4-31B-it  thinking ON   TP=8, replicas=1, concurrency=4
  qwen35-9b    Qwen/Qwen3.5-9B        thinking ON   TP=1, replicas=8, concurrency=32

All judge cells used the corrected full ordered response schema and the same
frozen pair Parquet at each checkpoint. Judge sampling was repetition_penalty=1.1,
temperature=0.6, max_completion_tokens=8192, and score clip maximum=7.
The Qwen 0.8B cell used three API retries.

Code and runtime provenance
---------------------------

Initial launch:
  source a5c198d25bee58d5f25dd2b40baec64749cba12f
  main   c9d9ab78d57b5169c4cf079a95b1cfcd08e69063
  snapshot /home/lancewicki/projects/turing-rl-sources/a5c198d25bee58d5f25dd2b40baec64749cba12f

Resume launch after the reuse-only continuation fix:
  source bbe8144a3f0d80fe5f3e06b033859eb646b31430
  main   94bcee7df71758b579f6a13b4872aeff44f9f3fc
  snapshot /home/lancewicki/projects/turing-rl-sources/bbe8144a3f0d80fe5f3e06b033859eb646b31430

Key judge environments:
  turing-rl-train: Python 3.12.13, torch 2.10.0+cu130,
    transformers 4.57.6, vLLM 0.18.0.
  turing-rl-gemma4-vllm-nightly: Python 3.12.13, torch 2.13.0,
    transformers 5.14.1, vLLM 0.26.1rc1.dev335+gc687c1abb.
  Full environment manifests and package hashes are retained under provenance/.

Jobs
----

Merge jobs for steps 12-120:
  18967, 18968, 18969, 18971, 18972, 18973, 18975, 18976, 18977, 18979.

Generator inference jobs for steps 0-120:
  18981, 18985, 18993, 18998, 19001, 19004, 19007, 19010, 19013, 19016, 19019.

Pair-building jobs for steps 0-120:
  18982, 18986, 18994, 18999, 19002, 19005, 19008, 19011, 19014, 19017, 19020.

Qwen 0.8B judge jobs for steps 0-120:
  19022, 19024, 19026, 19028, 19030, 19032, 19034, 19036, 19038, 19040, 19042.

Gemma 12B judge jobs for steps 12-120:
  19185, 19187, 19189, 19214, 19216, 19218, 19220, 19222, 19224, 19228.
  Step 0 was checksum-verified and reused from job 15974.

Gemma 31B judge jobs for steps 12-120:
  19231, 19233, 19247, 19259, 19261, 19263, 19272, 19303, 19310, 19313.
  Step 0 was checksum-verified and reused from job 16007.

Qwen 9B judge jobs for steps 12-120:
  19318, 19320, 19322, 19324, 19326, 19328, 19330, 19337, 19339, 19341.
  Step 0 was checksum-verified and reused from job 15880; its source generation
  was job 15805.

The complete Slurm accounting export, including continuation jobs and the
initial failed controller, is provenance/pipeline_jobs.psv.

Mechanical validation
---------------------

  PASS: held-out split guard, with zero overlap against SFT train, GRPO train,
        and GRPO validation rows or users.
  PASS: 44/44 judge cells contain exactly 440 unique pair keys.
  PASS: every cell is below the 2% parse-error threshold.
  PASS: all retained judge jobs completed with exit code 0:0.

Per-judge parse accounting is recorded in validation.txt.

Artifacts and SHA-256
--------------------

  16560994ed233a4c796280a1b47acd36951a62dcb5ca0713c4367a7cf11ab235  test_eval_judges_train10pct_20ep_qwen08b_reward_test50pct_full_schema.png
  951b7d8a45d0d9b0f69472f4454aa0bb654b081b4132523f3cd2f74b06d9ae29  summary_gemma4-12b.csv
  13da863fd84c3af16122ef2c87dd202d35b0067c180d56d4a0a20b42e8d0e5bc  summary_gemma4-31b.csv
  91442c0de9abf07cf99688cfaa938d61092e16fe9d7ffbe384871c8ed38df0e3  summary_qwen35-0.8b.csv
  30d68cd58cd52a1a14128d2ed3fe74f872833579b5c31afc893b6b998e3ac02e  summary_qwen35-9b.csv
  f32d3d3ad7336a8c6c08565909ce1a2ca080337596b8d39d36e9bd39fec02d79  validation.txt
  2befa3fba60fd4a5ab6785e4ce973d401dadf8759b64901e91ecb26b8dde2f37  plot/build_derived_artifacts.py
  6f6f1acec6cf06c425985fbba90b4a2a7659f450b3bfc81c8a600f3bfe3370eb  plot/plotstyle.py

Reproduction
------------

Rebuild the local derived plot and validation file:

  cd results/2026-08-25-test-eval-9b-train10pct-20ep-qwen08b-reward-test50pct-full-schema
  /Users/lancewicki/miniforge3/bin/python plot/build_derived_artifacts.py

Rebuild a summary from the cluster reward files, substituting CELL and MODE:

  /home/lancewicki/miniconda3/envs/turing-rl-train/bin/python \
    /home/lancewicki/projects/turing-rl-sources/bbe8144a3f0d80fe5f3e06b033859eb646b31430/scripts/summarize_test_eval.py \
    --eval_root /home/lancewicki/projects/turing-rl/results/2026-08-25-test-eval-9b-train10pct-20ep-qwen08b-reward-test50pct-full-schema \
    --cell CELL --mode MODE --expect_pairs 440 --out_csv summary_CELL.csv
