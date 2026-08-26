# split: PASS expect=heldout rows=440 users=123 parquet=/storage/home/lancewicki/projects/turing-rl/data/prism/full_s42_history_sft40_grpo60_test10/eval_subsets/test_seed42_n440.parquet

| checkpoint | n_scored | n_unique_pairs | n_likert | likert_mean | win_rate_ge5 | pct_7 | judge_accuracy | gen_win_rate | n_nontie | n_tie | n_parse_error |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 9b-train10pct-step72 | 440 | 440 | 440 | 4.4341 | 0.6114 | 2.73 | 0.3656 | 0.6344 | 424 | 16 | 0 |
| 9b-train10pct-step84 | 440 | 440 | 440 | 4.5864 | 0.6682 | 3.64 | 0.3147 | 0.6853 | 429 | 11 | 0 |
| 9b-train10pct-step96 | 440 | 440 | 439 | 4.6401 | 0.6583 | 2.51 | 0.3135 | 0.6865 | 421 | 18 | 1 |
| 9b-train10pct-step108 | 440 | 440 | 438 | 4.7078 | 0.6826 | 2.05 | 0.2778 | 0.7222 | 414 | 24 | 2 |
| 9b-train10pct-step120 | 440 | 440 | 439 | 4.7722 | 0.7084 | 3.64 | 0.2542 | 0.7458 | 417 | 22 | 1 |
