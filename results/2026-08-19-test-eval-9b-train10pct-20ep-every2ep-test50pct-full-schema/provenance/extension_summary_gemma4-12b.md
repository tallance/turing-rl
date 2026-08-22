# split: PASS expect=heldout rows=440 users=123 parquet=/storage/home/lancewicki/projects/turing-rl/data/prism/full_s42_history_sft40_grpo60_test10/eval_subsets/test_seed42_n440.parquet

| checkpoint | n_scored | n_unique_pairs | n_likert | likert_mean | win_rate_ge5 | pct_7 | judge_accuracy | gen_win_rate | n_nontie | n_tie | n_parse_error |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 9b-train10pct-step72 | 440 | 440 | 440 | 4.275 | 0.5455 | 0.23 | 0.36 | 0.64 | 375 | 65 | 0 |
| 9b-train10pct-step84 | 440 | 440 | 440 | 4.3091 | 0.5773 | 0.45 | 0.3437 | 0.6563 | 387 | 53 | 0 |
| 9b-train10pct-step96 | 440 | 440 | 440 | 4.4773 | 0.6227 | 0.23 | 0.2514 | 0.7486 | 366 | 74 | 0 |
| 9b-train10pct-step108 | 440 | 440 | 440 | 4.4023 | 0.5818 | 0.45 | 0.2809 | 0.7191 | 356 | 84 | 0 |
| 9b-train10pct-step120 | 440 | 440 | 440 | 4.4591 | 0.625 | 0.45 | 0.2568 | 0.7432 | 370 | 70 | 0 |
