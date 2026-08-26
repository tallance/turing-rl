# split: PASS expect=heldout rows=440 users=123 parquet=/storage/home/lancewicki/projects/turing-rl/data/prism/full_s42_history_sft40_grpo60_test10/eval_subsets/test_seed42_n440.parquet

| checkpoint | n_scored | n_unique_pairs | n_likert | likert_mean | win_rate_ge5 | pct_7 | judge_accuracy | gen_win_rate | n_nontie | n_tie | n_parse_error |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 9b-train10pct-step72 | 440 | 440 | 440 | 4.8341 | 0.7386 | 2.73 | 0.2442 | 0.7558 | 430 | 10 | 0 |
| 9b-train10pct-step84 | 440 | 440 | 440 | 4.9136 | 0.775 | 3.18 | 0.2215 | 0.7785 | 438 | 2 | 0 |
| 9b-train10pct-step96 | 440 | 440 | 440 | 4.9273 | 0.7773 | 3.41 | 0.212 | 0.788 | 434 | 6 | 0 |
| 9b-train10pct-step108 | 440 | 440 | 439 | 4.9112 | 0.7768 | 1.82 | 0.2033 | 0.7967 | 428 | 11 | 1 |
| 9b-train10pct-step120 | 440 | 440 | 438 | 4.9429 | 0.79 | 4.79 | 0.2009 | 0.7991 | 433 | 5 | 2 |
