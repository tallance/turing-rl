# split: PASS expect=heldout rows=440 users=123 parquet=/storage/home/lancewicki/projects/turing-rl/data/prism/full_s42_history_sft40_grpo60_test10/eval_subsets/test_seed42_n440.parquet

| checkpoint | n_scored | n_unique_pairs | n_likert | likert_mean | win_rate_ge5 | pct_7 | judge_accuracy | gen_win_rate | n_nontie | n_tie | n_parse_error |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 9b-gemma12btrain-step0 | 440 | 440 | 438 | 3.9749 | 0.4726 | 1.37 | 0.5152 | 0.4848 | 427 | 11 | 2 |
| 9b-gemma12btrain-step12 | 440 | 440 | 436 | 4.2408 | 0.5528 | 1.83 | 0.4343 | 0.5657 | 426 | 10 | 4 |
| 9b-gemma12btrain-step24 | 440 | 440 | 435 | 4.5195 | 0.6483 | 2.99 | 0.3318 | 0.6682 | 422 | 13 | 5 |
| 9b-gemma12btrain-step36 | 440 | 440 | 438 | 4.6416 | 0.7009 | 2.51 | 0.286 | 0.714 | 430 | 8 | 2 |
| 9b-gemma12btrain-step48 | 440 | 440 | 439 | 4.7335 | 0.7335 | 2.73 | 0.2529 | 0.7471 | 431 | 8 | 1 |
| 9b-gemma12btrain-step60 | 440 | 440 | 440 | 4.8955 | 0.7841 | 3.64 | 0.2014 | 0.7986 | 432 | 8 | 0 |
| 9b-gemma12btrain-step72 | 440 | 440 | 439 | 4.738 | 0.7198 | 3.42 | 0.2736 | 0.7264 | 435 | 4 | 1 |
| 9b-gemma12btrain-step84 | 440 | 440 | 440 | 4.7136 | 0.7568 | 2.27 | 0.2432 | 0.7568 | 440 | 0 | 0 |
| 9b-gemma12btrain-step96 | 440 | 440 | 436 | 4.6514 | 0.7294 | 2.98 | 0.2605 | 0.7395 | 430 | 6 | 4 |
| 9b-gemma12btrain-step108 | 440 | 440 | 437 | 4.897 | 0.8032 | 3.2 | 0.1875 | 0.8125 | 432 | 5 | 3 |
| 9b-gemma12btrain-step120 | 440 | 440 | 438 | 5.0251 | 0.8333 | 4.11 | 0.1551 | 0.8449 | 432 | 6 | 2 |
