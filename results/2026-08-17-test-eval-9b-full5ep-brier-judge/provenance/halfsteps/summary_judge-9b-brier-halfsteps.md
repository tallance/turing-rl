# split: PASS expect=heldout rows=880 users=128 parquet=/storage/home/lancewicki/projects/turing-rl/data/prism/full_s42_history_sft40_grpo60_test10/test.parquet

| checkpoint | n_scored | n_unique_pairs | n_likert | likert_mean | win_rate_ge5 | pct_7 | judge_accuracy | gen_win_rate | n_nontie | n_tie | n_parse_error |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 9b-full5ep-step0 | 880 | 880 | 877 | 2.5941 | 0.2543 | 24.86 | 0.7404 | 0.2596 | 859 | 18 | 3 |
| 9b-full5ep-step32 | 880 | 880 | 879 | 3.8714 | 0.4778 | 46.76 | 0.5205 | 0.4795 | 876 | 3 | 1 |
| 9b-full5ep-step96 | 880 | 880 | 880 | 5.375 | 0.7284 | 72.61 | 0.2708 | 0.7292 | 879 | 1 | 0 |
| 9b-full5ep-step160 | 880 | 880 | 879 | 5.5199 | 0.752 | 74.97 | 0.2454 | 0.7546 | 876 | 3 | 1 |
| 9b-full5ep-step224 | 880 | 880 | 879 | 4.8134 | 0.6371 | 63.03 | 0.3622 | 0.6378 | 878 | 1 | 1 |
| 9b-full5ep-step288 | 880 | 880 | 878 | 5.3041 | 0.7175 | 71.64 | 0.2825 | 0.7175 | 878 | 0 | 2 |
