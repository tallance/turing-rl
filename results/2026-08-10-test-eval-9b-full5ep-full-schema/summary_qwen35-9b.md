# split: PASS expect=heldout rows=880 users=128 parquet=/storage/home/lancewicki/projects/turing-rl/data/prism/full_s42_history_sft40_grpo60_test10/test.parquet

| checkpoint | n_scored | n_unique_pairs | n_likert | likert_mean | win_rate_ge5 | pct_7 | judge_accuracy | gen_win_rate | n_nontie | n_tie | n_parse_error |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 9b-full5ep-step0 | 880 | 880 | 870 | 3.9828 | 0.4678 | 1.61 | 0.5178 | 0.4822 | 844 | 26 | 10 |
| 9b-full5ep-step32 | 880 | 880 | 878 | 4.8132 | 0.7597 | 2.39 | 0.2244 | 0.7756 | 860 | 18 | 2 |
| 9b-full5ep-step64 | 880 | 880 | 875 | 5.0194 | 0.8286 | 3.77 | 0.1589 | 0.8411 | 862 | 13 | 5 |
| 9b-full5ep-step96 | 880 | 880 | 877 | 4.9943 | 0.8119 | 2.74 | 0.1759 | 0.8241 | 864 | 13 | 3 |
| 9b-full5ep-step128 | 880 | 880 | 875 | 5.0217 | 0.8251 | 2.97 | 0.1624 | 0.8376 | 862 | 13 | 5 |
| 9b-full5ep-step160 | 880 | 880 | 880 | 5.0841 | 0.85 | 3.52 | 0.1402 | 0.8598 | 870 | 10 | 0 |
| 9b-full5ep-step192 | 880 | 880 | 879 | 4.9522 | 0.818 | 3.3 | 0.1707 | 0.8293 | 867 | 12 | 1 |
| 9b-full5ep-step224 | 880 | 880 | 879 | 4.8999 | 0.7929 | 3.07 | 0.1933 | 0.8067 | 864 | 15 | 1 |
| 9b-full5ep-step256 | 880 | 880 | 876 | 4.7991 | 0.766 | 2.51 | 0.227 | 0.773 | 868 | 8 | 4 |
| 9b-full5ep-step288 | 880 | 880 | 878 | 4.9442 | 0.7984 | 2.96 | 0.1877 | 0.8123 | 863 | 15 | 2 |
| 9b-full5ep-step320 | 880 | 880 | 880 | 4.9659 | 0.8091 | 3.07 | 0.1788 | 0.8212 | 867 | 13 | 0 |
