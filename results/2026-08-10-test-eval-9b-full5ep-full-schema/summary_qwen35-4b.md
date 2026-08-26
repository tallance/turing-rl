# split: PASS expect=heldout rows=880 users=128 parquet=/storage/home/lancewicki/projects/turing-rl/data/prism/full_s42_history_sft40_grpo60_test10/test.parquet

| checkpoint | n_scored | n_unique_pairs | n_likert | likert_mean | win_rate_ge5 | pct_7 | judge_accuracy | gen_win_rate | n_nontie | n_tie | n_parse_error |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 9b-full5ep-step0 | 880 | 880 | 880 | 4.0398 | 0.4818 | 2.84 | 0.5006 | 0.4994 | 849 | 31 | 0 |
| 9b-full5ep-step32 | 880 | 880 | 879 | 4.8498 | 0.7679 | 3.64 | 0.2188 | 0.7812 | 864 | 15 | 1 |
| 9b-full5ep-step64 | 880 | 880 | 879 | 4.9613 | 0.7929 | 3.3 | 0.1979 | 0.8021 | 869 | 10 | 1 |
| 9b-full5ep-step96 | 880 | 880 | 880 | 4.9727 | 0.7955 | 2.84 | 0.1926 | 0.8074 | 867 | 13 | 0 |
| 9b-full5ep-step128 | 880 | 880 | 880 | 5.0818 | 0.825 | 4.2 | 0.1607 | 0.8393 | 865 | 15 | 0 |
| 9b-full5ep-step160 | 880 | 880 | 879 | 5.0694 | 0.8271 | 3.64 | 0.1634 | 0.8366 | 869 | 10 | 1 |
| 9b-full5ep-step192 | 880 | 880 | 879 | 5.0614 | 0.8339 | 3.3 | 0.1536 | 0.8464 | 866 | 13 | 1 |
| 9b-full5ep-step224 | 880 | 880 | 880 | 5.0568 | 0.8239 | 3.98 | 0.1657 | 0.8343 | 869 | 11 | 0 |
| 9b-full5ep-step256 | 880 | 880 | 879 | 4.9784 | 0.8009 | 3.41 | 0.188 | 0.812 | 867 | 12 | 1 |
| 9b-full5ep-step288 | 880 | 880 | 880 | 5.0375 | 0.817 | 3.86 | 0.1707 | 0.8293 | 867 | 13 | 0 |
| 9b-full5ep-step320 | 880 | 880 | 880 | 5.033 | 0.8091 | 4.32 | 0.1721 | 0.8279 | 860 | 20 | 0 |
