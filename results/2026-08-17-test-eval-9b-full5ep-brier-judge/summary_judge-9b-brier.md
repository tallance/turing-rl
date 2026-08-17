# split: PASS expect=heldout rows=880 users=128 parquet=/storage/home/lancewicki/projects/turing-rl/data/prism/full_s42_history_sft40_grpo60_test10/test.parquet

| checkpoint | n_scored | n_unique_pairs | n_likert | likert_mean | win_rate_ge5 | pct_7 | judge_accuracy | gen_win_rate | n_nontie | n_tie | n_parse_error |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 9b-full5ep-step0 | 880 | 880 | 877 | 2.5941 | 0.2543 | 24.86 | 0.7404 | 0.2596 | 859 | 18 | 3 |
| 9b-full5ep-step64 | 880 | 880 | 875 | 4.9829 | 0.664 | 66.17 | 0.3352 | 0.6648 | 874 | 1 | 5 |
| 9b-full5ep-step128 | 880 | 880 | 876 | 5.1324 | 0.6884 | 68.84 | 0.3116 | 0.6884 | 876 | 0 | 4 |
| 9b-full5ep-step192 | 880 | 880 | 878 | 5.2779 | 0.713 | 70.84 | 0.2854 | 0.7146 | 876 | 2 | 2 |
| 9b-full5ep-step256 | 880 | 880 | 877 | 5.293 | 0.7127 | 71.04 | 0.2824 | 0.7176 | 871 | 6 | 3 |
| 9b-full5ep-step320 | 880 | 880 | 877 | 5.8221 | 0.8039 | 80.27 | 0.1961 | 0.8039 | 877 | 0 | 3 |
