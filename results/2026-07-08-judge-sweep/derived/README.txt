Judge-sweep derived results
===========================

Plan:  docs/superpowers/plans/2026-07-08-judge-sweep-implementation.md
Spec:  docs/superpowers/specs/2026-07-08-judge-sweep-design.md
Decisions/deviations: docs/superpowers/post-plans/

WHAT'S HERE
  summary.md / summary.parquet   per-cell metrics: accuracy (picks true human,
                                 ties excluded), tie_rate, format_ok, budget_hit,
                                 position_bias, rating_mean, kappa_vs_anchor.
  per_pair_metrics.parquet       per-pair picked_human, all cells/modes.
  plots/accuracy_bar.png         grouped bar: accuracy per model, thinking off vs on
                                 (all models incl. anchor treated equally).
  plots/{accuracy,format_ok_rate,budget_hit_rate,position_bias_delta}.png
                                 metric-vs-size line plots (off vs on).
  family_decision.md             Task-17 family gate (-> qwen3.5).
  sampling_fidelity.md           Task-1 sampling policy.

INPUT DATA
  raw/sweep/<cell>/<mode>/reward/*.jsonl   one row per pair = one randomized-order
  judge call over the frozen 880-pair set raw/pairs/prism_heldout_880.parquet.
  Cells: qwen35-{4b,9b,27b,35b-a3b,122b,397b} + qwen3-8b, modes off/on.
  (fam_*/ and *-fp8/ dirs are Task-17 smoke / failed fp8 experiments; the analyzer
  skips any dir not in configs.judge_sweep_cells.SIZE_MAP.)

REPRODUCE (from repo root; cluster, env turing-rl-train)
  # 1. full sweep already produced raw/sweep/... via:
  #    FAMILY=qwen3.5 bash scripts/launch_judge_sweep.sh        (+ direct sbatch for
  #    qwen3-8b and the 122b/anchor cells; see .git/sdd-progress.md for job ids)
  # 2. analyze:
  python scripts/analyze_judge_sweep.py
  # 3. accuracy bar chart:
  python scripts/plot_accuracy_bar.py

NOTES
  - accuracy is vs ONE stochastic generator draw (T=0.6, 1 sample/pair); judge-vs-
    judge gaps <5pp are within generator sampling noise.
  - anchor = Qwen3.5-397B-A17B-GPTQ-Int4; kappa_vs_anchor is Cohen's kappa on the
    order-invariant picked_human decision, same thinking-mode, shared pairs.
