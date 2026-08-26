# Judge-family decision (Task 17) — **FAMILY = `qwen3.5`**

**Date:** 2026-07-13 · **Decider:** user, on the 4B family smoke below.

## Decision

The full judge sweep uses the **`qwen3.5`** family. The anchor is unchanged
(`Qwen/Qwen3.5-397B-A17B-GPTQ-Int4`) regardless of family.

Cell list (`configs/judge_sweep_cells.py`, `cell_list("qwen3.5")`):

| cell_name | model_id | size_b | arch | TP × replicas |
|---|---|---|---|---|
| qwen35-4b | Qwen/Qwen3.5-4B | 4 | Qwen3-Next (hybrid Mamba/GDN) | 1 × 8 |
| qwen35-9b | Qwen/Qwen3.5-9B | 9 | Qwen3-Next (hybrid) | 1 × 8 |
| qwen35-27b | Qwen/Qwen3.5-27B | 27 | Qwen3-Next (hybrid) | 2 × 4 |
| qwen35-35b-a3b | Qwen/Qwen3.5-35B-A3B-GPTQ-Int4 | 35 (3 active) | MoE-Int4 | 1 × 8 |
| qwen35-397b (anchor) | Qwen/Qwen3.5-397B-A17B-GPTQ-Int4 | 17 active | MoE-Int4 | 8 × 1 |

## Evidence — 4B family smoke (50 pairs, both thinking modes)

Two candidate 4B judges scored the same first 50 pairs of the frozen
`prism_heldout_880.parquet`, TP=1/REPLICAS=1 (single-GPU throughput =
apples-to-apples), concurrency 16, dedicated A100-40GB. Jobs 9650–9653.

**Throughput + parse (favors qwen3):**

| Cell | Mode | Wall (s) | vs Qwen3-4B | Parse (ok/err) |
|---|---|---|---|---|
| Qwen3-4B | off | 93.5 | 1.0× | 50/0 |
| Qwen3-4B | on | 191.4 | 1.0× | 50/0 |
| Qwen3.5-4B | off | 227.0 | 2.4× slower | 50/0 |
| Qwen3.5-4B | on | 562.1 | 2.9× slower | 50/0 |

**Judge quality — directional accuracy vs the true human side (favors qwen3.5):**

| Cell | Mode | frac 4-ties | non-tie n | dir. acc (excl. ties) |
|---|---|---|---|---|
| Qwen3-4B | off | 44% | 28 | 71.4% |
| Qwen3-4B | on | 28% | 36 | 72.2% |
| Qwen3.5-4B | off | 12% | 44 | 75.0% |
| Qwen3.5-4B | on | **2%** | 49 | **81.6%** |

("Directional accuracy" = fraction of non-tie calls where the judge picked the
actual human, `pick=A if rating≤3, B if ≥5`; ties `rating==4` excluded from the
accuracy but reported as `frac 4-ties`.)

## Rationale

The plan's primary criterion was tokens/sec at parse-rate parity, with
397B-agreement as tiebreak. Parse rate is at parity (100% both). Throughput
favors `qwen3` (2.4–2.9×), but **judge quality favors `qwen3.5` decisively on
two axes**:

1. **Accuracy:** Qwen3.5-4B is a better Turing discriminator (75.0/81.6% vs
   71.4/72.2%).
2. **Decisiveness:** Qwen3.5-4B returns rating-4 ties only 12%/2% of the time vs
   Qwen3-4B's 44%/28%. A judge that ties ~half the time gives almost no reward
   signal — critical for the downstream adversarial-GRPO goal, where the judge is
   the reward. Thinking-on Qwen3.5-4B is essentially never undecided (2% ties).

The user chose **quality over speed**: a better, more decisive judge is the
better starting point. Speed is a solvable feasibility concern (below); a weak,
hedging judge is not.

## Caveats

- **Decided on ground-truth accuracy + decisiveness, NOT 397B-anchor agreement.**
  The sweep's headline metric is κ vs the 397B judge; the anchor was **not** run
  on these 50 pairs, so anchor-agreement (the plan's nominal tiebreak) is not yet
  measured. Accuracy-vs-human is a strong proxy and the anchor is itself Qwen3.5,
  but the real κ is measured in the full sweep (Task 19/21).
- **Small n** (50 pairs; qwen3-4b/off rests on only 28 non-tie calls) — treat
  single-digit-% gaps as noisy.

## Implications for downstream tasks

- **Architecture discovery:** the small Qwen3.5 dense models (4B/9B/27B) are
  **Qwen3-Next hybrids** (Mamba/GDN linear attention, `qwen3_next.py`), a
  different architecture from the MoE 397B anchor and the 35B-A3B MoE cell. So the
  `qwen3.5` size axis is *not* architecturally homogeneous — the analyzer's
  size→agreement trend mixes a hybrid-attention regime (4B/9B/27B) with MoE
  (35B/397B). Note this when interpreting Task 21 plots.
- **Wall-time gate is now load-bearing (Task 18).** Qwen3-Next is 2.4–2.9× slower
  at 4B (thinking-on 4B = 562 s / 50 pairs). The larger hybrid cells (27B) and
  thinking-on cells could threaten the >4h full-sweep gate. **Run the CALIBRATION=1
  50-pair pass per cell first** and check the projected full-880 wall before
  launching the full sweep.
- **Serving env:** serve the Qwen3-Next cells from `turing-rl-train`
  (vLLM 0.18.0 supports the arch); the 397B anchor stays on `judge-vllm`. Both are
  already wired in `scripts/slurm/judge_sweep_cell.sh`.

## Reproduction

Models cached under `/home/lancewicki/data/hf_cache` (Qwen3.5-4B downloaded
2026-07-13). Smoke (from repo root on the cluster), per candidate × mode:

```bash
sbatch --gres=gpu:1 --mem=96G --cpus-per-task=16 --time=1:30:00 \
  --job-name=fam_<name>_<mode> \
  --export=ALL,MODEL=<model_id>,TP=1,REPLICAS=1,THINKING_MODE=<off|on>,CELL_NAME=fam_<name>,MAX_PAIRS=50,PORT_BASE=<unique> \
  scripts/slurm/judge_sweep_cell.sh
```

Dumps: `results/2026-07-08-judge-sweep/raw/sweep/fam_{qwen3-4b,qwen35-4b}/{off,on}/{reward,http}/`.
Throughput from each `run_metadata.json` (`wall_seconds`); accuracy computed from
the reward dumps (`rating_gt_first`/`human_side`).
