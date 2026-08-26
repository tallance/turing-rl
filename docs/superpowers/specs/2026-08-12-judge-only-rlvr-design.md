# Judge-Only RLVR — Training the Discriminator Against a Frozen Pair Set

**Date:** 2026-08-12 · **Status:** design (pre-plan) · **Author:** Tal Lancewicki (Mac agent)

> **Round 1 did not run as specified.** Two defects were found after four runs completed.
> The "Judge thinking: ON" decision below was silently not applied — the agent loop resolved the
> mode from an environment variable no launcher set, so every rollout trained with an empty
> `<think></think>` block. Separately, the format reward made partial JSON near-optimal, so
> all-37-fields was worth 0.025 of total reward. Both fixed in `ef217fa` (2026-08-18); the four
> completed runs are retained as a labelled thinking-OFF ablation.
> See `docs/judge-thinking-off-ablation.md` for mechanism, scope and measurements, and
> `results/2026-08-12-judge-only-rlvr/README.txt` for provenance.

## 1. Objective

Train the **discriminator alone** with GRPO to identify which of two candidate turns was written by
the real human. The generator is frozen at the SFT checkpoint; nothing about it is updated. This is
the judge-side counterpart to the completed generator-only runs, and it precedes both simulation and
alternating training.

**Question.** Does RLVR on verifiable pairwise labels lift a small judge's accuracy above its
zero-shot baseline, and how does that gain scale with model size (2B / 4B / 9B) and reward shape
(directional 0-1 vs. graded rating)?

Success is a defensible accuracy curve over size × reward shape against zero-shot baselines scored
on the same pairs. Producing a judge good enough for the adversarial loop is **not** required here.

The reward is fully local: the label is known by construction, so unlike generator GRPO there is no
judge server in the loop. Generator run 13634 spent 41.5 of 44.1 h (93.9%) waiting on the 397B
judge; that cost does not exist on this side.

## 2. Relation to the existing pipeline

Everything reuses the generator-side plumbing except the reward and the data builder.

| Item | Decision | Notes |
|---|---|---|
| Prompt | `TURING_PROMPT`, full 37-field schema | `shared/judge_prompts.py`; unchanged, so a trained judge is a drop-in for the reward path and eval harness |
| Judge thinking | **ON**, `--reasoning-parser qwen3` | Matches what generator RL actually ran (`PERSONA_JUDGE_ENABLE_THINKING=1`, `rl_generator_run_9b.sh:110`) |
| Rating derivation | Recomputed from the 6 dimension scores + 8 penalties | Same rule as `reward.py`; model-emitted arithmetic is not authoritative |
| Trainer | veRL 0.9, env `turing-rl-rl-qwen35`, FSDP2 | The path both completed 9B GRPO runs (13634, 14217) used |
| LoRA | r=64, α=32, `q/k/v/o/gate/up/down`, exclude `visual\|mtp`, `lora.merge=True` | Qwen3.5 2B/4B/9B are all hybrid Gated-DeltaNet (`full_attention_interval: 4`), so the 9B recipe transfers unchanged. Never LoRA the GDN backbone (arXiv:2604.22127) |
| GRPO hyperparameters | Start from `qwen3_9b_grpo_turing.yaml` | lr 1e-4, `kl_loss_coef` 1e-4, rollout T=1.0/top_p=1.0/top_k=-1, `train_batch_size` 64, `ppo_mini_batch_size` 64 |
| Structured decoding at train | **Not available** | veRL builds `SamplingParams` directly and has no schema field in 0.7.0–0.9.0.dev. Rollouts are unconstrained; format is handled by reward (§5) |
| Structured decoding at eval | `PERSONA_JUDGE_JSON_SCHEMA=1` | The eval harness goes through the OpenAI chat layer where `response_format` applies |

## 3. Data

### 3.1 Deterministic slice

Rows are selected by a pure hash of stable identifiers — no seed, no ordering dependence, no
dependence on how many rows precede a given row:

```
u = int.from_bytes(blake2b(f"{user_id}::{post_id}::{target_idx}".encode(),
                           digest_size=8).digest(), "big") / 2**64
```

Judge iteration 1 takes `u ∈ [0.0, 0.1)`. The remaining `[0.1, 1.0)` stays free so future
alternating iterations can draw disjoint slices without recomputing anything.

Hashing the `(user, post, turn)` triple rather than `user_id` alone is deliberate: users hold
varying row counts, so a 10% user slice would be lumpy, and the corpus is already partitioned by
user one level up (`data/prism/split_data.py` splits SFT / GRPO / heldout by user).

**Source:** `data/prism/full_s42_history_sft40_grpo60_test10/grpo/train.parquet` (4174 rows).
10% is 417; we round **down to 416 contexts** so that 416 × 4 generations × 2 orders = 3328 = 52×64
divides the batch exactly. This follows the `frac10ep10` precedent, which chose 384 = 6×64 over the
round 417 for the same reason. The slice ships as a column plus a manifest recording the row count,
the bounds, and a checksum.

### 3.2 Generation

Fake turns come from **`merged_ep3`** — `checkpoints/sft/qwen35_9b_prism_full_s42_bf16_fsdp_nopack_epochsave/merged_ep3`,
the Qwen3.5-9B SFT checkpoint that the 9B GRPO runs initialize from and that the eval harness calls
"step 0".

Sampling is **T=0.7, top_p=0.8, top_k=20, max_tokens=1024** (`launch_test_eval.sh:70-73`; Qwen3.5
model card = job 13634 `val_kwargs`). This is identical to how the 880 eval pairs were produced, so
training and eval pairs are distribution-matched.

> Note: `eval/generate_trained.py` carries an older per-domain default of prism T=0.6 / top_p=1.0 /
> top_k=-1, used by the 2026-07 sweeps. Two SFT-ep3 pair sets therefore exist on the 880. We use the
> 0.7/0.8/20 one throughout.

**k = 4** generations per context. Because top_p=0.8 / top_k=20 is fairly narrow, the four samples
may be near-duplicates and the augmentation may buy less than the row count suggests. The smoke
measures the exact-duplicate and near-duplicate rate; we raise k or widen sampling only if it is bad.

### 3.3 Row construction

Each `(context, human, generated_j)` pair is emitted in **both A/B orders** — 416 × 4 × 2 = **3328
train rows**. Both orders removes position bias from the training signal and yields
`order_consistency` as a free diagnostic. The `{source_copy_watchlist}` field is built exactly as at
eval time (`build_source_copy_warning`).

The label `y = 1` iff the human turn occupies slot B. By construction `human_is_B_rate = 0.5`; this
is asserted at build time and tracked during training as a sanity metric.

### 3.4 Val and eval

**Val:** `grpo/val.parquet`, **all 352 contexts** — the same file and same rows the generator
validated on. Same users as train, per-user *tail* turns (`_split_train_val_by_user_tail` at
`grpo_val_frac=0.1`). k=1 generation × 2 orders = **704 val rows**. Not hash-sliced: 10% of 352 is
35 contexts, which is roughly ±8% noise on an accuracy estimate.

**Eval:** the frozen **880-pair `step-0` set** — 128 heldout users, disjoint from every GRPO training
user, from `data/prism/full_s42_history_sft40_grpo60_test10/test.parquet`. Already built and already
scored by four judges.

## 4. Baselines

Zero-shot accuracy on the 880 `step-0` pairs (thinking-ON), from
`results/2026-08-03-test-eval-9b-half/`:

| Judge | judge_accuracy | ties /880 |
|---|---:|---:|
| qwen35-9b | 0.514 | 16 |
| qwen35-4b | 0.515 | 19 |
| gemma4-12b | 0.548 | 152 |
| qwen35-27b | 0.604 | 24 |

Those cells ran under `{"type":"json_object"}`; the full ordered schema landed later (commit
813cc4b). **All baselines are re-run under `PERSONA_JUDGE_JSON_SCHEMA=1`** and the re-run is the
published baseline. The in-flight gemma4-31b eval is part of that set. Still to add: zero-shot **2B**
(one `DO_GEN=0` cell) and optionally the 397B anchor.

Expected effect size, for calibration: J1 reports +4.5 to +10.3 accuracy points from RL over
zero-shot judges; TIR-Judge reports +15.9/+10.2 on Qwen3 4B/8B. A lift from ~0.51 to ~0.60 would be
consistent with that literature, not a disappointment.

## 5. Reward

### 5.1 Rating recovery ladder

The task reward is paid whenever a rating is **recoverable**, regardless of format quality. Rungs
reuse the helpers already in `shared/judge_utils.py`:

1. All 6 dimension scores + 8 penalties present → recompute base scores, penalties, `score_gap` →
   derive rating. **Authoritative.**
2. Else model-emitted `score_gap` → `_rating_from_turing_score_gap`
3. Else model-emitted `rating` → `_coerce_turing_rating`, clamped to 1–7
4. Else nothing recoverable → task reward 0

Rung 4 pays nothing because there is no prediction to score, not as a punishment. The rung reached is
logged per sample.

### 5.2 Format score

Four independent 0/1 components, summed and normalized to [0,1]:

- `fmt_json_valid` — a JSON object parses out of the completion
- `fmt_all_37` — all 37 required keys present, no extras
- `fmt_arith` — base scores, penalties, `score_gap` and `rating` all satisfy the prompt's formulas
- `fmt_rating_range` — rating is an integer in 1–7

`fmt_arith` is the load-bearing one: it forces the model to use the rubric rather than emit a rating
and backfill the fields around it.

### 5.3 Task reward arms

Let `r ∈ {1..7}` be the derived rating, `p = (r−1)/6`, and `y = 1` iff the human is in slot B.

**Arm A — directional 0-1.** The rating points at A when `r < 4` and at B when `r > 4`. Reward is
`1` if it points at the side actually holding the human turn, `0` if it points at the other side,
and **`0.5` if `r = 4`** (tie).

**Arm B — graded rating.** `1 − (p − y)²`. Dense credit for being directionally right *and*
confident: with the human in B, rating 7 pays 1.00, 6 pays 0.97, 5 pays 0.89, 4 pays 0.75, and 1 pays
0.00. This is a graded version of arm A over the 7 rating values, not a calibration claim.

### 5.4 Total

```
total = 0.9 · task_reward + 0.1 · format_score
```

Format weight 0.1 follows TIR-Judge (Qwen3 4B/8B, our exact scale). Format is deliberately a *minor*
term because it is a **training-time scaffold only** — at eval every model, trained or baseline, runs
under forced schema decoding, so format is free there and the published comparison is purely on
accuracy.

## 6. Metrics

Emitted through `reward_extra_info`, surfaced via the existing allowlist mechanism in
`training/grpo/verl_metric_patch.py`. Beyond veRL's standard set:

**Reward decomposition** — `task_reward`, `format_score` and each of its four components, `total`.
The user requirement that format and task reward be separately visible is served by logging them as
independent series, never only their sum.

**Accuracy** — `acc` (ties=0.5), `acc_nontie`, `tie_rate`, `brier` = `(p−y)²`, `conf_mean` =
`2·|p−0.5|` (0 at a tie, 1 at rating 1 or 7), rating histogram (fraction at each of 1–7).

**Bias and sanity** — `pred_B_rate`, `human_is_B_rate` (must hold at 0.5), `order_consistency` (the
fraction of pairs whose two orders name the same side).

**Validity** — `schema_valid_rate`, `parse_error_rate`, `arith_consistent_rate`, `recovery_rung`
distribution, `truncation_rate`.

**Group health** — `group_all_equal_rate`, `group_all_correct_rate`, `group_all_wrong_rate`. This is
the DAPO diagnostic and the earliest signal that dynamic sampling is needed (§9).

**Rubric behaviour** — per-side dimension means, i.e. whether the judge separates human from
generated on `immediate_target`, `human_goal`, and `communication_style` differently.

## 7. Phase 0 — zero-shot format probe

Runs **before** any training run is submitted, on ~200 val pairs per model, thinking-ON, for 2B / 4B
/ 9B, in three decoding regimes:

| Arm | Constraint | Measures |
|---|---|---|
| **A** | full 37-field `json_schema` | Truncation rate, tokens used, accuracy. Field count is 37 by construction; the risk is the token cap — the old rating-only schema overran 8192 in 10/16 cases |
| **B** | `{"type":"json_object"}` | How many of the 37 fields the model volunteers when structure is forced but content is not; accuracy |
| **C** | none (freeform) | **The training-rollout regime.** Parse rate, `fmt_all_37` rate, which recovery rung fires, accuracy |

Arm C must send **no `response_format`**. `build_chat_payload` already treats it as optional
(`shared/api_client.py:102`), but `reward.py::_resolve_response_format` currently returns either
`json_schema` or `json_object` and never `None`, so a third mode is added. Probing through the
default path would report near-total compliance and tell us nothing about unconstrained rollouts.

**Gate.** If arm C's `fmt_all_37` rate is around 50% or better, GRPO has enough format signal to
learn from. If a model comes in near zero, we know before spending GPU-days.

Comparing accuracy across A / B / C also tells us whether the format scaffold buys verdict *quality*
or only parseability — worth knowing before trusting the `fmt_all_37` weight.

`scripts/plot_field_compliance.py` already exists and is the starting point for the analysis.

**If the gate fails:** the fallback is a short **self-distilled format SFT** — run the same model
under forced schema decoding on the training slice, harvest its own valid 37-field outputs, and SFT
on them. Self-distillation keeps the content in the model's own voice and ability range, so it is a
pure format lesson rather than a capability transfer, and it needs no extra model served.
**Unfiltered** (not filtered to correct verdicts): filtering would make it rejection-sampling SFT,
which lifts accuracy before RL and muddies what the reward arms are buying. Filtered is recorded as a
separate lever, not the default.

## 8. Runs

**R0 — overfit gate.** 16 pairs, 4B, many epochs. Train accuracy must reach ~1.0. Verifies the whole
loop end to end before anything expensive. Follows the existing `build_overfit10.py` /
`overfit_gate_check.py` pattern.

**R1 — main grid.** {2B, 4B, 9B} × {arm A, arm B} = 6 runs, hyperparameters per §2.

**Baselines.** Per §4.

## 9. Risks

**Tie collapse.** Zero-shot skill is 0.51–0.60. Constant rating-4 pays 0.5 under arm A and 0.75 under
arm B, so hedging is close to competitive in both. Every RL-judge paper surveyed (J1, RM-R1, RRM,
DeepSeek-GRM) prohibits ties for exactly this reason; we keep 0.5 as specified and treat `tie_rate`
and the rating histogram as first-class tracked metrics. Documented fallback if collapse appears:
RLCR-style `0.5·correct + 0.5·(1−brier)`, which pins an accuracy floor under the graded term.

**Zero-variance groups.** A GRPO group whose rollouts all earn the same reward contributes no
gradient. At a uniform per-prompt success rate of 0.55 with G=4 the rate is only ~0.14, but that
assumes uniformity — if difficulty is bimodal (easy pairs always right, hard pairs always wrong) it
approaches 1. No controlled evidence exists that graded rewards alone fix this, and TIR-Judge
explicitly DAPO-filters. We measure `group_all_equal_rate` from the smoke onward; DAPO-style dynamic
sampling is the documented fallback.

**Rollout length.** The judge prompt is roughly 22k characters (~6k tokens) and thinking-ON adds up
to 8192 more, so `max_model_len` needs ~16k against the generator's 12288, on 40GB A100s. Training
completions are capped at 6144 and `truncation_rate` is watched in the smoke.

**Near-duplicate augmentation.** See §3.2.

**Baseline comparability.** Resolved by re-running all baselines under the full schema (§4).

## 10. Deferred

- **Position-consistency reward** (J1): credit only when both A/B orders are correct, both in the
  same batch. The only published training-time treatment of order bias. v2.
- **Qwen3.5-0.8B** as a fourth size point. Exists, Apache-2.0, same architecture; judged not worth
  the slot.
- Alternating / co-training with the generator. Out of scope by definition.

## 11. Open items at plan time

- Exact `max_model_len` and completion cap per model size, pending the Phase 0 token measurements.
- Whether 2B/4B can skip FSDP2 parameter offload (they should fit without it, which would be faster).
- Slurm shapes and the concurrency budget for the six R1 runs.
