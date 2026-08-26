# Single-token judge: design

Status: design, approved for planning
Date: 2026-08-26

## Question

Does the Turing judge need hidden thinking and a 37-field JSON verdict, or does a
bare prompt with a **single `A`/`B` output token** work as well? The answer decides
whether the judge protocol changes for all downstream work.

Scope is **measurement only**. Wiring a single-token judge into the generator's
reward path is a separate decision, made after this table exists.

## What is already measured

Both existing tables score the same frozen 880-pair set
(`gen_9b-full5ep-step0_880.parquet`). `accuracy` credits a correct side as 1, a wrong
side as 0, and a `rating == 4` tie as 0.5.

| judge | thinking ON | thinking OFF |
|---|---|---|
| judge-9b-graded-step52 (RL-trained) | 0.7341 | **0.7551** |
| judge-4b-graded-step52 (RL-trained) | 0.7011 | 0.6869 |
| qwen35-27b zero-shot | 0.5977 | 0.5233 |
| qwen35-9b zero-shot | 0.5182 | 0.4477 |
| qwen35-4b zero-shot | 0.5006 | 0.4562 |

Sources: `results/2026-08-12-judge-only-rlvr/judge_eval_880.csv` (thinking on) and
`results/2026-08-19-judge-only-rlvr-thinking-off-eval/judge_eval_880.csv` (both modes,
zero-shot; thinking off, trained).

Thinking-off degrades every zero-shot judge but not the trained ones — the trained 9B
is the best cell in the table with thinking off. That is the motivation for testing
whether the remaining scaffold (rubric text plus structured output) is also
replaceable by training.

## Decisions taken during design

| Decision | Choice | Reason |
|---|---|---|
| Trained-arm training method | Supervised LoRA cross-entropy on the answer token | GRPO over a one-token output has nothing to explore and no format to reward. The labels are already in the pair parquet. The comparison is therefore between two **packages** (thinking + schema + GRPO) vs (single token + CE); it does not decompose into format-vs-method, which is acceptable because the practical question is whether the cheap package suffices. |
| Prompt content | Bare: task, inputs, "Answer with a single letter, A or B" | The rubric was written to structure a *generated* analysis whose fields produce the rating. With no generation it can only act as static prefill conditioning, and it costs ~2-3k prompt tokens per call against a change whose point is cost. |
| Source-copy watchlist | Dropped from the new arm | Follows from the bare prompt. This makes the new arm differ from the old on an **input** as well as an output; recorded here so the difference is not mistaken for an output-format effect. |
| Verdict extraction | `max_tokens=1` + `logprobs`, softmax over A/B token variants | One forward pass, no constrained decoding, and yields a calibrated `p_a` for free. Fallback if tokenizer variant handling proves unreliable: vLLM `guided_choice: ["A","B"]`, accepting that its logprobs may be post-constraint renormalized. |
| Rejected: one-field JSON `{"answer":"A"}` | — | Retains the JSON dependency the experiment exists to remove, spends ~8 output tokens instead of 1, and gives no clean probability. |
| Rejected: full-rubric prompt with single-token output | — | Defends a hypothesis the design does not hold (see prompt row above), and the cell that matters is the trained one, where CE on labelled pairs swamps prompt conditioning. |

## Components

Five changes. No new subsystem.

| # | Location | Change |
|---|---|---|
| 1 | `shared/judge_prompts.py` | Add `TURING_SINGLE_TOKEN_PROMPT` beside `TURING_PROMPT` (literal below). Placeholders `user_history`, `context`, `response_a`, `response_b`; **no** `source_copy_watchlist`. Ends with the single-letter instruction. |
| 2 | `eval/metrics.py` (render+score site, currently line 602) | Branch on `JUDGE_PROMPT_STYLE` (`full` default, `single_token`). Single-token path: `max_completion_tokens=1`, no `response_format`, `logprobs=True`, `top_logprobs=20`, thinking off. |
| 3 | `scripts/build_judge_train_pairs.py` | Add `--prompt-style {full,bare}`. Only the `render_turing_prompt` call changes; slice, both-orders emission, labels and `extra_info` are untouched. |
| 4 | *(no new trainer)* | `training/sft/lora_sft.py:build_chat_template_sft_features` already masks to the assistant target span. With the assistant message set to `"A"` or `"B"` the target is one token. Needs only a pair-parquet → SFT-jsonl converter and a config. |
| 5 | `scripts/launch_judge_eval_matrix.sh` | Add `JUDGE_PROMPT_STYLE`, guarded the same way `THINKING_MODE` is: the style name must appear in `EVAL_ROOT`, and stale output directories are refused. |

`prompt_mode` / `TURING_PROMPT_MODE` in `training/grpo/turing_launcher_config.py` refer
to the **generator's** prompt, not the judge's. Do not overload them; `JUDGE_PROMPT_STYLE`
is a separate name on purpose.

### The bare prompt

Exact template, so implementation is not guessing at "bare":

```
One of the two candidate responses below was written by the real [HUMAN] user.
The other was written by an AI imitating that user.

## User History

<|User History|>
{user_history}
<|End User History|>

## Context

<|Context|>
{context}
<|End Context|>

## Response A

<|Response A|>
{response_a}
<|End Response A|>

## Response B

<|Response B|>
{response_b}
<|End Response B|>

Which response was written by the real [HUMAN]? Answer with a single letter, A or B,
and nothing else.
```

The input delimiters are kept identical to `TURING_PROMPT` so the only differences from
the existing arm are the removed rubric, the removed watchlist, and the removed output
schema — not the way the inputs are presented.

### Template-parity hazard

`lora_sft.py` strips the Qwen empty-think prefill. The chat template's `enable_thinking`
must therefore be pinned identically at training-render time and at eval-serve time. If
they differ, the supervised target sits at a different position from the one the judge
decodes at, and the trained arm is silently wrong. Enforced by an equality assertion
between the two renderers (see Tests) and caught in practice by the overfit gate.

## Eval matrix

One frozen pair set for everything:
`/home/lancewicki/projects/turing-rl/results/2026-08-10-test-eval-9b-full5ep-full-schema/raw/pairs/gen_9b-full5ep-step0_880.parquet`
(SHA-256 `95f48a9c52d85a6f6c49fd3387e60efe0e1ee5e436bd961f1884750ecfcf7783`).

**Reused, not re-run:** the cells already in the two CSVs above — five zero-shot models ×
thinking on/off (10), plus the GRPO-trained 4B/9B graded judges × thinking on/off (4).
The directional-arm rows present in those CSVs are carried through unchanged but are not
part of this comparison; the graded arms are the reference.

**New — one `single_token` column** (thinking off, bare prompt, one output token):

| cell | model | kind |
|---|---|---|
| `qwen35-4b-st` | Qwen/Qwen3.5-4B | zero-shot |
| `qwen35-9b-st` | Qwen/Qwen3.5-9B | zero-shot |
| `qwen35-27b-st` | Qwen/Qwen3.5-27B | zero-shot |
| `gemma4-12b-st` | google/gemma-4-12B-it | zero-shot |
| `gemma4-31b-st` | google/gemma-4-31B-it | zero-shot |
| `judge-4b-ce-st` | Qwen3.5-4B + LoRA CE | trained |
| `judge-9b-ce-st` | Qwen3.5-9B + LoRA CE | trained |

All five zero-shot models are kept so the new column is not ragged against the existing
table; at one output token their marginal cost is prefill only.

### Fairness control on the trained cells

The CE cells train on exactly the pair set the GRPO-trained judges used — same
`build_judge_train_pairs` slice, same both-orders emission, same labels — with only
`--prompt-style bare` changed. A val slice is held out of that training set for early
stopping. The 880 eval pairs come from held-out test users (`data/judge/slice.py` plus
the existing split guard), so they do not enter training.

CE has no step-count correspondence to GRPO's 52 steps. "Trained" means trained to best
val accuracy, with the val curve recorded. No arbitrary step count is pinned for the
sake of symmetry.

## Metrics

The headline `accuracy` definition is unchanged (1 / 0 / 0.5 for a tie), so the new
cells drop into the existing table. A tie scores what a forced coin-flip scores, so a
tie-free protocol remains comparable to tied ones. `tie_rate` is structurally 0 for
single-token cells.

New columns, all derived from `p_a`:

- `p_a` — renormalized probability over the A/B token variants (per row).
- `brier`, `auc` — calibration and ranking quality.
- `a_rate` — fraction of calls answering "A".
- `order_consistency` — fraction of `pair_id`s whose two presentation orders select the
  same underlying response.
- `hard_fail` — neither A nor B among the top-20 logprobs.

`a_rate` and `order_consistency` are not optional. A judge that always answers "A"
scores accuracy **exactly 0.5**, which is indistinguishable in the headline number from
genuine uncertainty but is a different object; it is identified by `a_rate` ≈ 1.0 and
`order_consistency` ≈ 0.

### Interval and comparison statistics

The existing `se` column is unpaired `sqrt(p(1-p)/n)` at n=880, but 880 is 440 pairs ×
2 orders and the two orders of one pair are not independent — effective n is nearer 440,
so that SE is optimistic by roughly √2. Add a bootstrap CI **clustered on `pair_id`** as
a new column; leave `se` untouched, since those CSVs are published artifacts.

Arm-vs-arm comparisons use a paired test (McNemar on the shared rows), which is more
powerful than comparing marginal CIs because the paired difference cancels shared pair
difficulty.

## Decision rule

Fixed before the run. Reference cell: `judge-9b-graded-step52`, thinking off, **0.7551**.

- `judge-9b-ce-st` ≥ 0.7551 − 0.02 on the paired test → **switch protocol.**
- Below that margin → **keep the schema**; the structured output is doing real work.
- Any cell with `a_rate` outside [0.3, 0.7] or `order_consistency` < 0.3 is reported as
  **degenerate** regardless of accuracy and does not count as a pass.

The 2-point non-inferiority margin follows from the cost ratio: ~1 output token against
up to 8192, and a trained judge costing minutes of LoRA CE on one GPU instead of an
8-GPU GRPO job.

`judge-4b-ce-st` is measured and reported against its own reference
(`judge-4b-graded-step52`, thinking off, 0.6869) but does not gate the decision; the 9B
cell does, because the 9B graded judge is the protocol currently in use.

## Error handling

1. **Hard fail** — recorded per row, never coin-flipped into accuracy. A cell with
   `hard_fail` > 1% is reported as failed: such failures concentrate on the longest
   inputs, so they bias rather than merely thin the result.
2. **Tokenizer variants** — `A` vs `▁A`, and Qwen and Gemma tokenize differently. Variant
   collection is per-tokenizer, with a test for each family in the matrix.
3. **Missing logprobs** — a one-prompt preflight at cell start fails the cell rather than
   scoring 880 pairs on junk.
4. **Template mismatch** — asserted equality between the training-time and eval-time
   rendered prompt for the same inputs.
5. **Wrong output directory** — `JUDGE_PROMPT_STYLE` must appear in `EVAL_ROOT`; stale
   output directories are refused. Prevents a single-token run being written into a
   full-schema results tree.
6. **Empty `user_history`** — existing behaviour retained: raise rather than score.

## Tests

Local, no GPU:

- Bare prompt renders all four placeholders, contains none of `Output Format`,
  `score_gap`, `rating`, and ends with the single-letter instruction.
- `p_a` extraction from synthetic logprob payloads: leading-space variants, both orders,
  the hard-fail path, and argmax agreeing with `p_a > 0.5`.
- Metrics on a hand-built case **including the always-A degenerate**, asserting accuracy
  0.5, `a_rate` 1.0, `order_consistency` 0.0.
- `--prompt-style bare` preserves row count, labels, both orders and ids; only the
  `prompt` content differs from `full`.
- Renderer parity: training-time render equals eval-time render for identical inputs.

GPU gates, both before the 7-cell matrix:

- **Overfit gate** — CE on ~16 pairs reaches ~1.0 train accuracy (mirrors
  `scripts/judge_overfit_gate.py`). Proves the label is wired to the correct token
  position; if it fails, every downstream number is meaningless.
- **1-cell smoke** — 20 pairs through zero-shot `qwen35-4b-st`: `hard_fail` == 0,
  logprobs present, both letters observed.

## Artifacts

Results package at `results/2026-08-26-single-token-judge/` with a `README.txt` carrying
provenance only — configuration and versions, job IDs and dates, cluster source paths,
artifact filenames and checksums, mechanical validation status, reproduction commands.
No interpretation of the numbers.

Contents: the merged comparison CSV across all arms (existing cells reused verbatim, new
cells appended), the accuracy plot, and per-cell timing from
`scripts/summarize_eval_timings.py` — which gives tokens-per-call and throughput for
free, without adding a separate throughput experiment.

## Out of scope

- Wiring a single-token judge into the generator's GRPO reward path (`REWARD_METRIC`).
  That decision follows this measurement.
- Reward-shape ablations (bounded vs log-based), which belong to the adversarial design.
- Any change to the existing full-schema judge path, which remains the default.
