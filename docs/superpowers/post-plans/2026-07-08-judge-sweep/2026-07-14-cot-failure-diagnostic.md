# Post-plan: thinking-ON CoT parse-failure diagnostic + 397B speed follow-up

Plan: `~/.claude/plans/smooth-strolling-lake.md`. Context: judge-sweep parse errors are
concentrated in thinking-ON on the slow big models; mechanism is a client **timeout**
(finish=None), not token-budget truncation. This post-plan records what we ran to
characterize the failure mode and to explore speeding up the 397B judge.

## What we found before running anything
- Raw `<think>` CoT is already dumped for **successful** calls at
  `raw/sweep/<cell>/on/http/*.jsonl` → `response.choices[0].message.reasoning` (vLLM 0.23
  key is `reasoning`, not `reasoning_content`). Failing calls timed out → no http row.
- `think_vs_answer_len.png`: two answer-side failure modes visible in existing dumps —
  (a) **empty answer** when thinking approaches the ~30k-char (8192-token) budget
  (thinking starves the answer); (b) **runaway repetition in the answer** (63 calls
  >10k chars, all finish=length, worst 3.5-4B at 874k chars).
- `generator_len_vs_judge.png`: a **verbose generator response does NOT cause parse
  failures** — corr(gen_len, judge_completion_tokens)=−0.65 (397B): long candidates are
  obvious fakes → judge decides fast. Failures skew to **short, human-like, hard** inputs
  where the judge deliberates longest → most likely to exceed the client timeout.

## Runs launched (all 397B, thinking ON, full 880, CONCURRENCY=8, timeout 1800s)
Injection: `PERSONA_JUDGE_SAMPLING` (JSON) is read by `reward.py:_openai_chat` and merged
top-level into the vLLM request via `build_chat_payload(sampling=...)`. `cell_env` does NOT
emit it but `os.environ.update` doesn't clear an exported value, so `--export=ALL` + a shell
env var works with **no code change**.

| job  | cell_name             | change                                   | hypothesis / purpose |
|------|-----------------------|------------------------------------------|----------------------|
| 9804 | `qwen35-397b-diag`    | replay 98 failing + 102 control, 1800s   | capture timed-out pairs' raw CoT (loop?) |
| 9805 | `qwen35-9b-diag`      | replay 44 failing + 156 control, 1800s   | cheap proxy (COMPLETED, 200/200) |
| 9824 | `qwen35-397b-t07`     | `{"temperature":0.7}` (model-card)        | does model-card temp change fail rate / accuracy? |
| 9825 | `qwen35-397b-reppen`  | `{"repetition_penalty":1.1}`              | if failures are loops, penalty should cut runaway/empty answers |
| 9826 | `qwen35-397b-specdec` | ngram speculative decoding (see below)    | speed follow-up (unrelated to parse failures) |

Baseline for comparison = existing `qwen35-397b/on` dumps. All 397B cells: full 880,
thinking ON, CONCURRENCY=8, timeout 1800s. Early signal: `reppen` scores noticeably faster
than `t07`, consistent with the penalty shortening runaway generations.

## Speed follow-up: speculative decoding (UNRELATED to the parse-failure issue)
Goal: quantify how much a server-side speculative-decoding config speeds up the 397B judge
and whether it costs any quality. Added an optional `SPEC_DECODE` hook to
`scripts/slurm/judge_sweep_cell.sh` (`--speculative-config <JSON>`, mirrors the QZ/RP/AR
pattern). vLLM 0.23 (judge-vllm) supports ngram speculative decoding — **no draft model**,
and it helps most exactly when the output repeats or copies the prompt (this judge does
both: it echoes the rubric structure and, on failures, loops).

- Run (job 9826): `CELL_NAME=qwen35-397b-specdec`,
  `SPEC_DECODE={"method":"ngram","num_speculative_tokens":5,"prompt_lookup_max":4,"prompt_lookup_min":2}`,
  full 880, thinking ON, same CONCURRENCY/timeout as baseline.
- **Time saved:** compare per-call `judge_latency_ms` / `completion_tokens` (tok/s) from the
  specdec dumps vs the baseline `qwen35-397b/on` dumps on the **same pairs** (matched by
  user_id/post_id/target_idx) — concurrency-independent.
- **Performance drop:** compare parse-error rate, accuracy, and rating distribution vs
  baseline. Note vLLM speculative decoding is **distribution-preserving** (rejection
  sampling), so the expected quality delta is ≈ sampling noise; the win is pure latency.
- Follow-ups if ngram underwhelms: `method:"suffix"` (suffix-decoding, better for
  self-repetition) or an EAGLE/draft-model config.

## Deviations
- D12: sampling overrides (temperature, repetition_penalty) injected via
  `PERSONA_JUDGE_SAMPLING` env for these diagnostic cells only — the frozen sweep policy
  (Task-1, no wire override) is unchanged for the main cells.
- D13: a fork agent autonomously committed a throughput-probe harness (commit 2076387) and
  launched an unauthorized job (9823); both reverted/cancelled. Speed work is done here
  instead, through the existing scoring path so dumps are comparable.
