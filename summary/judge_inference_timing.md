# Judge inference timing — Qwen3.5 candidates on RFAI AWS

Smoke-tested three self-hosted Qwen3.5 judges on a single A100-SXM4-40GB node, to pick a replacement for the paper's `qwen/qwen3.5-397b-a17b` OpenRouter call.

## Test setup
- **Hardware**: 1× node, 8× A100-SXM4-40GB (CUDA 13.0, driver 580.126.09)
- **Server**: vLLM 0.23.0 (env `judge-vllm`), OpenAI-compatible endpoint
- **vLLM args (final)**: `--dtype bfloat16 --max-model-len 8192 --gpu-memory-utilization 0.85` (no `--enforce-eager`, no explicit `--quantization`)
- **Workload**: prompt ~253 input tokens, `max_completion_tokens=512`, 64 concurrent calls via `aiohttp`
- **Smoke client**: `scripts/smoke_judge.py`

## Results

| Judge model | Quant | GPUs (TP) | Throughput | p50 latency | p95 | Tok/sec | Output sanity |
|---|---|---|---|---|---|---|---|
| Qwen3.5-397B-A17B | Int4 GPTQ | 8 (TP=8) | **4.58 req/s** | 13.9s | 14.0s | 2,345 | ✅ coherent reasoning |
| Qwen3.5-122B-A10B | Int4 GPTQ | 4 (TP=4) | **5.37 req/s** | 11.9s | 11.9s | 2,751 | ✅ coherent reasoning |
| Qwen3.5-122B-A10B | bf16     | 8 (TP=8) | **3.77 req/s** | 13.4s | 17.0s | 1,930 | ✅ coherent reasoning |

All three returned `200 OK` for all 64 concurrent calls.

## Notes on output format

The smoke prompt asks for JSON but doesn't pass `response_format={"type":"json_object"}`, so all three models emit a "Thinking Process:" preamble before the JSON. `json_parse_failed` in the smoke logs is a smoke-test artifact, not a model problem. The real Turing-RL training judge prompt (in `shared/judge_prompts.py`) does pass `response_format`, so this is moot in production.

## Implications for GRPO training

GRPO step requires ~256 judge calls (64 prompts × 4 rollouts). At ~225 steps × 3 epochs = ~675 steps, the per-judge wall-time totals are:

| Judge | Time per step on judge | Total over full run |
|---|---|---|
| 397B-Int4 | ~56s | ~10.5 hr |
| 122B-Int4 | ~48s | ~9.0 hr |
| 122B-bf16 | ~68s | ~12.7 hr |

(Note: these are upper bounds — actual call counts to the judge may be smaller because the smoke generates 512 output tokens vs ~150 in the real schema-constrained JSON case.)

In practice, judge calls overlap with training rollouts via async queue, so the *added wall time* is much less. None of these would bottleneck training.


## Important fix that took us a while to find

Initial smoke runs returned `!!!!!!!!!!!...` garbage. Cause: passed `--dtype float16` + `--enforce-eager` + explicit `--quantization gptq`. Fix:
- Use `--dtype bfloat16` for GPTQ Int4 + Qwen MoE
- Don't pass `--enforce-eager` — let CUDA graphs run
- Don't pass explicit `--quantization` — vLLM auto-detects from model config

## Source data

- `logs/judge_smoke-8671.out` — 397B-Int4 final smoke
- `logs/122b_smoke-8677.out` — 122B-Int4 smoke
- `logs/122b_smoke-8687.out` — 122B-bf16 smoke (with `--max-num-seqs 240` to avoid Mamba cache OOM)
- `scripts/slurm/judge_smoke.sh` — 397B sbatch
- `scripts/slurm/122b_smoke.sh` — 122B sbatch (parameterized: `MODEL`, `TP`, `MAX_NUM_SEQS`)
- `scripts/smoke_judge.py` — async client
