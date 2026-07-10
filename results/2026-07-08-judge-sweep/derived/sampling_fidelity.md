# OpenRouter Qwen3 sampling fidelity probe

## thinking-on (n=20)
- completion_tokens: mean=139 min=128 max=166
- sample params echo: "Alibaba"

## thinking-off (n=20)
- completion_tokens: mean=138 min=103 max=154
- sample params echo: "Alibaba"

## DECISION (frozen 2026-07-09)

**Policy:** use each served model's shipped `generation_config.json` defaults; do NOT override
sampling on the wire (vLLM applies generation_config automatically). Leave
`PERSONA_JUDGE_SAMPLING` UNSET for all cells.

**Frozen values** — confirmed identical for `Qwen/Qwen3-8B` and the
`Qwen/Qwen3.5-397B-A17B-GPTQ-Int4` anchor (cluster `cat generation_config.json`):
- temperature = 0.6
- top_p = 0.95
- top_k = 20
- min_p = 0.0  (absent from generation_config → vLLM default)
- do_sample = true

Applied to **both thinking modes and all cells** (judges + anchor + CoT). There is only one
generation_config per model (no separate thinking-off values), so — faithful to the paper's
no-override OpenRouter calls — thinking-off uses the same 0.6/0.95/20 rather than the Qwen
model-card's 0.7/0.8 recommendation. Thinking on/off is controlled solely via
`chat_template_kwargs.enable_thinking`.

**Why not the observed OpenRouter values:** unobservable — the probe echoed only the provider
name ("Alibaba"), not sampling. Since OpenRouter routed to Alibaba serving the same weights,
its defaults ARE these generation_config values; self-hosting vLLM with no override reproduces
the same behavior.

**Tooling implication:** Tasks 14/15 do not need a per-mode sampling table; `PERSONA_JUDGE_SAMPLING`
stays unset. If explicit per-cell recording is wanted, set it to
`{"temperature":0.6,"top_p":0.95,"top_k":20,"min_p":0.0}`.

**Cache note (reconcile at Task 14/17):** cached Qwen models are Qwen3-4B, Qwen3-8B,
Qwen3.5-122B-A10B (+GPTQ-Int4), Qwen3.5-397B-A17B-GPTQ-Int4. The plan's assumed Qwen3.5 sizes
(4B/9B/27B/35B-A3B) and Qwen3 14B/32B are NOT cached — the sweep size list must be reconciled
with what is actually available/downloadable.
