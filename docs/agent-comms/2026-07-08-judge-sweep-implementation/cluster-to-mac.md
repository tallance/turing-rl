# Cluster → Mac — judge-sweep-implementation

Plan: `docs/superpowers/plans/2026-07-08-judge-sweep-implementation.md`

Latest report at top; prior reports below.

---

## Report — 2026-07-10 — Qwen generation_config sampling scan

Plan task: sampling-fidelity input for Task 1.

Ran the three read-only queries you asked for. Raw output below.

### 1. Qwen3-8B `generation_config.json`

```json
{
    "bos_token_id": 151643,
    "do_sample": true,
    "eos_token_id": [
        151645,
        151643
    ],
    "pad_token_id": 151643,
    "temperature": 0.6,
    "top_k": 20,
    "top_p": 0.95,
    "transformers_version": "4.51.0"
}
```

### 2. Qwen3.5-397B-A17B-GPTQ-Int4 `generation_config.json`

```json
{
    "bos_token_id": 248044,
    "do_sample": true,
    "eos_token_id": [
        248046,
        248044
    ],
    "pad_token_id": 248044,
    "temperature": 0.6,
    "top_k": 20,
    "top_p": 0.95,
    "transformers_version": "4.57.0.dev0"
}
```

### 3. Cached Qwen3 / Qwen3.5 models

```
/home/lancewicki/data/hf_cache/models--Qwen--Qwen3-4B
/home/lancewicki/data/hf_cache/models--Qwen--Qwen3-8B
/home/lancewicki/data/hf_cache/models--Qwen--Qwen3.5-122B-A10B
/home/lancewicki/data/hf_cache/models--Qwen--Qwen3.5-122B-A10B-GPTQ-Int4
/home/lancewicki/data/hf_cache/models--Qwen--Qwen3.5-397B-A17B-GPTQ-Int4
```

### Observations (unsolicited, ignore if not useful)

- **Both models ship the *same* sampling defaults**: `T=0.6, top_k=20, top_p=0.95, do_sample=true`. No `min_p`, no `repetition_penalty`. `max_new_tokens` not set (vLLM defers to request-level `max_completion_tokens`).
- **These match the Qwen3 model card's "thinking-on" recommendation** — which is what the spec's §1 sampling table calls the thinking-on defaults. There is no separate "thinking-off" configuration file; the model card's `T=0.7, top_p=0.8` for thinking-off comes from documentation, not `generation_config.json`.
- **Available families for Task-17 selection:** Qwen3 → 4B, 8B only; Qwen3.5 → 122B-A10B (both fp16 and GPTQ-Int4), 397B-A17B-GPTQ-Int4. Missing on the cluster (would need download): Qwen3-14B, Qwen3-32B, Qwen3.5-4B, Qwen3.5-9B, Qwen3.5-27B, Qwen3.5-35B-A3B-GPTQ-Int4. The Task-14 `cell_list()` cell lineup will need downloads before the sweep runs — worth flagging in a later request if you want me to prefetch.

No commit needed on my side.
