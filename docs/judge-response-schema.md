# Judge response schema

`PERSONA_JUDGE_JSON_SCHEMA=1` selects one ordered, prompt-matched JSON Schema for
training rewards, served evaluations, offline sweeps, and diagnostics. It requires all
37 fields shown in `TURING_PROMPT`, disallows additional properties, and keeps `rating`
last. The source of truth is `shared/judge_prompts.py:TURING_RESPONSE_SCHEMA`.

The previous rating-only schema was incorrect: constrained decoders emitted `rating`
first and could then generate duplicate or malformed rating-like keys. In the paired
Gemma 4 31B smoke (Slurm 15172), the full schema produced 16/16 valid, stopped responses;
the rating-only control produced 5/16 valid responses and hit the 8192-token limit in
10/16 cases.

When the flag is unset, the reward path still uses `{"type":"json_object"}`. The completed
GRPO training run used that default, not the faulty rating-only schema, so this correction
does not require retraining.

Thinking text returned by local vLLM is in `choices[0].message.reasoning`; the field
`reasoning_content` may be absent. Full HTTP dumps retain the complete response. This is
distinct from the JSON verdict's `reasoning` field, which is a concise rubric explanation.

The reward code continues to recompute base scores, penalties, response scores, score gap,
and rating from the primitive dimension and penalty fields. Model-emitted arithmetic and
the model-emitted rating are not treated as authoritative when those primitive fields exist.
