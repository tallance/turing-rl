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

## The schema does not rescue a model that will not stop

A small model can fail here for a reason no response format addresses. Qwen3.5-0.8B was
measured as a candidate training judge (Slurm 18901) and was rejected:

| mode | `response_format` | usable | hard fail | hit the 8192 cap |
|---|---|---|---|---|
| training, 200 prompts | `json_object` | 0.175 | 0.825 | 0.825 |
| eval, 440 prompts x2 | `json_schema` | 0.200 / 0.193 | 0.800 / 0.807 | 0.800 / 0.807 |

Qwen3.5-9B scored the same prompts at a 0.0 hard-failure rate. The two schema-mode passes
agree to within 0.007, so this is not sampling noise.

Enabling the ordered schema changed nothing, because the output was not malformed: 165 of
200 responses ended with `finish_reason="length"`. The model never stopped thinking and was
truncated mid-verdict. A schema constrains the shape of a response, not the length of the
reasoning that precedes it, so it cannot fix this failure -- it is the same 8192-cap
symptom the Gemma 4 31B control showed above, but here it is the whole story rather than a
side effect of a bad schema.

Throughput fails too, in the opposite direction from intuition: 0.636 req/s at p50 199 s,
because nearly every request runs to the cap. A 0.8B judge is *slower* than the 9B, not
cheaper.

Disabling thinking (`PERSONA_JUDGE_ENABLE_THINKING=0`, Slurm 18913) confirms where the
runaway lives, and recovers most but not all of it:

| thinking | mode | usable | hard fail | throughput |
|---|---|---|---|---|
| on | `json_object` | 0.175 | 0.825 | 0.64 req/s, p50 199 s |
| off | `json_object` | 0.840 | 0.160 | 2.42 req/s, p50 32 s |
| off | `json_schema` | 0.886 / 0.855 | 0.114 / 0.145 | |

The concurrency sweep holds the same band -- 0.109 at 8, 0.125 at 16 and 32 -- so ~11-16%
is a floor, not a load artefact. Every remaining failure is still the 8192 cap: the verdict
alone, with no thinking block, can exhaust it.

So thinking-off is not a fix. It leaves roughly one judge call in seven unscored against
the 9B's zero, and those failures are not random -- they concentrate on the longest inputs,
which biases the reward rather than merely thinning it. It also changes the judge protocol
relative to every completed arm, so a run using it answers a different question.

For a deliberately small judge, `Qwen3.5-4B` is the better candidate: it is the smallest
model the eval sweep ever used, it is already cached, and it has existing thinking-on
sweep data, so it can be compared with the protocol left alone.

Thinking text returned by local vLLM is in `choices[0].message.reasoning`; the field
`reasoning_content` may be absent. Full HTTP dumps retain the complete response. This is
distinct from the JSON verdict's `reasoning` field, which is a concise rubric explanation.

The reward code continues to recompute base scores, penalties, response scores, score gap,
and rating from the primitive dimension and penalty fields. Model-emitted arithmetic and
the model-emitted rating are not treated as authoritative when those primitive fields exist.
