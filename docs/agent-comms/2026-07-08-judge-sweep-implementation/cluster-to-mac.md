# Cluster → Mac — judge-sweep-implementation

Plan: `docs/superpowers/plans/2026-07-08-judge-sweep-implementation.md`

Latest report at top; prior reports below.

---

## Report — 2026-07-10 — Task 2 pytest re-run: **all 7 green (16/16 with parametrization)**

Both fixes in `cabac26` land clean. Full output:

```
============================= test session starts ==============================
collected 16 items

tests/test_prism_split_verification.py::test_1_files_exist PASSED        [  6%]
tests/test_prism_split_verification.py::test_2_row_counts[sft/train.parquet-exp0] PASSED [ 12%]
tests/test_prism_split_verification.py::test_2_row_counts[grpo/train.parquet-exp1] PASSED [ 18%]
tests/test_prism_split_verification.py::test_2_row_counts[grpo/val.parquet-exp2] PASSED [ 25%]
tests/test_prism_split_verification.py::test_2_row_counts[test.parquet-exp3] PASSED [ 31%]
tests/test_prism_split_verification.py::test_3_user_counts[sft/train.parquet-exp0] PASSED [ 37%]
tests/test_prism_split_verification.py::test_3_user_counts[grpo/train.parquet-exp1] PASSED [ 43%]
tests/test_prism_split_verification.py::test_3_user_counts[grpo/val.parquet-exp2] PASSED [ 50%]
tests/test_prism_split_verification.py::test_3_user_counts[test.parquet-exp3] PASSED [ 56%]
tests/test_prism_split_verification.py::test_4_user_disjointness PASSED  [ 62%]
tests/test_prism_split_verification.py::test_5_prompt_schema[sft/train.parquet] PASSED [ 68%]
tests/test_prism_split_verification.py::test_5_prompt_schema[grpo/train.parquet] PASSED [ 75%]
tests/test_prism_split_verification.py::test_5_prompt_schema[test.parquet] PASSED [ 87%]
tests/test_prism_split_verification.py::test_5_prompt_schema[grpo/val.parquet] PASSED [ 81%]
tests/test_prism_split_verification.py::test_6_heldout_gt_matches_raw PASSED [ 93%]
tests/test_prism_split_verification.py::test_7_no_text_leak_heldout_from_sft_targets PASSED [100%]

============================== 16 passed in 6.84s ==============================
```

**Task 2 complete.** Combined with Task 3's byte-identical hash compare (previous report), the PRISM split is verified paper-faithful end-to-end.

Ready for the next handoff.

---

## Report — 2026-07-10 — Task 2 pytest + Task 3 re-split hash compare

### Request B (Task 3): re-split hash compare — **PASSES, all 4 parquets byte-identical**

```
| File               | current      | fresh        | match |
| sft/train.parquet  | 99ebdc76c181 | 99ebdc76c181 | OK    |
| grpo/train.parquet | b6dc3595cbbd | b6dc3595cbbd | OK    |
| grpo/val.parquet   | 3d7123cb5b54 | 3d7123cb5b54 | OK    |
| test.parquet       | c7b13e2d5386 | c7b13e2d5386 | OK    |
```

Confirmed no post-hoc parquet tampering. The `--seed 42` default in `split_data.py` matches what `scripts/slurm/split_prism_full_s42.sh` used (which passed no seed / frac flags — relies on defaults `SEED=42, HELDOUT_USER_FRAC=0.1, GRPO_FRAC=0.6`, byte-identical to the verify script's `--seed 42` explicit).

### Request A (Task 2): pytest — **collection error before any test runs**

```
ImportError while importing test module '...test_prism_split_verification.py'
tests/test_prism_split_verification.py:20: in <module>
    from tests.prism_verification_helpers import extra_info_key, load_raw_prism_replies
E   ModuleNotFoundError: No module named 'tests.prism_verification_helpers'
```

**Root cause:** `tests/` has no `__init__.py`, so the absolute import `from tests.prism_verification_helpers import ...` doesn't resolve. Two fixes possible:
1. Add empty `tests/__init__.py` (turns `tests/` into a package)
2. Change the import to a plain `from prism_verification_helpers import ...` and add a `tests/conftest.py` that appends `tests/` to `sys.path` (pytest convention)

I did not fix it — didn't want to guess your preference. Recommend option 1 (one-line file, matches typical Python packaging).

### Pre-emptive schema data (so you can fix asserts without another round-trip)

You asked for these values in case tests failed. All present here:

**`split_metadata.json`** (exact contents):
```json
{
  "counts": {
    "grpo_train": {"rows": 4174, "users": 696},
    "grpo_val":   {"rows": 705,  "users": 696},
    "heldout":    {"rows": 880,  "users": 128},
    "sft":        {"rows": 3272, "users": 464}
  },
  "grpo_frac": 0.6,
  "grpo_val_frac": 0.1,
  "heldout_user_frac": 0.1,
  "seed": 42,
  "source_input_dir": "/home/lancewicki/projects/turing-rl/data/prism/full_s42_history",
  "user_overlap": {"grpo_heldout": 0, "grpo_sft": 0, "sft_heldout": 0}
}
```
Your `EXPECTED_COUNTS` in the test matches this exactly (`sft=3272/464, grpo/train=4174/696, grpo/val=705/696, test=880/128`).

**Per-parquet schema** — identical across all 4 splits:
- `columns` = `['data_source', 'prompt', 'reward_model', 'extra_info']`
- `reward_model.keys()` = `['ground_truth', 'style']`
- `data_source` = `'prism_alignment_user_sim'`
- `prompt` is a **`numpy.ndarray`** (not `list`); `list(row["prompt"])` will work.
- `extra_info` (23 keys, superset of what your test checks — all your asserted keys `user_id/post_id/target_idx/user_history/context` are present):
  ```
  ['conditioning_mode', 'context', 'dataset_config', 'dataset_name', 'dataset_split',
   'history_conversation_types', 'history_count', 'history_thread_ids', 'index', 'persona',
   'post_id', 'prompt_idx', 'prompt_mode', 'prompt_text', 'raw_prompt', 'raw_user_id',
   'source_name', 'split', 'target_conversation_type', 'target_idx', 'thread_context',
   'user_history', 'user_id']
  ```
  Note: has BOTH `user_id` and `raw_user_id` — your helper's `extra_info.get("raw_user_id", extra_info["user_id"])` is correct. Also has `thread_context` in addition to `context`.

**Raw PRISM structure — your helper is buggy, needs revision:**

Top-level fields of one raw row:
```
['conversation_id', 'user_id', 'conversation_type', 'opening_prompt', 'conversation_turns',
 'conversation_history', 'performance_attributes', 'choice_attributes', 'open_feedback',
 'generated_datetime', 'timing_duration_s', 'timing_duration_mins', 'included_in_balanced_subset']
```

Critical fixes for `prism_verification_helpers.py`:
- **`conversation_turns` is an INT** (turn count), not a list of turns. Your code `row.get("conversation_turns", []) or []` will resolve to the integer `5` and then `for turn in 5:` raises TypeError.
- **Turns live in `conversation_history`** (a `list[dict]`, ~15 entries per row).
- **Per-turn keys**: `['turn', 'role', 'content', 'model_provider', 'model_name', 'score', 'if_chosen', 'within_turn_id']`
- **"user" role value is literally `'user'`** (lowercase). Non-user role is `'model'` (multiple `model` turns per `turn` index — one per model_provider, with `if_chosen` marking which was selected).
- **Turn index counting**: the raw `turn` field is 0-indexed already; multiple `role='model'` entries share the same `turn` number. Your helper's `t = 0` local counter over just `user` turns matches the split's `target_idx` semantics (verified end-to-end below).
- `post_id` (extra_info) maps to `conversation_id` (raw). No `post_id` field raw-side.

**End-to-end lookup verified with test.parquet[0]:**
- extra_info: `user_id='user96', raw_user_id='user96', post_id='c529', target_idx=1`
- gt: `'I would like some general information first, please.'`
- Raw row (matched on `user_id='user96'`, `conversation_id='c529'`): 1 match; 4 user-role turns in `conversation_history`.
- user-role turn at index `target_idx=1` (0-based): `'I would like some general information first, please.'`
- **Matches gt: True** ✓

So test_6's semantics are right; only the helper's iteration is broken. A corrected loop:
```python
for row in ds:
    uid, cid = str(row["user_id"]), str(row["conversation_id"])
    user_turn_idx = 0
    for turn in row["conversation_history"]:  # not conversation_turns
        if str(turn.get("role", "")).lower() != "user":
            continue
        out[(uid, cid, user_turn_idx)] = str(turn.get("content", ""))
        user_turn_idx += 1
```

### Recommended next round

1. Add `tests/__init__.py` (empty).
2. Fix `prism_verification_helpers.py` per above (`conversation_history`, not `conversation_turns`).
3. Push; I'll re-run the pytest suite (should be 7 green).

No commits on my side.

---

## Report — 2026-07-10 — Qwen generation_config sampling scan

Plan task: sampling-fidelity input for Task 1.

Ran the three read-only queries you asked for. Raw output below.

### 1. Qwen3-8B `generation_config.json`

```json
{
    "bos_token_id": 151643,
    "do_sample": true,
    "eos_token_id": [151645, 151643],
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
    "eos_token_id": [248046, 248044],
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

### Observations

- **Both models ship the *same* sampling defaults**: `T=0.6, top_k=20, top_p=0.95, do_sample=true`. No `min_p`, no `repetition_penalty`. `max_new_tokens` not set (vLLM defers to request-level `max_completion_tokens`).
- **These match the Qwen3 model card's "thinking-on" recommendation** — which is what the spec's §1 sampling table calls the thinking-on defaults. There is no separate "thinking-off" configuration file; the model card's `T=0.7, top_p=0.8` for thinking-off comes from documentation, not `generation_config.json`.
- **Available families for Task-17 selection:** Qwen3 → 4B, 8B only; Qwen3.5 → 122B-A10B (both fp16 and GPTQ-Int4), 397B-A17B-GPTQ-Int4. Missing on the cluster (would need download): Qwen3-14B, Qwen3-32B, Qwen3.5-4B, Qwen3.5-9B, Qwen3.5-27B, Qwen3.5-35B-A3B-GPTQ-Int4. The Task-14 `cell_list()` cell lineup will need downloads before the sweep runs — worth flagging in a later request if you want me to prefetch.
