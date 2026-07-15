# Our patches to the upstream repo

This file tracks every modification we make to files in
`/home/lancewicki/projects/turing-rl/` that originated upstream
(https://github.com/SusanWYS/turing-rl). Anything under `scripts/`, `CLAUDE.md`,
`summary/`, `.env`, or `our_patches.md` itself is **ours** and is not tracked here.

The intent is to keep this list small. Before patching an upstream file, prefer:
1. Wrapping in our own sbatch / Python script under `scripts/`
2. Overriding via env var or CLI flag if one exists
3. Only as a last resort: modify the upstream file (and document it here)

If a patch is temporary (e.g., flipped during a run, restored after via `trap`),
mark it `TEMP` and explain how/when it gets reverted. If it's permanent for the
duration of the repro, mark it `PERSISTENT`.

---

## TEMP: `training/sft/configs/qwen3_8b_lora.yaml` — `report_to`

- **Original**: `report_to: none` (line 17)
- **Patched to**: `report_to: wandb`
- **Where**: applied/reverted by `scripts/slurm/sft_smoke.sh` via inline `sed`
  + `trap EXIT` (with `.smoke-bak` snapshot).
- **Why**: `lora_sft.py` reads `report_to` from this YAML and exposes no CLI
  override. We want SFT loss/lr curves in wandb at
  `https://meta.wandb.io/lancewicki/turing-rl-smoke`. Their default of `none`
  is fine for their workflow but blocks our visibility goal.
- **Reverted**: yes, automatically at job end (success or failure).

---

## DELETED: `bash_scripts/grpo/train_grpo_smoke.sh` — smoke overrides NOT yet migrated

- **Status**: deleted (commit `3c1a475`) as misleading vs the canonical `train_grpo.sh`.
  The old smoke launchers `scripts/slurm/grpo_smoke.sh` / `grpo_smoke_8b.sh` still invoke it,
  so they are **broken — and are being treated as legacy, NOT fixed**: the RL-generator work
  (spec `docs/superpowers/specs/2026-07-15-rl-generator-vs-fixed-judge-design.md`) will use
  **fresh launch scripts** rather than these. Left in place for now (may still be referenced);
  delete the smoke launchers + `launch_grpo_smoke*.sh` once the fresh RL scripts land.
- **Overrides it carried** (kept as reference for writing the fresh scripts — 40GB/small-slice
  deltas worth reusing):
  - `actor_rollout_ref.model.use_remove_padding=false` — our env has no
    flash_attn (no cu130 wheel; see `scripts/slurm/train_env_install.sh`);
    verl's `unpad_input` requires it, so the sequence-packed path can't run.
  - `actor_rollout_ref.actor.use_remove_padding=false` — same reason.
  - `data.train_batch_size=32`, `ppo_mini_batch_size=32`,
    `ppo_micro_batch_size_per_gpu=1`, `rollout.n=2` — our 138-row smoke
    slice can't sustain their default 128/128/4/4.
  - `data.max_prompt_length=6144`, `rollout.max_model_len=7168`,
    `max_num_batched_tokens=8192`, `max_num_seqs=16`,
    `gpu_memory_utilization=0.45` — 40GB A100 safety (paper defaults assume
    80GB headroom).
  - `trainer.total_epochs=1`, `trainer.save_freq=2` — smoke scale.
  - `trainer.project_name=${WANDB_PROJECT:-turing-rl-smoke}` — route smoke
    runs to our wandb project.
  - Passed `"$@"` through as extra Hydra overrides so callers could tune ad hoc.

---

## PERSISTENT: `shared/api_client.py` — judge response dump hook

- **Original**: `post_chat_async` returned `_extract_chat_content(await resp.json())`
  inline; no place to observe judge payloads/responses.
- **Patched to**: two new helpers (`_should_dump_judge`, `_dump_judge_response`)
  and a small refactor of the return-path inside `post_chat_async` that calls
  them after parsing the response. Also adds `import hashlib`.
- **Env-gated (off by default)**:
  - `PERSONA_JUDGE_DUMP_RATE` (float, default `0.0`) — sample rate 0..1.
    Deterministic per-payload (md5-hashed) so a given payload is always
    dumped or always skipped.
  - `PERSONA_JUDGE_DUMP_DIR` (path) — parent dir for per-worker JSONL files.
    Files: `${DIR}/judge-{SLURM_JOB_ID}-{pid}.jsonl`.
  - When both unset, the helpers no-op; code path is functionally identical
    to upstream.
- **Row shape**:
  `{ts, worker_pid, latency_ms, model, payload_messages, response}` — full
  request messages + full response body. `.default=str` on the JSON writer
  guards against non-serializable fields.
- **Why**: to inspect what the Turing judge actually returns during GRPO
  training. First smoke run (job 8921) had ~100% JSON parse failures because
  we truncated responses at 512 tokens; without dumped bodies we couldn't
  tell that. Also useful for building post-hoc datasets and debugging
  reward-signal quality.
- **Sync path (`post_chat_sync`) intentionally not touched.** Its only
  caller (`data/sft/generate_cot.py`) already persists its output as parquet.
- **Reverted**: no. Persistent for the duration of the repro; safe because
  it's a no-op unless the env vars are set.

---

## PERSISTENT: `shared/api_client.py` — `_extract_chat_content` returns "" instead of raising

- **Original** (line 106): `raise ValueError("OpenAI response missing choices[0].message.content")`.
- **Patched to**: `return ""`.
- **Why**: with `--reasoning-parser deepseek_r1` on the judge server, if the
  model hits `finish_reason=length` inside `<think>` before emitting the
  closing tag, vLLM returns `choices[0].message.content=None`. The raise
  bubbles past `post_chat_async`'s retry loop (`ValueError` isn't caught by
  the `aiohttp.ClientError | asyncio.TimeoutError` filter — actually it IS
  caught, but after 3 attempts it re-raises and kills the whole GRPO step).
- **Effect after patch**: empty content flows down to `_extract_json` in
  `training/grpo/reward.py:360`, which returns `None`. Reward code retries
  the judge call up to 3 times; if all fail, `_turing_parse_failure_result`
  emits a `-0.15` fallback reward like any other parse failure. Training
  continues, wandb reports the failure rate as elevated `-0.15` incidence.
- **Reverted**: no. Persistent; the fallback is strictly safer than
  crashing.


## PERSISTENT: eval/generate_trained.py — --base_model flag

- **What**: Adds an additive `--base_model` flag to `eval/generate_trained.py`
  that runs the base `--model_id` with no LoRA adapter, enabling no-adapter
  base-model heldout generation for the generator sweep (scoring candidates
  from untrained base models like qwen3-8B / qwen3.5-9B alongside SFT'd ones).
- **How**: A new pure helper `resolve_adapter_for_run(checkpoint_dir, base_model)`
  short-circuits to `None` when `base_model=True`; otherwise it preserves the
  existing behavior (resolve latest checkpoint / adapter and raise
  `ValueError("No LoRA adapter found under ...")` if none is found). `main()`
  now calls this helper and prints "Base model (no adapter): <model_id>" when
  the adapter is `None`. Downstream (`build_llm_kwargs`, `build_vllm_lora_request`)
  already handle `adapter_path=None`.
- **Also**: `--checkpoint_dir` relaxed from `required=True` to
  `required=False, default=""` so base runs need not pass a dummy path. The
  output-naming block that derives a tag from the adapter path is guarded to
  only run when `args.adapter_path` is set (base runs always pass `--output`
  explicitly; a `model_id` basename is used as a fallback tag otherwise).
- **No-op unless `--base_model` is passed**: adapter runs behave exactly as
  before.
- **Reverted**: no. Persistent.
