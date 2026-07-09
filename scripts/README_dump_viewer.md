# Judge Dump Viewer

We record every judge call from GRPO smoke runs and browse them via a small
FastAPI viewer. Two dump types are written per run:

| Type   | Written by                                    | Contents                                       | Path                                      |
|--------|-----------------------------------------------|------------------------------------------------|-------------------------------------------|
| HTTP   | `shared/api_client.py:_dump_judge_response`   | Wire request/response (payload, usage, model)  | `${PERSONA_JUDGE_DUMP_DIR}/http/*.jsonl`  |
| Reward | `training/grpo/reward.py:_dump_reward_call`   | Semantic context (human_side A/B, reward, GT)  | `${PERSONA_JUDGE_DUMP_DIR}/reward/*.jsonl`|

Both key off `PERSONA_JUDGE_DUMP_RATE` (float in [0,1], default 0 = off) and
`PERSONA_JUDGE_DUMP_DIR`. The 8B smoke sets both dumps on at rate 1.0 (see
`scripts/slurm/grpo_smoke_8b.sh`).

**When to use which**

- Reward dump: everyday analysis. Shows which side (A or B) is the human, the
  ground-truth response, the generator's response, the parsed judge rubric,
  and the final reward with all four penalty components. Use this when asking
  "is the judge picking the human?", "which prompts get 0-reward penalties?",
  or "how did the reward evolve during training?".
- HTTP dump: wire-level debugging. Useful when the judge server itself
  misbehaves — bad JSON, wrong `finish_reason`, missing fields — and you need
  to see the raw payload we sent and the raw response we got. Rarely needed
  during normal analysis.

The viewer auto-detects schema per row and recurses into subdirs, so you point
it at the parent (`${PERSONA_JUDGE_DUMP_DIR}`) and get both types side by side.
Filter by schema in the sidebar (`schema: reward | http | all`).

## Running the viewer

**One important gotcha**: the viewer must run on the same host your Mac's SSH
tunnel lands on. That's your **login pod** (typically `lancewicki-login-0`).
Running it from a Claude sandbox pod or a compute node won't work — `ssh -L`
only forwards to `localhost` of the SSH endpoint.

### On the login pod

```bash
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY
cd /home/lancewicki/projects/turing-rl && \
  /home/lancewicki/miniconda3/envs/turing-rl-train/bin/python \
  scripts/dump_viewer.py \
    --dumps /home/lancewicki/tmp/judge_dumps_8b \
    --port 8082
```

Expected output:
```
loaded 512 rows from /home/lancewicki/tmp/judge_dumps_8b  (reward=256, http=256)
serving on http://127.0.0.1:8082/ (Ctrl-C to stop)
```

### From your Mac

```bash
ssh -L 8082:localhost:8082 rfai-research-aws-use2-1
```

Then browse to `http://localhost:8082/` in your Mac's browser.

## What the UI shows

- **Sidebar**: filterable list of rows. Each row is tagged `reward` (blue) or
  `http` (gray). For reward rows the columns are `idx | reward | rating |
  human`; for HTTP rows they're `idx | latency | tokens | ok`.
- **Header bar**: for reward rows, shows a `Human: A/B` badge, final reward,
  rating, and which ordering (`gt_first` or `gen_first`) was used.
- **Tabs (reward)**: `context`, `history`, `response` (generator's output),
  `ground_truth` (real human reply), `prompt` (the full Turing prompt sent to
  the judge), `raw` (raw judge response content, pre-JSON-extraction),
  `reasoning` (judge's `<think>` trace), `judge` (parsed rubric), `reward`
  (full breakdown), `metadata` (user_id/post_id/latency/finish_reason/etc).
- **Tabs (HTTP)**: `prompt`, `response`, `parsed`, `reasoning`, `metadata`
  (unchanged from before).

The default filter is `reward` since that's the schema you'll want for
analysis. Switch to `all` or `http` from the sidebar dropdown to see
HTTP-level dumps.

## Legacy dumps

Older dumps written before the `http/` subdir split (any `*.jsonl` directly
under `PERSONA_JUDGE_DUMP_DIR`) are still picked up — the viewer just scans
recursively and detects schema per row.
