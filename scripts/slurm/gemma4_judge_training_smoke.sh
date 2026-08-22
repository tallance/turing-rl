#!/bin/bash
#SBATCH --job-name=gemma_judge_smoke
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --mem=512G
#SBATCH --time=04:00:00
#SBATCH --output=/home/lancewicki/projects/turing-rl/logs/gemma_judge_smoke-%j.out
#SBATCH --partition=a100
#SBATCH --account=rfai
#
# Acceptance gates for using gemma-4-12B as the GRPO *training* judge.
#
# This gates a ~50 h run, and it exists because the gemma numbers we already trust were
# earned under conditions training does not reproduce:
#
#   - the eval sweep enables the 37-field ordered schema; training does not (json_object)
#   - the eval serves 8 single-GPU servers on 8 ports; training needs ONE endpoint, so
#     gemma has to run --data-parallel-size 8, which nothing has exercised
#
# The judge is brought up through the REAL serve script, not a hand-rolled vllm command,
# so a mistake in the serving branch fails here rather than 20 minutes into the real run.
#
# Gates (see the plan for the full rationale):
#   1 server identity  -- /v1/models advertises the canonical id; pinned snapshot in the log
#   2 training mode    -- parse outcomes with the schema UNSET. The go/no-go.
#   3 path equivalence -- replay the eval's own gemma prompts WITH the schema and compare
#                         per-prompt to what the 8-replica path recorded
#   4 DP fan-out       -- requests spread across all 8 engines for >=30 min, plus a
#                         concurrency sweep to choose JUDGE_CONC
#
# Env: none required. Optional SMOKE_N (gate 2 prompts, default 200),
#   SMOKE_CONC (default 32), REF_STEP (eval checkpoint to compare against, default step60).
set -uo pipefail
source "${TURING_RL_CODE_ROOT:?}/scripts/cluster_job_bootstrap.sh"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY
REPO=${TURING_RL_WORK_ROOT:?}
cd "$REPO"

# Defaults are gemma's, because that is what this was written for. Overriding SMOKE_MODEL and
# SMOKE_PARSER points the same battery at any judge the serve script can bring up -- gate 3
# self-skips when that model has no eval reference dump, which is the usual case.
MODEL=${SMOKE_MODEL:-google/gemma-4-12B-it}
TP=${SMOKE_TP:-1}
DP=${SMOKE_DP:-8}
REASONING_PARSER=${SMOKE_PARSER:-gemma4}
SMOKE_N=${SMOKE_N:-200}
SMOKE_CONC=${SMOKE_CONC:-32}
REF_STEP=${REF_STEP:-step60}

PY=/home/lancewicki/miniconda3/envs/turing-rl-train/bin/python
OUT=$REPO/results/judge-smoke/$(echo "$MODEL" | tr '/' '-')/${SLURM_JOB_ID:-manual}
mkdir -p "$OUT"
ENDPOINT_FILE=$OUT/judge_endpoint.txt
SERVE_LOG=$OUT/judge_server.log
rm -f "$ENDPOINT_FILE"

# Prompt sources. Gate 2 uses real TRAINING prompts (qwen-judged, so the dump also carries
# the qwen baseline for free). Gate 3 uses the eval's own gemma-judged prompts, whose
# recorded ratings are the reference the DP-8 path must reproduce.
TRAIN_DUMP="$REPO/results/grpo/rl-generator/9b_frac10_10ep_kl1e4_lr1e4_temp1/reward_dump/reward-*.jsonl"
EVAL_DUMP="$REPO/results/2026-08-12-test-eval-9b-train10pct-10ep-test50pct-full-schema/raw/9b-train10pct-${REF_STEP}/sweep/gemma4-12b/on/reward/reward-*.jsonl"

echo "============================================"
echo "$MODEL training-judge smoke  job=${SLURM_JOB_ID:-manual}"
echo "date=$(date) host=$(hostname)  out=$OUT"
echo "gate2 prompts: $SMOKE_N @ concurrency $SMOKE_CONC   gate3 reference: $REF_STEP"
echo "============================================"

# --- bring the judge up through the real serve script -------------------------------------
MODEL=$MODEL TP=$TP DP=$DP REASONING_PARSER=$REASONING_PARSER \
  JUDGE_ENDPOINT_FILE=$ENDPOINT_FILE \
  bash scripts/slurm/judge_serve_9b_replicas.sh > "$SERVE_LOG" 2>&1 &
SERVE_PID=$!
cleanup() { kill "$SERVE_PID" 2>/dev/null || true; }
trap cleanup EXIT TERM INT

echo ">> waiting for judge endpoint (up to 40 min warmup)..."
ok=0
for _ in $(seq 1 1200); do
  [ -s "$ENDPOINT_FILE" ] && { ok=1; break; }
  kill -0 "$SERVE_PID" 2>/dev/null || { echo "GATE1 FAIL: server died during warmup" >&2; tail -40 "$SERVE_LOG" >&2; exit 3; }
  sleep 2
done
[ $ok -eq 1 ] || { echo "GATE1 FAIL: timeout waiting for endpoint" >&2; tail -40 "$SERVE_LOG" >&2; exit 4; }
ENDPOINT=$(cat "$ENDPOINT_FILE")
echo ">> endpoint: $ENDPOINT"

# --- gate 1: identity ----------------------------------------------------------------------
{
  echo "=== GATE 1: server identity ==="
  echo "-- /v1/models --"
  curl -sf -m 10 "$ENDPOINT/models" | $PY -m json.tool 2>&1 | head -20
  echo "-- advertised id matches the canonical name? --"
  if curl -sf -m 10 "$ENDPOINT/models" | grep -qF "\"$MODEL\""; then
    echo "PASS: /v1/models advertises $MODEL"
  else
    echo "FAIL: /v1/models does not advertise $MODEL"
  fi
  # Only the gemma branch serves an offline snapshot by path, so only it has a pin to check.
  if [ "$REASONING_PARSER" = gemma4 ]; then
    echo "-- pinned snapshot actually served --"
    grep -m1 "gemma_snapshot=" "$SERVE_LOG" || echo "FAIL: no gemma_snapshot line in the server log"
  fi
  echo "-- DP engines that reported ready --"
  grep -oE "Engine [0-9]+" "$SERVE_LOG" | sort -u | tr '\n' ' '; echo
} | tee "$OUT/gate1_identity.txt"

# --- gate 2: TRAINING mode (schema unset) -- the go/no-go -----------------------------------
echo "=== GATE 2: judge validity under the training reward config ==="
# Exactly the env rl_generator_run_9b.sh exports, minus PERSONA_JUDGE_JSON_SCHEMA, which it
# does not set -- that omission is the thing under test.
export JUDGE_MODEL=$MODEL
export TURING_JUDGE_SCORE_CLIP_MAX=7
export PERSONA_JUDGE_SAMPLING='{"repetition_penalty":1.1,"temperature":0.6}'
# 1 is what rl_generator_run_9b.sh exports, so 1 is what "the training config" means and is
# the default here. SMOKE_THINKING=0 probes the one lever left for a model that fails gate 2
# by never stopping: the runaway lives in the thinking block. Note that answering with it off
# describes a DIFFERENT judge protocol than either completed arm uses, so a pass under 0 is
# not licence to compare the resulting run against them.
export PERSONA_JUDGE_ENABLE_THINKING=${SMOKE_THINKING:-1}
export PERSONA_JUDGE_MAX_COMPLETION_TOKENS=8192
export PERSONA_OPENAI_TIMEOUT_SECONDS=1800
export PERSONA_OPENAI_MAX_RETRIES=3
unset PERSONA_JUDGE_JSON_SCHEMA

PYTHONPATH=. $PY scripts/gemma4_judge_training_smoke.py \
  --endpoint "$ENDPOINT" --model "$MODEL" \
  --dump-glob "$TRAIN_DUMP" --split val --mode training \
  --n "$SMOKE_N" --concurrency "$SMOKE_CONC" \
  --out "$OUT/gate2_training_calls.jsonl" --summary "$OUT/gate2_training_summary.json" 2>&1 \
  | tee "$OUT/gate2_training.log"

# The qwen baseline on the SAME prompts needs no GPU: 15371 already recorded its own verdict
# for every one of them, so read the outcome mix straight out of the dump.
PYTHONPATH=. $PY - "$TRAIN_DUMP" "$OUT/gate2_qwen_baseline.json" <<'PYEOF' 2>&1 | tee -a "$OUT/gate2_training.log"
import glob, json, sys
from collections import Counter
rows, seen = [], set()
for path in sorted(glob.glob(sys.argv[1])):
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("split") != "val":
            continue
        key = (r.get("user_id"), r.get("post_id"), r.get("target_idx"))
        if key in seen:
            continue
        seen.add(key)
        rows.append(r)
n = len(rows)
scored = sum(r.get("turing_judge_score_raw") is not None for r in rows)
fin = Counter(r.get("judge_finish_reason") for r in rows)
out = {"judge": "qwen3.5-9b (recorded in job 15371)", "n_unique_val_prompts": n,
       "scored_rate": round(scored / n, 4) if n else 0.0,
       "hard_fail_rate": round(1 - scored / n, 4) if n else 1.0,
       "finish_length_rate": round(fin.get("length", 0) / n, 4) if n else 0.0,
       "finish_reasons": dict(fin)}
json.dump(out, open(sys.argv[2], "w"), indent=2)
print("[baseline] qwen on the same val prompts:", json.dumps(out, indent=2))
PYEOF

# --- gate 3: equivalence with the proven 8-replica path -------------------------------------
# Matched conditions: the reference was produced WITH the ordered schema, so replay with it
# enabled. Two passes give the judge's own noise floor, which is the only honest tolerance.
# NOTE for non-gemma judges: EVAL_DUMP names gemma's sweep output, so this always runs and
# is NOT an equivalence test for them -- the reference ratings came from a different model.
# What it still measures, usefully, is the served model's parse rate under the ordered
# schema, which is the mitigation one would reach for if gate 2 fails. The smoke labels the
# rating comparison "CROSS-JUDGE (not a gate)" in that case.
echo "=== GATE 3: schema-mode parse rate; vs the eval's 8-replica path when same-model ==="
if compgen -G "$EVAL_DUMP" > /dev/null; then
  PYTHONPATH=. $PY scripts/gemma4_judge_training_smoke.py \
    --endpoint "$ENDPOINT" --model "$MODEL" \
    --dump-glob "$EVAL_DUMP" --mode eval \
    --n 440 --concurrency "$SMOKE_CONC" --repeat 2 \
    --out "$OUT/gate3_eval_calls.jsonl" --summary "$OUT/gate3_eval_summary.json" 2>&1 \
    | tee "$OUT/gate3_eval.log"
else
  echo "SKIP: no eval reference dump at $EVAL_DUMP" | tee "$OUT/gate3_eval.log"
fi

# --- gate 4: concurrency sweep + DP fan-out --------------------------------------------------
echo "=== GATE 4: concurrency sweep and engine fan-out ==="
for c in 8 16 32 64; do
  PYTHONPATH=. $PY scripts/gemma4_judge_training_smoke.py \
    --endpoint "$ENDPOINT" --model "$MODEL" \
    --dump-glob "$TRAIN_DUMP" --split val --mode training \
    --n 64 --concurrency "$c" --seed $((100 + c)) \
    --out "$OUT/gate4_conc${c}_calls.jsonl" --summary "$OUT/gate4_conc${c}_summary.json" 2>&1 \
    | grep -E "req_per_s|p50_s|p95_s|hard_fail_rate|usable_rate" | sed "s/^/  conc=$c /"
done 2>&1 | tee "$OUT/gate4_concurrency.log"

# Fan-out is judged by REQUEST COUNTS per engine, not GPU utilisation: one long request keeps
# a GPU busy either way. The qwen judge's pathology was progressive collapse onto engine 000
# with an onset around 20 min, which is why the window has to be long rather than a snapshot.
{
  echo "=== GATE 4: per-engine request spread over the whole session ==="
  echo "-- lines logged per engine (proxy for how much work each one saw) --"
  grep -oE "Engine [0-9]+" "$SERVE_LOG" | sort | uniq -c | sort -k2
  echo
  echo "-- last observed Running/Waiting per engine --"
  for e in 000 001 002 003 004 005 006 007; do
    last=$(grep "Engine $e:" "$SERVE_LOG" | tail -1 | grep -oE "Running: [0-9]+ reqs, Waiting: [0-9]+ reqs")
    printf "  engine %s  %s\n" "$e" "${last:-<no traffic>}"
  done
  echo
  echo "-- server session length --"
  echo "  serve log lines: $(wc -l < "$SERVE_LOG")"
} | tee "$OUT/gate4_fanout.txt"

echo "============================================"
echo "smoke complete. artifacts in $OUT"
ls -la "$OUT"
echo "============================================"
