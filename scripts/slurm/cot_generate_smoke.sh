#!/bin/bash
#SBATCH --job-name=cot_gen
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/home/lancewicki/projects/turing-rl/logs/cot_generate_smoke-%j.out
#SBATCH --partition=a100
#SBATCH --account=rfai

# Runs data.sft.generate_cot then data.sft.build_sft_jsonl against the CoT server
# launched by cot_server.sh. The launcher exports COT_HOST + COT_PORT into this job.

set -uo pipefail

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY

if [ -z "${COT_HOST:-}" ]; then
  echo "ERROR: COT_HOST not set. This script is invoked by launch_cot_smoke.sh." >&2
  exit 2
fi
COT_PORT="${COT_PORT:-8000}"

# Source HF_TOKEN etc. (.env). We override the OpenAI-related vars so the repo's
# api_client targets our self-hosted server instead of OpenRouter.
if [ -f /home/lancewicki/projects/turing-rl/.env ]; then
  set -a
  # shellcheck disable=SC1091
  source /home/lancewicki/projects/turing-rl/.env
  set +a
fi

export OPENAI_API_BASE="http://${COT_HOST}:${COT_PORT}/v1"
export OPENAI_API_KEY="dummy-self-hosted"
# Drop the OpenRouter provider-routing extra (vllm doesn't understand it).
export OPENROUTER_PROVIDER_ORDER=""
unset OPENROUTER_API_KEY
# Generous timeout — Qwen3 thinking can be slow on long prompts.
export PERSONA_OPENAI_TIMEOUT_SECONDS=600
export PERSONA_OPENAI_MAX_RETRIES=3

export HF_HUB_DISABLE_XET=1
export HF_HOME=/home/lancewicki/data/hf_cache
export HF_HUB_CACHE=/home/lancewicki/data/hf_cache

PY=/home/lancewicki/miniconda3/envs/turing-rl-train/bin/python
REPO=/home/lancewicki/projects/turing-rl
IN_PARQUET="$REPO/data/prism/history_smoke/train.parquet"
OUT_DIR="$REPO/data/sft"
COT_PARQUET="$OUT_DIR/qwen3-8b_prism_smoke_sft_cot.parquet"
COT_JSONL="$OUT_DIR/qwen3-8b_prism_smoke_sft_cot.jsonl"
mkdir -p "$OUT_DIR"

echo "============================================"
echo "CoT generation + SFT JSONL build"
echo "Date:        $(date)"
echo "Host:        $(hostname)"
echo "CoT server:  $OPENAI_API_BASE"
echo "Model:       Qwen/Qwen3-8B (matches generate_cot.py default key 'qwen/qwen3-8b' over the wire)"
echo "Input:       $IN_PARQUET"
echo "Output:      $COT_PARQUET"
echo "             $COT_JSONL"
echo "============================================"

# Curl probe — verify the server returns the expected schema (separate .reasoning,
# clean .content). If this errors out, abort before generating 138 rows.
echo "=== curl probe (single chat completion) ==="
PROBE_BODY=$(curl -sf --max-time 120 -X POST "$OPENAI_API_BASE/chat/completions" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-8B",
    "messages": [
      {"role":"system","content":"You reconstruct a user'\''s reasoning."},
      {"role":"user","content":"[CONTEXT] Hello\n[REPLY] Hi back!\nWrite the reasoning."}
    ],
    "max_completion_tokens": 256,
    "reasoning": {"enabled": true}
  }')
PROBE_RC=$?
if [ $PROBE_RC -ne 0 ]; then
  echo "Curl probe failed (rc=$PROBE_RC). Server may be down or schema-incompatible." >&2
  exit 3
fi
echo "--- probe response (first 2000 chars) ---"
echo "$PROBE_BODY" | head -c 2000
echo
echo "--- probe diagnostics ---"
$PY -c "
import json,sys
d = json.loads('''$PROBE_BODY''')
choice = (d.get('choices') or [{}])[0]
msg = choice.get('message') or {}
print('message keys      :', sorted(msg.keys()))
print('has .reasoning    :', 'reasoning' in msg)
print('content len       :', len(msg.get('content') or ''))
content = msg.get('content') or ''
print('content has <think>:', '<think>' in content)
"
echo

cd "$REPO"

echo "=== 1/2 generate_cot on smoke parquet (138 rows) ==="
$PY -m data.sft.generate_cot \
  --input  "$IN_PARQUET" \
  --output "$COT_PARQUET" \
  --model  "Qwen/Qwen3-8B" \
  --max_completion_tokens 4096 \
  --max_regen_attempts 10
RC1=$?
echo "generate_cot exit: $RC1"

if [ $RC1 -ne 0 ]; then
  echo "generate_cot failed; not running build_sft_jsonl." >&2
  exit $RC1
fi

echo ""
echo "=== 2/2 build_sft_jsonl ==="
$PY -m data.sft.build_sft_jsonl \
  --input_parquet "$COT_PARQUET" \
  --output_jsonl  "$COT_JSONL"
RC2=$?
echo "build_sft_jsonl exit: $RC2"

if [ $RC2 -eq 0 ]; then
  echo ""
  echo "=== summary ==="
  echo "CoT parquet: $COT_PARQUET ($(stat -c%s "$COT_PARQUET") bytes)"
  echo "SFT JSONL:   $COT_JSONL  ($(wc -l <"$COT_JSONL") rows)"
  CoT_META="${COT_PARQUET}.cot_metadata.json"
  if [ -f "$CoT_META" ]; then
    echo "metadata:    $CoT_META"
    cat "$CoT_META"
  fi

  INSPECT_OUT=/home/lancewicki/tmp/cot_examples/cot_smoke_inspection.txt
  mkdir -p "$(dirname "$INSPECT_OUT")"
  echo ""
  echo "=== 3/3 inspect generated data -> $INSPECT_OUT ==="
  $PY <<PYEOF
import json
import pandas as pd

cot_pq = "$COT_PARQUET"
sft_jl = "$COT_JSONL"
out_path = "$INSPECT_OUT"

lines = []
def w(s=""):
    lines.append(str(s))

df = pd.read_parquet(cot_pq)
w("=" * 80)
w(f"CoT smoke inspection")
w(f"CoT parquet: {cot_pq}")
w(f"SFT JSONL:   {sft_jl}")
w("=" * 80)
w(f"rows                     : {len(df)}")
w(f"columns                  : {list(df.columns)}")

def has_reasoning(extra):
    if not isinstance(extra, dict):
        return False
    val = extra.get("ground_truth_reasoning")
    return bool(val and str(val).strip())

extras = df["extra_info"].tolist()
rows_with_reasoning  = sum(1 for e in extras if has_reasoning(e))
rows_failed_guard    = sum(1 for e in extras if isinstance(e, dict) and e.get("thinking_trace_failed_leakage_guard"))
regen_attempts       = [e.get("thinking_trace_num_regen_attempts", 0) for e in extras if isinstance(e, dict)]
reasoning_lengths    = [len(str(e.get("ground_truth_reasoning") or "")) for e in extras if isinstance(e, dict)]
think_leaks          = sum(1 for e in extras if isinstance(e, dict) and "<think>" in str(e.get("ground_truth_reasoning") or ""))

w(f"rows with reasoning      : {rows_with_reasoning}/{len(df)}")
w(f"rows failed leakage guard: {rows_failed_guard}")
if regen_attempts:
    w(f"regen attempts (min/avg/max): {min(regen_attempts)} / {sum(regen_attempts)/len(regen_attempts):.2f} / {max(regen_attempts)}")
if reasoning_lengths:
    w(f"reasoning length chars (min/avg/max): {min(reasoning_lengths)} / {sum(reasoning_lengths)//len(reasoning_lengths)} / {max(reasoning_lengths)}")
w(f"rows where reasoning contains <think>: {think_leaks} (expected 0 since vllm splits via reasoning-parser)")
w("")

# Show 3 full examples end-to-end (first, middle, last)
indices = [0, len(df)//2, len(df)-1]
for idx in indices:
    row = df.iloc[idx].to_dict()
    extra = dict(row["extra_info"] or {})
    rm = dict(row["reward_model"] or {})
    w("=" * 80)
    w(f"EXAMPLE row {idx}")
    w("=" * 80)
    w(f"user_id          : {extra.get('user_id')}")
    w(f"post_id          : {extra.get('post_id')}")
    w(f"regen_attempts   : {extra.get('thinking_trace_num_regen_attempts')}")
    w(f"failed_guard     : {extra.get('thinking_trace_failed_leakage_guard')}")
    w(f"trace_model      : {extra.get('thinking_trace_model')}")
    w(f"--- ground_truth ---")
    w(str(rm.get("ground_truth")))
    w(f"--- ground_truth_reasoning ---")
    w(str(extra.get("ground_truth_reasoning")))
    w("")

# SFT JSONL: count + show first 3 assistant targets
with open(sft_jl) as f:
    jsonl_lines = f.readlines()
w("=" * 80)
w(f"SFT JSONL inspection ({len(jsonl_lines)} rows)")
w("=" * 80)
for idx in indices:
    rec = json.loads(jsonl_lines[idx])
    asst = rec["messages"][-1]
    w(f"--- row {idx} assistant target ({len(asst['content'])} chars, role={asst['role']}) ---")
    w(asst["content"])
    w("")

with open(out_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print(f"wrote inspection to: {out_path}")
PYEOF
fi

echo "============================================"
echo "Done at $(date)"
echo "============================================"
exit $RC2
