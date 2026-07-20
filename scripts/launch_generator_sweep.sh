#!/bin/bash
# Single-node serialized generator sweep. Submits ONE dependency chain so at most one
# node is allocated at a time (2 nodes stay free for a concurrent agent).
#
# Chain per generator: [SFT] -> gen -> build -> {6 judge cells x 2 modes}.
# qwen3-8b-sft reuses the existing 880 pair-set (no gen/build).
#
# Serialization: each job depends afterany on the previous (PREV) so a 397B-on wall
# timeout doesn't abort the chain. WITHIN a generator, gen->build->first-sweep use
# afterok so we never sweep on a missing/failed pair-set.
#
#   bash scripts/launch_generator_sweep.sh            # submit the whole chain
#   DRY=1 bash scripts/launch_generator_sweep.sh      # print the plan, submit nothing
set -uo pipefail
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY
REPO=/home/lancewicki/projects/turing-rl
PY=/home/lancewicki/miniconda3/envs/turing-rl-train/bin/python
DRY=${DRY:-0}
GEN_ONLY=${GEN_ONLY:-}   # optional: run only this generator key (e.g. qwen35-9b-base)
cd "$REPO"

# The cot-failure fixes. Export into the ENV so `--export=ALL` carries it to each job.
# Do NOT put this in the `--export=ALL,VAR=..,VAR=..` comma-list: Slurm --export is
# comma-delimited and the JSON's internal comma would split/corrupt it.
export PERSONA_JUDGE_SAMPLING='{"repetition_penalty":1.1,"temperature":0.6}'
EXISTING_8BSFT_PAIRS=$REPO/results/2026-07-08-judge-sweep/raw/pairs/prism_heldout_880.parquet
# qwen3.5-9B SFT LoRA adapter (trained via sft_variant.sh MODEL=qwen35-9b; the FSDP end-save
# left only the tokenizer in final/, so the adapter weights were copied in from checkpoint-78).
SFT9B_ADP=$REPO/checkpoints/sft/qwen35_9b_prism_full_s42_bf16_fsdp_nopack/final

# Judge cells (cell_name model_id tp replicas), incl. the 397B anchor.
CELLS=$($PY -c "
from configs.judge_sweep_cells import cell_list
for c in cell_list('qwen3.5'):
    print(c['cell_name'], c['model_id'], c['tp'], c['replicas'])
")

# Running tail of the chain (set by CALLERS after each submit). Optionally SEED it with an
# existing job id via CHAIN_AFTER=<jid> so this launch queues BEHIND an already-running chain
# (keeps a single node in use across separately-launched generators).
PREV="${CHAIN_AFTER:-}"

# submit <dependency-or-empty> <sbatch-args...> ; echoes the job id ONLY.
# NOTE: callers invoke this via $(...), a subshell — so submit() must NOT set PREV
# (the assignment would be lost). Each call site does `X=$(submit ...); PREV=$X`.
submit () {
  local dep="$1"; shift
  # Plain string (NOT an array): an empty array expansion under `set -u` errors on
  # bash 3.2. deparg is empty (expands to nothing) for the head job, else one token
  # `--dependency=<dep>` with no spaces, so leaving it unquoted is safe.
  local deparg=""; [ -n "$dep" ] && deparg="--dependency=$dep"
  if [ "$DRY" = "1" ]; then
    echo "[DRY] sbatch $deparg $*" >&2
    echo "dry$RANDOM"; return 0
  fi
  sbatch --parsable $deparg "$@"
}

# Abort the chain if a submit returned no real job id. Otherwise PREV would go empty and
# every later job would launch with no dependency — un-serialized (multi-node) and against
# inputs that were never produced. In DRY mode any non-empty token is fine.
need_jid () {  # $1=captured id  $2=label
  if [ "$DRY" = "1" ]; then [ -n "$1" ] && return 0; fi
  case "$1" in
    ''|*[!0-9]*) echo "FATAL: sbatch failed for $2 (got '$1') — aborting chain" >&2; exit 1 ;;
  esac
}

# --- generator branch: gen -> build -> sweeps. $1=genkey $2=model_id $3=ckpt(""=base)
#     $4=pairs_override("" => build one) $5=sft_dep("" or afterok:<jid>)
run_generator () {
  local gk="$1" mid="$2" ckpt="$3" pairs_override="$4" sft_dep="$5" backend="${6:-vllm}"
  # GEN_ONLY filter: skip generators other than the requested one (single-generator launch).
  if [ -n "$GEN_ONLY" ] && [ "$GEN_ONLY" != "$gk" ]; then
    echo "skip $gk (GEN_ONLY=$GEN_ONLY)" >&2; return 0
  fi
  local pairs gate=""
  if [ -n "$pairs_override" ]; then
    pairs="$pairs_override"                       # reuse existing pairs, no gen/build
  else
    # gen: afterany on PREV to serialize; AND afterok on SFT if this gen needs it.
    local gendep=""; [ -n "$PREV" ] && gendep="afterany:$PREV"
    [ -n "$sft_dep" ] && gendep="${gendep:+$gendep,}$sft_dep"
    local gjid; gjid=$(submit "$gendep" --gres=gpu:8 --job-name=gen_${gk} \
      --export=ALL,GEN_KEY=$gk,MODEL_ID=$mid,CKPT=$ckpt,BACKEND=$backend scripts/slurm/generator_infer.sh)
    need_jid "$gjid" "gen $gk"; PREV="$gjid"; echo "submitted gen $gk -> $gjid" >&2
    local bjid; bjid=$(submit "afterok:$gjid" --gres=gpu:0 --job-name=build_${gk} \
      --export=ALL,GEN_KEY=$gk scripts/slurm/build_pairs.sh)
    need_jid "$bjid" "build $gk"; PREV="$bjid"; echo "submitted build $gk -> $bjid" >&2
    pairs=$REPO/results/2026-07-15-generator-sweep/raw/pairs/gen_${gk}_880.parquet
    gate="afterok:$bjid"      # first sweep waits afterok on build; rest afterany on PREV
  fi
  # Per-generator subtree of BARE judge-cell names => a standard sweep the existing
  # analyzers handle unchanged.
  local SWROOT=$REPO/results/2026-07-15-generator-sweep/raw/$gk/sweep
  while read -r cell_name model_id tp replicas; do
    [ -z "$cell_name" ] && continue
    local gpus=$((tp * replicas))
    for mode in off on; do
      local dep=""; [ -n "$PREV" ] && dep="afterany:$PREV"
      if [ -n "$gate" ]; then dep="$gate"; gate=""; fi   # first sweep uses the build gate
      # PERSONA_JUDGE_SAMPLING is carried by --export=ALL (exported above) — NOT in the
      # comma-list (its JSON comma would corrupt Slurm --export).
      local sjid; sjid=$(submit "$dep" --gres=gpu:$gpus --job-name=gsw_${gk}_${cell_name}_${mode} \
        --export=ALL,MODEL=$model_id,TP=$tp,REPLICAS=$replicas,THINKING_MODE=$mode,\
CELL_NAME=$cell_name,PAIRS=$pairs,SWEEP_ROOT=$SWROOT \
        scripts/slurm/judge_sweep_cell.sh)
      need_jid "$sjid" "sweep $gk $cell_name $mode"; PREV="$sjid"
      echo "submitted sweep $gk $cell_name $mode -> $sjid (gpu:$gpus)" >&2
    done
  done <<< "$CELLS"
}

# ---- generators (4; qwen35-9b-sft now trained — see below) ----
# qwen35-9b-sft: the adapter (SFT9B_ADP) was trained in a dedicated transformers-5.x env
# (turing-rl-sft-qwen35). vLLM 0.18 can't LoRA-serve its Gated-DeltaNet adapter, so it
# generates via BACKEND=hf (transformers+PEFT). The judge sweep for it is unchanged.
run_generator qwen3-8b-base  Qwen/Qwen3-8B    ""            ""                       ""  vllm
run_generator qwen35-9b-base Qwen/Qwen3.5-9B  ""            ""                       ""  vllm
run_generator qwen3-8b-sft   Qwen/Qwen3-8B    ""            "$EXISTING_8BSFT_PAIRS"  ""  vllm
run_generator qwen35-9b-sft  Qwen/Qwen3.5-9B  "$SFT9B_ADP"  ""                       ""  hf

echo "chain tail job: $PREV" >&2
