#!/bin/bash
#SBATCH --job-name=sft_variant
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:8
#SBATCH --mem=0
#SBATCH --time=20:00:00
#SBATCH --output=/home/lancewicki/projects/turing-rl/logs/sft_variant-%j.out
#SBATCH --partition=a100
#SBATCH --account=rfai

# Parameterized multi-GPU (torchrun, 8-GPU) SFT launcher. Select the recipe via
# VARIANT env: qlora_r64 | bf16_fsdp | bf16_fa2. Each variant differs only by CLI
# flags — the committed qwen3_8b_lora.yaml stays read-only (no concurrent-sed race).
#   Pass VARIANT=bf16_fsdp through scripts/cluster_launch.sh and submit_snapshot_job.sh.
# SMOKE=1 does a fast config check (--exit_after_trainer_build --max_train_examples 64).
#
# Optional overrides, all unset by default (unset = the yaml / full dataset decides, i.e.
# today's exact arg list):
#   DATA, OUT, RUN_TAG      - see the inline comments below.
#   NOPACK=1                - --no_packing (forced on for the judge aliases).
#   EPOCHS=<n>              - --num_epochs <n>. The yaml says 3; an overfit gate needs many
#                             more over its handful of examples.
#   MAX_TRAIN_EXAMPLES=<n>  - --max_train_examples <n>. Takes precedence over the 64 that
#                             SMOKE=1 would otherwise pass, so the flag is never duplicated.
#   HF_HUB_OFFLINE=0        - allow a Hub round-trip. Defaults to 1; see the hf-env block.
#
# Also trains the judge discriminator's cross-entropy run: MODEL=qwen35-4b-judge or
# qwen35-9b-judge, with DATA pointed at the judge CE jsonl and OUT at a judge checkpoint
# dir. Both judge aliases force NOPACK=1 (see the MODEL case) — the CE target is a single
# A/B token and packing would let an example read its neighbour's answer.

set -uo pipefail
source "${TURING_RL_CODE_ROOT:?}/scripts/cluster_job_bootstrap.sh"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY

REPO=${TURING_RL_WORK_ROOT:?}

# Source any .env (WANDB creds, HF_TOKEN, etc.)
if [ -f "$REPO/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "$REPO/.env"
  set +a
fi

# >>> hf-env: extracted verbatim and executed by tests/test_sft_variant_launcher.py, so
# keep it to environment exports with no cluster calls. >>>
export HF_HOME=/home/lancewicki/data/hf_cache
export HF_HUB_CACHE=/home/lancewicki/data/hf_cache
export HF_HUB_DISABLE_XET=1
# HF_HUB_OFFLINE defaults ON deliberately. Do NOT "clean this line up".
# This cache stores the Qwen3.5 weights under a NONSTANDARD shard name —
# model.safetensors-00001-of-00002.safetensors — where the Hub's own convention is
# model-00001-of-00002.safetensors. The cached model.safetensors.index.json references the
# nonstandard names consistently, so an OFFLINE load of a fully-present model succeeds.
# ONLINE, transformers resolves the file list from the Hub, does not find the indexed shard
# name among the real (standard-named) files, and dies with
#   OSError: Qwen/Qwen3.5-4B does not appear to have a file named
#            model.safetensors-00001-of-00002.safetensors
# on a model that is 100% cached with zero *.incomplete blobs (job 19315, dead one minute
# into an 8-GPU allocation; 19316 passed straight through with this set). Qwen3.5-9B has the
# same layout, so this is not 4B-specific.
# Overridable for a genuine first-download run; ON by default because every model this
# launcher trains is pre-cached, and failing fast beats pulling 10 GB mid-allocation.
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
# <<< hf-env <<<
export PYTHONUNBUFFERED=1
export WANDB_MODE=online
export WANDB_PROJECT=turing-rl-sft
export WANDB_RUN_GROUP=sft-variants
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1

# >>> resolve-config: pure resolution, no cluster calls. tests/test_sft_variant_launcher.py
# extracts this block verbatim and executes it, so keep side effects out of it. >>>
PY=/home/lancewicki/miniconda3/envs/turing-rl-train/bin/python
# DATA override: the judge CE run trains on a different jsonl. Unset = the generator default.
DATA=${DATA:-$TURING_RL_GENERATED_DATA_ROOT/sft/prism_full_s42_sft_cot.jsonl}

VARIANT=${VARIANT:?set VARIANT=qlora_r64|bf16_fsdp|bf16_fa2}
SMOKE=${SMOKE:-0}
# NOPACK=1 disables trl sequence packing. Under sdpa, packing=True leaks attention
# across packed conversations (trl delegates isolation to FlashAttention varlen, which
# sdpa lacks). --no_packing gives clean per-conversation attention. Writes to a distinct
# "_nopack" output dir so it never resumes a packed run. DELIBERATE deviation from
# upstream (which packs) — documented in our_patches.md.
NOPACK=${NOPACK:-0}

# EPOCHS / MAX_TRAIN_EXAMPLES: lora_sft.py overrides the launcher could not previously reach.
# Empty = unset = not passed at all, so every existing invocation keeps its exact arg list.
# Validated here rather than left to argparse: these reach a multi-hour GPU job, and a
# typo'd EPOCHS=3O (letter O) must fail at submit, not minutes into a node allocation.
EPOCHS=${EPOCHS:-}
MAX_TRAIN_EXAMPLES=${MAX_TRAIN_EXAMPLES:-}
_positive_int_or_die() {  # name value; empty value = unset, accepted
  case "$2" in
    "") ;;
    *[!0-9]*) echo "bad $1=$2 (expected a positive integer)"; exit 2 ;;
    *) [ "$2" -gt 0 ] || { echo "bad $1=$2 (expected a positive integer)"; exit 2; } ;;
  esac
}
_positive_int_or_die EPOCHS "$EPOCHS"
_positive_int_or_die MAX_TRAIN_EXAMPLES "$MAX_TRAIN_EXAMPLES"

MODEL=${MODEL:-qwen3-8b}   # qwen3-8b | qwen35-9b | qwen35-4b-judge | qwen35-9b-judge |
                           # gemma4-12b-judge
# Per-model: output stem, python env, the FSDP auto-wrap decoder class, and (Gemma only) a
# corrected HF cache root. Empty default so `set -u` is satisfied for the Qwen aliases.
HF_HUB_CACHE_OVERRIDE=
# qwen3.5 needs its own transformers-5.x env (model_type=qwen3_5 unsupported by the 4.57.6
# in turing-rl-train) and a different decoder class (both verified via probe_qwen35.py).
# The judge aliases are Qwen3.5 too, so they take the same env and decoder class; their
# stems carry a judge_ prefix so a judge checkpoint can never land on a generator path.
case "$MODEL" in
  qwen3-8b)
    STEM=qwen3_8b
    PY=/home/lancewicki/miniconda3/envs/turing-rl-train/bin/python
    FSDP_LAYER_CLS=Qwen3DecoderLayer ;;
  qwen35-9b)
    STEM=qwen35_9b
    PY=/home/lancewicki/miniconda3/envs/turing-rl-sft-qwen35/bin/python
    FSDP_LAYER_CLS=Qwen3_5DecoderLayer ;;
  # Judge discriminator CE. NOPACK is FORCED, not defaulted: the supervised target is a
  # single A/B token at the end of each example, and under sdpa trl's packing=True leaks
  # attention across packed conversations — an example would read its neighbour's answer
  # letter. Leaving that to the caller is one forgotten env var away from a run that
  # measures nothing, so the alias decides it.
  qwen35-4b-judge)
    STEM=judge_qwen35_4b
    PY=/home/lancewicki/miniconda3/envs/turing-rl-sft-qwen35/bin/python
    FSDP_LAYER_CLS=Qwen3_5DecoderLayer
    NOPACK=1 ;;
  qwen35-9b-judge)
    STEM=judge_qwen35_9b
    PY=/home/lancewicki/miniconda3/envs/turing-rl-sft-qwen35/bin/python
    FSDP_LAYER_CLS=Qwen3_5DecoderLayer
    NOPACK=1 ;;
  # Gemma judge CE. Reuses the qwen35 SFT env on purpose: transformers 5.14.1 there already
  # resolves model_type=gemma4_unified, so no Gemma-specific training env is needed (the
  # gemma4 env in cluster_workflow.py is a vLLM SERVING env and is not in the sft profile).
  # The wrap class differs from the Qwen aliases because the container differs:
  # Gemma4UnifiedForConditionalGeneration nests 48 Gemma4UnifiedTextDecoderLayer under
  # model.language_model. NOPACK is forced for the same reason as the Qwen judges.
  gemma4-12b-judge)
    STEM=judge_gemma4_12b
    PY=/home/lancewicki/miniconda3/envs/turing-rl-sft-qwen35/bin/python
    FSDP_LAYER_CLS=Gemma4UnifiedTextDecoderLayer
    NOPACK=1
    # This cache holds TWO layouts. Qwen sits at the top level, which is why the hf-env
    # block above points HF_HUB_CACHE there; Gemma sits under hub/. Left alone, the Gemma
    # id still resolves — to a 31 MB config+tokenizer stub at the top level with NO weights
    # — so the run dies at model load AFTER Slurm has handed over 8 GPUs. Verified: under
    # the default HF_HUB_CACHE, config.json resolves and model.safetensors does not.
    # Overriding only for this alias keeps the Qwen aliases (which would break under hub/)
    # exactly as they were, and resolves snapshot 707f0a3b… — the same revision
    # judge_sweep_cell.sh pins for the zero-shot gemma4-12b cell, so the trained judge and
    # its baseline share a base. Resolved here as a plain value and exported below, so this
    # block stays the side-effect-free resolution its fence promises.
    HF_HUB_CACHE_OVERRIDE=/home/lancewicki/data/hf_cache/hub ;;
  *) echo "bad MODEL=$MODEL"; exit 2 ;;
esac

case "$VARIANT" in
  qlora_r64)
    DEFAULT_OUT=$REPO/checkpoints/sft/${STEM}_prism_full_s42_qlora_r64
    export WANDB_NAME=sft-qlora-r64
    ;;
  bf16_fsdp)
    DEFAULT_OUT=$REPO/checkpoints/sft/${STEM}_prism_full_s42_bf16_fsdp
    export WANDB_NAME=sft-bf16-fsdp
    ;;
  bf16_fa2)
    DEFAULT_OUT=$REPO/checkpoints/sft/${STEM}_prism_full_s42_bf16_fa2
    export WANDB_NAME=sft-bf16-fa2
    ;;
  *)
    echo "bad VARIANT=$VARIANT (expected qlora_r64|bf16_fsdp|bf16_fa2)"; exit 2 ;;
esac

# OUT override: the per-VARIANT default bakes the generator's prism_full_s42 dataset name,
# which is wrong for a judge run. Unset OUT keeps today's exact path. The _nopack / RUN_TAG
# suffixes below still apply on top, to an overridden OUT as much as to the default.
OUT=${OUT:-$DEFAULT_OUT}

if [ "$NOPACK" = "1" ]; then
  OUT="${OUT}_nopack"
  export WANDB_NAME="${WANDB_NAME}-nopack"
fi

# RUN_TAG (optional): a distinct suffix for the output dir + WANDB name so a re-run with a
# changed recipe (e.g. save_strategy=epoch + stop-token supervision) does NOT resume/clobber a
# prior run's dir under --resume_from_checkpoint auto. Default empty = original behavior.
RUN_TAG=${RUN_TAG:-}
if [ -n "$RUN_TAG" ]; then
  OUT="${OUT}_${RUN_TAG}"
  export WANDB_NAME="${WANDB_NAME}-${RUN_TAG}"
fi

# Build the arg list as an array so the quoted multi-word FSDP value survives intact.
ARGS=(--model "$MODEL" --data_path "$DATA" --output_dir "$OUT" --max_seq_length 8192
      --resume_from_checkpoint auto --report_to wandb --no_torch_compile)

case "$VARIANT" in
  qlora_r64) ARGS+=(--force_qlora --attn_implementation sdpa) ;;
  # bf16 variants pass --no_qlora explicitly so the recipe is self-describing and
  # robust to yaml drift (4-bit bnb + FSDP full_shard is a broken combo).
  bf16_fsdp) ARGS+=(--no_qlora --attn_implementation sdpa --fsdp "full_shard auto_wrap" --fsdp_transformer_layer_cls "$FSDP_LAYER_CLS") ;;
  bf16_fa2)  ARGS+=(--no_qlora --attn_implementation flash_attention_2) ;;
esac

# Iterative SFT: merge a previous LoRA into the base, then train a FRESH LoRA on top. This
# is how judge iter2 continues from iter1 instead of restarting from Qwen3.5-9B.
#
# The existence check is the whole point. --resume_from_checkpoint auto silently finds
# nothing when a path is wrong, and a typo'd BASE_ADAPTER would likewise train from BASE --
# producing a judge that looks entirely plausible, trains cleanly, and is simply not
# iteration 2. Refuse instead, the same way the frac10 arm refuses a colliding override.
BASE_ADAPTER=${BASE_ADAPTER:-}
if [ -n "$BASE_ADAPTER" ]; then
  [ -f "$BASE_ADAPTER/adapter_config.json" ] || {
    echo "ERROR: BASE_ADAPTER=$BASE_ADAPTER has no adapter_config.json." >&2
    echo "       Point it at a PEFT adapter dir (e.g. .../judge_qwen35_9b_ce_nopack/checkpoint-144)." >&2
    exit 2; }
  ARGS+=(--base_adapter "$BASE_ADAPTER")
fi

[ "$SMOKE" = "1" ] && ARGS+=(--exit_after_trainer_build)
# SMOKE's own cap yields to an explicit MAX_TRAIN_EXAMPLES so --max_train_examples is never
# passed twice; SMOKE=1 on its own still emits the same 64 it always did.
[ "$SMOKE" = "1" ] && [ -z "$MAX_TRAIN_EXAMPLES" ] && ARGS+=(--max_train_examples 64)
[ "$NOPACK" = "1" ] && ARGS+=(--no_packing)
[ -n "$EPOCHS" ] && ARGS+=(--num_epochs "$EPOCHS")
[ -n "$MAX_TRAIN_EXAMPLES" ] && ARGS+=(--max_train_examples "$MAX_TRAIN_EXAMPLES")
# <<< resolve-config <<<

# Applied outside the pure block; see the gemma4-12b-judge case for why it is needed.
[ -n "$HF_HUB_CACHE_OVERRIDE" ] && export HF_HUB_CACHE="$HF_HUB_CACHE_OVERRIDE"

mkdir -p "$OUT"
[ -f "$DATA" ] || { echo "ERROR: missing $DATA"; exit 2; }
cd "$REPO"

echo "============================================"
echo "SFT variant (torchrun, 8-GPU)"
echo "Date:    $(date)"
echo "Host:    $(hostname)"
echo "VARIANT: $VARIANT"
echo "MODEL:   $MODEL"
echo "Data:    $DATA"
echo "Output:  $OUT"
echo "Smoke:   $SMOKE"
echo "NoPack:  $NOPACK"
echo "Epochs:  ${EPOCHS:-<yaml>}"
echo "MaxEx:   ${MAX_TRAIN_EXAMPLES:-<all>}"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader | head -1
echo "============================================"

$PY -u -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 \
    -m training.sft.lora_sft "${ARGS[@]}"
RC=$?

echo ""
echo "============================================"
echo "SFT variant $VARIANT exit: $RC"
echo "Date:                     $(date)"
if [ $RC -eq 0 ]; then
  echo ""
  echo "=== output dir contents ==="
  ls -la "$OUT" 2>/dev/null
  echo ""
  echo "=== adapter files ==="
  find "$OUT" -maxdepth 3 -name 'adapter_model.safetensors' -printf '%p (%s bytes)\n'
fi
echo "============================================"
exit $RC
