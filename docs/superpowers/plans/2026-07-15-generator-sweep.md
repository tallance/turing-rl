# Generator Sweep Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Score generators through the full qwen3.5 judge matrix (with the cot-failure
fixes), serialized on a single node, and plot one accuracy/parse-error curve per generator.
Target set = {qwen3-8B, qwen3.5-9B} × {base, SFT}, **but the qwen3.5-9B SFT cell is deferred**
(the SFT env lacks `model_type=qwen3_5` support) — this run does **3 generators**: qwen3-8B
base, qwen3.5-9B base, qwen3-8B SFT (existing pairs).

**Architecture:** Reuse the judge-sweep machinery unchanged (`judge_sweep_cell.sh`,
`run_judge_sweep_cell.py`, `build_judge_pairs.py`, `configs/judge_sweep_cells.cell_list`).
New pieces: a 9B SFT config, an additive `--base_model` flag on `eval/generate_trained.py`,
a per-generator generation sbatch, a single dependency-chain orchestrator, and a
comparison analyzer. Everything runs as one `--dependency` chain so ≤1 node is ever
allocated (2 nodes stay free for a concurrent agent).

**Tech Stack:** Python 3 (pandas/numpy/matplotlib, pytest), vLLM (`turing-rl-train` +
`judge-vllm` envs), TRL `SFTTrainer` + PEFT LoRA, Slurm (A100-40GB, partition `a100`).

**Spec:** `docs/superpowers/specs/2026-07-15-generator-sweep-design.md`.

**Conventions (from repo CLAUDE.md):**
- Mac is sole author: edit → commit → `scripts/sync_to_cluster.sh` → run on cluster via the
  tunnel `ssh -p 2223 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null lancewicki@localhost "<cmd>"`.
- Read remote files via SSH `cat`, not the Read tool.
- Run the `preflight-job-check` skill before any `sbatch`; ≤10 concurrent jobs; never
  `scancel` another agent's job without approval.
- Additive commits only. Cluster repo: `/home/lancewicki/projects/turing-rl` (`$REPO`).
- Python: `/home/lancewicki/miniconda3/envs/turing-rl-train/bin/python` (numpy/pandas/pytest
  live here; the Mac has none — **all pytest runs on the cluster**).

**Generator registry (used throughout):**

| gen key | `--model_id` | ckpt / base | pairs |
|---|---|---|---|
| `qwen3-8b-base`  | `Qwen/Qwen3-8B`   | `--base_model` | generate |
| `qwen35-9b-base` | `Qwen/Qwen3.5-9B` | `--base_model` | generate |
| `qwen35-9b-sft`  | `Qwen/Qwen3.5-9B` | `checkpoints/sft/qwen35_9b_prism_full_s42_bf16_fsdp_nopack` (trained in Task 2) | generate |
| `qwen3-8b-sft`   | `Qwen/Qwen3-8B`   | existing `checkpoints/sft/qwen3_8b_prism_full_s42_bf16_fsdp_nopack` | **reuse** `results/2026-07-08-judge-sweep/raw/pairs/prism_heldout_880.parquet` |

Results root: `results/2026-07-15-generator-sweep/` (call it `$GROOT`).
Judge cells: `cell_list("qwen3.5")` = `qwen35-4b, qwen35-9b, qwen35-27b, qwen35-35b-a3b,
qwen35-122b, qwen35-397b` × modes `{off,on}`.

**Output layout — one standard sweep subtree per generator.** Each generator's cells use
**bare judge-cell names** under a per-generator `SWEEP_ROOT`:
`$GROOT/raw/<genkey>/sweep/<judgecell>/<mode>/{reward,http}`. This makes every generator's
subtree a self-contained judge sweep (cell names are in `SIZE_MAP`), so the existing
`analyze_judge_sweep.py` + `plot_field_compliance.py` run on it unchanged to produce the
**full per-model plot set** (`accuracy`, `accuracy_penalized`, `parse_error_rate`,
`tie_rate`, `budget_hit_rate`, `kappa_vs_anchor`, `position_bias_delta`,
`position_bias_signed`, `rating_distribution`, `rubric_complete_by_model`) per generator.

---

## Task 1: `--base_model` flag on `eval/generate_trained.py` (base-model generation)

Base generators have no adapter; `generate_trained.py` currently raises at line ~645. Add an
additive flag that runs the base model with `enable_lora=False`.

**Files:**
- Modify: `eval/generate_trained.py` (argparse + adapter-resolution block in `main` + the
  adapter-name output block)
- Test: `tests/test_generate_trained_base.py` (new)
- Modify: `our_patches.md`

**Step 1: Write the failing test.** The resolution logic is inline in `main`; extract the
decision into a tiny pure helper so it's unit-testable.

Create `tests/test_generate_trained_base.py`:

```python
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval.generate_trained import resolve_adapter_for_run


def test_base_model_flag_forces_no_adapter():
    # base_model=True => adapter is None regardless of checkpoint_dir
    assert resolve_adapter_for_run("checkpoints/whatever", base_model=True) is None


def test_missing_adapter_raises_when_not_base():
    try:
        resolve_adapter_for_run("/definitely/not/a/checkpoint/dir", base_model=False)
    except ValueError as e:
        assert "No LoRA adapter" in str(e)
    else:
        raise AssertionError("expected ValueError for missing adapter")
```

**Step 2: Run it, verify it fails** (on cluster):

```bash
ssh -p 2223 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null lancewicki@localhost \
  "cd /home/lancewicki/projects/turing-rl && /home/lancewicki/miniconda3/envs/turing-rl-train/bin/python -m pytest tests/test_generate_trained_base.py -q"
```
Expected: FAIL — `ImportError: cannot import name 'resolve_adapter_for_run'`.

**Step 3: Implement.** In `eval/generate_trained.py`, add the helper near
`find_latest_checkpoint` (after line ~156):

```python
def resolve_adapter_for_run(checkpoint_dir: str, base_model: bool) -> str | None:
    """Return the adapter path for this run, or None for base-model generation.

    base_model=True short-circuits to None (no LoRA). Otherwise resolve the latest
    checkpoint / adapter under checkpoint_dir and raise if none is found (existing
    behavior)."""
    if base_model:
        return None
    adapter_path = find_latest_checkpoint(checkpoint_dir)
    if adapter_path is None:
        adapter_path = resolve_adapter_path(checkpoint_dir)
    if adapter_path is None:
        raise ValueError(f"No LoRA adapter found under {checkpoint_dir}")
    return adapter_path
```

Add the argparse flag (next to `--checkpoint_dir`, and relax it to not required so base runs
needn't pass a dummy):

```python
    parser.add_argument("--base_model", action="store_true",
                        help="Generate from the base --model_id with no LoRA adapter.")
```
Change `--checkpoint_dir` `required=True` → `required=False, default=""`.

Replace the resolution block in `main` (lines ~641-646) with:

```python
    args.adapter_path = resolve_adapter_for_run(args.checkpoint_dir, args.base_model)
    if args.adapter_path is None:
        print(f"Base model (no adapter): {args.model_id}")
    else:
        print(f"Using checkpoint/adapter: {args.adapter_path}")
```

Guard the adapter-name output block (~line 688, `adapter_clean = args.adapter_path...`) so it
only runs `if args.adapter_path:` — we always pass `--output` explicitly, so a `None`
adapter must not be split.

**Step 4: Run tests, verify pass** (same pytest command as Step 2). Expected: 2 passed.

**Step 5: Document + commit.** Append to `our_patches.md` a `PERSISTENT` entry: "add
`--base_model` to `eval/generate_trained.py` for no-adapter base generation (generator
sweep); no-op unless the flag is passed."

```bash
git add eval/generate_trained.py tests/test_generate_trained_base.py our_patches.md
git commit -m "feat: --base_model flag for base-model heldout generation"
```

---

## Task 2: qwen3.5-9B SFT config + launcher branch

**Files:**
- Create: `training/sft/configs/qwen35_9b_lora.yaml`
- Modify: `scripts/slurm/sft_variant.sh`
- Test: `tests/test_qwen35_9b_sft_config.py` (new)

**Step 1: Write the failing test** — assert the 9B config matches paper Table 5 LoRA and
targets the 9B base.

Create `tests/test_qwen35_9b_sft_config.py`:

```python
import os, yaml
CFG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "training", "sft", "configs", "qwen35_9b_lora.yaml")


def test_qwen35_9b_lora_config():
    with open(CFG) as f:
        c = yaml.safe_load(f)
    assert c["lora_r"] == 64 and c["lora_alpha"] == 128 and c["lora_dropout"] == 0.05
    assert c["use_qlora"] is False          # bf16, paper Table 5
    assert c["num_epochs"] == 3
```

**Step 2: Run it, verify it fails** (cluster pytest): FAIL — file not found.

**Step 3: Implement.** Create `training/sft/configs/qwen35_9b_lora.yaml` (clone of the 8B
config; LoRA/optim identical — only the base model differs, and that is passed via the
launcher `--model`, so this file is byte-identical to `qwen3_8b_lora.yaml` except the header
comment):

```yaml
# Qwen3.5-9B LoRA SFT config (generator sweep).
# Identical hyperparameters to qwen3_8b_lora.yaml (paper Table 5): LoRA r=64/alpha=128,
# bf16 (no QLoRA), 3 epochs. Base model is selected by the launcher (--model qwen35-9b).
lora_r: 64
lora_alpha: 128
lora_dropout: 0.05
use_qlora: false

num_epochs: 3
batch_size: 1
gradient_accumulation_steps: 16
learning_rate: 2e-4
lr_scheduler: cosine
warmup_ratio: 0.05
weight_decay: 0.01

gradient_checkpointing: true
logging_steps: 10
report_to: none
save_strategy: steps
save_steps: 10
save_total_limit: 2
```

**How `lora_sft.py` selects model + config (verified):** `--model <alias>` →
`model_id = MODEL_MAP[args.model]` (line 348) and the config is **auto-derived** from the
alias: `config_name = args.model.replace("-","_").replace(".","_") + "_lora.yaml"` (line 339),
loaded from `training/sft/configs/`. So `--model qwen35-9b` auto-loads `qwen35_9b_lora.yaml`
— **there is no `--config` flag**. But `MODEL_MAP` only contains `qwen3-8b`, so we must add
the 9B entry. `get_lora_targets` maps any `"qwen"` name to the qwen3 target modules (works
for the 9B dense model). `--data_path`/`--output_dir` are passed explicitly and override the
alias-derived defaults.

**Edit `training/sft/lora_sft.py` (additive — one dict entry).** Add to `MODEL_MAP`:
```python
MODEL_MAP = {
    "qwen3-8b": "Qwen/Qwen3-8B",
    "qwen35-9b": "Qwen/Qwen3.5-9B",
}
```
Document this in `our_patches.md` (append a new section; upstream file).

**Parameterize `scripts/slurm/sft_variant.sh`** via a `MODEL` env (default `qwen3-8b` keeps
current behavior — no `--config` needed):
- After `VARIANT=...`/`NOPACK=...` add:
  ```bash
  MODEL=${MODEL:-qwen3-8b}   # qwen3-8b | qwen35-9b
  case "$MODEL" in
    qwen3-8b)  STEM=qwen3_8b ;;
    qwen35-9b) STEM=qwen35_9b ;;
    *) echo "bad MODEL=$MODEL"; exit 2 ;;
  esac
  ```
- In each `case "$VARIANT"` OUT line, replace the hard-coded `qwen3_8b` stem with `$STEM`
  (e.g. `OUT=$REPO/checkpoints/sft/${STEM}_prism_full_s42_bf16_fsdp`).
- In the `ARGS=(--model qwen3-8b ...)` line, replace `qwen3-8b` with `$MODEL`. **Do NOT add
  a `--config` flag** (auto-derived). Everything else (`--data_path`, FSDP flags, `--no_qlora`,
  etc.) is unchanged.

**Step 4: Verify** — `python3 -m py_compile` is N/A for yaml/shell; run `bash -n
scripts/slurm/sft_variant.sh` (clean) and confirm the MODEL_MAP edit compiles:
`python3 -m py_compile training/sft/lora_sft.py`. Cluster pytest for the config test is
deferred to the Task 7 gate. (A config-only SMOKE run — `SMOKE=1 MODEL=qwen35-9b
VARIANT=bf16_fsdp NOPACK=1 sbatch scripts/slurm/sft_variant.sh` — can validate config-load +
trainer-build after sync, but is optional; the real training run is submitted by the chain.)

**Step 5: Commit** (stage explicit paths only; append our_patches.md at EOF):
```bash
git add training/sft/configs/qwen35_9b_lora.yaml scripts/slurm/sft_variant.sh \
        training/sft/lora_sft.py our_patches.md tests/test_qwen35_9b_sft_config.py
git commit -m "feat: qwen3.5-9B SFT (config + MODEL_MAP entry + MODEL-parameterized launcher)"
```

---

## Task 3: model-id support for the 9B + per-generator generation sbatch

A parameterized 1-GPU generation job (base or adapter), templated on `heldout_inference.sh`,
plus the model-id support needed so `generate_trained.py` accepts `Qwen/Qwen3.5-9B`.

**Files:**
- Modify: `shared/model_ids.py` (support the 9B generator id)
- Modify: `our_patches.md` (APPEND ONLY)
- Create: `scripts/slurm/generator_infer.sh`

**Step 0: Support `Qwen/Qwen3.5-9B` in `shared/model_ids.py` (additive).**
`normalize_model_id()` RAISES for any id not in `SUPPORTED_MODEL_IDS`
(`{Qwen/Qwen3-8B, Qwen/Qwen3.5-397B-A17B}`), so `generate_trained.py --model_id
Qwen/Qwen3.5-9B` would fail. Add the 9B as a supported id + alias:
```python
DEFAULT_MODEL_ID = "Qwen/Qwen3-8B"
QWEN3_5_MOE_MODEL_ID = "Qwen/Qwen3.5-397B-A17B"
QWEN3_5_9B_MODEL_ID = "Qwen/Qwen3.5-9B"
SUPPORTED_MODEL_IDS = {DEFAULT_MODEL_ID, QWEN3_5_MOE_MODEL_ID, QWEN3_5_9B_MODEL_ID}
MODEL_ID_ALIASES = {
    "qwen3-8b": DEFAULT_MODEL_ID,
    "qwen3.5-397b": QWEN3_5_MOE_MODEL_ID,
    "qwen35-9b": QWEN3_5_9B_MODEL_ID,
}
```
`qwen3-8B` base needs no change (already `DEFAULT_MODEL_ID`). Document in `our_patches.md`
(append). Note: vLLM in `turing-rl-train` already served Qwen3.5-9B as a judge, so offline
text generation is expected to load; a failure here surfaces only on the (early) 9B-base gen
job and does not block the qwen3-8B generators.

**Step 1: Implement.** Create `scripts/slurm/generator_infer.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=gen_infer
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/home/lancewicki/projects/turing-rl/logs/gen_infer-%j.out
#SBATCH --partition=a100
#SBATCH --account=rfai
# Per-generator heldout candidate generation for the generator sweep.
# Required env: GEN_KEY MODEL_ID   Optional: CKPT (empty => --base_model)
# Uses gpu:8 (one whole node) only so the single-node chain never overlaps a scoring job;
# vLLM uses TP=1 by default so 7 GPUs idle here — acceptable (chain serialization matters
# more than packing). Set --vllm_tensor_parallel_size 1.
set -uo pipefail
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY
export HF_HOME=/home/lancewicki/data/hf_cache HF_HUB_CACHE=/home/lancewicki/data/hf_cache PYTHONUNBUFFERED=1
PY=/home/lancewicki/miniconda3/envs/turing-rl-train/bin/python
REPO=/home/lancewicki/projects/turing-rl
GEN_KEY=${GEN_KEY:?set GEN_KEY}
MODEL_ID=${MODEL_ID:?set MODEL_ID}
CKPT=${CKPT:-}
TEST=$REPO/data/prism/full_s42_history_sft40_grpo60_test10/test.parquet
OUT_DIR=$REPO/results/2026-07-15-generator-sweep/raw/generator/$GEN_KEY
OUT=$OUT_DIR/heldout_inference.pkl
mkdir -p "$OUT_DIR"; cd "$REPO"
[ -f "$TEST" ] || { echo "ERROR: missing $TEST"; exit 2; }

BASE=(); [ -z "$CKPT" ] && BASE=(--base_model)
CK=(); [ -n "$CKPT" ] && CK=(--checkpoint_dir "$CKPT")

echo "=== generator_infer: GEN_KEY=$GEN_KEY MODEL_ID=$MODEL_ID CKPT=${CKPT:-<base>} ==="
$PY -u -m eval.generate_trained "${BASE[@]}" "${CK[@]}" --test_parquet "$TEST" \
    --model_id "$MODEL_ID" --gen_num 1 --output "$OUT" --conditioning_mode history \
    --vllm_tensor_parallel_size 1 --vllm_gpu_memory_utilization 0.6 --vllm_max_num_seqs 32
RC=$?
$PY -c "import json,os; json.dump({'gen_key':'$GEN_KEY','model_id':'$MODEL_ID',\
'checkpoint_dir':'${CKPT:-}','base_model':$([ -z "$CKPT" ] && echo true || echo false),\
'test_parquet':'$TEST','gen_num':1,'output':'$OUT',\
'slurm_job_id':os.environ.get('SLURM_JOB_ID')}, open('$OUT_DIR/gen_metadata.json','w'), indent=2)"
echo "=== exit: $RC ==="; exit $RC
```

**Step 2: Verify** `bash -n scripts/slurm/generator_infer.sh` is clean; confirm the TEST
path exists on the cluster:
```bash
ssh ... "ls -la $REPO/data/prism/full_s42_history_sft40_grpo60_test10/test.parquet"
```

**Step 3: Commit** (stage explicit paths; append our_patches.md at EOF).
```bash
git add shared/model_ids.py scripts/slurm/generator_infer.sh our_patches.md
git commit -m "feat: support Qwen3.5-9B model-id + per-generator generation sbatch"
```

---

## Task 4: pair-build sbatch (tiny CPU job)

`build_judge_pairs.py` is CPU-only; wrap it so it can be a chained job.

**Files:**
- Create: `scripts/slurm/build_pairs.sh`

**Step 1: Implement.** Create `scripts/slurm/build_pairs.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=build_pairs
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=/home/lancewicki/projects/turing-rl/logs/build_pairs-%j.out
#SBATCH --partition=a100
#SBATCH --account=rfai
# Build a generator's (human, generated) 880 pair-set. Required env: GEN_KEY
set -uo pipefail
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY
PY=/home/lancewicki/miniconda3/envs/turing-rl-train/bin/python
REPO=/home/lancewicki/projects/turing-rl
GEN_KEY=${GEN_KEY:?set GEN_KEY}
PKL=$REPO/results/2026-07-15-generator-sweep/raw/generator/$GEN_KEY/heldout_inference.pkl
TEST=$REPO/data/prism/full_s42_history_sft40_grpo60_test10/test.parquet
OUT=$REPO/results/2026-07-15-generator-sweep/raw/pairs/gen_${GEN_KEY}_880.parquet
mkdir -p "$(dirname "$OUT")"; cd "$REPO"
$PY scripts/build_judge_pairs.py --inference_pkl "$PKL" --test_parquet "$TEST" --out "$OUT"
RC=$?; echo "=== exit: $RC ==="; exit $RC
```

**Step 2: Verify** `bash -n` clean.

**Step 3: Commit.**
```bash
git add scripts/slurm/build_pairs.sh
git commit -m "feat: pair-build sbatch wrapper (generator sweep)"
```

---

## Task 5: the single-node dependency-chain orchestrator

The core new piece. Submits SFT → (gen → build → 12 sweeps) per generator, all serialized
via `--dependency`, ≤1 node ever active.

**Files:**
- Modify: `scripts/slurm/judge_sweep_cell.sh` (make `SWEEP_ROOT` env-overridable)
- Create: `scripts/launch_generator_sweep.sh`

**Step 0: Make `SWEEP_ROOT` overridable.** In `scripts/slurm/judge_sweep_cell.sh:64`,
change the hardcoded assignment to honor an env override (additive — default unchanged):

```bash
# was: SWEEP_ROOT=$REPO/results/2026-07-08-judge-sweep/raw/sweep
SWEEP_ROOT=${SWEEP_ROOT:-$REPO/results/2026-07-08-judge-sweep/raw/sweep}
```
Verify: `ssh ... "grep -n 'SWEEP_ROOT=' $REPO/scripts/slurm/judge_sweep_cell.sh"`. Commit
this small edit with the orchestrator in Step 3.

**Step 1: Implement.** Create `scripts/launch_generator_sweep.sh`:

```bash
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
cd "$REPO"

# The cot-failure fixes. Export into the ENV so `--export=ALL` carries it to each job.
# Do NOT put this in the `--export=ALL,VAR=..,VAR=..` comma-list: Slurm --export is
# comma-delimited and the JSON's internal comma would split/corrupt it.
export PERSONA_JUDGE_SAMPLING='{"repetition_penalty":1.1,"temperature":0.6}'
EXISTING_8BSFT_PAIRS=$REPO/results/2026-07-08-judge-sweep/raw/pairs/prism_heldout_880.parquet

# Judge cells (cell_name model_id tp replicas), incl. the 397B anchor.
CELLS=$($PY -c "
from configs.judge_sweep_cells import cell_list
for c in cell_list('qwen3.5'):
    print(c['cell_name'], c['model_id'], c['tp'], c['replicas'])
")

PREV=""   # running tail of the chain (set by CALLERS after each submit)

# submit <dependency-or-empty> <sbatch-args...> ; echoes the job id ONLY.
# NOTE: callers invoke this via $(...), a subshell — so submit() must NOT set PREV
# (the assignment would be lost). Each call site does `X=$(submit ...); PREV=$X`.
submit () {
  local dep="$1"; shift
  local depflag=(); [ -n "$dep" ] && depflag=(--dependency="$dep")
  if [ "$DRY" = "1" ]; then
    echo "[DRY] sbatch ${depflag[*]} $*" >&2
    echo "dry$RANDOM"; return 0
  fi
  sbatch --parsable "${depflag[@]}" "$@"
}

# --- generator branch: gen -> build -> sweeps. $1=genkey $2=model_id $3=ckpt(""=base)
#     $4=pairs_override("" => build one) $5=sft_dep("" or afterok:<jid>)
run_generator () {
  local gk="$1" mid="$2" ckpt="$3" pairs_override="$4" sft_dep="$5"
  local pairs gate=""
  if [ -n "$pairs_override" ]; then
    pairs="$pairs_override"                       # reuse existing pairs, no gen/build
  else
    # gen: afterany on PREV to serialize; AND afterok on SFT if this gen needs it.
    local gendep=""; [ -n "$PREV" ] && gendep="afterany:$PREV"
    [ -n "$sft_dep" ] && gendep="${gendep:+$gendep,}$sft_dep"
    local gjid; gjid=$(submit "$gendep" --gres=gpu:8 --job-name=gen_${gk} \
      --export=ALL,GEN_KEY=$gk,MODEL_ID=$mid,CKPT=$ckpt scripts/slurm/generator_infer.sh)
    PREV="$gjid"; echo "submitted gen $gk -> $gjid" >&2
    local bjid; bjid=$(submit "afterok:$gjid" --gres=gpu:0 --job-name=build_${gk} \
      --export=ALL,GEN_KEY=$gk scripts/slurm/build_pairs.sh)
    PREV="$bjid"; echo "submitted build $gk -> $bjid" >&2
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
      PREV="$sjid"; echo "submitted sweep $gk $cell_name $mode -> $sjid (gpu:$gpus)" >&2
    done
  done <<< "$CELLS"
}

# ---- generators (3 for now; qwen35-9b-sft DEFERRED — see note below) ----
# DEFERRED: the qwen3.5-9B SFT generator needs an SFT env that supports model_type=qwen3_5
# (turing-rl-train transformers 4.57.6 does not; model is also multimodal). When that lands,
# re-add the SFT job + `run_generator qwen35-9b-sft Qwen/Qwen3.5-9B "$SFT_OUT" "" "$SFT_DEP"`.
run_generator qwen3-8b-base  Qwen/Qwen3-8B    ""  ""                       ""
run_generator qwen35-9b-base Qwen/Qwen3.5-9B  ""  ""                       ""
run_generator qwen3-8b-sft   Qwen/Qwen3-8B    ""  "$EXISTING_8BSFT_PAIRS"  ""

echo "chain tail job: $PREV" >&2
```

> **Deferred #3:** the `run_generator` signature keeps its 5th arg (`sft_dep`) unused here so
> re-adding the SFT branch later is a one-line change. No SFT job is submitted now.

**Step 2: Verify** `bash -n scripts/launch_generator_sweep.sh` clean, then dry-run **on the
cluster** (needs `configs.judge_sweep_cells` import) and eyeball the plan:
```bash
ssh ... "cd $REPO && DRY=1 bash scripts/launch_generator_sweep.sh"
```
Expected (3 generators; #3 qwen35-9b-sft deferred): for qwen3-8b-base and qwen35-9b-base a
gen+build then 12 sweep lines each; for qwen3-8b-sft (reuses existing pairs) 12 sweep lines,
no gen/build. **36 sweep submissions + 2 gen + 2 build total; no SFT job.**

**Step 3: Commit.**
```bash
git add scripts/slurm/judge_sweep_cell.sh scripts/launch_generator_sweep.sh
git commit -m "feat: single-node dependency-chain orchestrator + SWEEP_ROOT override"
```

---

## Task 6: analyzer — full per-generator plot set + cross-generator comparison

Two responsibilities: (A) for **each** generator, run the existing `analyze_judge_sweep.py`
and `plot_field_compliance.py` on that generator's subtree → the **full 10-plot set** in
`derived/<gen>/plots/`; (B) read each generator's `summary.parquet` and draw comparison
plots (one line per generator) in `derived/compare/plots/`.

**Files:**
- Create: `scripts/analyze_generator_sweep.py`
- Test: `tests/test_analyze_generator_sweep.py`

**Step 1: Write the failing test** — the net-new pure logic is (a) discovering generator
subtrees and (b) assembling comparison rows from per-gen summary parquets.

Create `tests/test_analyze_generator_sweep.py`:

```python
import os, sys
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.analyze_generator_sweep import discover_generators, comparison_rows


def test_discover_generators(tmp_path):
    for g in ("qwen3-8b-base", "qwen35-9b-sft"):
        (tmp_path / g / "sweep" / "qwen35-397b" / "on" / "reward").mkdir(parents=True)
    (tmp_path / "pairs").mkdir()                       # non-generator dir ignored
    assert discover_generators(tmp_path) == ["qwen35-9b-sft", "qwen3-8b-base"] or \
           sorted(discover_generators(tmp_path)) == ["qwen3-8b-base", "qwen35-9b-sft"]


def test_comparison_rows(tmp_path):
    # one fake per-generator summary.parquet -> flat comparison rows tagged by generator
    d = tmp_path / "derived" / "qwen3-8b-base"; d.mkdir(parents=True)
    pd.DataFrame([{"cell": "qwen35-397b", "mode": "on", "accuracy": 0.72,
                   "accuracy_penalized": 0.70, "parse_error_rate": 0.03}]
                 ).to_parquet(d / "summary.parquet")
    rows = comparison_rows(tmp_path / "derived", ["qwen3-8b-base"])
    assert rows[0]["generator"] == "qwen3-8b-base"
    assert rows[0]["judge"] == "qwen35-397b" and rows[0]["accuracy"] == 0.72
```

**Step 2: Run it, verify it fails** (cluster pytest): FAIL — module/function missing.

**Step 3: Implement.** Create `scripts/analyze_generator_sweep.py`:

```python
"""Generator-sweep analyzer.

(A) Per generator, run the existing analyzers on that generator's standard sweep subtree
    (raw/<gen>/sweep/<judgecell>/<mode>) -> the full per-model plot set in derived/<gen>/.
(B) Read each derived/<gen>/summary.parquet and draw cross-generator comparison plots
    (one line per generator) in derived/compare/plots/.
"""
from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO_ROOT))
import pandas as pd
from configs.judge_sweep_cells import SIZE_MAP

PY = sys.executable
GEN_LABELS = {
    "qwen3-8b-base": "qwen3-8B base", "qwen3-8b-sft": "qwen3-8B SFT",
    "qwen35-9b-base": "qwen3.5-9B base", "qwen35-9b-sft": "qwen3.5-9B SFT",
}
GEN_ORDER = ["qwen3-8b-base", "qwen3-8b-sft", "qwen35-9b-base", "qwen35-9b-sft"]
# Comparison metrics (must be columns in analyze_judge_sweep's summary.parquet).
CMP_METRICS = [
    ("accuracy", "accuracy | parse ok (picks true human)", (0.45, 0.85), 0.5),
    ("accuracy_penalized", "accuracy (parse-fail counted wrong)", (0.45, 0.85), 0.5),
    ("parse_error_rate", "parse-error rate", None, None),
    ("tie_rate", "tie rate (rating==4)", None, None),
]


def discover_generators(raw_root: Path) -> list[str]:
    """Sub-dirs of raw_root that contain a sweep/ dir (i.e. a generator subtree)."""
    return sorted(p.name for p in raw_root.iterdir()
                  if p.is_dir() and (p / "sweep").is_dir())


def comparison_rows(derived_root: Path, generators: list[str]) -> list[dict]:
    """Flatten each generator's summary.parquet into rows tagged by generator + judge."""
    rows: list[dict] = []
    for gen in generators:
        pq = derived_root / gen / "summary.parquet"
        if not pq.exists():
            print(f"[gen-analyzer] no summary for {gen} (skipping in comparison)", flush=True)
            continue
        for rec in pd.read_parquet(pq).to_dict("records"):
            rec = dict(rec); rec["generator"] = gen; rec["judge"] = rec.get("cell")
            rows.append(rec)
    return rows


def run_per_generator_analyzers(raw_root: Path, derived_root: Path, gen: str) -> None:
    """Full per-model plot set for one generator via the existing two analyzers."""
    graw, gderived = raw_root / gen, derived_root / gen
    subprocess.run([PY, "scripts/analyze_judge_sweep.py",
                    "--raw_root", str(graw), "--derived_root", str(gderived)],
                   cwd=REPO_ROOT, check=True)
    subprocess.run([PY, "scripts/plot_field_compliance.py",
                    "--raw_root", str(graw), "--out_dir", str(gderived / "plots")],
                   cwd=REPO_ROOT, check=True)


def write_comparison_plots(rows: list[dict], out_dir: Path) -> None:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    out_dir.mkdir(parents=True, exist_ok=True)
    by = {(r["generator"], r["judge"], r["mode"]): r for r in rows}
    for mode in ("off", "on"):
        for metric, ylab, ylim, ref in CMP_METRICS:
            fig, ax = plt.subplots(figsize=(7, 5))
            for gen in GEN_ORDER:
                pts = [(SIZE_MAP[j], by[(gen, j, mode)].get(metric))
                       for (g, j, m) in by if g == gen and m == mode
                       and j in SIZE_MAP and by[(gen, j, mode)].get(metric) is not None]
                if not pts:
                    continue
                pts.sort(); xs, ys = zip(*pts)
                ax.plot(xs, ys, marker="o", label=GEN_LABELS.get(gen, gen))
            if ref is not None:
                ax.axhline(ref, ls="--", c="gray", lw=1)
            if ylim:
                ax.set_ylim(*ylim)
            ax.set_xscale("log"); ax.set_xlabel("judge active-params (B)")
            ax.set_ylabel(ylab); ax.set_title(f"{metric} — thinking {mode} (per generator)")
            ax.legend(); fig.tight_layout()
            fig.savefig(out_dir / f"cmp_{metric}_{mode}.png", dpi=130); plt.close(fig)


def main() -> None:
    base = REPO_ROOT / "results" / "2026-07-15-generator-sweep"
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_root", type=Path, default=base / "raw")
    ap.add_argument("--derived_root", type=Path, default=base / "derived")
    args = ap.parse_args()

    gens = discover_generators(args.raw_root)
    print(f"[gen-analyzer] generators: {gens}", flush=True)
    for gen in gens:                              # (A) full per-model plot set
        run_per_generator_analyzers(args.raw_root, args.derived_root, gen)
    rows = comparison_rows(args.derived_root, gens)   # (B) cross-generator comparison
    write_comparison_plots(rows, args.derived_root / "compare" / "plots")
    pd.DataFrame(rows).to_parquet(args.derived_root / "compare" / "comparison_summary.parquet",
                                  index=False)
    print(f"[gen-analyzer] per-gen plots in derived/<gen>/plots; comparison in "
          f"derived/compare/plots ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
```

> Confirm `analyze_judge_sweep.py`'s `summary.parquet` columns include `cell`, `mode`,
> `accuracy`, `accuracy_penalized`, `parse_error_rate`, `tie_rate` (read `write_summary` /
> `aggregate_cell` at `scripts/analyze_judge_sweep.py:121,204`). Adjust `CMP_METRICS` / the
> `judge = rec["cell"]` mapping if a column is named differently.

**Step 2: Run tests, verify pass** (cluster pytest). Expected: 2 passed. (The subprocess
per-generator path is integration-tested in Task 8, not unit-tested here.)

**Step 3: Commit.**
```bash
git add scripts/analyze_generator_sweep.py tests/test_analyze_generator_sweep.py
git commit -m "feat: generator-sweep analyzer (full per-gen plot set + comparison)"
```

---

## Task 7: deploy + launch the chain

**Step 1: Sync** the committed HEAD to the cluster:
```bash
scripts/sync_to_cluster.sh
ssh ... "cat /home/lancewicki/projects/turing-rl/DEPLOYED_SHA"   # == local HEAD
```

**Step 2: Run all cluster pytest** (Tasks 1/2/6):
```bash
ssh ... "cd $REPO && /home/lancewicki/miniconda3/envs/turing-rl-train/bin/python -m pytest \
  tests/test_generate_trained_base.py tests/test_qwen35_9b_sft_config.py tests/test_analyze_generator_sweep.py -q"
```
Expected: all pass.

**Step 3: Dry-run the orchestrator** (Task 5 Step 2) and eyeball the plan.

**Step 4: Preflight, then launch.** Run the `preflight-job-check` skill for
`generator_infer.sh`, `build_pairs.sh`, `sft_variant.sh` (MODEL=qwen35-9b), and
`judge_sweep_cell.sh`, then:
```bash
ssh ... "touch /tmp/sbatch_preflight_ok && cd $REPO && bash scripts/launch_generator_sweep.sh"
```
Record every job id.

**Step 5: Verify the chain is single-node.** `squeue --me` should show ≤1 RUNNING sweep job
at a time (others PENDING with `(Dependency)`):
```bash
ssh ... "squeue --me -o '%.10i %.24j %.8T %.10r'"
```

---

## Task 8: analyze + write up (after the chain lands)

The chain is multi-day. When enough cells have landed:

**Step 1:** Run the analyzer:
```bash
ssh ... "cd $REPO && /home/lancewicki/miniconda3/envs/turing-rl-train/bin/python scripts/analyze_generator_sweep.py"
```

**Step 2:** Sanity-check per generator: `gen_<gen>_880.parquet` `.meta.json` shows 880 rows
and `exact_match_frac < 0.01`; 397B-on parse-error ≈0.03 (rep_pen fix working).

**Step 3:** Confirm each generator got the **full plot set** (10 PNGs) and pull to the Mac:
```bash
# per-generator plot sets + the comparison set
ssh ... "for g in qwen3-8b-base qwen3-8b-sft qwen35-9b-base qwen35-9b-sft; do \
  echo \$g: \$(ls $REPO/results/2026-07-15-generator-sweep/derived/\$g/plots/*.png 2>/dev/null | wc -l) plots; done"
scp -P 2223 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -r \
  "lancewicki@localhost:$REPO/results/2026-07-15-generator-sweep/derived" \
  results/2026-07-15-generator-sweep/
```
Expected per generator: `accuracy`, `accuracy_penalized`, `parse_error_rate`, `tie_rate`,
`budget_hit_rate`, `kappa_vs_anchor`, `position_bias_delta`, `position_bias_signed`,
`rating_distribution`, `rubric_complete_by_model`. Plus `derived/compare/plots/cmp_*.png`.

**Step 4:** Write `results/2026-07-15-generator-sweep/README.txt` (repro commands, input
paths, job ids) per the reports-repro rule, and a short post-plan under
`docs/superpowers/post-plans/`. Commit.

---

## Verification summary

- Unit tests (cluster): base-model flag, 9B config, discover_generators + comparison_rows —
  all pass.
- `bash -n` clean on all new/edited shell scripts.
- Dry-run shows exactly: 2 gen + 2 build + 36 sweeps (3 generators; qwen35-9b-sft deferred, no SFT job).
- `squeue` confirms ≤1 node in use throughout the chain.
- Each generator's pairs = 880 rows, `exact_match_frac < 0.01`.
- 397B-on parse-error ≈0.03 (fix), ratings parse.
- **Every generator has the full 10-plot per-model set** in `derived/<gen>/plots/`, plus the
  cross-generator `derived/compare/plots/cmp_*.png` (one line per generator).
- Full 12 cells run for every generator — **no cell dropped** without explicit approval.
