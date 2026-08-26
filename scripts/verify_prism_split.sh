#!/bin/bash
# Re-run ONLY data/prism/split_data.py on the cached PRISM raw + current build,
# SHA-256 each output parquet, compare to the current split. NO build.py rerun
# (spec Section 2: build re-run is the expensive step and the determinism test
# already covers pipeline determinism; this catches post-hoc parquet tampering).
# Run through scripts/cluster_launch.sh so code comes from an immutable snapshot.
set -uo pipefail
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY
REPO=${TURING_RL_WORK_ROOT:?run via scripts/cluster_launch.sh}
PY=/home/lancewicki/miniconda3/envs/turing-rl-train/bin/python
TMP=$(mktemp -d /tmp/prism-verify.XXXXXX)
CUR_SPLIT=$TURING_RL_INPUT_DATA_ROOT/prism/full_s42_history_sft40_grpo60_test10
FRESH=$TMP/full_s42_history_sft40_grpo60_test10
OUT=$REPO/results/2026-07-08-judge-sweep/derived/split_verification.md
mkdir -p "$FRESH" "$(dirname "$OUT")"
export HF_HOME=/home/lancewicki/data/hf_cache HF_HUB_CACHE=/home/lancewicki/data/hf_cache
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
cd "$REPO"
echo "=== re-split only (upstream data.prism.split_data) ==="
# CONFIRMED flags (data/prism/split_data.py): --input-dir, --output-dir,
# --heldout-user-frac, --grpo-frac, --seed. To reproduce the CURRENT split
# byte-identically, pass the SAME args scripts/slurm/split_prism_full_s42.sh used
# (mirror that script rather than relying on defaults).
$PY -u -m data.prism.split_data --input-dir "$TURING_RL_INPUT_DATA_ROOT/prism/full_s42_history" \
    --output-dir "$FRESH" --seed 42 || { echo "split failed"; exit 2; }
STATUS=0
{ echo "# PRISM split verification (re-split hash compare)"; echo;
  echo "Date: $(date -Iseconds)"; echo;
  echo '| File | current | fresh | match |'; echo '|---|---|---|---|'; } > "$OUT"
for f in sft/train.parquet grpo/train.parquet grpo/val.parquet test.parquet; do
    cur=$CUR_SPLIT/$f; fresh=$FRESH/$f
    if [ ! -f "$cur" ] || [ ! -f "$fresh" ]; then
        echo "| $f | MISSING | MISSING | FAIL |" >> "$OUT"; STATUS=1; continue; fi
    cs=$(sha256sum "$cur" | cut -d' ' -f1); fs=$(sha256sum "$fresh" | cut -d' ' -f1)
    if [ "$cs" = "$fs" ]; then
        echo "| $f | ${cs:0:12} | ${fs:0:12} | OK |" >> "$OUT"
    else
        echo "| $f | ${cs:0:12} | ${fs:0:12} | FAIL |" >> "$OUT"; STATUS=1
    fi
done
echo >> "$OUT"; echo "## pytest 7-check suite" >> "$OUT"
if $PY -m pytest tests/test_prism_split_verification.py -v > "$(dirname "$OUT")/split_verification_pytest.out" 2>&1; then
    echo "all 7 checks passed" >> "$OUT"
else
    echo "FAILURES (see split_verification_pytest.out)" >> "$OUT"; STATUS=1
fi
cat "$OUT"; rm -rf "$TMP"; exit $STATUS
