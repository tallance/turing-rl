#!/usr/bin/env bash
# Sync turing-rl SOURCE to the RFAI V3 cluster over the localhost:2223 SSH tunnel.
#
# Model: the Mac is the SOLE author; the cluster is a compute mirror. We ship the
# committed git tree (so what runs on the cluster always maps to a SHA) and stamp
# that SHA on the cluster as DEPLOYED_SHA. We NEVER touch cluster-generated
# artifacts (checkpoints/, results/, logs/, wandb/) — those are gitignored, so they
# aren't in the archive, and `tar -x` only overlays, it never deletes.
#
# Usage:
#   scripts/sync_to_cluster.sh                 # full sync of committed HEAD (default; requires clean tree)
#   scripts/sync_to_cluster.sh --wip           # sync the WHOLE working tree incl. uncommitted edits, NO commit (iteration)
#   scripts/sync_to_cluster.sh path/a.py b.sh  # quick DEBUG push of specific files (allows dirty; not authoritative)
#
# The full sync REFUSES a dirty tree so DEPLOYED_SHA is meaningful. The partial
# mode is for tight debug loops only — it stamps DEPLOYED_SHA as "<sha>-dirty" so
# provenance stays honest. Always run a full sync before any run whose output you keep.
set -uo pipefail

LOCAL="${TURING_RL_LOCAL:-/Users/lancewicki/Projects/turing-rl}"
REMOTE="${TURING_RL_REMOTE:-/home/lancewicki/projects/turing-rl}"
SSH_OPTS=(-p 2223 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=10)
HOST="lancewicki@localhost"

cd "$LOCAL" || { echo "ERROR: no local repo at $LOCAL" >&2; exit 1; }

# --- tunnel up? ---
if ! ssh "${SSH_OPTS[@]}" "$HOST" true 2>/dev/null; then
  echo "ERROR: cluster tunnel (localhost:2223) is down." >&2
  echo "  Open it in a separate terminal: ssh -L 2223:localhost:22 rfai-research-aws-use2-1 -N" >&2
  echo "  (if it refuses: cloud_corp hpc login rfai-research-aws-use2-1, then reopen)" >&2
  exit 1
fi

verify_syntax() {  # $1 = space-separated file list piped via stdin; runs py_compile / bash -n on cluster
  local kind="$1"
  ssh "${SSH_OPTS[@]}" "$HOST" "cd '$REMOTE' && fail=0; while IFS= read -r f; do
      [ -f \"\$f\" ] || continue
      case \"\$f\" in
        *.py) python3 -m py_compile \"\$f\" 2>/dev/null || { echo \"    PYFAIL \$f\"; fail=1; } ;;
        *.sh) bash -n \"\$f\" 2>/dev/null || { echo \"    SHFAIL \$f\"; fail=1; } ;;
      esac
    done; exit \$fail"
}

# ============================ WIP MODE (--wip) ============================
# Ship the whole current working tree (committed + uncommitted tracked edits) with
# NO commit. Uses `git stash create` to snapshot the dirty tree into a throwaway
# commit object (does not touch your working tree/index), so only tracked files
# ship (gitignore honored). Brand-new UNTRACKED files are excluded — `git add`
# them first if needed. Stamps DEPLOYED_SHA as "<sha>-wip" (not reproducible).
if [ "${1:-}" = "--wip" ]; then
  base="$(git rev-parse HEAD)"
  wip_ref="$(git stash create || true)"   # empty when tree is clean
  [ -n "$wip_ref" ] || wip_ref=HEAD
  echo "[sync] WIP sync of current working tree (uncommitted tracked edits) -> $HOST:$REMOTE"
  echo "       base=$base  (brand-new untracked files are NOT included — git add them first)"
  if ! git archive --format=tar "$wip_ref" | ssh "${SSH_OPTS[@]}" "$HOST" "tar -xf - -C '$REMOTE'"; then
    echo "ERROR: archive/extract failed" >&2; exit 4
  fi
  printf '%s-wip\n' "$base" | ssh "${SSH_OPTS[@]}" "$HOST" "cat > '$REMOTE/DEPLOYED_SHA'"
  echo "[sync] wrote DEPLOYED_SHA=${base}-wip (iteration build — run a full sync before any kept run)"
  git ls-files -- '*.py' '*.sh' | verify_syntax "wip"; vrc=$?
  [ "$vrc" -eq 0 ] && echo "[verify] all tracked .py/.sh compile clean on cluster ✓" \
                    || { echo "[verify] SYNTAX FAILURES above — fix before submitting jobs" >&2; }
  echo "[sync] wip done @ ${base}-wip"
  exit $vrc
fi

# ============================ PARTIAL DEBUG MODE ============================
if [ "$#" -gt 0 ]; then
  echo "[sync] PARTIAL debug push of $# file(s) — NOT authoritative; run a full sync before any kept run."
  for f in "$@"; do
    [ -f "$LOCAL/$f" ] || { echo "  ERROR: missing local file: $f" >&2; exit 3; }
    ssh "${SSH_OPTS[@]}" "$HOST" "mkdir -p '$REMOTE/$(dirname "$f")' && cat > '$REMOTE/$f'" < "$LOCAL/$f" \
      && echo "  pushed $f"
  done
  printf '%s\n' "$@" | verify_syntax "partial"; vrc=$?
  sha="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
  printf '%s-dirty (partial: %s)\n' "$sha" "$*" | ssh "${SSH_OPTS[@]}" "$HOST" "cat > '$REMOTE/DEPLOYED_SHA'"
  echo "[sync] DEPLOYED_SHA marked dirty (partial push). verify rc=$vrc"
  exit $vrc
fi

# ============================ FULL SYNC (default) ============================
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "ERROR: working tree has uncommitted changes — commit first so DEPLOYED_SHA is meaningful," >&2
  echo "       or pass specific file paths for a throwaway debug push." >&2
  git status --short >&2
  exit 2
fi

SHA="$(git rev-parse HEAD)"
echo "[sync] full source sync of committed HEAD=$SHA -> $HOST:$REMOTE"
echo "       (cluster artifacts checkpoints/ results/ logs/ wandb/ are untouched)"

# Ship the committed tree; tar overlays onto the cluster (no deletes).
if ! git archive --format=tar HEAD | ssh "${SSH_OPTS[@]}" "$HOST" "tar -xf - -C '$REMOTE'"; then
  echo "ERROR: archive/extract failed" >&2; exit 4
fi

printf '%s\n' "$SHA" | ssh "${SSH_OPTS[@]}" "$HOST" "cat > '$REMOTE/DEPLOYED_SHA'"
echo "[sync] wrote DEPLOYED_SHA=$SHA on cluster"

# Verify syntax of every tracked .py / .sh on the cluster (syntax-only; safe, fast).
git ls-files -- '*.py' '*.sh' | verify_syntax "full"; vrc=$?
if [ "$vrc" -eq 0 ]; then
  echo "[verify] all tracked .py/.sh compile clean on cluster ✓"
else
  echo "[verify] SYNTAX FAILURES above — fix before submitting jobs" >&2
fi
echo "[sync] done @ $SHA"
exit $vrc
