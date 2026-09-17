"""Extract a fixed 100-pair subset of the step-320 test-eval reward rows.

Runs ON THE CLUSTER (that is where the per-pair reward dumps live), streams JSON
Lines to stdout, diagnostics to stderr:

    ssh <cluster> 'python3 -' < scripts/extract_step320_subset.py > pairs.jsonl

Each output line carries the fully-rendered ``judge_prompt`` the five open-weight
judges actually saw, plus their ratings. A frontier judge can then be sent the
SAME prompt verbatim, so the only thing that differs between judges is the model.

Why the cross-cell assertions matter: the whole comparison rests on every judge
having been shown an identical prompt with the generation on an identical side.
If that ever stopped being true the numbers would still look plausible, so it is
checked rather than assumed.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import random
import sys

EVAL_ROOT = os.environ.get(
    "EVAL_ROOT",
    "/home/lancewicki/projects/turing-rl/results/"
    "2026-08-10-test-eval-9b-full5ep-full-schema",
)
GEN_KEY = os.environ.get("GEN_KEY", "9b-full5ep-step320")
ROOT = os.environ.get("STEP320_ROOT", "%s/raw/%s/sweep" % (EVAL_ROOT, GEN_KEY))
# JSON list of [user_id, post_id, target_idx] triples. Set it to score a later
# checkpoint on the SAME pairs as an earlier one: the trajectory is then paired,
# so a change between checkpoints cannot be a change of sample. Without it the
# keys are sampled fresh from this checkpoint.
KEYS_JSON = os.environ.get("KEYS_JSON")
# KEYS_FILE is the same list read from a path instead of the environment. A
# 200-key list is ~15 KB of JSON; passing that inline through an ssh command
# string is where quoting bugs live, and a silently truncated list would just
# look like a smaller sample rather than an error.
if not KEYS_JSON and os.environ.get("KEYS_FILE"):
    KEYS_JSON = open(os.environ["KEYS_FILE"]).read()
# Which cells must be present, and which to cross-check against each other. The
# default is the five open-weight judges of the step-320 sweep. Override it for a
# root that ran a different set: the alternating-lineage roots have exactly one
# full-schema `on` cell (gemma4-12b), so CELLS=gemma4-12b. With a single cell the
# cross-cell comparisons below degenerate to no-ops, while the per-pair
# judge_prompt presence check -- the one that actually matters for replay -- still
# fires.
CELLS = os.environ.get(
    "CELLS", "qwen35-4b,qwen35-9b,qwen35-27b,gemma4-12b,gemma4-31b").split(",")
# Which cell to lift the replayed prompt from, in preference order. No single
# cell records one for every pair and which cells do varies by checkpoint
# (at step 320 qwen35-4b stores none; at step 0 it stores all 880 while
# qwen35-9b misses 10). Since every cell that records a prompt records a
# byte-identical one -- asserted below, per pair -- taking the first available
# is safe, and preferring the trained-against judge keeps it stable where it can.
PROMPT_CELLS = os.environ.get(
    "PROMPT_CELLS", "qwen35-9b,qwen35-27b,gemma4-31b,gemma4-12b,qwen35-4b").split(",")
MODE = "on"
# A non-"full" prompt style nests one level deeper: judge_sweep_cell.sh:175 appends the style to
# the mode dir, so the rating lineage writes on/rating_only/reward. Kept as a separate env rather
# than folded into MODE so the two stay readable against that line.
STYLE = os.environ.get("STYLE", "")
MODE_DIR = MODE if not STYLE else os.path.join(MODE, STYLE)
N_SAMPLE = int(os.environ.get("N_SAMPLE", "100"))
EXPECT_PAIRS = int(os.environ.get("EXPECT_PAIRS", "880"))
SEED = int(os.environ.get("SEED", "0"))
# SLIM=1 drops the prompt and conversation text, leaving just keys, orientation
# and the incumbent ratings. That is all the golden pipeline test needs, and it
# turns a ~34 MB all-pairs dump into ~200 KB.
SLIM = os.environ.get("SLIM") == "1"

KEY_FIELDS = ("user_id", "post_id", "target_idx")
# Carried through for the qualitative read; judge_prompt already embeds them.
PAIR_FIELDS = ("judge_prompt", "generated_is_b", "context", "user_history",
               "ground_truth", "response")
RATING_FIELDS = ("turing_judge_score_raw", "rating_gt_first", "rating_gen_first")


def key_of(row):
    return tuple(str(row.get(f, "")) for f in KEY_FIELDS)


def load_cell(cell):
    """First row per pair key, plus a count of duplicate keys seen."""
    pattern = os.path.join(ROOT, cell, MODE_DIR, "reward", "*.jsonl")
    shards = sorted(glob.glob(pattern))
    if not shards:
        raise SystemExit("FAIL: no reward shards at %s" % pattern)
    rows, dupes = {}, 0
    for shard in shards:
        with open(shard) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                k = key_of(row)
                if k in rows:
                    dupes += 1
                    continue
                rows[k] = row
    sys.stderr.write("%-12s %d shards, %d unique pairs, %d dupes\n"
                     % (cell, len(shards), len(rows), dupes))
    return rows


def main():
    per_cell = {c: load_cell(c) for c in CELLS}

    for cell, rows in per_cell.items():
        if len(rows) != EXPECT_PAIRS:
            raise SystemExit("FAIL: %s has %d unique pairs, expected %d"
                             % (cell, len(rows), EXPECT_PAIRS))

    keysets = {c: set(r) for c, r in per_cell.items()}
    base = keysets[CELLS[0]]
    for cell in CELLS[1:]:
        if keysets[cell] != base:
            raise SystemExit("FAIL: %s key set differs from %s (%d symmetric diff)"
                             % (cell, CELLS[0], len(keysets[cell] ^ base)))

    # The pairing premise, checked rather than assumed:
    #   * generated_is_b, response and ground_truth identical in EVERY cell --
    #     this is what makes the judges comparable at all;
    #   * judge_prompt identical in every cell that records one. qwen35-4b dumps
    #     it as null, so it is exempted by presence, not by name.
    no_prompt = {}
    prompt_from = {}
    for k in base:
        ref = per_cell[CELLS[0]][k]
        for cell in CELLS[1:]:
            row = per_cell[cell][k]
            for field in ("generated_is_b", "response", "ground_truth"):
                if row.get(field) != ref.get(field):
                    raise SystemExit("FAIL: %s differs on %s for pair %s"
                                     % (cell, field, k))
        prompts = {c: per_cell[c][k].get("judge_prompt") for c in CELLS}
        for c, p in prompts.items():
            if not p:
                no_prompt[c] = no_prompt.get(c, 0) + 1
        distinct = {p for p in prompts.values() if p}
        if len(distinct) > 1:
            raise SystemExit("FAIL: %d distinct judge_prompt values for pair %s "
                             "across cells that record one" % (len(distinct), k))
        src = next((c for c in PROMPT_CELLS if prompts.get(c)), None)
        if src is None:
            raise SystemExit("FAIL: no cell recorded a judge_prompt for pair %s" % (k,))
        prompt_from[k] = src
    sys.stderr.write(
        "cross-cell check OK on all %d pairs: identical generated_is_b, response and "
        "ground_truth; every pair has a judge_prompt and all cells recording one agree. "
        "Missing-prompt rows per cell: %s\n"
        % (len(base), ", ".join("%s=%d" % (c, n) for c, n in sorted(no_prompt.items())) or "none"))

    if KEYS_JSON:
        chosen = [tuple(str(x) for x in k) for k in json.loads(KEYS_JSON)]
        missing = [k for k in chosen if k not in base]
        if missing:
            raise SystemExit("FAIL: %d requested keys are absent from %s, e.g. %s"
                             % (len(missing), GEN_KEY, missing[0]))
        source = "KEYS_JSON"
    elif N_SAMPLE >= len(base):
        chosen = sorted(base)
        source = "all"
    else:
        chosen = random.Random(SEED).sample(sorted(base), N_SAMPLE)
        source = "seed=%d" % SEED
    digest = hashlib.sha256(
        json.dumps(sorted(chosen)).encode()).hexdigest()
    sys.stderr.write("%s: selected %d pairs (%s), sorted-key-list sha256=%s\n"
                     % (GEN_KEY, len(chosen), source, digest))

    for k in chosen:
        ref = per_cell[prompt_from[k]][k]
        out = {f: ref.get(f) for f in KEY_FIELDS}
        fields = ("generated_is_b",) if SLIM else PAIR_FIELDS
        out.update({f: ref.get(f) for f in fields})
        if not SLIM:
            out["prompt_source_cell"] = prompt_from[k]
            out["gen_key"] = GEN_KEY
        out["incumbent"] = {
            cell: {f: per_cell[cell][k].get(f) for f in RATING_FIELDS}
            for cell in CELLS
        }
        sys.stdout.write(json.dumps(out) + "\n")


if __name__ == "__main__":
    main()
