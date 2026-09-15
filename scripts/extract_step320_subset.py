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

ROOT = os.environ.get(
    "STEP320_ROOT",
    "/home/lancewicki/projects/turing-rl/results/"
    "2026-08-10-test-eval-9b-full5ep-full-schema/raw/9b-full5ep-step320/sweep",
)
CELLS = ["qwen35-4b", "qwen35-9b", "qwen35-27b", "gemma4-12b", "gemma4-31b"]
# The judge the GRPO run was trained against: its stored prompt is the canonical
# one to replay. (qwen35-4b dumps judge_prompt as null, so it cannot be the
# source; the other four store byte-identical prompts -- asserted below.)
PROMPT_CELL = "qwen35-9b"
MODE = "on"
N_SAMPLE = int(os.environ.get("N_SAMPLE", "100"))
EXPECT_PAIRS = int(os.environ.get("EXPECT_PAIRS", "880"))
SEED = int(os.environ.get("SEED", "0"))

KEY_FIELDS = ("user_id", "post_id", "target_idx")
# Carried through for the qualitative read; judge_prompt already embeds them.
PAIR_FIELDS = ("judge_prompt", "generated_is_b", "context", "user_history",
               "ground_truth", "response")
RATING_FIELDS = ("turing_judge_score_raw", "rating_gt_first", "rating_gen_first")


def key_of(row):
    return tuple(str(row.get(f, "")) for f in KEY_FIELDS)


def load_cell(cell):
    """First row per pair key, plus a count of duplicate keys seen."""
    pattern = os.path.join(ROOT, cell, MODE, "reward", "*.jsonl")
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
    no_prompt = set()
    for k in base:
        ref = per_cell[CELLS[0]][k]
        for cell in CELLS[1:]:
            row = per_cell[cell][k]
            for field in ("generated_is_b", "response", "ground_truth"):
                if row.get(field) != ref.get(field):
                    raise SystemExit("FAIL: %s differs on %s for pair %s"
                                     % (cell, field, k))
        prompts = {c: per_cell[c][k].get("judge_prompt") for c in CELLS}
        no_prompt |= {c for c, p in prompts.items() if not p}
        distinct = {p for p in prompts.values() if p}
        if len(distinct) > 1:
            raise SystemExit("FAIL: %d distinct judge_prompt values for pair %s "
                             "across cells that record one" % (len(distinct), k))
        if not prompts.get(PROMPT_CELL):
            raise SystemExit("FAIL: %s has no judge_prompt for pair %s"
                             % (PROMPT_CELL, k))
    sys.stderr.write(
        "cross-cell check OK on all %d pairs: identical generated_is_b, response "
        "and ground_truth; identical judge_prompt across the %d cells that record "
        "one (no judge_prompt: %s)\n"
        % (len(base), len(CELLS) - len(no_prompt), ", ".join(sorted(no_prompt)) or "none"))

    chosen = random.Random(SEED).sample(sorted(base), N_SAMPLE)
    digest = hashlib.sha256(
        json.dumps(sorted(chosen)).encode()).hexdigest()
    sys.stderr.write("selected %d pairs, seed=%d, sorted-key-list sha256=%s\n"
                     % (len(chosen), SEED, digest))

    for k in chosen:
        ref = per_cell[PROMPT_CELL][k]
        out = {f: ref.get(f) for f in KEY_FIELDS}
        out.update({f: ref.get(f) for f in PAIR_FIELDS})
        out["prompt_source_cell"] = PROMPT_CELL
        out["incumbent"] = {
            cell: {f: per_cell[cell][k].get(f) for f in RATING_FIELDS}
            for cell in CELLS
        }
        sys.stdout.write(json.dumps(out) + "\n")


if __name__ == "__main__":
    main()
