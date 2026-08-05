"""Per-example conversation trajectory dump for the held-out test-set eval.

The training-time analogue is scripts/dump_conversation_trajectory.py, which walks a single
run's reward_dump and chunks each example's rollouts into epochs. Here the x-axis is the GRPO
CHECKPOINT STEP instead: every checkpoint generated exactly one turn per test prompt, so for a
given (user_id, post_id, target_idx) we can line the generated turns up as a function of step
and read the drift off top-to-bottom.

Samples N distinct users deterministically (seeded), one turn per user, from the pairs that
every checkpoint scored. The primary judge supplies the Likert shown per step; any extra judge
cells are shown alongside, and are strictly comparable because all judges scored byte-identical
pair sets.

Usage:
  python scripts/dump_test_eval_trajectory.py \
      --eval_root results/2026-08-03-test-eval-9b-half \
      --out results/2026-08-03-test-eval-9b-half/test_conversation_trajectory.txt \
      [--cell qwen35-9b] [--extra_cells qwen35-4b,qwen35-27b] [--n_users 10] [--seed 42]
"""
from __future__ import annotations

import argparse
import glob
import json
import random
import re
import sys
from pathlib import Path

KEY_FIELDS = ("user_id", "post_id", "target_idx")
# The reward dumps carry the full judge prompt and chain of thought (~75KB/row, ~1GB over the
# 15 cells). None of that is printed here, so drop everything but these on the way in.
KEEP_FIELDS = KEY_FIELDS + ("response", "ground_truth", "context", "user_history",
                            "turing_judge_score_raw")


def load_rows(reward_dir: Path) -> list[dict]:
    """Same loader as scripts/summarize_test_eval.py: every *.jsonl in the reward dir."""
    rows = []
    for f in sorted(glob.glob(str(reward_dir / "*.jsonl"))):
        for line in open(f):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows.append({k: r.get(k) for k in KEEP_FIELDS})
    return rows


def row_key(r: dict) -> tuple:
    return tuple(str(r.get(f, "")) for f in KEY_FIELDS)


def step_of(checkpoint: str) -> int:
    m = re.search(r"step(\d+)", checkpoint)
    if not m:
        raise SystemExit(f"FAIL: cannot read a step number out of checkpoint dir {checkpoint!r}")
    return int(m.group(1))


def likert(r: dict):
    """Order-normalized 1-7 judge rating; 0/None means the judge output did not parse."""
    v = r.get("turing_judge_score_raw")
    if v is None:
        return None
    try:
        n = int(round(float(v)))
    except (TypeError, ValueError):
        return None
    return None if n == 0 else n


def tag_of(lk) -> str:
    if lk is None:
        return "parsefail"
    if lk >= 5:
        return "win"
    if lk == 4:
        return "tie"
    return "human-wins"


def load_cell(root: Path, cell: str, mode: str) -> dict[int, dict[tuple, dict]]:
    """step -> {pair key -> judge row}. Raises on a duplicated key within a checkpoint."""
    found = {p.parents[3].name: p for p in root.glob(f"raw/*/sweep/{cell}/{mode}/reward")}
    if not found:
        raise SystemExit(f"FAIL: no reward dirs under {root}/raw/*/sweep/{cell}/{mode}")
    by_step: dict[int, dict[tuple, dict]] = {}
    for ckpt, reward_dir in found.items():
        step = step_of(ckpt)
        if step in by_step:
            raise SystemExit(f"FAIL: two checkpoint dirs map to step {step} for cell {cell}")
        seen: dict[tuple, dict] = {}
        dupes = 0
        for r in load_rows(reward_dir):
            k = row_key(r)
            if k in seen:
                dupes += 1
                continue
            seen[k] = r
        if dupes:
            # Steps 0/8/16 of the 9B cell legitimately span two Slurm jobs (original run plus a
            # re-judge of timed-out pairs). That re-judge only filled gaps, so a key scored twice
            # would mean something else went wrong and the sample would not be well defined.
            raise SystemExit(f"FAIL: {cell} {ckpt} has {dupes} duplicated pair keys")
        by_step[step] = seen
    return by_step


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval_root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cell", default="qwen35-9b",
                    help="Primary judge; its Likert drives the [win]/[tie]/[human-wins] tag")
    ap.add_argument("--extra_cells", default="qwen35-4b,qwen35-27b",
                    help="Comma-separated extra judge cells shown alongside ('' to disable)")
    ap.add_argument("--mode", default="on")
    ap.add_argument("--n_users", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--include_history", action="store_true",
                    help="Also print the [USER HISTORY] block the generator was conditioned on")
    ap.add_argument("--history_chars", type=int, default=2000)
    a = ap.parse_args()

    root = Path(a.eval_root)
    primary = load_cell(root, a.cell, a.mode)
    steps = sorted(primary)
    extras = [c.strip() for c in a.extra_cells.split(",") if c.strip()]
    extra_cells = {c: load_cell(root, c, a.mode) for c in extras}
    for c, cell in extra_cells.items():
        if sorted(cell) != steps:
            raise SystemExit(f"FAIL: cell {c} covers steps {sorted(cell)}, primary covers {steps}")

    # Only sample pairs every checkpoint scored, so a sample never has a hole in its trajectory.
    common = set.intersection(*(set(primary[s]) for s in steps))
    if not common:
        raise SystemExit("FAIL: no pair key is present at every step")

    # Deterministic pick: sorted candidate lists, and a per-user RNG so the turn drawn for a user
    # does not shift when a different user is added or dropped upstream.
    users = sorted({k[0] for k in common})
    if len(users) < a.n_users:
        raise SystemExit(f"FAIL: only {len(users)} users available, asked for {a.n_users}")
    picked = sorted(random.Random(a.seed).sample(users, a.n_users))
    chosen = []
    for u in picked:
        cands = sorted(k for k in common if k[0] == u)
        chosen.append(random.Random(f"{a.seed}:{u}").choice(cands))

    judge_names = [a.cell] + extras
    lines: list[str] = []
    lines.append(f"# Conversation trajectory by GRPO step — {a.eval_root}")
    lines.append(f"# {len(chosen)} test-set examples: {a.n_users} users sampled at seed {a.seed}, "
                 f"one turn each, from the {len(common)} pairs scored at every step.")
    lines.append(f"# Steps: {', '.join(str(s) for s in steps)} (step 0 = pre-RL SFT init). "
                 f"1 generation per prompt, temperature 0.7 / top_p 0.8 / top_k 20.")
    lines.append(f"# Likert is the order-normalized 1-7 judge rating (higher = generated turn "
                 f"looks more human). win = >=5, tie = 4, parse-fail = None.")
    lines.append(f"# Primary judge {a.cell}"
                 + (f"; also shown: {', '.join(extras)}." if extras else "."))
    lines.append("")

    for si, key in enumerate(chosen, 1):
        head = primary[steps[0]][key]
        lines.append("=" * 100)
        lines.append(f"SAMPLE {si}/{len(chosen)}  key={key[0]}/{key[1]}/t{key[2]}")
        lines.append("=" * 100)
        if a.include_history:
            hist = str(head.get("user_history", "")).rstrip()
            if len(hist) > a.history_chars:
                hist = hist[:a.history_chars] + f"\n... [truncated at {a.history_chars} chars]"
            lines.append(hist)
            lines.append("")
        lines.append("[CONTEXT — conversation so far]")
        lines.append(str(head.get("context", "")).rstrip())
        lines.append("")
        lines.append("[GROUND TRUTH — real human turn]")
        lines.append(str(head.get("ground_truth", "")).strip())
        lines.append("")
        lines.append("[GENERATED TURN BY GRPO STEP]")

        traj = []
        for s in steps:
            r = primary[s][key]
            gen = str(r.get("response", "")).replace("\n", " ").strip()
            lk = likert(r)
            traj.append(lk)
            extra_bits = []
            for c in extras:
                er = extra_cells[c][s].get(key)
                if er is None:
                    extra_bits.append(f"{c}=NA")
                    continue
                # All judges scored byte-identical pair sets; if the text differs, the columns
                # would be describing different generations and must not be printed side by side.
                if str(er.get("response", "")).replace("\n", " ").strip() != gen:
                    raise SystemExit(
                        f"FAIL: {c} step {s} key {key} judged a different generated turn than "
                        f"{a.cell}; the pair sets are not the same sample")
                extra_bits.append(f"{c}={likert(er)}")
            suffix = f"  ({'  '.join(extra_bits)})" if extra_bits else ""
            lines.append(f"  --- step {s:>2} ---  Likert={lk} [{tag_of(lk)}]{suffix}")
            lines.append(f"    {gen}")
        lines.append("")
        lines.append("  [Likert by step, "
                     + " / ".join(judge_names) + "]")
        for c in judge_names:
            cell = primary if c == a.cell else extra_cells[c]
            seq = "  ".join(f"{s}:{likert(cell[s][key]) if key in cell[s] else 'NA'}"
                            for s in steps)
            lines.append(f"    {c:<12} {seq}")
        lines.append("")

    Path(a.out).write_text("\n".join(lines) + "\n")
    print(f"wrote {a.out}  ({len(chosen)} examples, steps {steps}, "
          f"{len(common)} common pairs, {len(users)} users)", file=sys.stderr)


if __name__ == "__main__":
    main()
