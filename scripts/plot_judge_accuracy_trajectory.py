"""Judge accuracy vs GRPO step: does the generator train its way past each judge?

The two-panel figure plots how GENEROUS each judge is. This plots whether each
judge is still RIGHT: the fraction of non-tie pairs where it picks the real
human. 0.5 is chance; below it the judge is not noisy but reliably fooled.

Every judge at every step is scored on the identical 100 pair keys, so a change
along x is a change in the generator or the judge, never a change of sample.
Incumbent ratings are read straight out of the per-step pairs files; the
frontier judges are read from their scored reward rows.

  PYTHONPATH=plot python plot/plot_judge_accuracy_trajectory.py --eval_root .
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from judges import CURVE_JUDGES, POINT_JUDGES  # noqa: E402
from plotstyle import INK, apply_rc, declutter, style_axes  # noqa: E402


def _repo_root() -> Path:
    """Locate the checkout holding scripts/eval_rl_generator.py.

    This file is also copied into a results package's plot/ directory for
    provenance, where the usual parents[1] guess lands on the results dir, so
    walk up looking for the real thing and let TURING_RL_ROOT override.
    """
    env = os.environ.get("TURING_RL_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    here = Path(__file__).resolve()
    for cand in here.parents:
        if (cand / "scripts" / "eval_rl_generator.py").is_file():
            return cand
    raise SystemExit("FAIL: cannot find the turing-rl checkout; set TURING_RL_ROOT")


REPO_ROOT = _repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.claude_judge_run import row_from_incumbent  # noqa: E402
from scripts.eval_rl_generator import directional_accuracy  # noqa: E402

STEP_RE = re.compile(r"pairs_step(\d+)_")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval_root", required=True)
    ap.add_argument("--gen_prefix", default="9b-full5ep-step")
    ap.add_argument("--stem", default="judge_accuracy_trajectory")
    ap.add_argument("--title", default="Does the generator train its way past each judge?")
    a = ap.parse_args()

    root = Path(a.eval_root)
    pair_files = {}
    for p in sorted(glob.glob(str(root / "pairs_step*_100.jsonl"))):
        m = STEP_RE.search(Path(p).name)
        if m:
            pair_files[int(m.group(1))] = p
    if not pair_files:
        raise SystemExit(f"FAIL: no pairs_step*_100.jsonl under {root}")
    steps = sorted(pair_files)

    per_step = {s: [json.loads(l) for l in open(pair_files[s]) if l.strip()]
                for s in steps}
    n_pairs = {len(v) for v in per_step.values()}
    if len(n_pairs) != 1:
        raise SystemExit(f"FAIL: pair files differ in size {sorted(n_pairs)}")
    keysets = {s: {(r["user_id"], r["post_id"], str(r["target_idx"]))
                   for r in v} for s, v in per_step.items()}
    if len({frozenset(v) for v in keysets.values()}) != 1:
        raise SystemExit("FAIL: the per-step pair files are not the same keys; "
                         "the trajectory would mix a judge effect with a sample effect")
    n = n_pairs.pop()

    series, table = {}, []
    for j in CURVE_JUDGES:
        pts = []
        for s in steps:
            rows = [row_from_incumbent(rec, j.cell) for rec in per_step[s]]
            acc = directional_accuracy(rows)
            pts.append((s, acc["accuracy"]))
            table.append({"judge": j.label, "cell": j.cell, "step": s,
                          "judge_accuracy": round(acc["accuracy"], 4),
                          "n_nontie": acc["n_nontie"], "n_tie": acc["n_tie"]})
        series[j.cell] = pts

    for j in POINT_JUDGES:
        pts = []
        for s in steps:
            f = (root / "raw" / f"{a.gen_prefix}{s}" / "sweep" / j.cell / "on"
                 / "reward" / "scores.jsonl")
            if not f.is_file():
                continue
            rows = [json.loads(l) for l in open(f) if l.strip()]
            acc = directional_accuracy(rows)
            pts.append((s, acc["accuracy"]))
            table.append({"judge": j.label, "cell": j.cell, "step": s,
                          "judge_accuracy": round(acc["accuracy"], 4),
                          "n_nontie": acc["n_nontie"], "n_tie": acc["n_tie"]})
        if pts:
            series[j.cell] = pts

    csv_path = root / f"{a.stem}.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["judge", "cell", "step",
                                           "judge_accuracy", "n_nontie", "n_tie"])
        w.writeheader()
        w.writerows(sorted(table, key=lambda r: (r["cell"], r["step"])))

    apply_rc()
    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    ax.axhline(0.5, color=INK["primary"], lw=1.2, ls=(0, (4, 3)), zorder=1)
    ax.annotate("0.5 = chance", xy=(steps[-1], 0.5), xytext=(0, 5),
                textcoords="offset points", color=INK["primary"], fontsize=9,
                va="bottom", ha="right", zorder=1)

    ends = []
    for j in CURVE_JUDGES:
        xs = [s for s, _ in series[j.cell]]
        ys = [v for _, v in series[j.cell]]
        ax.plot(xs, ys, color=j.color, linestyle=j.linestyle, lw=3.0 if j.emph else 2.0,
                marker=j.marker, markersize=j.size, markerfacecolor=j.color,
                markeredgecolor=INK["surface"], markeredgewidth=2,
                zorder=5 if j.emph else 3,
                label=f"{j.label} judge" + ("  (trained against)" if j.emph else ""))
        ends.append((ys[-1], j.label, j.emph, j.color))

    for j in POINT_JUDGES:
        if j.cell not in series:
            continue
        xs = [s for s, _ in series[j.cell]]
        ys = [v for _, v in series[j.cell]]
        # Thin dashed connector when there is more than one point: still clearly
        # distinct from the solid curves, but the trajectory is readable.
        ax.plot(xs, ys, color=j.color, linestyle=(0, (2, 2)) if len(xs) > 1 else "none",
                lw=1.6, marker=j.marker, markersize=j.size, markerfacecolor="none",
                markeredgecolor=j.color, markeredgewidth=2.2, zorder=7,
                label=f"{j.label} judge")
        ends.append((ys[-1], j.label, False, j.color))

    # Direct labels in the SERIES colour, not ink. The house style puts them in
    # ink because the adjacent mark carries identity -- but six endpoints inside
    # 0.14-0.42 force declutter to push labels well off their own marks, so
    # adjacency stops being reliable and the colour has to carry it instead.
    lo, hi = ax.get_ylim()
    placed = declutter([(y - lo) / (hi - lo) for y, _, _, _ in ends])
    for (y, label, emph, color), yf in zip(ends, placed):
        ax.annotate(label + ("*" if emph else ""),
                    xy=(steps[-1], lo + yf * (hi - lo)),
                    xytext=(9, 0), textcoords="offset points",
                    color=color,
                    fontsize=10.5, fontweight="bold" if emph else "normal",
                    va="center", ha="left", zorder=8, annotation_clip=False)

    style_axes(ax, "", "GRPO step", "judge accuracy: picked the real human", steps)
    ax.set_xlim(steps[0] - 6, steps[-1] + 34)

    fig.suptitle(a.title, x=0.008, ha="left", color=INK["primary"],
                 fontsize=14, fontweight="bold", y=0.995)
    fig.text(0.008, 0.925,
             f"Held-out test set, {n} pairs, identical pair keys at every step and for "
             f"every judge.\nBelow the dashed line the judge is not merely noisy -- it "
             f"reliably prefers the generated turn.\n* = the judge the run was trained "
             f"against.",
             ha="left", va="top", color=INK["muted"], fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.86))

    p = root / f"{a.stem}.png"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    print(f"wrote {csv_path}")
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
