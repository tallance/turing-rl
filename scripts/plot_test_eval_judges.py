"""Plot the held-out test-set eval: one figure, three judges, two panels.

Reads the per-judge tables written by scripts/summarize_test_eval.py
(summary_qwen35-<size>.csv) and draws GRPO step on x against two panels:

  left   mean judge Likert rating (1-7); 4 = "cannot tell", the tie point
  right  win rate, i.e. fraction of pairs rated >= 5 (judge prefers the
         generated turn); 0.5 = parity

The 9B curve is emphasised because that is the judge the GRPO run was trained
against (job 13634); 4B and 27B are held-out judges that never shaped the
reward, so they are the transfer check.

All three judges score the SAME 880 generations per checkpoint -- the pair-sets
are built once and reused -- so differences between curves are a property of the
judge, not of the sample.

Usage:
  python scripts/plot_test_eval_judges.py \
      --eval_root results/2026-08-03-test-eval-9b-half \
      [--out_dir <dir>] [--dark]
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Reference categorical palette, slots 1-3 (see the dataviz skill's palette.md).
# Assigned in fixed slot order by entity, never by rank -- adding or dropping a
# judge must not repaint the others.
LIGHT = {
    "surface": "#fcfcfb", "primary": "#0b0b0b", "secondary": "#52514e",
    "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7",
    "series": {"4b": "#2a78d6", "9b": "#eb6834", "27b": "#1baf7a"},
}
DARK = {
    "surface": "#1a1a19", "primary": "#ffffff", "secondary": "#c3c2b7",
    "muted": "#898781", "grid": "#2c2c2a", "axis": "#383835",
    "series": {"4b": "#3987e5", "9b": "#d95926", "27b": "#199e70"},
}

# Draw order puts the emphasised judge last so it sits on top of the others.
JUDGES = [
    ("4b", "qwen35-4b", "4B", False),
    ("27b", "qwen35-27b", "27B", False),
    ("9b", "qwen35-9b", "9B", True),
]

PANELS = [
    ("likert_mean", "Mean judge rating", "Likert 1-7", 4.0, "4 = cannot tell"),
    ("win_rate_ge5", "Generator win rate", "fraction rated >= 5", 0.5, "0.5 = parity"),
]

STEP_RE = re.compile(r"step(\d+)$")

# Minimum vertical gap between two direct labels, as a fraction of axes height.
# 4B and 9B end within 0.005 of each other on the win-rate panel, so without
# this their labels render on top of one another.
LABEL_GAP = 0.075


def declutter(y_fracs: list[float], gap: float = LABEL_GAP) -> list[float]:
    """Nudge label positions apart, preserving their vertical order.

    Takes/returns axes-fraction y positions. Order is preserved so a label
    always stays on the same side of its neighbours as the curve it names.
    """
    order = sorted(range(len(y_fracs)), key=lambda i: y_fracs[i])
    out = list(y_fracs)
    for a, b in zip(order, order[1:]):
        if out[b] - out[a] < gap:
            out[b] = out[a] + gap
    overflow = out[order[-1]] - 1.0
    if overflow > 0:                       # pushed past the top -- slide the stack down
        for i in order:
            out[i] -= overflow
    return out


def read_summary(path: Path) -> list[dict]:
    """Return [{step, likert_mean, win_rate_ge5}, ...] sorted by step."""
    rows = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            m = STEP_RE.search(r["checkpoint"])
            if not m:
                raise SystemExit(f"FAIL: cannot parse a step from {r['checkpoint']!r} in {path}")
            rows.append({
                "step": int(m.group(1)),
                "likert_mean": float(r["likert_mean"]),
                "win_rate_ge5": float(r["win_rate_ge5"]),
                "n_scored": int(r["n_scored"]),
            })
    return sorted(rows, key=lambda d: d["step"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval_root", required=True)
    ap.add_argument("--out_dir", default=None, help="defaults to --eval_root")
    ap.add_argument("--dark", action="store_true", help="render the dark-mode variant")
    ap.add_argument("--stem", default="test_eval_judges")
    a = ap.parse_args()

    root = Path(a.eval_root)
    out_dir = Path(a.out_dir) if a.out_dir else root
    out_dir.mkdir(parents=True, exist_ok=True)
    C = DARK if a.dark else LIGHT

    data, n_pairs = {}, set()
    for key, cell, _label, _emph in JUDGES:
        p = root / f"summary_{cell}.csv"
        if not p.exists():
            raise SystemExit(f"FAIL: missing {p} -- run scripts/summarize_test_eval.py --cell {cell}")
        data[key] = read_summary(p)
        n_pairs.update(r["n_scored"] for r in data[key])

    # Every judge must be on the same pair count, or the curves are not comparable.
    if len(n_pairs) != 1:
        raise SystemExit(f"FAIL: judges scored different pair counts {sorted(n_pairs)}; "
                         f"the curves would not be comparable")
    n = n_pairs.pop()

    steps = [r["step"] for r in data["9b"]]
    for key, rows in data.items():
        if [r["step"] for r in rows] != steps:
            raise SystemExit(f"FAIL: {key} covers steps {[r['step'] for r in rows]}, expected {steps}")

    plt.rcParams.update({
        "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"],
        "figure.facecolor": C["surface"], "axes.facecolor": C["surface"],
        "savefig.facecolor": C["surface"],
    })
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6))

    for ax, (field, title, ylab, ref, ref_note) in zip(axes, PANELS):
        # Reference line first, so data marks draw over it.
        ax.axhline(ref, color=C["muted"], lw=1, ls=(0, (4, 3)), zorder=1)

        ends = []
        for key, _cell, label, emph in JUDGES:
            ys = [r[field] for r in data[key]]
            ax.plot(steps, ys,
                    color=C["series"][key],
                    lw=3.2 if emph else 2.0,
                    marker="o", markersize=9 if emph else 7,
                    markerfacecolor=C["series"][key],
                    # 2px surface ring keeps overlapping markers separable.
                    markeredgecolor=C["surface"], markeredgewidth=2,
                    zorder=5 if emph else 3,
                    label=f"{label} judge" + ("  (trained against)" if emph else ""),
                    solid_capstyle="round")
            ends.append((ys[-1], label, emph))

        # Anchor the note to the RIGHT end of the reference line: at the last step
        # every curve sits well above it in both panels, so nothing overprints the
        # text. (At the left end the rising 4B/9B curves cross straight through it.)
        ax.annotate(ref_note, xy=(steps[-1], ref), xytext=(0, 5), textcoords="offset points",
                    color=C["muted"], fontsize=8.5, va="bottom", ha="right", zorder=1)

        # Direct labels at the line ends, in ink rather than the series colour --
        # the adjacent mark carries identity. Spread them so they never overlap.
        lo, hi = ax.get_ylim()
        placed = declutter([(y - lo) / (hi - lo) for y, _, _ in ends])
        for (y, label, emph), yf in zip(ends, placed):
            ax.annotate(label + ("*" if emph else ""),
                        xy=(steps[-1], lo + yf * (hi - lo)),
                        xytext=(9, 0), textcoords="offset points",
                        color=C["primary"] if emph else C["secondary"],
                        fontsize=10.5, fontweight="bold" if emph else "normal",
                        va="center", ha="left", zorder=6, annotation_clip=False)

        ax.set_title(title, color=C["primary"], fontsize=12.5, fontweight="bold",
                     loc="left", pad=10)
        ax.set_xlabel("GRPO step", color=C["secondary"], fontsize=10.5)
        ax.set_ylabel(ylab, color=C["secondary"], fontsize=10.5)
        ax.set_xticks(steps)
        ax.set_xlim(steps[0] - 1.5, steps[-1] + 4.5)   # headroom for the direct labels
        ax.grid(True, axis="y", color=C["grid"], lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(colors=C["muted"], labelsize=9.5)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(C["axis"])

    handles, labels = axes[0].get_legend_handles_labels()
    # Legend order follows the judge size ladder, not the draw order.
    idx = [labels.index(l) for l in sorted(labels, key=lambda s: int(re.match(r"(\d+)", s).group(1)))]
    leg = fig.legend([handles[i] for i in idx], [labels[i] for i in idx],
                     loc="lower center", ncol=3, frameon=False,
                     bbox_to_anchor=(0.5, -0.015), fontsize=10.5)
    for t in leg.get_texts():
        t.set_color(C["secondary"])

    fig.suptitle("9B GRPO generator on the held-out test set", x=0.008, ha="left",
                 color=C["primary"], fontsize=14, fontweight="bold", y=1.005)
    fig.text(0.008, 0.925,
             f"{n} pairs per checkpoint, 128 unseen users. All three judges score the "
             f"same generations.  * = judge the run was trained against.",
             ha="left", color=C["muted"], fontsize=9.5)

    fig.tight_layout(rect=(0, 0.07, 1, 0.90))

    suffix = "_dark" if a.dark else ""
    for ext in ("png", "pdf"):
        p = out_dir / f"{a.stem}{suffix}.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
