"""Plot the held-out test-set eval: one figure, five judge curves, two panels.

Reads the per-judge tables written by scripts/summarize_test_eval.py
(summary_<cell>.csv) and draws GRPO step on x against two panels:

  left   mean judge Likert rating (1-7); 4 = "cannot tell", the tie point
  right  win rate, i.e. fraction of pairs rated >= 5 (judge prefers the
         generated turn); 0.5 = parity

The 9B curve is emphasised because that is the judge the GRPO run was trained
against; 4B, 27B, Gemma 4 12B, and Gemma 4 31B are held-out judges.

All five judges score the SAME generations per checkpoint -- the pair-sets
are built once and reused -- so differences between curves are a property of the
judge, not of the sample.

POINT JUDGES (--point_cell) overlay a judge evaluated at a single checkpoint on
a smaller pair set, e.g. a frontier model scored on 100 pairs at the last step.
They are deliberately routed around the shared-pair-count and shared-origin
guards, which exist to stop incomparable CURVES being drawn together -- so they
are drawn markers-only, hollow, and the pair-count asymmetry is written into the
subtitle automatically rather than relying on the caller to pass --subtitle.

Provenance: patched copy of
results/2026-08-10-test-eval-9b-full5ep-full-schema/plot/plot_test_eval_judges.py
(frozen package, unchanged); the only additions are the point-judge channel.

Usage:
  python scripts/plot_test_eval_judges.py \
      --eval_root results/2026-08-03-test-eval-9b-half [--out_dir <dir>] \
      [--point_cell claude-opus-5 --point_cell claude-sonnet-5]
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from plotstyle import INK, SLOT, apply_rc, declutter, style_axes  # noqa: E402

# Categorical slots assigned in fixed order by entity, never by rank -- adding or
# dropping a judge must not repaint the others. Draw order puts the emphasised
# judge last so it sits on top.
JUDGES = [
    ("4b", "qwen35-4b", "4B", SLOT[1], False, "-", "o"),
    ("27b", "qwen35-27b", "27B", SLOT[3], False, "-", "o"),
    ("gemma4", "gemma4-12b", "Gemma 4 12B", INK["secondary"], False, "--", "D"),
    ("gemma31", "gemma4-31b", "Gemma 4 31B", "#8b5fbf", False, "-", "s"),
    ("9b", "qwen35-9b", "9B", SLOT[2], True, "-", "o"),
]
LEGEND_ORDER = ("4b", "9b", "27b", "gemma4", "gemma31")

# Single-checkpoint overlays. Hollow marker + no line, so they never read as a
# trajectory and never look like they share the curves' pair count.
POINT_STYLE = {
    "claude-opus-5": ("Opus 5", "#b5451f", "*", 20),
    "claude-sonnet-5": ("Sonnet 5", "#1f6f8b", "P", 12),
}
POINT_FALLBACK = ("#6b6a66", "X", 12)

PANELS = [
    ("likert_mean", "Mean judge rating", "Likert 1-7", 4.0, "4 = cannot tell"),
    ("win_rate_ge5", "Generator win rate", "fraction rated >= 5", 0.5, "0.5 = parity"),
]

STEP_RE = re.compile(r"step(\d+)$")


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
    ap.add_argument("--stem", default="test_eval_judges")
    ap.add_argument("--title", default="9B GRPO generator on the held-out test set",
                    help="figure headline; set it to name the specific GRPO run")
    ap.add_argument("--subtitle", default=None,
                    help="figure subtitle; {n} is substituted with the pair count")
    ap.add_argument("--point_cell", action="append", default=[],
                    help="cell drawn as a single marker, exempt from the "
                         "shared-pair-count and shared-origin guards (repeatable)")
    a = ap.parse_args()

    root = Path(a.eval_root)
    out_dir = Path(a.out_dir) if a.out_dir else root
    out_dir.mkdir(parents=True, exist_ok=True)

    data, n_pairs = {}, set()
    for key, cell, _label, _color, _emph, _linestyle, _marker in JUDGES:
        p = root / f"summary_{cell}.csv"
        if not p.exists():
            raise SystemExit(f"FAIL: missing {p} -- run scripts/summarize_test_eval.py --cell {cell}")
        data[key] = read_summary(p)
        n_pairs.update(r["n_scored"] for r in data[key])

    # Every CURVE judge must be on the same pair count, or the curves are not
    # comparable. Point judges are handled separately and deliberately exempt.
    if len(n_pairs) != 1:
        raise SystemExit(f"FAIL: judges scored different pair counts {sorted(n_pairs)}; "
                         f"the curves would not be comparable")
    n = n_pairs.pop()

    steps_of = {k: [r["step"] for r in rows] for k, rows in data.items()}
    steps = sorted({s for v in steps_of.values() for s in v})
    for k, v in steps_of.items():
        if v != sorted(set(v)):
            raise SystemExit(f"FAIL: {k} has duplicate or unsorted steps {v}")
        if v[0] != steps[0]:
            raise SystemExit(f"FAIL: {k} starts at step {v[0]}, others start at {steps[0]}; "
                             f"curves would not share an origin")
    partial = {k: v for k, v in steps_of.items() if v != steps}

    points = {}
    for cell in a.point_cell:
        p = root / f"summary_{cell}.csv"
        if not p.exists():
            raise SystemExit(f"FAIL: missing {p} for --point_cell {cell}")
        rows = read_summary(p)
        label, color, marker, size = (POINT_STYLE.get(cell, (cell,) + POINT_FALLBACK))
        points[cell] = {"rows": rows, "label": label, "color": color,
                        "marker": marker, "size": size}

    apply_rc()
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6))

    for ax, (field, title, ylab, ref, ref_note) in zip(axes, PANELS):
        # Reference line first, so data marks draw over it.
        ax.axhline(ref, color=INK["muted"], lw=1, ls=(0, (4, 3)), zorder=1)

        ends = []
        for key, _cell, label, color, emph, linestyle, marker in JUDGES:
            xs = steps_of[key]
            ys = [r[field] for r in data[key]]
            ax.plot(xs, ys,
                    color=color,
                    linestyle=linestyle,
                    lw=3.2 if emph else 2.0,
                    marker=marker, markersize=9 if emph else 7,
                    markerfacecolor=color,
                    # 2px surface ring keeps overlapping markers separable.
                    markeredgecolor=INK["surface"], markeredgewidth=2,
                    zorder=5 if emph else 3,
                    label=f"{label} judge" + ("  (trained against)" if emph else ""),
                    solid_capstyle="round")
            ends.append((ys[-1], label, emph, xs[-1]))

        for cell, pj in points.items():
            xs = [r["step"] for r in pj["rows"]]
            ys = [r[field] for r in pj["rows"]]
            ax.plot(xs, ys,
                    # Faint dotted connector once there is more than one point:
                    # readable as a trend, still unmistakably not one of the
                    # solid 880-pair curves.
                    color=pj["color"],
                    linestyle=(0, (2, 3)) if len(xs) > 1 else "none", lw=1.4,
                    marker=pj["marker"], markersize=pj["size"],
                    markerfacecolor="none",
                    markeredgecolor=pj["color"], markeredgewidth=2.2,
                    zorder=7,
                    label=f"{pj['label']} judge  ({pj['rows'][0]['n_scored']} pairs)")
            ends.append((ys[-1], pj["label"], False, xs[-1]))

        # Anchor the note to the RIGHT end of the reference line: at the last step
        # every curve sits well above it in both panels, so nothing overprints the
        # text. (At the left end the rising 4B/9B curves cross straight through it.)
        ax.annotate(ref_note, xy=(steps[-1], ref), xytext=(0, 5), textcoords="offset points",
                    color=INK["muted"], fontsize=8.5, va="bottom", ha="right", zorder=1)

        # Direct labels at the line ends, in ink rather than the series colour --
        # the adjacent mark carries identity. Spread them so they never overlap.
        lo, hi = ax.get_ylim()
        placed = declutter([(y - lo) / (hi - lo) for y, _, _, _ in ends])
        for (y, label, emph, xend), yf in zip(ends, placed):
            ax.annotate(label + ("*" if emph else "") + ("" if xend == steps[-1] else " →"),
                        xy=(xend, lo + yf * (hi - lo)),
                        xytext=(9, 0), textcoords="offset points",
                        color=INK["primary"] if emph else INK["secondary"],
                        fontsize=10.5, fontweight="bold" if emph else "normal",
                        va="center", ha="left", zorder=6, annotation_clip=False)

        style_axes(ax, title, "GRPO step", ylab, steps)
        ax.set_xlim(steps[0] - 1.5, steps[-1] + 4.5)   # headroom for the direct labels

    handles, labels = axes[0].get_legend_handles_labels()
    # Legend order follows the Qwen size ladder, then the separate Gemma family,
    # then any point judges (which are not in LEGEND_ORDER) last.
    key_for_label = {
        f"{label} judge" + ("  (trained against)" if emph else ""): key
        for key, _cell, label, _color, emph, _linestyle, _marker in JUDGES
    }
    def _rank(lbl):
        key = key_for_label.get(lbl)
        return LEGEND_ORDER.index(key) if key in LEGEND_ORDER else len(LEGEND_ORDER)
    idx = sorted(range(len(labels)), key=lambda i: _rank(labels[i]))
    leg = fig.legend([handles[i] for i in idx], [labels[i] for i in idx],
                     loc="lower center", ncol=4, frameon=False,
                     bbox_to_anchor=(0.5, -0.015), fontsize=10.5)
    for t in leg.get_texts():
        t.set_color(INK["secondary"])

    fig.suptitle(a.title, x=0.008, ha="left",
                 color=INK["primary"], fontsize=14, fontweight="bold", y=0.995)
    default_subtitle = (f"{n} pairs per checkpoint, 128 unseen users. All five judges "
                        f"score the same generations.  * = judge the run was trained against.")
    subtitle = a.subtitle.format(n=n) if a.subtitle else default_subtitle
    if points:
        # Stated unconditionally. A reader comparing a hollow marker against a
        # curve is comparing 100 pairs against 880, and the figure has to say so
        # even when the caller passed their own --subtitle.
        bits = ", ".join(
            f"{pj['label']} {pj['rows'][0]['n_scored']} pairs at step "
            f"{', '.join(str(r['step']) for r in pj['rows'])}"
            for pj in points.values())
        subtitle += (f"\nHollow markers are single-checkpoint judges on a SMALLER pair "
                     f"set ({bits}); not on the {n}-pair curve basis.")
    if partial:
        names = {k: l for k, _c, l, _col, _e, _ls, _m in JUDGES}
        bits = ", ".join(f"{names[k]} to step {v[-1]}" for k, v in sorted(partial.items()))
        subtitle += (f"\nIN PROGRESS: {bits}; their later cells are still queued, so those "
                     f"curves stop early (arrow = more coming).")
    fig.text(0.008, 0.925,
             subtitle,
             ha="left", va="top", color=INK["muted"], fontsize=9.5)

    # Header (title + subtitle) lives inside the figure box, so the rect top has
    # to clear it -- tight_layout does not see fig.text/suptitle.
    fig.tight_layout(rect=(0, 0.07, 1, 0.88))

    p = out_dir / f"{a.stem}.png"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
