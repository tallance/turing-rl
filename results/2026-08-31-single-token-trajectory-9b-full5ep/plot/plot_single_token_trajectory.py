"""Companion figure: the same generator trajectory scored by single-token judges.

Same layout, axis order and palette as ``plot_test_eval_judges.py`` in the full-schema
package, so the two figures can be read side by side. The two zero-shot judges keep the
colour their full-schema counterparts use there; the CE judge is new to this figure and is
emphasised.

The panels are NOT the same quantities as the full-schema figure's, because single-token
has no 1-7 rating:

  left   mean p(generated)  -- the graded signal, read from the judge's A/B logprobs.
                               Replaces "mean judge rating"; a mean likert here would be
                               1 + 6*win_rate, since a single-token verdict maps to 1 or 7.
  right  generator win rate -- fraction of pairs where the judge picked the generated
                               response. This one IS directly comparable to the
                               full-schema figure's right panel: same quantity, no tie
                               bucket.

  python plot/plot_single_token_trajectory.py --eval_root <package dir>
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plotstyle import INK, SLOT, apply_rc, declutter, style_axes  # noqa: E402

# Fixed slots BY ENTITY, never by rank. The two zero-shot judges reuse the colour their
# full-schema counterparts carry in the adjacent figure (9B = slot 2, Gemma 4 12B = the
# dashed secondary), so a reader moving between the figures tracks the same model.
JUDGES = [
    ("gemma12", "gemma4-12b-st", "Gemma 4 12B", INK["secondary"], False, "--", "D"),
    ("qwen9b", "qwen35-9b-st", "9B", SLOT[2], False, "-", "o"),
    ("ce9b", "judge-9b-ce-st", "9B CE", SLOT[1], True, "-", "^"),
]
LEGEND_ORDER = ("qwen9b", "ce9b", "gemma12")

# The generator was trained against Qwen3.5-9B with the FULL schema, thinking ON. That
# judge is not on this figure -- qwen35-9b-st is the same base model under the
# single-token protocol. "*" therefore means the same thing here as in the full-schema
# figures (it points at the training judge) rather than being reused for a second meaning;
# the subtitle carries the protocol caveat.
TRAINED_AGAINST = "qwen9b"
TRAINING_JUDGE = "Qwen3.5-9B, full schema, thinking ON"

PANELS = [
    ("p_gen_mean", "Mean p(generated)", "P(judge picks the generated turn)",
     0.5, "0.5 = no preference"),
    ("gen_win_rate", "Generator win rate", "fraction picked over the human",
     0.5, "0.5 = parity"),
]


def read_summary(path: Path) -> list[dict]:
    rows = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            rows.append({
                "step": int(r["step"]),
                "p_gen_mean": float(r["p_gen_mean"]),
                "gen_win_rate": float(r["gen_win_rate"]),
                "n_scored": int(r["n_scored"]),
            })
    return sorted(rows, key=lambda d: d["step"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval_root", required=True)
    ap.add_argument("--out_dir", default=None, help="defaults to --eval_root")
    ap.add_argument("--stem", default="test_eval_single_token")
    ap.add_argument("--title",
                    default="9B GRPO generator, 10%-dataset run to 20 epochs "
                            "- single-token judges")
    ap.add_argument("--subtitle", default=None, help="{n} is substituted with the pair count")
    a = ap.parse_args()

    root = Path(a.eval_root)
    out_dir = Path(a.out_dir) if a.out_dir else root
    out_dir.mkdir(parents=True, exist_ok=True)

    data, n_pairs = {}, set()
    for key, cell, _label, _color, _emph, _ls, _m in JUDGES:
        path = root / f"summary_{cell}.csv"
        if not path.exists():
            raise SystemExit(
                f"FAIL: missing {path} -- run scripts/build_single_token_trajectory.py"
            )
        data[key] = read_summary(path)
        n_pairs.update(r["n_scored"] for r in data[key])

    if len(n_pairs) != 1:
        raise SystemExit(f"FAIL: judges scored different pair counts {sorted(n_pairs)}; "
                         f"the curves would not be comparable")
    n = n_pairs.pop()

    steps_of = {k: [r["step"] for r in rows] for k, rows in data.items()}
    steps = sorted({s for v in steps_of.values() for s in v})
    for key, value in steps_of.items():
        if value != steps:
            raise SystemExit(
                f"FAIL: {key} covers {value}, others cover {steps}. A judge missing later "
                f"checkpoints would draw a short curve that reads as the run ending."
            )

    apply_rc()
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6))

    for ax, (field, title, ylab, ref, ref_note) in zip(axes, PANELS):
        ax.axhline(ref, color=INK["muted"], lw=1, ls=(0, (4, 3)), zorder=1)

        ends = []
        for key, _cell, label, color, emph, linestyle, marker in JUDGES:
            xs = steps_of[key]
            ys = [r[field] for r in data[key]]
            ax.plot(xs, ys, color=color, linestyle=linestyle,
                    lw=3.2 if emph else 2.0,
                    marker=marker, markersize=9 if emph else 7,
                    markerfacecolor=color,
                    markeredgecolor=INK["surface"], markeredgewidth=2,
                    zorder=5 if emph else 3,
                    label=f"{label} judge" + ("  (trained, single-token)" if emph else ""),
                    solid_capstyle="round")
            ends.append((ys[-1], label, emph, key))

        # Anchored LEFT, unlike the full-schema figure. These curves rise from below the
        # reference line to above it, so the right end is where the direct labels crowd
        # in and the strongest judge can finish sitting on 0.5 exactly.
        # The surface-coloured bbox matters: these curves CROSS the reference line early,
        # so without it a rising curve strikes straight through the text.
        ax.annotate(ref_note, xy=(steps[0], ref), xytext=(0, 5), textcoords="offset points",
                    color=INK["muted"], fontsize=8.5, va="bottom", ha="left", zorder=4,
                    bbox=dict(facecolor=INK["surface"], edgecolor="none", pad=1.5))

        lo, hi = ax.get_ylim()
        placed = declutter([(y - lo) / (hi - lo) for y, _, _, _ in ends])
        for (y, label, emph, key), yf in zip(ends, placed):
            ax.annotate(label + ("*" if key == TRAINED_AGAINST else ""),
                        xy=(steps[-1], lo + yf * (hi - lo)),
                        xytext=(9, 0), textcoords="offset points",
                        color=INK["primary"] if emph else INK["secondary"],
                        fontsize=10.5, fontweight="bold" if emph else "normal",
                        va="center", ha="left", zorder=6, annotation_clip=False)

        style_axes(ax, title, "GRPO step", ylab, steps)
        ax.set_xlim(steps[0] - 1.5, steps[-1] + 4.5)

    handles, labels = axes[0].get_legend_handles_labels()
    key_for_label = {
        f"{label} judge" + ("  (trained, single-token)" if emph else ""): key
        for key, _cell, label, _color, emph, _ls, _m in JUDGES
    }
    idx = sorted(range(len(labels)), key=lambda i: LEGEND_ORDER.index(key_for_label[labels[i]]))
    leg = fig.legend([handles[i] for i in idx], [labels[i] for i in idx],
                     loc="lower center", ncol=3, frameon=False,
                     bbox_to_anchor=(0.5, -0.015), fontsize=10.5)
    for t in leg.get_texts():
        t.set_color(INK["secondary"])

    fig.suptitle(a.title, x=0.008, ha="left",
                 color=INK["primary"], fontsize=14, fontweight="bold", y=0.995)
    default_subtitle = (
        f"{n} pairs per checkpoint, every 2 epochs, on the SAME generations the "
        f"full-schema figure scores.  Judges answer with one A/B token, no rubric and no "
        f"schema.\n"
        f"* The generator was trained against {TRAINING_JUDGE}. That judge is NOT plotted "
        f"here: the 9B line is the same base model under the single-token protocol."
    )
    fig.text(0.008, 0.925, a.subtitle.format(n=n) if a.subtitle else default_subtitle,
             ha="left", va="top", color=INK["muted"], fontsize=9.5)

    fig.tight_layout(rect=(0, 0.07, 1, 0.86))
    path = out_dir / f"{a.stem}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
