"""Six judges, two protocols, on the single-token-trained generator trajectory.

Adapted from plot_single_token_trajectory.py in the sibling packages. The difference is
that this generator was trained against a SINGLE-TOKEN judge, so qwen35-9b-st is the
training judge and is actually on the figure -- the sibling figures mark "*" with a caveat
because their training judge (full-schema 9B) is not plotted. Here "*" needs no caveat.

Panels differ because the two protocols share only one quantity:

  left   mean p(generated)  -- single-token judges ONLY. Read from the A/B logprobs; a
                               full-schema judge has no such quantity, and its 1-7 rating
                               is not the same thing on a different scale.
  right  generator win rate -- ALL SIX. The one genuinely shared measurement: the fraction
                               of pairs where the judge picked the generated turn.

Colour identifies the MODEL, fill identifies the PROTOCOL: single-token judges are drawn
with filled markers, full-schema judges with hollow ones in the same colour. So the 9B
appears twice in the right panel, same colour, and the pair is legible as one model under
two protocols rather than two unrelated judges.

  python plot/plot_st_trained_generator.py --eval_root <package dir>
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import NamedTuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plotstyle import INK, SLOT, apply_rc, declutter, style_axes  # noqa: E402

SINGLE_TOKEN = "single_token"
FULL_SCHEMA = "full"


class Judge(NamedTuple):
    """Named rather than an 8-tuple: this is unpacked in four places, and a positional
    slip between `color` and `marker` would render silently."""
    key: str
    cell: str
    label: str
    color: str
    emph: bool
    linestyle: str
    marker: str
    protocol: str


# Colours are fixed BY MODEL and match the sibling packages, so a reader moving between
# figures tracks the same model. The two full-schema judges reuse their own model's colour.
JUDGES = [
    Judge("gemma12", "gemma4-12b-st", "Gemma 4 12B", INK["secondary"], False, "--", "D", SINGLE_TOKEN),
    Judge("qwen9b", "qwen35-9b-st", "9B", SLOT[2], False, "-", "o", SINGLE_TOKEN),
    Judge("ce9b", "judge-9b-ce-st", "9B CE", SLOT[1], True, "-", "o", SINGLE_TOKEN),
    Judge("ce12b", "judge-gemma12b-ce-st", "Gemma 12B CE", SLOT[3], True, "-", "D", SINGLE_TOKEN),
    Judge("qwen9b_fs", "qwen35-9b", "9B full-schema", SLOT[2], False, ":", "o", FULL_SCHEMA),
    Judge("gemma12_fs", "gemma4-12b", "Gemma 12B full-schema", INK["secondary"], False, ":", "D", FULL_SCHEMA),
]
LEGEND_ORDER = ("qwen9b", "ce9b", "gemma12", "ce12b", "qwen9b_fs", "gemma12_fs")

# This generator WAS trained against the single-token 9B, so unlike the sibling figures the
# starred judge is on the plot and needs no "not shown here" caveat.
TRAINED_AGAINST = "qwen9b"
TRAINING_JUDGE = "Qwen3.5-9B, single-token"

STEP_RE = re.compile(r"step(\d+)$")


def read_summary(path: Path) -> list[dict]:
    """Read either summary schema.

    The single-token builder emits a `step` column; summarize_test_eval.py emits only
    `checkpoint`. Parse the step as an INT either way -- the grid contains 12, 108 and 120,
    which a string sort orders 0, 108, 12, 120, 24 and renders as a scrambled curve that
    still looks like a plausible trajectory.
    """
    rows = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            if r.get("step"):
                step = int(r["step"])
            else:
                m = STEP_RE.search(r["checkpoint"])
                if not m:
                    raise SystemExit(
                        f"FAIL: cannot parse a step from {r['checkpoint']!r} in {path}"
                    )
                step = int(m.group(1))
            row = {
                "step": step,
                "gen_win_rate": float(r["gen_win_rate"]),
                "n_scored": int(r["n_scored"]),
            }
            # Only the single-token arm has a graded probability.
            if r.get("p_gen_mean"):
                row["p_gen_mean"] = float(r["p_gen_mean"])
            rows.append(row)
    return sorted(rows, key=lambda d: d["step"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval_root", required=True)
    ap.add_argument("--out_dir", default=None, help="defaults to --eval_root")
    ap.add_argument("--stem", default="test_eval_st_trained_generator")
    ap.add_argument("--title",
                    default="9B GRPO generator trained against a SINGLE-TOKEN judge "
                            "- 10%-dataset run to 20 epochs")
    ap.add_argument("--subtitle", default=None, help="{n} is substituted with the pair count")
    a = ap.parse_args()

    root = Path(a.eval_root)
    out_dir = Path(a.out_dir) if a.out_dir else root
    out_dir.mkdir(parents=True, exist_ok=True)

    data, n_pairs = {}, set()
    for j in JUDGES:
        path = root / f"summary_{j.cell}.csv"
        if not path.exists():
            raise SystemExit(f"FAIL: missing {path}")
        data[j.key] = read_summary(path)
        n_pairs.update(r["n_scored"] for r in data[j.key])
        if j.protocol == SINGLE_TOKEN and not all("p_gen_mean" in r for r in data[j.key]):
            raise SystemExit(f"FAIL: {j.cell} is single-token but has no p_gen_mean column")

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
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.6))

    panels = [
        ("p_gen_mean", "Mean p(generated)", "P(judge picks the generated turn)",
         "0.5 = no preference", [j for j in JUDGES if j.protocol == SINGLE_TOKEN]),
        ("gen_win_rate", "Generator win rate", "fraction picked over the human",
         "0.5 = parity", JUDGES),
    ]

    for ax, (field, title, ylab, ref_note, judges) in zip(axes, panels):
        ax.axhline(0.5, color=INK["muted"], lw=1, ls=(0, (4, 3)), zorder=1)

        ends = []
        for j in judges:
            xs = steps_of[j.key]
            ys = [r[field] for r in data[j.key]]
            filled = j.protocol == SINGLE_TOKEN
            ax.plot(xs, ys, color=j.color, linestyle=j.linestyle,
                    lw=3.2 if j.emph else 2.0,
                    marker=j.marker, markersize=9 if j.emph else 7,
                    # Hollow marker = full schema. Colour still names the model, so the
                    # same model under both protocols reads as a pair.
                    markerfacecolor=j.color if filled else INK["surface"],
                    markeredgecolor=INK["surface"] if filled else j.color,
                    markeredgewidth=2 if filled else 1.6,
                    zorder=5 if j.emph else 3,
                    label=j.label + (" judge" if filled else " judge (thinking ON)"),
                    solid_capstyle="round")
            ends.append((ys[-1], j))

        # Anchored LEFT: these curves rise from below the reference line to above it, so
        # the right end is where the direct labels crowd in. The surface bbox keeps a
        # rising curve from striking through the text where it crosses.
        ax.annotate(ref_note, xy=(steps[0], 0.5), xytext=(0, 5), textcoords="offset points",
                    color=INK["muted"], fontsize=8.5, va="bottom", ha="left", zorder=4,
                    bbox=dict(facecolor=INK["surface"], edgecolor="none", pad=1.5))

        lo, hi = ax.get_ylim()
        placed = declutter([(y - lo) / (hi - lo) for y, _ in ends])
        for (y, j), yf in zip(ends, placed):
            ax.annotate(j.label + ("*" if j.key == TRAINED_AGAINST else ""),
                        xy=(steps[-1], lo + yf * (hi - lo)),
                        xytext=(9, 0), textcoords="offset points",
                        color=INK["primary"] if j.emph else INK["secondary"],
                        fontsize=9.5, fontweight="bold" if j.emph else "normal",
                        va="center", ha="left", zorder=6, annotation_clip=False)

        style_axes(ax, title, "GRPO step", ylab, steps)
        ax.set_xlim(steps[0] - 1.5, steps[-1] + 4.5)

    handles, labels = axes[1].get_legend_handles_labels()
    key_for_label = {
        j.label + (" judge" if j.protocol == SINGLE_TOKEN else " judge (thinking ON)"): j.key
        for j in JUDGES
    }
    idx = sorted(range(len(labels)), key=lambda i: LEGEND_ORDER.index(key_for_label[labels[i]]))
    leg = fig.legend([handles[i] for i in idx], [labels[i] for i in idx],
                     loc="lower center", ncol=3, frameon=False,
                     bbox_to_anchor=(0.5, -0.06), fontsize=10)
    for t in leg.get_texts():
        t.set_color(INK["secondary"])

    fig.suptitle(a.title, x=0.008, ha="left",
                 color=INK["primary"], fontsize=14, fontweight="bold", y=0.995)
    default_subtitle = (
        f"{n} pairs per checkpoint, every 2 epochs. Filled marker = single-token judge "
        f"(one A/B token); hollow = full schema, thinking ON.\n"
        f"* The generator was trained against {TRAINING_JUDGE} -- that judge IS plotted "
        f"here. Left panel is single-token only: a full-schema judge has no p(generated)."
    )
    fig.text(0.008, 0.925, a.subtitle.format(n=n) if a.subtitle else default_subtitle,
             ha="left", va="top", color=INK["muted"], fontsize=9.5)

    fig.tight_layout(rect=(0, 0.10, 1, 0.86))
    path = out_dir / f"{a.stem}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
