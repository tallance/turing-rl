"""Single-panel win-rate figure: the four ZERO-SHOT judges only.

A reduced variant of plot_st_trained_generator.py for the write-up. Drops the two
CE-trained judges and the p(generated) panel, leaving the one quantity every judge
measures the same way, on the four judges that were never trained on this task.

Two models x two protocols, so each model appears twice: colour names the model, marker
fill names the protocol (filled = single-token, hollow = full schema, thinking ON).

read_summary and Judge are imported rather than copied -- that keeps the two-schema
handling and the numeric step parsing in one place.

  python plot/plot_winrate_zeroshot.py --eval_root <package dir>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plotstyle import INK, SLOT, apply_rc, declutter, style_axes  # noqa: E402
from plot_st_trained_generator import (  # noqa: E402
    FULL_SCHEMA,
    SINGLE_TOKEN,
    Judge,
    read_summary,
)

JUDGES = [
    Judge("qwen9b", "qwen35-9b-st", "9B single-token", SLOT[2], True, "-", "o", SINGLE_TOKEN),
    Judge("qwen9b_fs", "qwen35-9b", "9B full-schema", SLOT[2], False, ":", "o", FULL_SCHEMA),
    Judge("gemma12", "gemma4-12b-st", "Gemma 12B single-token", INK["secondary"], False, "--", "D", SINGLE_TOKEN),
    Judge("gemma12_fs", "gemma4-12b", "Gemma 12B full-schema", INK["secondary"], False, ":", "D", FULL_SCHEMA),
]
TRAINED_AGAINST = "qwen9b"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval_root", required=True)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--stem", default="winrate_zeroshot")
    ap.add_argument("--title", default="Generator win rate under four zero-shot judges")
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
    if len(n_pairs) != 1:
        raise SystemExit(f"FAIL: judges scored different pair counts {sorted(n_pairs)}")
    n = n_pairs.pop()

    steps_of = {k: [r["step"] for r in rows] for k, rows in data.items()}
    steps = sorted({s for v in steps_of.values() for s in v})
    for key, value in steps_of.items():
        if value != steps:
            raise SystemExit(f"FAIL: {key} covers {value}, others cover {steps}")

    apply_rc()
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.axhline(0.5, color=INK["muted"], lw=1, ls=(0, (4, 3)), zorder=1)

    ends = []
    for j in JUDGES:
        ys = [r["gen_win_rate"] for r in data[j.key]]
        filled = j.protocol == SINGLE_TOKEN
        ax.plot(steps_of[j.key], ys, color=j.color, linestyle=j.linestyle,
                lw=3.2 if j.emph else 2.0,
                marker=j.marker, markersize=9 if j.emph else 7,
                markerfacecolor=j.color if filled else INK["surface"],
                markeredgecolor=INK["surface"] if filled else j.color,
                markeredgewidth=2 if filled else 1.6,
                zorder=5 if j.emph else 3, solid_capstyle="round")
        ends.append((ys[-1], j))

    # Anchored right: the curves all start near 0.5 and fan out, so the left end is where
    # the note would sit on top of them. By the last step 0.5 is empty space.
    ax.annotate("0.5 = parity", xy=(steps[-1], 0.5), xytext=(0, 5),
                textcoords="offset points", color=INK["muted"], fontsize=8.5,
                va="bottom", ha="right", zorder=4,
                bbox=dict(facecolor=INK["surface"], edgecolor="none", pad=1.5))

    lo, hi = ax.get_ylim()
    placed = declutter([(y - lo) / (hi - lo) for y, _ in ends])
    for (y, j), yf in zip(ends, placed):
        ax.annotate(j.label + ("*" if j.key == TRAINED_AGAINST else ""),
                    xy=(steps[-1], lo + yf * (hi - lo)), xytext=(9, 0),
                    textcoords="offset points",
                    color=INK["primary"] if j.emph else INK["secondary"],
                    fontsize=10, fontweight="bold" if j.emph else "normal",
                    va="center", ha="left", zorder=6, annotation_clip=False)

    style_axes(ax, "", "GRPO step", "fraction picked over the human", steps)
    ax.set_xlim(steps[0] - 1.5, steps[-1] + 4.5)

    fig.suptitle(a.title, x=0.008, ha="left", color=INK["primary"],
                 fontsize=13.5, fontweight="bold", y=0.995)
    default_subtitle = (
        f"{n} held-out pairs per checkpoint. Filled = single-token judge, "
        f"hollow = full schema (thinking ON).\n"
        f"* = the judge the generator was trained against."
    )
    fig.text(0.008, 0.915, a.subtitle.format(n=n) if a.subtitle else default_subtitle,
             ha="left", va="top", color=INK["muted"], fontsize=9)

    fig.tight_layout(rect=(0, 0, 0.74, 0.85))
    path = out_dir / f"{a.stem}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
