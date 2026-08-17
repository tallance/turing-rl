"""Shared light-mode chart style for the test-eval figures.

Two figures land in the same results directory, so the palette and the label
placement live here rather than being copied into each script -- otherwise the
same quantity can end up a different colour in adjacent figures.

Categorical hues are the reference palette's slots 1-3 (see the dataviz skill's
palette.md), which are the three that validate all-pairs in both modes. They are
assigned in fixed slot order BY ENTITY, never by rank: adding or dropping a
series must not repaint the others.
"""
from __future__ import annotations

# Chrome & ink.
INK = {
    "surface": "#fcfcfb",
    "primary": "#0b0b0b",
    "secondary": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
}

# Categorical slots 1-3.
SLOT = {1: "#2a78d6", 2: "#eb6834", 3: "#1baf7a"}

# Minimum vertical gap between two direct labels, as a fraction of axes height.
LABEL_GAP = 0.075


def apply_rc() -> None:
    """Set the figure-wide rcParams. Call before creating the figure."""
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        # Concrete faces only -- "system-ui"/"-apple-system" are CSS keywords and
        # make matplotlib emit a findfont warning per text object.
        "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"],
        "figure.facecolor": INK["surface"],
        "axes.facecolor": INK["surface"],
        "savefig.facecolor": INK["surface"],
    })


def declutter(y_fracs: list[float], gap: float = LABEL_GAP) -> list[float]:
    """Nudge direct-label positions apart, preserving their vertical order.

    Takes/returns axes-fraction y positions. Order is preserved so a label always
    stays on the same side of its neighbours as the curve it names.
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


def style_axes(ax, title: str, xlabel: str, ylabel: str, xticks: list[int]) -> None:
    """Titles, ticks, recessive grid, two spines. Common to every panel."""
    ax.set_title(title, color=INK["primary"], fontsize=12.5, fontweight="bold",
                 loc="left", pad=10)
    ax.set_xlabel(xlabel, color=INK["secondary"], fontsize=10.5)
    ax.set_ylabel(ylabel, color=INK["secondary"], fontsize=10.5)
    ax.set_xticks(xticks)
    ax.grid(True, axis="y", color=INK["grid"], lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK["muted"], labelsize=9.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK["axis"])
