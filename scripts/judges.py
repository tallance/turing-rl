"""One definition of who the judges are: cell name, labels, and how they draw.

Three figures and the comparison table all need the same judge identities. The
palette already lives in plotstyle.py for exactly this reason -- otherwise "the
same quantity can end up a different colour in adjacent figures" -- and the same
argument applies to which cells exist and what they are called. Adding a judge
should be one edit here, not four.

Colours are assigned in fixed slot order BY ENTITY, never by rank: adding or
dropping a judge must not repaint the others.

CURVE judges have a summary row per checkpoint on the full pair set. POINT
judges are evaluated at selected checkpoints on a smaller pair set, so they are
drawn as hollow markers and are deliberately exempt from the figure guards that
keep the curves comparable -- see plot_test_eval_judges.py.
"""
from typing import NamedTuple

from plotstyle import INK, SLOT


class Judge(NamedTuple):
    cell: str           # sweep cell directory, and the summary_<cell>.csv stem
    key: str            # short internal handle
    label: str          # short label for figures
    long_label: str     # full label for tables
    color: str
    linestyle: str
    marker: str
    size: float         # marker size; point judges need a larger glyph
    emph: bool          # the judge the GRPO run was trained against
    legend_rank: int


# Draw order, emphasised judge last so it sits on top.
CURVE_JUDGES = [
    Judge("qwen35-4b", "4b", "4B", "Qwen3.5 4B", SLOT[1], "-", "o", 7, False, 0),
    Judge("qwen35-27b", "27b", "27B", "Qwen3.5 27B", SLOT[3], "-", "o", 7, False, 2),
    Judge("gemma4-12b", "gemma4", "Gemma 4 12B", "Gemma 4 12B",
          INK["secondary"], "--", "D", 7, False, 3),
    Judge("gemma4-31b", "gemma31", "Gemma 4 31B", "Gemma 4 31B",
          "#8b5fbf", "-", "s", 7, False, 4),
    Judge("qwen35-9b", "9b", "9B", "Qwen3.5 9B *", SLOT[2], "-", "o", 9, True, 1),
]

POINT_JUDGES = [
    Judge("claude-opus-5", "opus", "Opus 5", "Claude Opus 5",
          "#b5451f", "none", "*", 18, False, 5),
    Judge("claude-sonnet-5", "sonnet", "Sonnet 5", "Claude Sonnet 5",
          "#1f6f8b", "none", "P", 11, False, 6),
]

ALL_JUDGES = CURVE_JUDGES + POINT_JUDGES
BY_CELL = {j.cell: j for j in ALL_JUDGES}


def label_for(cell: str, long: bool = False) -> str:
    """Display name for a cell, falling back to the raw cell for unknown ones."""
    judge = BY_CELL.get(cell)
    if judge is None:
        return cell
    return judge.long_label if long else judge.label
