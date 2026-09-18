"""Every prompt-style validator accepts the same set.

The style list is duplicated across four files -- three shell scripts and one Python constant --
because bash and Python cannot share one. Adding letter_only, only rl_generator_run_9b.sh was
updated, and the omission surfaced as `FATAL: PROMPT_STYLE must be full|single_token|rating_only`
at pair-build submit time. That was the cheap failure; the expensive one is a validator that is
missing a style it should REJECT, or a sweep cell silently scoring the wrong protocol.

Pinned as a set rather than a literal string so the copies can differ in wording and ordering,
which they do.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# rl_generator_run_9b.sh is the reference: it is the only one whose arms carry behaviour
# (single_token forces thinking off), so it cannot drift without something visibly breaking.
EXPECTED = {"full", "single_token", "rating_only", "letter_only"}

SHELL_VALIDATORS = (
    "scripts/launch_judge_pairs.sh",
    "scripts/slurm/judge_sweep_cell.sh",
    "scripts/slurm/rl_generator_run_9b.sh",
)


def _styles_in_error_message(path: str) -> set[str]:
    text = (ROOT / path).read_text()
    m = re.search(r"(?:PROMPT_STYLE|JUDGE_PROMPT_STYLE) must be ([a-z_|]+)", text)
    assert m, f"{path} has no style validator error message"
    return set(m.group(1).split("|"))


@pytest.mark.parametrize("path", SHELL_VALIDATORS)
def test_shell_validators_accept_every_style(path):
    assert _styles_in_error_message(path) == EXPECTED


@pytest.mark.parametrize("path", SHELL_VALIDATORS)
def test_the_case_arms_match_the_error_message(path):
    """An error message listing a style the case block does not actually accept is worse than
    useless: it says the value is valid while the script exits 2."""
    text = (ROOT / path).read_text()
    accepted = set()
    for arm in re.findall(r"^\s{2}([a-z_|]+)\)", text, re.M):
        accepted.update(arm.split("|"))

    assert EXPECTED <= accepted, f"{path} rejects {sorted(EXPECTED - accepted)}"


def test_the_python_sweep_constant_agrees():
    import sys

    sys.path.insert(0, str(ROOT))
    from scripts.run_judge_sweep_cell import PROMPT_STYLES

    assert set(PROMPT_STYLES) == EXPECTED
