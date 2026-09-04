"""Guards on SERIALIZE_JUDGES in scripts/launch_test_eval.sh.

The launcher cannot be executed here: it resolves its cell list through a hardcoded cluster
python path. So the dependency logic is tested two ways -- a structural property over the
file (no unconditional chaining can reappear), and by lifting the actual guard line out of
the script and running it under bash, so the assertions cannot drift from the code.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "launch_test_eval.sh"


def _lines() -> list[str]:
    return SCRIPT.read_text().splitlines()


def _code_lines() -> list[str]:
    return [ln for ln in _lines() if not ln.strip().startswith("#")]


def _guard_line() -> str:
    """The one line that builds the serialization dependency, lifted verbatim."""
    hits = [ln for ln in _code_lines() if "afterany" in ln]
    assert len(hits) == 1, f"expected exactly one afterany line, got {hits}"
    return hits[0].strip()


def test_chaining_cannot_reappear_unconditionally():
    """A property, not a copy of the line: any afterany must be behind the flag.

    Written this way on purpose -- asserting the exact text would still pass if someone
    added a *second*, unguarded chain elsewhere in the loop.
    """
    assert "SERIALIZE_JUDGES" in _guard_line()


def test_the_real_data_dependency_is_never_conditional():
    # afterok on the pair build is what stops a cell running before its pairs exist. Only the
    # capacity chain is opt-out; if this became conditional the sweep would race.
    afterok = [ln.strip() for ln in _code_lines() if "afterok:" in ln]
    assert afterok, "the pair-build dependency vanished"
    for line in afterok:
        assert "SERIALIZE_JUDGES" not in line, line


def test_default_still_serializes_and_bad_values_are_refused():
    text = SCRIPT.read_text()
    assert "SERIALIZE_JUDGES=${SERIALIZE_JUDGES:-1}" in text, "default must stay 1"
    assert "FATAL: SERIALIZE_JUDGES must be 0 or 1" in text


def _dep_for(serialize: str, prev: str, bjid: str = "999") -> str:
    """Run the script's own guard line and report the dependency string it builds."""
    script = "\n".join([
        "set -uo pipefail",
        f'SERIALIZE_JUDGES="{serialize}"',
        f'PREV="{prev}"',
        f'bjid="{bjid}"',
        'dep=""',
        '[ -n "$bjid" ] && dep="afterok:$bjid"',
        _guard_line(),
        'printf "%s" "$dep"',
    ])
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=False)
    assert out.returncode == 0, out.stderr
    return out.stdout


def test_serialized_chains_behind_the_previous_cell():
    assert _dep_for("1", "12345") == "afterok:999,afterany:12345"
    # First cell of a run has no predecessor, so only the build dependency applies.
    assert _dep_for("1", "") == "afterok:999"


def test_unserialized_keeps_the_build_dependency_and_drops_the_chain():
    dep = _dep_for("0", "12345")
    assert dep == "afterok:999", dep
    assert "afterany" not in dep


def test_unserialized_still_waits_on_pairs_when_it_is_the_first_cell():
    assert _dep_for("0", "", bjid="777") == "afterok:777"
