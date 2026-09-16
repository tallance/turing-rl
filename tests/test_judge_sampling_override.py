"""The opt-in judge decode override for sweep cells.

Task 1 froze the sweep policy to "no wire sampling override; vLLM uses each model's
generation_config.json defaults" so cells stay comparable across the matrix. Measuring a
different decode policy therefore has to be explicit, and has to leave the default untouched --
otherwise every previously published cell silently becomes non-reproducible.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_judge_sweep_cell import cell_env  # noqa: E402

QWEN_THINKING = (
    '{"temperature":1.0,"top_p":0.95,"top_k":20,"min_p":0.0,'
    '"presence_penalty":1.5,"repetition_penalty":1.0}'
)


def test_no_override_by_default():
    """The frozen policy. If this ever emits a value, every historical cell's decode changes."""
    env = cell_env(model_id="Qwen/Qwen3.5-9B", mode="on", out_dir="/tmp/x")

    assert "PERSONA_JUDGE_SAMPLING" not in env


def test_the_ignored_sampling_argument_still_does_not_leak():
    """`sampling` is accepted for interface compatibility and deliberately dropped; only the
    explicit override may set the wire value."""
    env = cell_env(
        model_id="Qwen/Qwen3.5-9B", mode="on", out_dir="/tmp/x",
        sampling={"temperature": 0.7},
    )

    assert "PERSONA_JUDGE_SAMPLING" not in env


def test_an_explicit_override_is_emitted_verbatim():
    env = cell_env(
        model_id="Qwen/Qwen3.5-9B", mode="on", out_dir="/tmp/x",
        style="rating_only", sampling_override=QWEN_THINKING,
    )

    assert env["PERSONA_JUDGE_SAMPLING"] == QWEN_THINKING


def test_the_override_composes_with_the_rating_only_env():
    """Both must hold at once: the 37-field schema dropped AND the decode pinned."""
    env = cell_env(
        model_id="Qwen/Qwen3.5-9B", mode="on", out_dir="/tmp/x",
        style="rating_only", sampling_override=QWEN_THINKING,
    )

    assert "PERSONA_JUDGE_JSON_SCHEMA" not in env
    assert env["PERSONA_JUDGE_ENABLE_THINKING"] == "1"
    assert env["JUDGE_PROMPT_STYLE"] == "rating_only"
    assert env["PERSONA_JUDGE_SAMPLING"] == QWEN_THINKING


def test_the_cell_script_forwards_it_only_when_set():
    """Unset must leave the client invocation byte-identical to the historical one."""
    script = (ROOT / "scripts" / "slurm" / "judge_sweep_cell.sh").read_text()

    assert '[ -n "${JUDGE_SAMPLING:-}" ] && EXTRA+=(--judge_sampling "$JUDGE_SAMPLING")' in script
    # Not given a default anywhere, which would defeat the point.
    assert "JUDGE_SAMPLING:-{" not in script
    assert "JUDGE_SAMPLING=$" not in script
