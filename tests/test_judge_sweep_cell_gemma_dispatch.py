"""Extra guards on judge_sweep_cell.sh's model dispatch.

tests/test_gemma_judge_sft.py already covers the four main routes (zero-shot Gemma,
trained Gemma, trained Qwen, unknown id / anchor). This file adds the cases that pin the
dispatch against a plausible future "simplification":

  - the decision reads the CONFIG, in both directions, so it cannot be rewritten as a
    directory-name match without failing;
  - the pinned snapshot SHAs are the ones actually served;
  - a MODEL path that does not exist does not crash the dispatch, so the later existence
    check owns that error and reports it as such.

The dispatch block is extracted and executed rather than pattern-matched, so a
restructure that changes behaviour fails here.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CELL_SCRIPT = ROOT / "scripts" / "slurm" / "judge_sweep_cell.sh"

GEMMA_CONFIG = {
    "architectures": ["Gemma4UnifiedForConditionalGeneration"],
    "model_type": "gemma4_unified",
}
QWEN_CONFIG = {
    "architectures": ["Qwen3_5ForConditionalGeneration"],
    "model_type": "qwen3_5",
}


def _dispatch(model: str) -> dict[str, str]:
    """Run the script's model-dispatch block for MODEL and report what it selected."""
    text = CELL_SCRIPT.read_text()
    start = text.index("IS_GEMMA4=0")
    # Stop at the environment checks: they assert cluster-only paths exist and would exit
    # here. What this file covers is the dispatch decision above them, so close the `if`
    # ourselves rather than stubbing binaries that only exist on the cluster.
    end = text.index('[ -x "$GEMMA_VLLM" ]', start)
    block = text[start:end] + "fi\n"

    script = f"MODEL={model!r}\n{block}\n" + (
        'printf "IS_GEMMA4=%s\\nGEMMA_MODEL_PATH=%s\\nPY_SERVER=%s\\n" '
        '"$IS_GEMMA4" "${GEMMA_MODEL_PATH:-}" "${PY_SERVER:-}"\n'
    )
    proc = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    return dict(
        line.split("=", 1) for line in proc.stdout.strip().splitlines() if "=" in line
    )


def _model_dir(tmp_path: Path, name: str, config: dict) -> Path:
    model = tmp_path / name
    model.mkdir(exist_ok=True)
    (model / "config.json").write_text(json.dumps(config))
    return model


def test_a_gemma_checkpoint_routes_by_config_whatever_it_is_named(tmp_path: Path) -> None:
    # Nothing in this name says Gemma. A name-based rule would send it to a vLLM that
    # cannot load the architecture, and the failure would land after the allocation.
    model = _model_dir(tmp_path, "judge_ce_epoch3", GEMMA_CONFIG)
    got = _dispatch(str(model))
    assert got["IS_GEMMA4"] == "1"
    assert got["GEMMA_MODEL_PATH"] == str(model)


def test_a_gemma_named_directory_that_is_not_gemma_is_not_captured(tmp_path: Path) -> None:
    # The converse: the word "gemma" in a path must not route a Qwen checkpoint onto the
    # Gemma server.
    model = _model_dir(tmp_path, "gemma_style_qwen_judge", QWEN_CONFIG)
    got = _dispatch(str(model))
    assert got["IS_GEMMA4"] == "0"
    assert got["PY_SERVER"].endswith("turing-rl-train/bin/python")


@pytest.mark.parametrize(
    "model,snapshot",
    [
        ("google/gemma-4-12B-it", "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7"),
        ("google/gemma-4-31B-it", "842da3794eaa0b77d5f08bae87a17459d91ff475"),
    ],
)
def test_the_hub_gemma_ids_serve_their_pinned_snapshot_sha(model: str, snapshot: str) -> None:
    got = _dispatch(model)
    assert got["IS_GEMMA4"] == "1"
    assert got["GEMMA_MODEL_PATH"].endswith(f"snapshots/{snapshot}")


def test_a_nonexistent_model_path_does_not_crash_the_dispatch(tmp_path: Path) -> None:
    # The config probe must tolerate a missing directory: the existence check further down
    # owns that error, and a crash here would surface it as an opaque shell failure.
    got = _dispatch(str(tmp_path / "absent"))
    assert got["IS_GEMMA4"] == "0"
    assert got["PY_SERVER"].endswith("turing-rl-train/bin/python")
