"""Which vLLM a model is served on, for judge_sweep_cell.sh.

Gemma 4 only loads on the nightly environment; the pinned 0.18.0 in turing-rl-train
cannot register the architecture. A locally merged Gemma judge is a directory path
rather than a hub id, so the hub-id cases do not catch it and it would otherwise fall
through to a server that fails at model load.

The dispatch block is extracted and executed rather than pattern-matched, so the test
fails if the branch is restructured in a way that changes behaviour.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CELL_SCRIPT = ROOT / "scripts" / "slurm" / "judge_sweep_cell.sh"


def _dispatch(model: str) -> dict[str, str]:
    """Run the script's model-dispatch block for MODEL and report what it selected."""
    text = CELL_SCRIPT.read_text()
    start = text.index("IS_GEMMA4=0")
    # Stop at the environment checks: they assert cluster-only paths exist and would exit
    # here. What this test covers is the dispatch decision above them, so close the `if`
    # ourselves rather than stubbing out binaries that only exist on the cluster.
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


def _model_dir(tmp_path: Path, architectures: list[str]) -> Path:
    model = tmp_path / "merged"
    model.mkdir(exist_ok=True)
    (model / "config.json").write_text(json.dumps({"architectures": architectures}))
    return model


def test_a_locally_merged_gemma_serves_on_the_gemma_vllm(tmp_path: Path) -> None:
    model = _model_dir(tmp_path, ["Gemma4UnifiedForConditionalGeneration"])
    got = _dispatch(str(model))
    assert got["IS_GEMMA4"] == "1"
    # Serve the directory itself, not a hub cache snapshot.
    assert got["GEMMA_MODEL_PATH"] == str(model)


def test_a_locally_merged_qwen_does_not_get_the_gemma_path(tmp_path: Path) -> None:
    # The Qwen CE judges are also local directories; they must keep the old behaviour.
    model = _model_dir(tmp_path, ["Qwen3_5ForConditionalGeneration"])
    got = _dispatch(str(model))
    assert got["IS_GEMMA4"] == "0"
    assert got["PY_SERVER"].endswith("turing-rl-train/bin/python")


def test_the_decision_follows_the_config_not_the_directory_name(tmp_path: Path) -> None:
    # A Gemma checkpoint whose directory says nothing about Gemma still routes correctly;
    # a name-based rule would send it to a vLLM that cannot load the architecture.
    model = tmp_path / "judge_ce_epoch3"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps({"architectures": ["Gemma4UnifiedForConditionalGeneration"]})
    )
    assert _dispatch(str(model))["IS_GEMMA4"] == "1"

    # And the converse: "gemma" in the name does not make a non-Gemma config Gemma.
    decoy = tmp_path / "gemma_style_qwen"
    decoy.mkdir()
    (decoy / "config.json").write_text(json.dumps({"architectures": ["Qwen3_5ForCausalLM"]}))
    assert _dispatch(str(decoy))["IS_GEMMA4"] == "0"


@pytest.mark.parametrize(
    "model,snapshot",
    [
        ("google/gemma-4-12B-it", "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7"),
        ("google/gemma-4-31B-it", "842da3794eaa0b77d5f08bae87a17459d91ff475"),
    ],
)
def test_the_hub_gemma_ids_still_resolve_to_their_pinned_snapshots(
    model: str, snapshot: str
) -> None:
    got = _dispatch(model)
    assert got["IS_GEMMA4"] == "1"
    assert got["GEMMA_MODEL_PATH"].endswith(f"snapshots/{snapshot}")


def test_a_plain_hub_qwen_id_is_unaffected() -> None:
    got = _dispatch("Qwen/Qwen3.5-9B")
    assert got["IS_GEMMA4"] == "0"
    assert got["PY_SERVER"].endswith("turing-rl-train/bin/python")


def test_the_397b_anchor_keeps_its_pinned_environment() -> None:
    got = _dispatch("Qwen/Qwen3.5-397B-A17B-GPTQ-Int4")
    assert got["IS_GEMMA4"] == "0"
    assert got["PY_SERVER"].endswith("judge-vllm/bin/python")


def test_a_nonexistent_model_path_does_not_crash_the_dispatch(tmp_path: Path) -> None:
    # The config probe must tolerate a missing directory: the later existence checks own
    # that error, and a crash here would report it as a shell failure instead.
    got = _dispatch(str(tmp_path / "absent"))
    assert got["IS_GEMMA4"] == "0"
