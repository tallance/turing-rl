"""CELLS/PROMPT_CELLS are read from the environment at import time.

The default must stay byte-identical to the hardcoded step-320 list it replaced:
a typo there would not fail loudly, it would quietly drop or rename a judge and
the sweep would still produce plausible-looking numbers from fewer cells.
"""
import importlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

STEP320_CELLS = ["qwen35-4b", "qwen35-9b", "qwen35-27b", "gemma4-12b", "gemma4-31b"]
STEP320_PROMPT_CELLS = ["qwen35-9b", "qwen35-27b", "gemma4-31b", "gemma4-12b", "qwen35-4b"]


def _load(monkeypatch, **env):
    for k in ("CELLS", "PROMPT_CELLS", "KEYS_JSON", "KEYS_FILE", "STYLE"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    import scripts.extract_step_subset as mod
    return importlib.reload(mod)


def test_default_reproduces_the_step320_lists(monkeypatch):
    mod = _load(monkeypatch)
    assert mod.CELLS == STEP320_CELLS
    assert mod.PROMPT_CELLS == STEP320_PROMPT_CELLS


def test_single_cell_override_for_the_lineage_roots(monkeypatch):
    """The alternating-lineage roots have one full-schema `on` cell."""
    mod = _load(monkeypatch, CELLS="gemma4-12b", PROMPT_CELLS="gemma4-12b")
    assert mod.CELLS == ["gemma4-12b"]
    assert mod.PROMPT_CELLS == ["gemma4-12b"]
    # A single cell must leave nothing to cross-check, which is what makes the
    # existing cross-cell assertions safe to keep rather than special-case.
    assert mod.CELLS[1:] == []


def test_multi_cell_override_splits_on_comma(monkeypatch):
    mod = _load(monkeypatch, CELLS="a,b,c")
    assert mod.CELLS == ["a", "b", "c"]


def test_keys_file_is_read_when_keys_json_is_unset(monkeypatch, tmp_path):
    p = tmp_path / "keys.json"
    p.write_text('[["u1","p1","0"],["u2","p2","1"]]')
    monkeypatch.delenv("KEYS_JSON", raising=False)
    monkeypatch.setenv("KEYS_FILE", str(p))
    mod = _load(monkeypatch, KEYS_FILE=str(p))
    assert json.loads(mod.KEYS_JSON) == [["u1", "p1", "0"], ["u2", "p2", "1"]]


def test_keys_json_wins_over_keys_file(monkeypatch, tmp_path):
    """An explicit inline list is not silently overridden by a stale file."""
    p = tmp_path / "keys.json"
    p.write_text('[["from","file","0"]]')
    mod = _load(monkeypatch, KEYS_JSON='[["from","env","0"]]', KEYS_FILE=str(p))
    assert json.loads(mod.KEYS_JSON) == [["from", "env", "0"]]


def test_style_nests_the_mode_dir_like_judge_sweep_cell(monkeypatch):
    """Mirrors judge_sweep_cell.sh:175 -- a non-full style adds one path level.

    Without this the extractor globs sweep/<cell>/on/reward and silently finds nothing for a
    rating_only run, which reads as "no shards" rather than "wrong path".
    """
    mod = _load(monkeypatch)
    assert mod.MODE_DIR == "on"
    mod = _load(monkeypatch, STYLE="rating_only")
    assert mod.MODE_DIR == "on/rating_only"


@pytest.fixture(autouse=True)
def _restore(monkeypatch):
    """Leave the module in its default state for any later importer."""
    yield
    for k in ("CELLS", "PROMPT_CELLS", "KEYS_JSON", "KEYS_FILE", "STYLE"):
        monkeypatch.delenv(k, raising=False)
    importlib.reload(sys.modules["scripts.extract_step_subset"])
