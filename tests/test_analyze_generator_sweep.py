import os, sys
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.analyze_generator_sweep import discover_generators, comparison_rows


def test_discover_generators(tmp_path):
    for g in ("qwen3-8b-base", "qwen35-9b-sft"):
        (tmp_path / g / "sweep" / "qwen35-397b" / "on" / "reward").mkdir(parents=True)
    (tmp_path / "pairs").mkdir()                       # non-generator dir ignored
    assert discover_generators(tmp_path) == ["qwen35-9b-sft", "qwen3-8b-base"] or \
           sorted(discover_generators(tmp_path)) == ["qwen3-8b-base", "qwen35-9b-sft"]


def test_comparison_rows(tmp_path):
    # one fake per-generator summary.parquet using the REAL persisted schema
    # (write_summary renames accuracy->acc_parse_ok, accuracy_penalized->acc_penalized,
    #  parse_error_rate->parse_error) -> flat comparison rows tagged by generator.
    d = tmp_path / "derived" / "qwen3-8b-base"; d.mkdir(parents=True)
    pd.DataFrame([{"cell": "qwen35-397b", "mode": "on", "acc_parse_ok": 0.72,
                   "acc_penalized": 0.70, "parse_error": 0.03, "tie_rate": 0.05}]
                 ).to_parquet(d / "summary.parquet")
    rows = comparison_rows(tmp_path / "derived", ["qwen3-8b-base"])
    assert rows[0]["generator"] == "qwen3-8b-base"
    assert rows[0]["judge"] == "qwen35-397b" and rows[0]["acc_parse_ok"] == 0.72
