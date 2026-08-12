"""Unit tests for the zero-shot judge format probe.

Network calls are out of scope here; these lock the regime mapping and the summary
arithmetic, which are the parts that decide the Phase 0 gate.
"""

import json

import pytest

from shared.judge_prompts import TURING_RESPONSE_PROPERTIES
from scripts.probe_judge_format import (
    REGIMES,
    dump_row,
    probe_record,
    response_format_for_regime,
    summarize_probe,
)


def _full_verdict(rating: int = 7) -> str:
    data = {}
    for name, schema in TURING_RESPONSE_PROPERTIES.items():
        data[name] = "text" if schema["type"] == "string" else 0.0
    data["immediate_target_score_b"] = 1.0
    data["human_goal_score_b"] = 1.0
    data["communication_style_score_b"] = 1.0
    data["base_score_a"] = 0.0
    data["base_score_b"] = 3.0
    data["penalty_a"] = 0.0
    data["penalty_b"] = 0.0
    data["response_a_score"] = 0.0
    data["response_b_score"] = 3.0
    data["score_gap"] = 3.0
    data["rating"] = rating
    return json.dumps(data)


def test_regimes_are_the_three_documented_ones():
    assert REGIMES == ("json_schema", "json_object", "freeform")


def test_freeform_sends_no_response_format():
    assert response_format_for_regime("freeform") is None


def test_json_object_sends_the_loose_constraint():
    assert response_format_for_regime("json_object") == {"type": "json_object"}


def test_json_schema_sends_the_full_ordered_schema():
    fmt = response_format_for_regime("json_schema")
    assert fmt["type"] == "json_schema"
    assert fmt["json_schema"]["schema"]["required"] == list(TURING_RESPONSE_PROPERTIES)


def test_unknown_regime_raises():
    with pytest.raises(ValueError):
        response_format_for_regime("nonsense")


def test_probe_record_scores_a_well_formed_verdict():
    record = probe_record(_full_verdict(7), "stop", human_is_b=True)
    assert record["fmt_all_fields"] == 1.0
    assert record["recovered"] == 1.0
    assert record["correct"] == 1.0
    assert record["truncated"] == 0.0
    assert record["rung"] == "dimensions"


def test_probe_record_marks_a_length_stop_as_truncated():
    assert probe_record(_full_verdict(), "length", human_is_b=True)["truncated"] == 1.0


def test_probe_record_handles_an_unusable_completion():
    record = probe_record("nothing useful", "stop", human_is_b=True)
    assert record["recovered"] == 0.0
    assert record["fmt_all_fields"] == 0.0
    assert record["correct"] == 0.0


def test_summary_averages_the_gate_metrics():
    records = [
        probe_record(_full_verdict(7), "stop", human_is_b=True),
        probe_record("nothing useful", "stop", human_is_b=True),
    ]
    summary = summarize_probe(records)
    assert summary["n"] == 2
    assert summary["fmt_all_fields_rate"] == 0.5
    assert summary["recovered_rate"] == 0.5
    assert summary["accuracy"] == 0.5
    assert summary["truncation_rate"] == 0.0
    assert summary["rung_counts"]["dimensions"] == 1
    assert summary["rung_counts"]["none"] == 1


def test_summary_of_no_records_is_empty_not_a_crash():
    assert summarize_probe([])["n"] == 0


def test_dump_row_emits_the_five_canonical_analysis_columns():
    row = {
        "prompt": [{"role": "user", "content": "x"}],
        "extra_info": {"pair_id": "p1::g0", "order": "human_b", "human_is_b": True},
    }
    record = probe_record(_full_verdict(7), "stop", human_is_b=True)
    assert dump_row("qwen35-4b", row, record) == {
        "model": "qwen35-4b",
        "pair_id": "p1::g0",
        "order": "human_b",
        "rating": 7,
        "human_is_b": True,
    }


def test_dump_row_carries_a_null_rating_when_nothing_was_recovered():
    row = {
        "prompt": [{"role": "user", "content": "x"}],
        "extra_info": {"pair_id": "p1::g0", "order": "human_a", "human_is_b": False},
    }
    record = probe_record("garbage", "stop", human_is_b=False)
    assert dump_row("m", row, record)["rating"] is None
