import json

import pandas as pd
import pytest

from scripts.plot_rubric_trajectory import (
    RAW_FIELDS,
    parse_json_object,
    raw_generated_field,
    summarize_trajectories,
)


def _raw(**fields):
    return json.dumps(fields)


def test_raw_generated_field_uses_raw_generated_side_without_defaulting_missing_values():
    row_a = {
        "generated_is_b": False,
        "judge_raw_content": _raw(
            immediate_target_score_a=0.25,
            immediate_target_score_b=0.75,
        ),
    }
    row_b = {**row_a, "generated_is_b": True}

    assert raw_generated_field(row_a, "immediate_target_score") == 0.25
    assert raw_generated_field(row_b, "immediate_target_score") == 0.75
    assert raw_generated_field(row_a, "assistant_like_penalty") is None


def test_raw_generated_field_rejects_invalid_values_and_parser_accepts_wrapped_json():
    parsed = parse_json_object('prefix ```json\n{"rating": 4}\n``` suffix')
    assert parsed == {"rating": 4}

    for value in (True, "0.4", -0.1, 1.1, None):
        row = {
            "generated_is_b": False,
            "judge_raw_content": _raw(immediate_target_score_a=value),
        }
        assert raw_generated_field(row, "immediate_target_score") is None


def test_summary_reports_field_errors_and_uses_common_pair_support_across_epochs():
    cells = {}
    for epoch in range(4):
        cells[("qwen35-9b", epoch, "judge")] = pd.DataFrame(
            {
                "pair_id": ["common", "missing_once", "epoch_only"],
                **{
                    field: (
                        [0.2 + epoch * 0.1, None if epoch == 2 else 0.9, 0.4]
                        if field == "immediate_target_score"
                        else [0.0, 0.0, 0.0]
                    )
                    for field in RAW_FIELDS
                },
            }
        )

    summary = summarize_trajectories(cells, judges=["judge"])
    score = summary[
        (summary["model_key"] == "qwen35-9b")
        & (summary["judge"] == "judge")
        & (summary["field"] == "immediate_target_score")
    ].sort_values("epoch")

    assert score["n_calls"].tolist() == [3, 3, 3, 3]
    assert score["n_valid"].tolist() == [3, 3, 2, 3]
    assert score["field_error_fraction"].tolist() == pytest.approx([0.0, 0.0, 1 / 3, 0.0])
    assert score["available_mean"].tolist() == pytest.approx([0.5, 1.6 / 3, 0.4, 0.6])
    assert score["paired_n"].tolist() == [2, 2, 2, 2]
    assert score["paired_mean"].tolist() == pytest.approx([0.3, 0.35, 0.4, 0.45])
