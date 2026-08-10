import json

from scripts.experiments.compare_gemma31_schemas import (
    CURRENT_SCHEMA,
    FULL_PROPERTIES,
    FULL_SCHEMA,
    build_body,
    evenly_spaced_indices,
    formulas_valid,
    schema_valid,
)


def _full_object():
    value = {}
    for key, spec in FULL_PROPERTIES.items():
        if spec["type"] == "string":
            value[key] = "ok"
        elif spec["type"] == "integer":
            value[key] = 4
        else:
            value[key] = 0.0
    value.update(
        immediate_target_score_a=0.5,
        human_goal_score_a=0.5,
        communication_style_score_a=0.5,
        immediate_target_score_b=0.6,
        human_goal_score_b=0.5,
        communication_style_score_b=0.4,
        base_score_a=1.5,
        base_score_b=1.5,
        response_a_score=1.5,
        response_b_score=1.5,
        score_gap=0.0,
    )
    return value


def test_full_schema_is_ordered_rating_last_and_strict():
    assert FULL_SCHEMA["required"] == list(FULL_PROPERTIES)
    assert FULL_SCHEMA["required"][-1] == "rating"
    assert FULL_SCHEMA["additionalProperties"] is False
    assert len(FULL_PROPERTIES) == 37


def test_only_response_schema_differs_between_paired_bodies():
    messages = [{"role": "user", "content": "prompt"}]
    current = build_body(messages, "current_minimal", "model", 123)
    full = build_body(messages, "full_prompt_schema", "model", 123)
    current_rf = current.pop("response_format")
    full_rf = full.pop("response_format")
    assert current == full
    assert current_rf["json_schema"]["schema"] == CURRENT_SCHEMA
    assert full_rf["json_schema"]["schema"] == FULL_SCHEMA


def test_full_schema_validation_and_formulas():
    value = _full_object()
    assert schema_valid(value, "full_prompt_schema")
    assert formulas_valid(value)
    reordered = json.loads(json.dumps(value, sort_keys=True))
    assert not schema_valid(reordered, "full_prompt_schema")
    value["score_gap"] = 0.5
    assert formulas_valid(value) is False


def test_evenly_spaced_selection_includes_endpoints():
    indices = evenly_spaced_indices(64, 16)
    assert indices[0] == 0
    assert indices[-1] == 63
    assert len(indices) == len(set(indices)) == 16
