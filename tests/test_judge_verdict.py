"""Unit tests for judge verdict parsing and the rating-recovery ladder."""

import json

from shared.judge_prompts import TURING_RESPONSE_PROPERTIES
from training.grpo.judge_verdict import (
    TURING_FIELDS,
    derive_rating,
    extract_json_object,
    parse_judge_verdict,
)


def _verdict(**overrides) -> dict:
    """A well-formed verdict where B scores 3.0 and A scores 0.0 -> gap 3.0 -> rating 7."""
    data = {}
    for name, schema in TURING_RESPONSE_PROPERTIES.items():
        if schema["type"] == "string":
            data[name] = "text"
        elif schema["type"] == "integer":
            data[name] = 4
        else:
            data[name] = 0.0
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
    data["rating"] = 7
    data.update(overrides)
    return data


def test_turing_fields_come_from_the_schema():
    assert TURING_FIELDS == tuple(TURING_RESPONSE_PROPERTIES)


def test_extract_json_object_handles_a_fenced_block():
    assert extract_json_object('```json\n{"rating": 5}\n```') == {"rating": 5}


def test_extract_json_object_handles_prose_around_the_object():
    assert extract_json_object('here you go: {"rating": 5} hope that helps') == {"rating": 5}


def test_extract_json_object_returns_none_instead_of_raising():
    assert extract_json_object("no json here") is None
    assert extract_json_object(None) is None
    assert extract_json_object("[1, 2, 3]") is None


def test_derive_rating_recomputes_gap_and_rating():
    rating, gap = derive_rating(_verdict())
    assert rating == 7
    assert gap == 3.0


def test_derive_rating_applies_the_penalty_formula():
    # All four B penalties at 1.0 -> penalty_b = (4/4)*3 = 3.0 -> b_score = max(0, 3-3) = 0.
    data = _verdict(
        source_copy_penalty_b=1.0,
        wrong_target_or_role_penalty_b=1.0,
        unsupported_adversarial_reframing_penalty_b=1.0,
        assistant_like_penalty_b=1.0,
    )
    rating, gap = derive_rating(data)
    assert gap == 0.0
    assert rating == 4


def test_perfect_verdict_scores_every_format_component():
    v = parse_judge_verdict(json.dumps(_verdict()))
    assert v.rating == 7
    assert v.recovery_rung == "dimensions"
    assert v.fmt_json_valid and v.fmt_all_fields and v.fmt_arith and v.fmt_rating_range
    assert v.format_score == 1.0


def test_missing_field_loses_all_fields_but_keeps_the_rating():
    data = _verdict()
    del data["reasoning"]
    v = parse_judge_verdict(json.dumps(data))
    assert v.rating == 7
    assert v.recovery_rung == "dimensions"
    assert v.fmt_json_valid and not v.fmt_all_fields
    assert v.recovered


def test_extra_field_loses_the_all_fields_component():
    v = parse_judge_verdict(json.dumps(_verdict(surprise="nope")))
    assert not v.fmt_all_fields
    assert v.rating == 7


def test_bad_arithmetic_loses_only_the_arith_component():
    v = parse_judge_verdict(json.dumps(_verdict(score_gap=-3.0, rating=1)))
    assert v.rating == 7  # derived from the dimensions, not the model's own claim
    assert not v.fmt_arith
    assert v.fmt_json_valid and v.fmt_all_fields


def test_score_gap_rung_when_dimensions_are_absent():
    v = parse_judge_verdict(json.dumps({"score_gap": 1.5, "reasoning": "x"}))
    assert v.recovery_rung == "score_gap"
    assert v.rating == 6
    assert not v.fmt_arith


def test_rating_field_rung_when_only_the_rating_survives():
    v = parse_judge_verdict(json.dumps({"rating": 2, "reasoning": "x"}))
    assert v.recovery_rung == "rating_field"
    assert v.rating == 2
    assert v.fmt_rating_range


def test_rating_text_rung_when_json_is_unparseable():
    v = parse_judge_verdict('the verdict is "rating": 6 and that is final')
    assert v.recovery_rung == "rating_text"
    assert v.rating == 6
    assert not v.fmt_json_valid


def test_unrecoverable_completion():
    v = parse_judge_verdict("total gibberish with no verdict")
    assert v.rating is None
    assert v.recovery_rung == "none"
    assert not v.recovered
    assert v.format_score == 0.0


def test_none_completion_is_unrecoverable():
    assert parse_judge_verdict(None).recovery_rung == "none"


def test_out_of_range_rating_field_fails_the_range_component():
    v = parse_judge_verdict(json.dumps({"rating": 99, "reasoning": "x"}))
    assert not v.fmt_rating_range
    assert v.recovery_rung == "none"
