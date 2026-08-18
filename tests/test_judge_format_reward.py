"""The format reward must pay for the ordered 37-field schema, densely and un-gameably.

Regression cover for a reward-shaping defect. `format_score` was a flat mean over four booleans
against `0.9*task + 0.1*format`, making all-37-fields worth 0.025 of total reward: a correct
compact `{"score_gap","rating"}` scored 0.95 while a full verdict with imperfect arithmetic
scored 0.975. Both 2B runs found the shortcut -- `fmt_all_fields` fell to 0.000 while the
aggregate format reward *rose* 0.33 -> 0.44. The model was optimising the reward as written.
"""

import json

import pytest

from shared.judge_prompts import TURING_RESPONSE_PROPERTIES

from training.grpo.judge_verdict import (
    TURING_FIELDS,
    JudgeVerdict,
    derive_rating,
    ordered_prefix_coverage,
    parse_judge_verdict,
    strict_parse_answer,
    top_level_json_keys,
)


def _full_verdict_dict():
    """A schema-perfect verdict whose stated totals match its own primitives.

    The rating is *derived* rather than asserted, so `fmt_arith` holds: a fixture that declares a
    rating its own numbers do not imply would silently cost 0.10 of the format score and make
    every "full credit" assertion below wrong for the wrong reason.
    """
    values = {
        name: "because" if schema["type"] == "string" else 0.0
        for name, schema in TURING_RESPONSE_PROPERTIES.items()
    }
    values["rating"] = derive_rating(values)[0]
    return values


def _full_verdict_json():
    return json.dumps(_full_verdict_dict())


COMPACT = '{"score_gap": 1.5, "rating": 6}'


# --- the shortcut this work exists to close ------------------------------------------------


def test_full_schema_beats_compact_json_by_the_intended_margin():
    full = parse_judge_verdict(_full_verdict_json())
    compact = parse_judge_verdict(COMPACT)

    # Under the old flat mean this gap was 0.25 of a 0.1-weighted term, i.e. 0.025 of total
    # reward. It is now ~0.9 of the format term, i.e. ~0.09 of total.
    assert full.format_score == pytest.approx(1.0, abs=0.01)
    assert compact.format_score == pytest.approx(0.1, abs=0.01)
    assert (full.format_score - compact.format_score) * 0.1 == pytest.approx(0.09, abs=0.005)


def test_compact_json_still_earns_full_task_credit():
    """Format shapes format. It must never gate the accuracy signal."""
    compact = parse_judge_verdict(COMPACT)

    assert compact.rating is not None
    assert compact.recovery_rung == "score_gap"


# --- density: truncation is the case the dense term exists for -----------------------------


def test_truncated_verdict_earns_its_ordered_prefix():
    """The 2B saturates the response cap, so truncation is the common case, not an edge case."""
    cut_field = "human_goal_score_a"
    survivors = TURING_FIELDS.index(cut_field)
    full = _full_verdict_json()
    truncated = full[: full.index(f'"{cut_field}"')]

    verdict = parse_judge_verdict(truncated)

    # Whatever survives the cut is credited; parse-then-score yielded exactly 0.0 here.
    assert survivors > 0
    assert verdict.fmt_ordered_coverage == pytest.approx(survivors / 37, abs=0.001)
    assert verdict.fmt_ordered_coverage > 0.0
    assert verdict.fmt_strict_json is False
    assert verdict.fmt_exact_schema is False


def test_coverage_grows_monotonically_with_emitted_prefix():
    full = _full_verdict_json()
    cuts = [full.index(f'"{name}"') for name in TURING_FIELDS[1:6]]
    coverages = [parse_judge_verdict(full[:cut]).fmt_ordered_coverage for cut in cuts]

    assert coverages == sorted(coverages)
    assert coverages[0] < coverages[-1]


# --- order-awareness and un-gameability ----------------------------------------------------


def test_reversed_field_order_scores_no_coverage():
    reversed_json = json.dumps({k: v for k, v in reversed(list(_full_verdict_dict().items()))})

    verdict = parse_judge_verdict(reversed_json)

    assert verdict.fmt_ordered_coverage == pytest.approx(0.0)
    assert verdict.fmt_exact_schema is False


def test_field_names_mentioned_in_reasoning_earn_nothing():
    """Coverage must not be farmable by talking about the schema."""
    prose = json.dumps(
        {
            "reasoning": " ".join(f'"{name}": 1.0,' for name in TURING_FIELDS[:10]),
            "rating": 6,
        }
    )

    verdict = parse_judge_verdict(prose)

    assert verdict.fmt_ordered_coverage == pytest.approx(0.0)


def test_extra_keys_break_exact_schema_but_not_coverage():
    payload = _full_verdict_dict()
    payload["bonus_field"] = 1.0

    verdict = parse_judge_verdict(json.dumps(payload))

    assert verdict.fmt_ordered_coverage == pytest.approx(1.0)
    assert verdict.fmt_exact_schema is False


def test_missing_middle_field_truncates_the_prefix():
    dropped = 4
    payload = {k: v for k, v in _full_verdict_dict().items() if k != TURING_FIELDS[dropped]}

    verdict = parse_judge_verdict(json.dumps(payload))

    assert verdict.fmt_ordered_coverage == pytest.approx(dropped / 37, abs=0.001)


def test_wrong_types_fail_exact_schema():
    numeric_field = next(
        n for n, sch in TURING_RESPONSE_PROPERTIES.items() if sch["type"] == "number"
    )
    payload = _full_verdict_dict()
    payload[numeric_field] = "not a number"

    verdict = parse_judge_verdict(json.dumps(payload))

    assert verdict.fmt_strict_json is True
    assert verdict.fmt_exact_schema is False


# --- the thinking-marker interaction -------------------------------------------------------


def test_reasoning_then_answer_earns_full_format_credit():
    """With thinking ON the completion is `reasoning...</think>\\n\\n{json}`.

    `</think>` is an ordinary vocabulary token and survives skip_special_tokens, so scoring the
    whole string would fail json.loads on every well-formed answer -- zeroing strict validity and
    exact schema, 0.30 of the format term, on exactly the runs the thinking fix enables.
    """
    completion = f"Let me weigh both turns.\n</think>\n\n{_full_verdict_json()}"

    verdict = parse_judge_verdict(completion)

    assert verdict.format_score == pytest.approx(1.0, abs=0.01)
    assert verdict.fmt_strict_json is True
    assert verdict.fmt_exact_schema is True


def test_reasoning_naming_fields_with_a_compact_answer_earns_no_coverage():
    """The other half of the interaction: reasoning must not be scored as if it were the answer."""
    named = " ".join(f'"{name}"' for name in TURING_FIELDS[:20])
    completion = f"I will report {named}.\n</think>\n\n{COMPACT}"

    verdict = parse_judge_verdict(completion)

    assert verdict.fmt_ordered_coverage == pytest.approx(0.0)
    assert verdict.format_score == pytest.approx(0.1, abs=0.01)


def test_only_the_final_think_marker_splits_the_answer():
    completion = f"a</think>b</think>\n{_full_verdict_json()}"

    assert parse_judge_verdict(completion).fmt_exact_schema is True


def test_thinking_off_completions_are_unaffected():
    """No marker means the whole completion is the answer -- the retained ablation runs' shape."""
    assert parse_judge_verdict(_full_verdict_json()).fmt_exact_schema is True


# --- strict parsing must not use the tolerant extractor ------------------------------------


def test_strict_parse_rejects_prose_wrapped_json():
    wrapped = f"Here is my verdict: {_full_verdict_json()} -- hope that helps."

    parsed, exact = strict_parse_answer(wrapped)

    assert parsed is None
    assert exact is False
    # ...but the lenient ladder still recovers a rating from it, so task reward is unharmed.
    assert parse_judge_verdict(wrapped).rating is not None


def test_strict_parse_accepts_surrounding_whitespace():
    parsed, exact = strict_parse_answer(f"\n  {_full_verdict_json()}  \n")

    assert parsed is not None
    assert exact is True


# --- the streaming key scanner --------------------------------------------------------------


def test_scanner_ignores_nested_and_quoted_keys():
    payload = '{"a": {"b": 1}, "c": "text with \\"d\\": 2 inside", "e": [{"f": 3}], "g": 4}'

    assert top_level_json_keys(payload) == ["a", "c", "e", "g"]


def test_scanner_survives_truncation_mid_string():
    """Only COMPLETED entries count, so "b" -- whose value is unterminated -- is not yet emitted."""
    assert top_level_json_keys('{"a": 1, "b": "unterminated') == ["a"]


def test_scanner_requires_a_delimited_value():
    """A key with no value earns nothing. This is the key-only farming exploit, in miniature."""
    assert top_level_json_keys('{"a":,"b":}') == []
    assert top_level_json_keys('{"a": 1,"b":}') == ["a"]


def test_scanner_returns_nothing_without_an_object():
    assert top_level_json_keys("no json here") == []
    assert top_level_json_keys(None) == []


def test_coverage_of_empty_key_list_is_zero():
    assert ordered_prefix_coverage([]) == pytest.approx(0.0)


def test_verdict_defaults_keep_old_constructions_working():
    """The three scored fields default, so callers constructing verdicts directly still work."""
    verdict = JudgeVerdict(
        rating=4,
        recovery_rung="rating_field",
        fmt_json_valid=True,
        fmt_all_fields=False,
        fmt_arith=False,
        fmt_rating_range=True,
    )

    assert verdict.format_score == pytest.approx(0.0)


# --- reward-hacking regression cover ---------------------------------------------------------
#
# Each case below scored at or above an honest answer before being closed. The invariant that
# ties them together: a correct, complete, in-range verdict must strictly dominate every
# degenerate alternative. Numbers assume a correct rating, i.e. task reward 1.0.


def _total(text, task=1.0):
    verdict = parse_judge_verdict(text)
    effective_task = task if verdict.recovered else 0.0
    return 0.9 * effective_task + 0.1 * verdict.format_score


def _absurd_verdict():
    """Internally consistent arithmetic over values the rubric cannot produce."""
    payload = _full_verdict_dict()
    for field in ("immediate_target_score_b", "human_goal_score_b", "communication_style_score_b"):
        payload[field] = 10.0        # declared maximum is 1.0
    payload["base_score_b"] = 30.0   # declared maximum is 3.0
    payload["response_b_score"] = 30.0
    payload["score_gap"] = 30.0      # declared range is [-3, 3]
    payload["rating"] = derive_rating(payload)[0]
    return json.dumps(payload)


def test_key_only_output_earns_nothing():
    """`{"field":,"field":,...}` is 37 keys, no values, invalid JSON. It scored 0.96."""
    key_only = "{" + ",".join(f'"{name}":' for name in TURING_FIELDS) + "}"

    assert parse_judge_verdict(key_only).fmt_ordered_coverage == pytest.approx(0.0)
    assert _total(key_only) == pytest.approx(0.0)


def test_reasoning_only_completion_earns_no_task_reward():
    """Reasoning to a full verdict then emitting no answer scored 0.91, tying honest compact.

    judge_reward scores an unrecovered verdict at task 0.0, so refusing to read the reasoning
    block makes actually answering necessary rather than merely worth 0.1.
    """
    completion = f"thinking {_full_verdict_json()} done\n</think>\n\n"
    verdict = parse_judge_verdict(completion)

    assert verdict.recovered is False
    assert verdict.recovery_rung == "none"
    assert _total(completion) == pytest.approx(0.0)


def test_out_of_range_values_lose_coverage_schema_and_arithmetic():
    """Scores of 10 and a gap of 30 are internally consistent, and scored a perfect 1.0."""
    verdict = parse_judge_verdict(_absurd_verdict())

    assert verdict.fmt_exact_schema is False
    assert verdict.fmt_arith is False, "arithmetic over impossible values is not consistency"
    assert verdict.fmt_ordered_coverage < 0.2


def test_non_finite_numbers_are_not_valid_json():
    """json.loads accepts NaN/Infinity by default; a NaN score_gap satisfied every check."""
    payload = _full_verdict_dict()
    payload["score_gap"] = float("nan")
    text = json.dumps(payload)

    verdict = parse_judge_verdict(text)

    assert verdict.fmt_strict_json is False
    assert verdict.fmt_exact_schema is False


def test_rating_must_be_a_strict_in_range_integer():
    for bad in (99, 0, 6.0, True):
        payload = _full_verdict_dict()
        payload["rating"] = bad
        assert parse_judge_verdict(json.dumps(payload)).fmt_exact_schema is False, bad


def test_honest_output_strictly_dominates_every_known_exploit():
    """The invariant. If this fails, some degenerate policy pays at least as well as answering."""
    honest = _total(_full_verdict_json())
    nan_payload = _full_verdict_dict()
    nan_payload["score_gap"] = float("nan")

    exploits = {
        "key-only": "{" + ",".join(f'"{n}":' for n in TURING_FIELDS) + "}",
        "reasoning-only": f"thinking {_full_verdict_json()} done\n</think>\n\n",
        "out-of-range": _absurd_verdict(),
        "non-finite": json.dumps(nan_payload),
        "compact": COMPACT,
    }
    for name, text in exploits.items():
        assert _total(text) < honest, f"{name} scores {_total(text):.4f} vs honest {honest:.4f}"
