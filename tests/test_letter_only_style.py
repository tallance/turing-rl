"""The letter_only judge: thinking ON, `{"answer": "A"|"B"}`, no tie reachable.

Loop 2's rating judge collapsed into hedging -- 78.1% of verdicts were rating 4. Under `graded`
a tie pays a flat 0.75, far above the 0.5 of information it carries, and because that payout is
CONSTANT a GRPO group that ties throughout has zero advantage and yields no gradient at all.
Learning stalled on exactly the hard examples.

This style removes the tie from the output space, so a hard example resolves 1/0 with real
variance inside the group. The tests that matter most here are the ones pinning that property
(no input yields rating 4) and the ones pinning that a non-answer FAILS rather than landing on
4 by accident -- the same bug shape fixed in 20cd6f8 on the generator side.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shared.judge_prompts import (  # noqa: E402
    TURING_LETTER_ONLY_PROMPT,
    TURING_PROMPT_HEADER,
)
from shared.judge_utils import letter_to_rating  # noqa: E402
from training.grpo.judge_reward import directional_task_reward  # noqa: E402
from training.grpo.judge_verdict import (  # noqa: E402
    PROMPT_STYLE_LETTER_ONLY,
    PROMPT_STYLE_RATING_ONLY,
    parse_judge_verdict,
)

THINK = "the judge reasons at length</think>"


def _verdict(answer: str):
    return parse_judge_verdict(
        f"{THINK}{answer}",
        thinking_enabled=True,
        prompt_style=PROMPT_STYLE_LETTER_ONLY,
    )


# --- the helper ------------------------------------------------------------------------------


@pytest.mark.parametrize("value", ["A", "a", " A ", "A\n"])
def test_a_means_response_a_is_human(value):
    assert letter_to_rating(value) == 1


@pytest.mark.parametrize("value", ["B", "b", " B ", "B\n"])
def test_b_means_response_b_is_human(value):
    assert letter_to_rating(value) == 7


@pytest.mark.parametrize("value", ["", "AB", "maybe A", "C", 4, None, "4"])
def test_anything_else_is_not_an_answer(value):
    """Strict on purpose: a loose parse here would invent verdicts out of noise."""
    assert letter_to_rating(value) is None


# --- the judge-training parser ---------------------------------------------------------------


def test_the_answer_field_is_read():
    got = _verdict('{"answer": "B"}')

    assert got.rating == 7
    assert got.recovery_rung == "letter"
    assert got.recovered


def test_the_other_letter_too():
    assert _verdict('{"answer": "A"}').rating == 1


def test_an_unclosed_think_block_has_no_answer():
    """Thinking never closed, so no answer exists -- judge_reward scores this task 0.0."""
    got = parse_judge_verdict(
        "reasoning that never ends",
        thinking_enabled=True,
        prompt_style=PROMPT_STYLE_LETTER_ONLY,
    )

    assert got.rating is None
    assert got.recovery_rung == "unclosed_thinking"


def test_a_prose_wrapped_letter_still_recovers():
    """Tolerant about packaging, strict about the verdict existing."""
    got = _verdict('Here is my answer: {"answer": "B"}')

    assert got.rating == 7
    assert got.recovered


@pytest.mark.parametrize(
    "answer",
    ['{"answer": "maybe"}', "{}", '{"rating": 4}', "no idea", '{"answer": 4}'],
)
def test_a_non_answer_is_not_recovered(answer):
    got = _verdict(answer)

    assert not got.recovered, answer
    assert got.rating != 4, f"{answer} landed on a tie"


def test_no_input_yields_a_tie():
    """THE property this loop exists for. A tie is not merely discouraged, it is unreachable."""
    for answer in [
        '{"answer": "A"}', '{"answer": "B"}', '{"answer": "C"}', '{"answer": ""}',
        '{"rating": 4}', "{}", "", "4", "tie", '{"answer": null}',
    ]:
        assert _verdict(answer).rating != 4, answer


def test_the_mapping_lands_on_the_right_side_of_the_tie_rating():
    """directional_task_reward splits on rating vs 4, so A=1/B=7 must straddle it."""
    assert directional_task_reward(1, human_is_b=False) == 1.0
    assert directional_task_reward(1, human_is_b=True) == 0.0
    assert directional_task_reward(7, human_is_b=True) == 1.0
    assert directional_task_reward(7, human_is_b=False) == 0.0


# --- format score ----------------------------------------------------------------------------


def test_an_exact_answer_object_scores_full_format():
    assert _verdict('{"answer": "A"}').format_score == pytest.approx(1.0)


def test_tidiness_is_what_the_format_term_grades():
    exact = _verdict('{"answer": "A"}').format_score
    wrapped = _verdict('Sure! {"answer": "A"}').format_score

    assert wrapped < exact
    assert wrapped > 0.0, "a recoverable verdict still earns partial format credit"


def test_the_other_styles_are_untouched():
    """Additive: rating_only keeps its own 0.5/0.3/0.2 split."""
    rating = parse_judge_verdict(
        f'{THINK}{{"rating": 6}}',
        thinking_enabled=True,
        prompt_style=PROMPT_STYLE_RATING_ONLY,
    )

    assert rating.rating == 6
    assert rating.format_score == pytest.approx(1.0)


# --- the prompt ------------------------------------------------------------------------------


def test_the_prompt_builds_on_the_shared_header():
    assert TURING_LETTER_ONLY_PROMPT.startswith(TURING_PROMPT_HEADER)


def test_the_prompt_asks_for_the_answer_field():
    assert '{{"answer": "<A or B>"}}' in TURING_LETTER_ONLY_PROMPT


def test_the_prompt_carries_no_rating_anchors():
    """A leftover 1-7 scale would invite a rating the parser then scores as unrecovered."""
    assert "Rating Scale" not in TURING_LETTER_ONLY_PROMPT
    assert "Cannot tell" not in TURING_LETTER_ONLY_PROMPT
    assert "integer from 1 to 7" not in TURING_LETTER_ONLY_PROMPT


def test_the_prompt_defines_what_the_letters_mean():
    assert "A means Response A was written by the real [HUMAN]." in TURING_LETTER_ONLY_PROMPT


# --- generator-serving side --------------------------------------------------------------------


def _serve(monkeypatch, payload: str) -> dict:
    import training.grpo.reward as R

    async def fake_post(session, body, **kwargs):
        return payload

    monkeypatch.setattr(R, "post_chat_async", fake_post)
    monkeypatch.setenv("JUDGE_PROMPT_STYLE", PROMPT_STYLE_LETTER_ONLY)
    monkeypatch.setenv("PERSONA_ENABLE_THINKING", "1")
    return asyncio.run(
        R._score_pairwise_likert_with_info(
            None, "key", "generated turn", "human turn", "hist", "ctx",
            prompt_template=TURING_LETTER_ONLY_PROMPT,
        )
    )["judge_randomized"]


def test_the_served_judge_answer_is_scored(monkeypatch):
    got = _serve(monkeypatch, f'{THINK}{{"answer": "A"}}')

    assert got["parse_error"] is False
    assert got["rating"] == 1


@pytest.mark.parametrize("payload", ['{"answer": "maybe"}', "{}", '{"answer": 4}'])
def test_a_served_non_answer_fails_rather_than_tying(monkeypatch, payload):
    """Guards the 20cd6f8 fix: an empty body used to score a confident 4.0."""
    got = _serve(monkeypatch, f"{THINK}{payload}")

    assert got["parse_error"] is True, payload
    assert got["rating"] != 4, payload
