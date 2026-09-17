"""A judge body that parses but carries no verdict must fail, not score a tie.

Every _coerce_* in the Likert scorer defaults to 0.0, so an object with none of the expected
fields gives base_score_a == base_score_b == 0, score_gap == 0, and
_rating_from_turing_score_gap(0.0) == 4 -- a confident "cannot tell" with parse_error unset.

Measured before the fix:

    {}                      rating=4  parse_error=False  score=4.0
    {"foo": 1, "bar": "x"}  rating=4  parse_error=False  score=4.0
    not JSON at all         rating=0  parse_error=True   score=0.0

Non-JSON was caught; parseable-but-empty was not. In generator RL that is a mid-scale 4.0
handed out for a judge response containing no judgement, where a failure earns 0.0 -- so a
judge degenerating into empty objects would pay the generator instead of tripping the error
path.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import training.grpo.reward as R  # noqa: E402
from shared.judge_prompts import TURING_PROMPT  # noqa: E402


def _score(monkeypatch, payload: str) -> dict:
    async def fake_post(session, body, **kwargs):
        return payload

    monkeypatch.setattr(R, "post_chat_async", fake_post)
    return asyncio.run(
        R._score_pairwise_likert_with_info(
            None, "key", "generated turn", "human turn", "hist", "ctx",
            prompt_template=TURING_PROMPT,
        )
    )


def _verdict(out: dict) -> dict:
    return out["judge_randomized"]


@pytest.mark.parametrize("payload", ['{}', '{"foo": 1, "bar": "x"}', '{"reasoning": "hmm"}'])
def test_a_body_with_no_verdict_is_a_parse_failure(monkeypatch, payload):
    got = _verdict(_score(monkeypatch, payload))

    assert got["parse_error"] is True, payload
    assert got["rating"] != 4, f"{payload} scored a confident tie"


def test_the_tie_rating_is_still_reachable_when_the_judge_means_it(monkeypatch):
    """4 must stay available as a real verdict -- the fix is about absence, not about 4."""
    got = _verdict(_score(monkeypatch, '{"rating": 4}'))

    assert got["rating"] == 4
    assert got["parse_error"] is False


def test_an_explicit_rating_alone_is_a_valid_body(monkeypatch):
    """This is the rating_only judge's entire output."""
    got = _verdict(_score(monkeypatch, '{"rating": 7}'))

    assert got["rating"] == 7
    assert got["parse_error"] is False


def test_a_score_gap_alone_is_also_a_failure_here(monkeypatch):
    """Unlike judge_verdict.py, this path has no score_gap rung: it always RECOMPUTES the gap
    from the dimension fields and never reads data["score_gap"]. So a score-gap-only body was
    scoring 4.0 as well, for the same reason. Failing is the honest outcome -- adding a
    score_gap rung here would be a behaviour change, not a bug fix."""
    got = _verdict(_score(monkeypatch, '{"score_gap": 2.5}'))

    assert got["parse_error"] is True


def test_dimension_fields_alone_are_a_valid_body(monkeypatch):
    """Scores present but no stated rating: the rating is derived from them."""
    body = ('{"immediate_target_score_a": 0.1, "human_goal_score_a": 0.1,'
            ' "communication_style_score_a": 0.1, "immediate_target_score_b": 0.9,'
            ' "human_goal_score_b": 0.9, "communication_style_score_b": 0.9}')
    got = _verdict(_score(monkeypatch, body))

    assert got["parse_error"] is False
    assert got["rating"] > 4, "B scores far higher, so the verdict should favour B"


def test_non_json_still_fails(monkeypatch):
    """Regression: the path that already worked must keep working."""
    got = _verdict(_score(monkeypatch, "the judge rambled and said nothing"))

    assert got["parse_error"] is True
