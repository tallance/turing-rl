"""The rating_only style on the GENERATOR's reward path.

rating_only is the full arm with a smaller prompt: the judge reasons and answers
``{"rating": N}``, which the Likert scorer already understands. So the whole change is which
template is sent, and these tests pin exactly that -- plus the two things a future tidy-up is
most likely to break: that the single-token arm is NOT reached, and that an unknown style
still raises instead of quietly scoring with the 37-field judge.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import training.grpo.reward as R  # noqa: E402
from shared.judge_prompts import (  # noqa: E402
    TURING_PROMPT,
    TURING_RATING_ONLY_PROMPT,
)


def _capture_template(monkeypatch):
    """Run score_turing_with_info against a stub and return the template it passed down."""
    seen: dict[str, str] = {}

    async def fake_pairwise(session, api_key, response, ground_truth, user_history, context,
                            *, prompt_template, **kwargs):
        seen["template"] = prompt_template
        return {"score": 7.0, "source_copy": False, "assistant_like": False,
                "wrong_target_or_role": False, "unsupported_adversarial_reframing": False}

    monkeypatch.setattr(R, "_score_pairwise_likert_with_info", fake_pairwise)
    asyncio.run(R.score_turing_with_info(None, "key", "gen", "human", "hist", "ctx"))
    return seen["template"]


def test_rating_only_sends_the_rating_template(monkeypatch):
    monkeypatch.setenv("JUDGE_PROMPT_STYLE", "rating_only")
    assert _capture_template(monkeypatch) is TURING_RATING_ONLY_PROMPT


def test_full_still_sends_the_full_template(monkeypatch):
    """The default arm must be untouched: it is the reference cell for every comparison."""
    monkeypatch.delenv("JUDGE_PROMPT_STYLE", raising=False)
    assert _capture_template(monkeypatch) is TURING_PROMPT


def test_rating_only_is_an_accepted_style(monkeypatch):
    monkeypatch.setenv("JUDGE_PROMPT_STYLE", "rating_only")
    assert R.resolve_judge_prompt_style() == R.PROMPT_STYLE_RATING_ONLY


def test_a_typo_near_the_new_style_still_raises(monkeypatch):
    """The loud-failure property is the point of resolve_judge_prompt_style: a style that
    silently fell back to "full" would produce a complete, healthy-looking run of the wrong
    experiment."""
    for typo in ("rating-only", "ratingonly", "rating"):
        monkeypatch.setenv("JUDGE_PROMPT_STYLE", typo)
        with pytest.raises(ValueError, match="JUDGE_PROMPT_STYLE"):
            R.resolve_judge_prompt_style()


def test_rating_only_does_not_reach_the_single_token_arm(monkeypatch):
    """It scores through the Likert path. Routing it to the single-token scorer would ask a
    thinking judge for one token and read a verdict out of the think opener."""
    monkeypatch.setenv("REWARD_METRIC", "turing")
    monkeypatch.setenv("JUDGE_PROMPT_STYLE", "rating_only")

    async def boom(*args, **kwargs):
        raise AssertionError("single-token scorer must not be called for rating_only")

    async def full_scorer(*args, **kwargs):
        return {"score": 7.0, "source_copy": False, "assistant_like": False,
                "wrong_target_or_role": False, "unsupported_adversarial_reframing": False}

    monkeypatch.setattr(R, "score_turing_single_token_with_info", boom)
    monkeypatch.setattr(R, "score_turing_with_info", full_scorer)
    monkeypatch.setattr(R, "_get_session", lambda: None)
    monkeypatch.setattr(R, "resolve_judge_api_key", lambda: "key")

    got = asyncio.run(R.compute_score("ds", "a generated turn", "the human turn", {}))

    # ...and it gains no single-token columns, exactly like the full arm.
    assert "p_human" not in got and "letter_is_a" not in got


def test_the_rating_template_map_covers_every_style_that_reaches_it():
    """single_token is absent on purpose (it is dispatched earlier). If a style is ever added
    without a template, this fails here rather than with a KeyError mid-run."""
    dispatched_elsewhere = {R.PROMPT_STYLE_SINGLE_TOKEN}
    expected = set(R.PROMPT_STYLES) - dispatched_elsewhere

    assert set(R._JUDGE_PROMPT_TEMPLATES) == expected
