"""JUDGE_PROMPT_STYLE selects the judge protocol; unset must change nothing."""

from unittest.mock import patch

import pytest

from eval import metrics

_ARGS = dict(
    context="[OTHER]: hello",
    response_a="candidate a",
    response_b="candidate b",
    user_history="[HUMAN]: past turn",
)


def _choice(pairs, sampled=None):
    """A minimal OpenAI choice carrying one position's top_logprobs.

    ``sampled`` is the token the model actually emitted. Left out by default so the
    fixtures also cover the transport that does not return it.
    """
    position = {"top_logprobs": [{"token": t, "logprob": lp} for t, lp in pairs]}
    if sampled is not None:
        position["token"] = sampled
    return {"logprobs": {"content": [position]}}


def _capture(monkeypatch, env, *, text_reply=None, choice_reply=None):
    """Run one scoring call, returning the kwargs the HTTP layer was handed.

    Both transports are patched because the two paths use different ones: the
    full-schema path needs response text, the single-token path needs the choice
    object so it can read logprobs.
    """
    seen = {}

    def fake_text(kwargs):
        seen.update(kwargs)
        return text_reply

    def fake_choice(kwargs):
        seen.update(kwargs)
        return choice_reply

    for k, v in env.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(metrics, "post_chat_sync", fake_text)
    monkeypatch.setattr(metrics, "post_chat_choice_sync", fake_choice, raising=False)
    return seen


def test_default_path_is_unchanged(monkeypatch):
    seen = _capture(monkeypatch, {}, text_reply='{"rating": 5, "score_gap": 0.5}')
    metrics._turing_api_call(**_ARGS, max_tokens=2048)
    assert seen["response_format"] == {"type": "json_object"}
    assert seen["max_completion_tokens"] == 2048
    assert "logprobs" not in seen
    assert "## Criteria" in seen["messages"][0]["content"]


def test_single_token_request_shape(monkeypatch):
    seen = _capture(monkeypatch, {"JUDGE_PROMPT_STYLE": "single_token"},
                    choice_reply=_choice([("A", -0.1), ("B", -2.0)]))
    metrics._turing_api_call(**_ARGS, return_details=True)
    assert seen["max_completion_tokens"] == 1
    assert "response_format" not in seen
    assert seen["logprobs"] is True
    assert seen["top_logprobs"] == 20
    assert "## Criteria" not in seen["messages"][0]["content"]


def test_single_token_result_carries_letter_and_p_a(monkeypatch):
    _capture(monkeypatch, {"JUDGE_PROMPT_STYLE": "single_token"},
             choice_reply=_choice([("A", -0.1), ("B", -2.0)]))
    out = metrics._turing_api_call(**_ARGS, return_details=True)
    assert out["letter"] == "A"
    assert out["p_a"] > 0.5
    assert out["rating"] == 1          # 1 == "definitely A" on the existing scale
    assert out["parse_error"] is None
    # ab_mass, not a top-k residual: see Verdict.off_ab_mass for why.
    assert out["ab_mass"] > 0.01
    assert out["off_ab_mass"] == pytest.approx(1.0 - out["ab_mass"])
    assert "residual_mass" not in out


def test_scorer_threads_the_sampled_token_into_the_structural_check(monkeypatch):
    """The scorer must forward choice[...]["token"], not just the top_logprobs.

    The top-k below is a clean, high-mass A. Only the sampled token reveals that the
    model was emitting a think tag and this is not a verdict position at all.
    """
    from shared.single_token_verdict import HardFail

    _capture(monkeypatch, {"JUDGE_PROMPT_STYLE": "single_token"},
             choice_reply=_choice([("A", -0.1), ("B", -2.0)], sampled="<think>"))
    with pytest.raises(HardFail, match="not an A/B verdict"):
        metrics._turing_api_call(**_ARGS, return_details=True)


def test_scorer_accepts_a_sampled_verdict_token(monkeypatch):
    _capture(monkeypatch, {"JUDGE_PROMPT_STYLE": "single_token"},
             choice_reply=_choice([("A", -0.1), ("B", -2.0)], sampled="A"))
    out = metrics._turing_api_call(**_ARGS, return_details=True)
    assert out["letter"] == "A"


def test_scorer_hard_fails_below_the_mass_floor(monkeypatch):
    """A stray " a" at 1e-9 must not be scored as a certain A."""
    import math

    from shared.single_token_verdict import HardFail

    _capture(monkeypatch, {"JUDGE_PROMPT_STYLE": "single_token"},
             choice_reply=_choice([("<think>", math.log(0.60)),
                                   ("Answer", math.log(0.399)),
                                   (" a", math.log(1e-9))]))
    with pytest.raises(HardFail):
        metrics._turing_api_call(**_ARGS, return_details=True)


def test_single_token_maps_b_to_rating_seven(monkeypatch):
    _capture(monkeypatch, {"JUDGE_PROMPT_STYLE": "single_token"},
             choice_reply=_choice([("A", -2.0), ("B", -0.1)]))
    out = metrics._turing_api_call(**_ARGS, return_details=True)
    assert out["letter"] == "B"
    assert out["rating"] == 7


def test_hard_fail_propagates_and_is_not_retried(monkeypatch):
    """A hard fail is a property of the input, not a transient. Retrying would hide it
    from the hard_fail column and bias the accuracy toward the shorter inputs."""
    from shared.single_token_verdict import HardFail

    calls = []

    def fake_choice(kwargs):
        calls.append(kwargs)
        return _choice([("Neither", -0.1)])

    monkeypatch.setenv("JUDGE_PROMPT_STYLE", "single_token")
    monkeypatch.setattr(metrics, "post_chat_choice_sync", fake_choice, raising=False)
    with pytest.raises(HardFail):
        metrics._turing_api_call(**_ARGS, return_details=True)
    assert len(calls) == 1


def test_unknown_style_is_rejected(monkeypatch):
    monkeypatch.setenv("JUDGE_PROMPT_STYLE", "freeform")
    with pytest.raises(ValueError, match="full|single_token"):
        metrics._turing_api_call(**_ARGS)
