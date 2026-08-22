# tests/test_reward_extract_json.py
import json

import training.grpo.reward as R


def test_huge_integer_literal_does_not_escape_as_an_exception():
    # Job 18916 died at step 7 of 120 because the Qwen3.5-0.8B judge emitted a 7726-digit
    # number. json.loads raises a PLAIN ValueError past Python 3.11's 4300-digit int/str
    # limit -- not a JSONDecodeError -- so it escaped _extract_json, propagated out of the
    # agent loop, and killed the whole run over one bad verdict.
    payload = '{"rating": ' + "9" * 7726 + "}"

    # guard the premise: this really does raise, and really is not a JSONDecodeError
    try:
        json.loads(payload)
        raised = None
    except ValueError as exc:
        raised = exc
    assert raised is not None, "premise broken: huge int no longer raises"
    assert not isinstance(raised, json.JSONDecodeError)

    assert R._extract_json(payload) is None


def test_ordinary_malformed_json_still_returns_none():
    # JSONDecodeError subclasses ValueError, so widening the handler must not have
    # narrowed the original behaviour.
    assert R._extract_json("{not json at all") is None
    assert R._extract_json("") is None
    assert R._extract_json(None) is None


def test_valid_verdict_still_parses():
    assert R._extract_json('{"rating": 5}') == {"rating": 5}
    assert R._extract_json('```json\n{"rating": 6}\n```') == {"rating": 6}
    # prose on both sides of the object, which is the common judge output shape
    assert R._extract_json('here you go {"rating": 7} hope that helps') == {"rating": 7}
