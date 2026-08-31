"""The single-token judge as a GENERATOR REWARD.

The load-bearing thing here is orientation. ``p_a`` is the judge's probability that the
HUMAN is response A, but the reward is the probability the GENERATED turn looks human, and
which side the generated turn sits on flips per pair. Getting that backwards produces a
run that trains perfectly well straight into the wrong optimum, with no error anywhere.

Mirrors the concerns tests/test_eval_parity.py pins for the Likert protocol.
"""
from __future__ import annotations

import ast
import asyncio
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.grpo import single_token_reward as stf  # noqa: E402


def _logprobs(p_a: float, p_b: float, other: dict[str, float] | None = None) -> list[dict]:
    entries = [{"token": "A", "logprob": math.log(p_a)}, {"token": "B", "logprob": math.log(p_b)}]
    for token, prob in (other or {}).items():
        entries.append({"token": token, "logprob": math.log(prob)})
    return entries


class _FakeChoice:
    """One captured judge call: records the payload, returns a canned distribution."""

    def __init__(self, entries: list[dict], token: str = "A") -> None:
        self.entries = entries
        self.token = token
        self.payloads: list[dict] = []
        self.calls = 0

    async def __call__(self, session, payload, semaphore=None, api_key=None):
        self.calls += 1
        self.payloads.append(payload)
        return {"logprobs": {"content": [{"token": self.token, "top_logprobs": self.entries}]}}


def _score(monkeypatch, fake, *, response="gen turn", ground_truth="human turn", **kw):
    monkeypatch.setattr(stf, "post_chat_choice_async", fake)
    monkeypatch.setattr(stf, "get_judge_call_meta", lambda: {})
    return asyncio.run(
        stf.score_turing_single_token_with_info(
            None, "key", response, ground_truth, "history", "context", **kw
        )
    )


def _force_side(monkeypatch, generated_is_b: bool) -> None:
    """Pin which side the generated turn lands on, so orientation is testable directly."""
    monkeypatch.setattr(stf, "_stable_turing_generated_is_b", lambda *a, **k: generated_is_b)


# --- orientation: the reward must follow the generated turn, not side A ------------------

def test_p_human_is_p_a_when_the_generated_turn_is_a(monkeypatch):
    _force_side(monkeypatch, False)          # generated on side A
    fake = _FakeChoice(_logprobs(0.8, 0.2))
    got = _score(monkeypatch, fake)
    assert got["p_a"] == pytest.approx(0.8)
    assert got["p_human"] == pytest.approx(0.8)   # judge says A is human; generated IS A
    assert got["randomized_order"] == "gen_first"
    assert got["rating_gen_first"] == stf.RATING_FOR_A and got["rating_gt_first"] is None


def test_p_human_is_one_minus_p_a_when_the_generated_turn_is_b(monkeypatch):
    _force_side(monkeypatch, True)           # generated on side B
    fake = _FakeChoice(_logprobs(0.8, 0.2))
    got = _score(monkeypatch, fake)
    assert got["p_a"] == pytest.approx(0.8)
    assert got["p_human"] == pytest.approx(0.2)   # judge says A is human; generated is B
    assert got["randomized_order"] == "gt_first"
    assert got["rating_gt_first"] == stf.RATING_FOR_A and got["rating_gen_first"] is None


def test_a_confident_wrong_judge_gives_the_generator_a_high_reward(monkeypatch):
    """The whole point: reward is high when the judge is FOOLED, not when it is right."""
    _force_side(monkeypatch, True)           # generated on side B
    fake = _FakeChoice(_logprobs(0.02, 0.98))  # judge is sure the human is B == the generated
    got = _score(monkeypatch, fake)
    assert got["p_human"] == pytest.approx(0.98)
    assert got["letter"] == "B"


# --- hard fails ---------------------------------------------------------------------------

def test_hard_fail_scores_neutral_and_never_raises(monkeypatch):
    _force_side(monkeypatch, True)
    # A/B carry ~0 mass: the judge answered something else entirely.
    fake = _FakeChoice(_logprobs(1e-9, 1e-9, {"Based": 0.6, "Looking": 0.3}), token="Based")
    got = _score(monkeypatch, fake)
    assert got["hard_fail"] is True
    assert got["p_human"] == stf.HARD_FAIL_P_HUMAN == 0.5
    assert got["letter"] is None and got["p_a"] is None
    assert got["rating_gt_first"] is None and got["rating_gen_first"] is None
    assert got["hard_fail_reason"] and "floor" in got["hard_fail_reason"]


def test_missing_top_logprobs_raises_rather_than_scoring(monkeypatch):
    """A misconfigured server must not read as a 100%-abstention run training on 0.5."""
    async def no_logprobs(session, payload, semaphore=None, api_key=None):
        return {"logprobs": {"content": [{"token": "A"}]}}

    _force_side(monkeypatch, True)
    with pytest.raises(RuntimeError, match="top_logprobs"):
        _score(monkeypatch, no_logprobs)


# --- request shape -------------------------------------------------------------------------

def test_thinking_is_off_even_when_the_env_asks_for_it(monkeypatch):
    """The generator run scripts export PERSONA_JUDGE_ENABLE_THINKING=1. With a one-token
    budget that spends the token on the think opener and no verdict ever exists."""
    monkeypatch.setenv("PERSONA_JUDGE_ENABLE_THINKING", "1")
    _force_side(monkeypatch, True)
    fake = _FakeChoice(_logprobs(0.7, 0.3))
    got = _score(monkeypatch, fake)
    payload = fake.payloads[0]
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert got["enable_thinking"] is False


def test_payload_asks_for_one_token_and_the_full_top_k(monkeypatch):
    _force_side(monkeypatch, True)
    fake = _FakeChoice(_logprobs(0.7, 0.3))
    _score(monkeypatch, fake)
    payload = fake.payloads[0]
    assert payload["max_completion_tokens"] == 1
    assert payload["logprobs"] is True
    assert payload["top_logprobs"] == stf.TOP_LOGPROBS == 20


def test_exactly_one_judge_call_no_parse_retry(monkeypatch):
    """The full-schema path retries on malformed JSON. If that loop is ever copied in here,
    the hard-fail rate stops measuring anything."""
    _force_side(monkeypatch, True)
    fake = _FakeChoice(_logprobs(1e-9, 1e-9, {"Based": 0.9}), token="Based")
    _score(monkeypatch, fake)
    assert fake.calls == 1


# --- drift guards against the eval arm -------------------------------------------------------

def test_agrees_with_the_eval_arm_on_the_pieces_that_must_not_diverge():
    from eval import single_token_judge as stj

    assert stf.TOP_LOGPROBS == stj.TOP_LOGPROBS
    assert stf.RATING_FOR_A == stj.RATING_FOR_A
    assert stf.RATING_FOR_B == stj.RATING_FOR_B
    # Same objects, not merely equal text/behaviour: a copy would let one arm be edited alone.
    assert stf.TURING_SINGLE_TOKEN_PROMPT is stj.TURING_SINGLE_TOKEN_PROMPT
    assert stf._stable_turing_generated_is_b is stj._stable_turing_generated_is_b
    assert stf.extract_verdict is stj.extract_verdict


def test_reward_module_does_not_import_the_eval_arm():
    """Training must not depend on eval; both share via shared/ instead."""
    tree = ast.parse((ROOT / "training" / "grpo" / "single_token_reward.py").read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert not {name for name in imported if name.startswith("eval")}


# --- the dump stays readable by the existing tooling -------------------------------------------

def test_emitted_ratings_flow_through_the_existing_accuracy_scorer(monkeypatch):
    """We emit 1/7 ratings so a single-token run's dumps stay analyzable by the same
    directional_accuracy the Likert runs use. 4 never appears, so nothing is ever a tie."""
    from scripts.eval_rl_generator import directional_accuracy

    rows = []
    for generated_is_b, entries in (
        (True, _logprobs(0.9, 0.1)),    # human=A, judge picks A -> judge correct
        (False, _logprobs(0.9, 0.1)),   # human=B, judge picks A -> judge wrong
    ):
        _force_side(monkeypatch, generated_is_b)
        rows.append(_score(monkeypatch, _FakeChoice(entries)))

    acc = directional_accuracy(rows)
    assert acc["n_nontie"] == 2 and acc["correct"] == 1
    assert acc["accuracy"] == 0.5
