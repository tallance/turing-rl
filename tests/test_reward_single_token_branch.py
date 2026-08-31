"""compute_score's single-token branch: reward wiring, style gate, and the dump row.

The Likert arm multiplies a 1-7 rating through ``clip_turing_judge_score`` and ``(r-1)/6``.
The single-token arm must NOT: ``p_human`` is already a probability, so routing it through
that transform would rescale it silently. These tests pin that separation, since a later
tidy-up unifying the two paths is the obvious way to break it.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import training.grpo.reward as R  # noqa: E402


def _fake_scorer(p_human: float, *, hard_fail: bool = False, letter: str = "B"):
    async def scorer(session, api_key, response, ground_truth, user_history, context, **kw):
        return {
            "score": 7.0,
            "rating_gt_first": 7,
            "rating_gen_first": None,
            "rating_randomized": 7,
            "generated_is_b": True,
            "randomized_order": "gt_first",
            "judge_prompt": "PROMPT",
            "judge_raw_content": letter,
            "judge_latency_ms": 12,
            "judge_finish_reason": "length",
            "judge_usage": {},
            "judge_prompt_style": "single_token",
            "p_human": p_human,
            "p_a": 1.0 - p_human,
            "ab_mass": 0.99,
            "off_ab_mass": 0.01,
            "letter": letter,
            "human_is_b": False,
            "hard_fail": hard_fail,
            "hard_fail_reason": None,
            "sampled_token": letter,
            "sampled_token_is_ab": True,
            "enable_thinking": False,
        }
    return scorer


def _run(monkeypatch, p_human=0.75, **env):
    monkeypatch.setenv("REWARD_METRIC", "turing")
    monkeypatch.setenv("JUDGE_PROMPT_STYLE", "single_token")
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(R, "score_turing_single_token_with_info", _fake_scorer(p_human))
    monkeypatch.setattr(R, "_get_session", lambda: None)
    monkeypatch.setattr(R, "resolve_judge_api_key", lambda: "key")
    return asyncio.run(R.compute_score("ds", "a generated turn", "the human turn", {}))


def test_reward_is_p_human_not_the_likert_transform(monkeypatch):
    got = _run(monkeypatch, p_human=0.75)
    # unadjusted is p_human itself; adjusted applies only the raw-reward scale.
    assert got["unadjusted_raw_reward"] == pytest.approx(0.75)
    assert got["adjusted_raw_reward"] == pytest.approx(0.75 * R.TURING_RAW_REWARD_SCALE)
    assert got["p_human"] == pytest.approx(0.75)


def test_the_likert_clip_does_not_move_a_single_token_reward(monkeypatch):
    """CLIP_MAX rescales the full arm. It must not touch p_human."""
    low = _run(monkeypatch, p_human=0.75, TURING_JUDGE_SCORE_CLIP_MAX="5")
    high = _run(monkeypatch, p_human=0.75, TURING_JUDGE_SCORE_CLIP_MAX="7")
    assert low["unadjusted_raw_reward"] == high["unadjusted_raw_reward"] == pytest.approx(0.75)
    # ...while the recorded Likert column still honours the clip, for dump continuity.
    assert low["turing_judge_score_clipped"] == 5.0
    assert high["turing_judge_score_clipped"] == 7.0


def test_metrics_for_the_new_columns_are_emitted(monkeypatch):
    got = _run(monkeypatch, p_human=0.75)
    assert got["hard_fail"] == 0.0
    assert got["letter_is_a"] == 0.0          # fake scorer returns "B"
    assert {"p_human", "hard_fail", "letter_is_a"} <= set(got)


def test_full_schema_result_gains_no_single_token_columns(monkeypatch):
    """The default arm's metric set must be unchanged, not padded with constant zeros."""
    monkeypatch.setenv("REWARD_METRIC", "turing")
    monkeypatch.delenv("JUDGE_PROMPT_STYLE", raising=False)

    async def full_scorer(*a, **k):
        return {"score": 7.0, "source_copy": False, "assistant_like": False,
                "wrong_target_or_role": False, "unsupported_adversarial_reframing": False}

    monkeypatch.setattr(R, "score_turing_with_info", full_scorer)
    monkeypatch.setattr(R, "_get_session", lambda: None)
    monkeypatch.setattr(R, "resolve_judge_api_key", lambda: "key")
    got = asyncio.run(R.compute_score("ds", "a generated turn", "the human turn", {}))
    assert "p_human" not in got and "hard_fail" not in got and "letter_is_a" not in got


def test_unknown_prompt_style_is_rejected(monkeypatch):
    monkeypatch.setenv("JUDGE_PROMPT_STYLE", "singletoken")
    with pytest.raises(ValueError, match="JUDGE_PROMPT_STYLE"):
        R.resolve_judge_prompt_style()


def test_style_defaults_to_full(monkeypatch):
    monkeypatch.delenv("JUDGE_PROMPT_STYLE", raising=False)
    assert R.resolve_judge_prompt_style() == R.PROMPT_STYLE_FULL


# --- dump row -------------------------------------------------------------------------

def test_single_token_row_carries_the_extras_and_the_base_keys():
    row = R._build_reward_dump_row(
        judge_prompt_style="single_token", p_human=0.75, p_a=0.25, letter="B",
        hard_fail=False, generated_is_b=True,
    )
    assert set(R._REWARD_DUMP_KEYS) <= set(row)
    for key in R._SINGLE_TOKEN_DUMP_KEYS:
        assert key in row
    assert row["p_human"] == 0.75 and row["letter"] == "B"


def test_full_schema_row_shape_is_untouched():
    row = R._build_reward_dump_row(generated_is_b=True, rating_gt_first=7)
    assert set(row) == set(R._REWARD_DUMP_KEYS)
    assert "p_human" not in row


def test_base_keys_stay_equal_to_the_eval_arms_pinned_copy():
    """eval/single_token_judge.py holds a hand copy of _REWARD_DUMP_KEYS. The single-token
    extras are kept OUT of that tuple so the two arms' base contract stays identical."""
    from eval.single_token_judge import _BASE_DUMP_KEYS

    assert _BASE_DUMP_KEYS == R._REWARD_DUMP_KEYS
