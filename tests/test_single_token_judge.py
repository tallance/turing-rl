"""The standalone single-token scorer: request shape, dump contract, hard fails.

The dump-contract tests deliberately run the emitted rows through the REAL consumer
(``scripts.analyze_judge_sweep.per_call_features``) instead of asserting on key names.
The whole point of this module is that a single-token cell must be comparable against a
published full-schema number, and only the consumer can testify to that.
"""

from __future__ import annotations

import ast
import asyncio
import json
import math
from pathlib import Path

import pytest

from eval import single_token_judge as stj
from scripts.analyze_judge_sweep import per_call_features
from shared.judge_utils import _stable_turing_generated_is_b, sanitize_prompt_text
from shared.single_token_verdict import MIN_AB_MASS

ROOT = Path(__file__).resolve().parents[1]

_PAIR = dict(
    response="generated turn",
    ground_truth="the real human turn",
    user_history="[HUMAN]: past turn",
    context="[OTHER]: hello",
)


def _choice(pairs, sampled=None):
    """A minimal OpenAI choice carrying one decoded position's logprobs."""
    position = {
        "token": sampled,
        "top_logprobs": [{"token": t, "logprob": lp} for t, lp in pairs],
    }
    return {"logprobs": {"content": [position]}}


def _run(monkeypatch, choice, *, env=None, **overrides):
    """Score one pair against a canned choice; returns (row, captured_payload)."""
    seen: dict = {}

    async def fake_post(session, payload, *, semaphore, api_key=None, max_retries=None):
        seen.update(payload)
        if isinstance(choice, Exception):
            raise choice
        return choice

    monkeypatch.setenv("JUDGE_MODEL", "Qwen/Qwen3.5-9B")
    monkeypatch.setenv("PERSONA_DISABLE_OPENROUTER_EXTRAS", "1")
    for key, value in (env or {}).items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(stj, "post_chat_choice_async", fake_post)

    kwargs = {**_PAIR, **overrides}
    row = asyncio.run(
        stj.score_single_token_with_info(object(), "EMPTY", **kwargs)
    )
    return row, seen


# --------------------------------------------------------------------------- request


def test_request_pins_thinking_off_and_asks_for_logprobs(monkeypatch):
    _, payload = _run(monkeypatch, _choice([("A", -0.1), ("B", -2.0)], sampled="A"))

    assert payload["max_completion_tokens"] == 1
    assert payload["logprobs"] is True
    assert payload["top_logprobs"] == 20
    assert "response_format" not in payload
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    # The single-token tail, not the 37-field one.
    assert "## Criteria" not in payload["messages"][0]["content"]
    assert "single letter" in payload["messages"][0]["content"]


def test_thinking_stays_off_even_when_the_cell_env_asks_for_it(monkeypatch):
    """With a one-token budget, thinking-on spends that token on the think opener and
    the verdict never exists -- silently. The pin must not be overridable by env."""
    _, payload = _run(
        monkeypatch,
        _choice([("A", -0.1), ("B", -2.0)], sampled="A"),
        env={"PERSONA_JUDGE_ENABLE_THINKING": "1"},
    )

    assert payload["chat_template_kwargs"] == {"enable_thinking": False}


def test_missing_judge_model_is_refused(monkeypatch):
    monkeypatch.delenv("JUDGE_MODEL", raising=False)
    monkeypatch.setattr(stj, "post_chat_choice_async", None)

    with pytest.raises(ValueError, match="JUDGE_MODEL"):
        asyncio.run(stj.score_single_token_with_info(object(), "EMPTY", **_PAIR))


def test_the_watchlist_placeholder_is_rendered(monkeypatch):
    """A stray '{source_copy_watchlist}' in the prompt would change what the judge sees
    relative to the full arm and confound the comparison."""
    _, payload = _run(monkeypatch, _choice([("A", -0.1), ("B", -2.0)], sampled="A"))

    assert "{" not in payload["messages"][0]["content"].replace("{{", "").replace("}}", "")


# ------------------------------------------------------------------- verdict / rating


def test_a_verdict_maps_to_rating_one_and_b_to_seven(monkeypatch):
    a_row, _ = _run(monkeypatch, _choice([("A", -0.1), ("B", -2.0)], sampled="A"))
    b_row, _ = _run(monkeypatch, _choice([("A", -2.0), ("B", -0.1)], sampled="B"))

    assert a_row["letter"] == "A" and a_row["p_a"] > 0.5
    assert b_row["letter"] == "B" and b_row["p_a"] < 0.5
    ratings = {
        r["rating_gt_first"] if r["generated_is_b"] else r["rating_gen_first"]
        for r in (a_row, b_row)
    }
    assert ratings == {stj.RATING_FOR_A, stj.RATING_FOR_B}


def test_exactly_one_rating_slot_is_populated(monkeypatch):
    """The analyzer reads rating_gt_first and falls back to rating_gen_first. Both set
    would make the recorded ordering unrecoverable."""
    row, _ = _run(monkeypatch, _choice([("A", -0.1), ("B", -2.0)], sampled="A"))

    populated = [k for k in ("rating_gt_first", "rating_gen_first") if row[k] is not None]
    assert len(populated) == 1
    assert populated[0] == ("rating_gt_first" if row["generated_is_b"] else "rating_gen_first")


def test_ab_mass_and_p_a_are_recorded(monkeypatch):
    row, _ = _run(monkeypatch, _choice([("A", -0.1), ("B", -2.0)], sampled="A"))

    assert row["ab_mass"] > MIN_AB_MASS
    assert row["off_ab_mass"] == pytest.approx(1.0 - row["ab_mass"])
    assert row["sampled_token"] == "A"
    assert row["judge_prompt_style"] == "single_token"
    assert row["enable_thinking"] is False


# ------------------------------------------------------------------ analyzer contract


def test_a_correct_verdict_reads_as_correct_through_the_real_analyzer(monkeypatch):
    """End-to-end on the contract: the judge names the human's slot, and
    per_call_features -- the function that computes the published accuracy -- must
    score it 1."""
    row, _ = _run(monkeypatch, _choice([("A", -0.1), ("B", -2.0)], sampled="A"))
    # human sits on A exactly when the generated answer is B.
    if not row["generated_is_b"]:
        row, _ = _run(
            monkeypatch,
            _choice([("A", -2.0), ("B", -0.1)], sampled="B"),
        )

    features = per_call_features(row)

    assert features["picked_human"] == 1
    assert features["parse_error"] is False
    assert features["human_side"] in ("A", "B")
    assert features["randomized_order"] in ("gt_first", "gen_first")


def test_a_wrong_verdict_reads_as_wrong_through_the_real_analyzer(monkeypatch):
    row, _ = _run(monkeypatch, _choice([("A", -0.1), ("B", -2.0)], sampled="A"))
    if row["generated_is_b"]:
        row, _ = _run(monkeypatch, _choice([("A", -2.0), ("B", -0.1)], sampled="B"))

    assert per_call_features(row)["picked_human"] == 0


def test_dump_row_lands_in_the_reward_directory_the_analyzer_globs(monkeypatch, tmp_path):
    monkeypatch.setenv("PERSONA_JUDGE_DUMP_RATE", "1.0")
    monkeypatch.setenv("PERSONA_REWARD_DUMP_DIR", str(tmp_path / "reward"))

    _run(monkeypatch, _choice([("A", -0.1), ("B", -2.0)], sampled="A"))

    files = sorted((tmp_path / "reward").glob("*.jsonl"))
    assert len(files) == 1
    written = [json.loads(line) for line in files[0].read_text().splitlines() if line]
    assert len(written) == 1
    assert per_call_features(written[0])["picked_human"] in (0, 1)


def test_dumping_is_off_without_the_rate_env(monkeypatch, tmp_path):
    monkeypatch.delenv("PERSONA_JUDGE_DUMP_RATE", raising=False)
    monkeypatch.setenv("PERSONA_REWARD_DUMP_DIR", str(tmp_path / "reward"))

    _run(monkeypatch, _choice([("A", -0.1), ("B", -2.0)], sampled="A"))

    assert not (tmp_path / "reward").exists()


def test_base_dump_keys_match_the_reward_dump_contract():
    """``_BASE_DUMP_KEYS`` is a hand copy of reward.py's list (it cannot be imported
    without importing the reward scoring path). Pin it so the copy cannot drift."""
    from training.grpo.reward import _REWARD_DUMP_KEYS

    assert stj._BASE_DUMP_KEYS == _REWARD_DUMP_KEYS


def test_the_module_does_not_import_the_reward_path():
    """The eval protocol must not be coupled to the generator's reward module."""
    tree = ast.parse((ROOT / "eval" / "single_token_judge.py").read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)

    assert not [name for name in imported if name.split(".")[0] == "training"]


# ------------------------------------------------------------------ order randomization


def test_order_randomization_matches_the_full_arm(monkeypatch):
    """Same function, same seed material, and hashed AFTER sanitization -- reward.py
    sanitizes first too. A different rule in either arm confounds the comparison."""
    raw = "generated turn 0\x07 with a control char"
    clean = sanitize_prompt_text(raw)
    seed = dict(user_id="u", post_id="p", target_idx=0)
    # The fixture is only discriminating while these two disagree; assert that up front
    # so the test cannot quietly stop testing anything.
    assert _stable_turing_generated_is_b(clean, **seed) != _stable_turing_generated_is_b(raw, **seed)

    row, _ = _run(
        monkeypatch,
        _choice([("A", -0.1), ("B", -2.0)], sampled="A"),
        response=raw,
        **seed,
    )

    assert row["generated_is_b"] == _stable_turing_generated_is_b(clean, **seed)


@pytest.mark.parametrize("target_idx,expect_generated_is_b", [(0, False), (4, True)])
def test_both_orderings_occur_and_agree_with_human_side(
    monkeypatch, target_idx, expect_generated_is_b
):
    row, payload = _run(
        monkeypatch,
        _choice([("A", -0.1), ("B", -2.0)], sampled="A"),
        response=f"gen{target_idx}",
        user_id="u",
        post_id="p",
        target_idx=target_idx,
    )

    assert row["generated_is_b"] is expect_generated_is_b
    assert row["human_side"] == ("A" if expect_generated_is_b else "B")
    assert row["human_is_b"] is not expect_generated_is_b
    assert row["randomized_order"] == ("gt_first" if expect_generated_is_b else "gen_first")
    # The prompt must actually place the responses the way the row claims.
    prompt = payload["messages"][0]["content"]
    human_first = prompt.index(_PAIR["ground_truth"]) < prompt.index(f"gen{target_idx}")
    assert human_first is expect_generated_is_b


# ------------------------------------------------------------------------- hard fails


def test_hard_fail_is_recorded_not_raised(monkeypatch):
    """Hard fails are a headline number of this protocol and they concentrate on the
    longest inputs. Raising would fail the whole shard AND delete the evidence.

    A stray " a" at 1e-9 renormalizes to a *certain* A if the floor is not applied, so
    this fixture also pins MIN_AB_MASS being honoured on the eval path.
    """
    row, _ = _run(
        monkeypatch,
        _choice(
            [("<think>", math.log(0.60)), ("Answer", math.log(0.399)), (" a", math.log(1e-9))],
            sampled=" a",
        ),
    )

    assert row["hard_fail"] is True
    assert row["letter"] is None and row["p_a"] is None
    assert row["rating_gt_first"] is None and row["rating_gen_first"] is None
    assert "below the" in row["hard_fail_reason"]


def test_a_hard_failed_row_abstains_in_the_analyzer_rather_than_counting_as_wrong(monkeypatch):
    row, _ = _run(monkeypatch, _choice([("Neither", -0.1)], sampled="Neither"))

    assert per_call_features(row)["picked_human"] is None


def test_the_sampled_token_is_threaded_into_the_structural_check(monkeypatch):
    """The top-k below is a clean, high-mass A. Only the sampled token reveals that the
    model was emitting a think tag and this is not a verdict position at all."""
    row, _ = _run(monkeypatch, _choice([("A", -0.1), ("B", -2.0)], sampled="<think>"))

    assert row["hard_fail"] is True
    assert "not an A/B verdict" in row["hard_fail_reason"]


def test_the_verdict_is_not_retried(monkeypatch):
    calls = []

    async def fake_post(session, payload, *, semaphore, api_key=None, max_retries=None):
        calls.append(payload)
        return _choice([("Neither", -0.1)], sampled="Neither")

    monkeypatch.setenv("JUDGE_MODEL", "Qwen/Qwen3.5-9B")
    monkeypatch.setattr(stj, "post_chat_choice_async", fake_post)
    asyncio.run(stj.score_single_token_with_info(object(), "EMPTY", **_PAIR))

    assert len(calls) == 1


def test_a_response_without_logprobs_fails_the_shard(monkeypatch):
    """Distinct from a hard fail: no logprobs at all is a server/request misconfiguration
    that would otherwise produce a 100%-hard-fail cell that still exits 0."""
    with pytest.raises(RuntimeError, match="no logprobs"):
        _run(monkeypatch, {"message": {"content": "A"}})
