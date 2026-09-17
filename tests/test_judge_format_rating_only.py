"""The rating_only format score.

The full-schema format term grades COMPLETENESS -- how far along the 37-field ordered schema a
verdict got. A rating_only answer has one field, so that axis is binary and measures nothing.
What does vary is PACKAGING: across job 23152, `fmt_strict_json` ranged 0.906-1.0 while
`judge_recovered` stayed 0.957-1.0, i.e. ~5% of rollouts produced a perfectly usable rating in
an untidy wrapper. These tests pin a score that grades that axis instead.

    0.5 * strict_json    the answer is nothing but a JSON object
    0.3 * exact_schema   its keys are exactly ("rating",)
    0.2 * rating_range   a rating key holding an integer 1-7 (tolerant parse)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.grpo.judge_verdict import (  # noqa: E402
    PROMPT_STYLE_FULL,
    PROMPT_STYLE_RATING_ONLY,
    PROMPT_STYLES,
    RATING_ONLY_FIELDS,
    parse_judge_verdict,
    resolve_prompt_style,
)

THINK = "some reasoning</think>"


def _rating_only(answer: str):
    return parse_judge_verdict(
        THINK + answer, thinking_enabled=True, prompt_style=PROMPT_STYLE_RATING_ONLY
    )


# --- the ladder ---------------------------------------------------------------------------


def test_a_bare_rating_object_scores_full_marks():
    verdict = _rating_only('{"rating": 6}')

    assert verdict.rating == 6
    assert verdict.format_score == pytest.approx(1.0)


def test_an_extra_field_loses_the_exact_schema_share():
    """The prompt asks for exactly one object and nothing else. A reasoning field alongside the
    rating is still clean JSON and still usable, so it keeps 0.7 rather than being failed."""
    verdict = _rating_only('{"reasoning": "A reads human", "rating": 6}')

    assert verdict.rating == 6
    assert verdict.format_score == pytest.approx(0.7)


def test_a_fenced_or_prose_wrapped_rating_keeps_only_the_tolerant_share():
    """This is the ~5% population. The verdict is recoverable, so the TASK reward is unaffected;
    only the tidiness term is docked."""
    for answer in ('```json\n{"rating": 6}\n```', 'My verdict: {"rating": 6}'):
        verdict = _rating_only(answer)
        assert verdict.rating == 6, answer
        assert verdict.format_score == pytest.approx(0.2), answer


def test_prose_with_no_json_object_scores_zero_format():
    verdict = _rating_only("I would say the answer is 6")

    assert verdict.format_score == pytest.approx(0.0)


def test_unclosed_thinking_scores_zero_format():
    verdict = parse_judge_verdict(
        'reasoning that never closes {"rating": 6}',
        thinking_enabled=True,
        prompt_style=PROMPT_STYLE_RATING_ONLY,
    )

    assert not verdict.recovered
    assert verdict.format_score == pytest.approx(0.0)


def test_an_out_of_range_rating_is_neither_recovered_nor_exact():
    verdict = _rating_only('{"rating": 9}')

    assert verdict.rating is None
    # 9 fails the declared 1-7 bound, so neither the schema nor the range share is earned.
    assert verdict.format_score == pytest.approx(0.5)


def test_format_score_never_exceeds_one():
    assert _rating_only('{"rating": 1}').format_score <= 1.0


# --- the full-schema arm must not move ------------------------------------------------------


def test_the_compact_shortcut_still_scores_one_tenth_under_the_full_schema():
    """Regression pin. The 37-field weights exist because both 2B runs collapsed onto a compact
    {"score_gap", "rating"} answer when format was a flat mean over booleans."""
    verdict = parse_judge_verdict(
        THINK + '{"score_gap": 1.0, "rating": 6}', thinking_enabled=True
    )

    assert verdict.format_score == pytest.approx(0.10)


def test_the_full_schema_is_the_default_style():
    """A caller that passes nothing must get the historical behaviour, not the new one."""
    verdict = parse_judge_verdict(THINK + '{"rating": 6}', thinking_enabled=True)

    # Under the 37-field schema this is the compact shortcut: strict JSON only.
    assert verdict.format_score == pytest.approx(0.10)


# --- style resolution -----------------------------------------------------------------------


def test_style_defaults_to_full(monkeypatch):
    monkeypatch.delenv("JUDGE_PROMPT_STYLE", raising=False)
    assert resolve_prompt_style() == PROMPT_STYLE_FULL


def test_style_reads_the_same_env_var_the_rest_of_the_pipeline_uses(monkeypatch):
    # The real run trains with thinking on (data.apply_chat_template_kwargs.enable_thinking),
    # and the driver propagates it here. Without it compute_score treats the whole completion
    # as the answer, so the </think> split never happens and even a perfect answer scores 0.2 --
    # a real failure mode, not a test artifact: a run that lost this env var would silently
    # train against a format term stuck near its floor.
    monkeypatch.setenv("PERSONA_ENABLE_THINKING", "1")
    monkeypatch.setenv("JUDGE_PROMPT_STYLE", "rating_only")
    assert resolve_prompt_style() == PROMPT_STYLE_RATING_ONLY


def test_an_unknown_style_raises_rather_than_falling_back(monkeypatch):
    """Falling back to "full" would score a rating_only run against the 37-field schema: every
    rollout pinned at 0.10, a dead format term, and a complete healthy-looking run."""
    monkeypatch.setenv("JUDGE_PROMPT_STYLE", "rating-only")
    with pytest.raises(ValueError, match="JUDGE_PROMPT_STYLE"):
        resolve_prompt_style()


def test_single_token_is_not_a_judge_grpo_style():
    """single_token is an eval/generator protocol with no <think> block and no JSON body; there
    is nothing for this module to score, so it must not be silently accepted here."""
    assert "single_token" not in PROMPT_STYLES


def test_the_style_names_match_the_reward_paths_copy():
    """judge_verdict cannot import training.grpo.reward (that pulls aiohttp and veRL at import
    time and this module must stay importable anywhere), so the names are a hand copy. Same
    pinning the eval arm's _BASE_DUMP_KEYS gets."""
    from training.grpo.reward import PROMPT_STYLE_FULL as R_FULL
    from training.grpo.reward import PROMPT_STYLE_RATING_ONLY as R_RATING

    assert PROMPT_STYLE_FULL == R_FULL
    assert PROMPT_STYLE_RATING_ONLY == R_RATING


def test_rating_only_fields_is_just_the_rating():
    assert RATING_ONLY_FIELDS == ("rating",)


# --- how it reaches the reward ----------------------------------------------------------------


def test_the_reward_mixes_task_and_format_at_the_configured_weights(monkeypatch):
    """0.9/0.1 are the module defaults, which the rating_only runs now use rather than
    overriding format to zero."""
    import asyncio

    from training.grpo.judge_reward import compute_score

    # The real run trains with thinking on (data.apply_chat_template_kwargs.enable_thinking),
    # and the driver propagates it here. Without it compute_score treats the whole completion
    # as the answer, so the </think> split never happens and even a perfect answer scores 0.2 --
    # a real failure mode, not a test artifact: a run that lost this env var would silently
    # train against a format term stuck near its floor.
    monkeypatch.setenv("PERSONA_ENABLE_THINKING", "1")
    monkeypatch.setenv("JUDGE_PROMPT_STYLE", "rating_only")
    monkeypatch.setenv("JUDGE_REWARD_ARM", "graded")
    monkeypatch.delenv("JUDGE_TASK_WEIGHT", raising=False)
    monkeypatch.delenv("JUDGE_FORMAT_WEIGHT", raising=False)

    # human is B, judge says 7 -> graded task reward 1.0; clean answer -> format 1.0.
    got = asyncio.run(compute_score("prism_judge", THINK + '{"rating": 7}', "B"))

    assert got["judge_task_reward"] == pytest.approx(1.0)
    assert got["judge_format_score"] == pytest.approx(1.0)
    assert got["judge_total"] == pytest.approx(0.9 * 1.0 + 0.1 * 1.0)


def test_the_trainer_derives_the_style_from_the_config_name():
    """One fact, one source. The trainer reads the config and the reward reads the style; if
    they could be set independently there is a combination that scores a rating_only corpus
    against the 37-field schema and still completes cleanly."""
    script = (ROOT / "scripts" / "slurm" / "judge_grpo_train.sh").read_text()

    assert "qwen35_judge_grpo)        export JUDGE_PROMPT_STYLE=full" in script
    assert "qwen35_judge_rating_grpo) export JUDGE_PROMPT_STYLE=rating_only" in script
    # Never taken from the caller: no ${JUDGE_PROMPT_STYLE:-...} default anywhere.
    assert "JUDGE_PROMPT_STYLE:-" not in script


def test_the_trainer_does_not_pin_the_reward_weights():
    """0.9/0.1 are judge_reward's defaults. Round 1 overrode them to 1.0/0.0 because the format
    term was meaningless for this schema; now that it is not, the overrides must be gone."""
    script = (ROOT / "scripts" / "slurm" / "judge_grpo_train.sh").read_text()

    for line in script.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        assert "JUDGE_FORMAT_WEIGHT=0.0" not in stripped
        assert "JUDGE_TASK_WEIGHT=1.0" not in stripped


def test_a_wrong_but_tidy_answer_still_earns_the_format_share(monkeypatch):
    """The point of an additive format term: answering cleanly is worth something even when the
    verdict is wrong, while never answering is worth nothing."""
    import asyncio

    from training.grpo.judge_reward import compute_score

    # The real run trains with thinking on (data.apply_chat_template_kwargs.enable_thinking),
    # and the driver propagates it here. Without it compute_score treats the whole completion
    # as the answer, so the </think> split never happens and even a perfect answer scores 0.2 --
    # a real failure mode, not a test artifact: a run that lost this env var would silently
    # train against a format term stuck near its floor.
    monkeypatch.setenv("PERSONA_ENABLE_THINKING", "1")
    monkeypatch.setenv("JUDGE_PROMPT_STYLE", "rating_only")
    monkeypatch.setenv("JUDGE_REWARD_ARM", "directional")
    monkeypatch.delenv("JUDGE_TASK_WEIGHT", raising=False)
    monkeypatch.delenv("JUDGE_FORMAT_WEIGHT", raising=False)

    wrong = asyncio.run(compute_score("prism_judge", THINK + '{"rating": 1}', "B"))
    silent = asyncio.run(compute_score("prism_judge", "never closes", "B"))

    assert wrong["judge_task_reward"] == pytest.approx(0.0)
    assert wrong["judge_total"] == pytest.approx(0.1)
    assert silent["judge_total"] == pytest.approx(0.0)
    assert wrong["judge_total"] > silent["judge_total"]
