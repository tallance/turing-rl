"""The single-token judge prompt shares its inputs with TURING_PROMPT by construction."""

import hashlib

import pytest

from scripts.build_judge_train_pairs import render_turing_prompt
from shared.judge_prompts import (
    TURING_PROMPT,
    TURING_PROMPT_HEADER,
    TURING_SINGLE_TOKEN_PROMPT,
)

# sha256 of TURING_PROMPT as it stood at 2026-08-26, before the header/tail split.
# Pins the refactor as text-preserving: the full-schema arm supplies the reference cell
# for the switch decision, so it must not move by a single character.
TURING_PROMPT_SHA256 = "a09b4e268cca2616f2f20af4e233998cc56fb33fb05fb89c476a38001018731d"

# The composed hash alone has a compensating-boundary hole: moving a line of prose from
# the header down into the full-schema tail leaves TURING_PROMPT byte-identical, so its
# hash still passes -- while that line silently vanishes from the single-token arm, which
# is header + a different tail. The line most at risk is judge_prompts.py's "One candidate
# is the real [HUMAN] response. The other candidate is AI-generated.", the sentence the
# design names as most load-bearing for a one-token classifier. Pinning the header and the
# single-token prompt separately closes the boundary.
TURING_PROMPT_HEADER_SHA256 = (
    "7eabb1ea7be1c4c80c1e6107cda05b29f3fc1402f527a43ef26169d3fd7b81e4"
)
TURING_SINGLE_TOKEN_PROMPT_SHA256 = (
    "e2c790c0bc8edc907cb778581e0015ef6cdc3c5ea1570ca7fa80b79a74816823"
)


def test_refactor_preserves_turing_prompt_exactly():
    assert hashlib.sha256(TURING_PROMPT.encode()).hexdigest() == TURING_PROMPT_SHA256


def test_header_text_is_pinned_independently_of_the_composed_prompt():
    assert (hashlib.sha256(TURING_PROMPT_HEADER.encode()).hexdigest()
            == TURING_PROMPT_HEADER_SHA256)


def test_single_token_prompt_text_is_pinned():
    assert (hashlib.sha256(TURING_SINGLE_TOKEN_PROMPT.encode()).hexdigest()
            == TURING_SINGLE_TOKEN_PROMPT_SHA256)


def test_the_load_bearing_framing_sentence_is_in_the_single_token_arm():
    # Named explicitly rather than left to the hash so a future edit reads a sentence,
    # not a hex mismatch.
    sentence = ("One candidate is the real [HUMAN] response. "
                "The other candidate is AI-generated.")
    assert sentence in TURING_PROMPT_HEADER
    assert sentence in TURING_SINGLE_TOKEN_PROMPT


def test_both_templates_start_with_the_shared_header():
    assert TURING_PROMPT.startswith(TURING_PROMPT_HEADER)
    assert TURING_SINGLE_TOKEN_PROMPT.startswith(TURING_PROMPT_HEADER)


def test_header_ends_at_the_evaluation_procedure_boundary():
    assert TURING_PROMPT_HEADER.rstrip().endswith("<|End Source-Copy Watchlist|>")
    assert "## Evaluation Procedure" not in TURING_PROMPT_HEADER


def test_header_carries_every_input_placeholder():
    for field in ("user_history", "context", "response_a", "response_b",
                  "source_copy_watchlist"):
        assert "{" + field + "}" in TURING_PROMPT_HEADER


def test_single_token_prompt_has_the_verdict_tail_and_no_rubric():
    assert TURING_SINGLE_TOKEN_PROMPT.rstrip().endswith("Your output:")
    assert "Answer with a single letter, A or B" in TURING_SINGLE_TOKEN_PROMPT
    for leaked in ("## Evaluation Procedure", "## Criteria", "## Penalty Checks",
                   "score_gap", "immediate_target_score_a"):
        assert leaked not in TURING_SINGLE_TOKEN_PROMPT


_FIELDS = dict(
    user_history="[HUMAN]: earlier turn",
    context="[OTHER]: something happened",
    response_a="first candidate",
    response_b="second candidate",
)


def test_single_token_render_shares_the_full_prompt_header():
    full = render_turing_prompt(**_FIELDS)
    single = render_turing_prompt(**_FIELDS, prompt_style="single_token")
    # Everything the model reads before the verdict instruction is byte-identical.
    head = full.split("## Evaluation Procedure")[0]
    assert single.startswith(head)


def test_single_token_render_drops_the_rubric_and_schema():
    single = render_turing_prompt(**_FIELDS, prompt_style="single_token")
    for marker in ("score_gap", "immediate_target_score_a", "## Criteria",
                   "## Penalty Checks", "rating"):
        assert marker not in single
    assert single.rstrip().endswith("Your output:")
    assert "Answer with a single letter, A or B" in single


def test_single_token_render_keeps_the_watchlist_block():
    single = render_turing_prompt(**_FIELDS, prompt_style="single_token")
    assert "<|Source-Copy Watchlist|>" in single


def test_unknown_prompt_style_is_rejected():
    with pytest.raises(ValueError, match="prompt_style"):
        render_turing_prompt(**_FIELDS, prompt_style="nonsense")
