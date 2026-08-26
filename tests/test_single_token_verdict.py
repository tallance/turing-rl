"""Verdict + probability extraction from a one-token logprobs payload."""

import math

import pytest

from shared.single_token_verdict import HardFail, extract_verdict

def _payload(pairs):
    """OpenAI-shaped top_logprobs: [(token, logprob), ...] for the single position."""
    return [{"token": t, "logprob": lp} for t, lp in pairs]


def test_picks_a_when_a_is_more_probable():
    v = extract_verdict(_payload([("A", -0.1), ("B", -2.3)]))
    assert v.letter == "A"
    assert v.p_a > 0.5


def test_picks_b_when_b_is_more_probable():
    v = extract_verdict(_payload([("A", -3.0), ("B", -0.05)]))
    assert v.letter == "B"
    assert v.p_a < 0.5


def test_sentencepiece_leading_space_is_stripped():
    # ▁B must carry the decision. If ▁ were not stripped, B's mass would be 0 and the
    # only classified token would be A, so the assertion flips.
    v = extract_verdict(_payload([("A", -3.0), ("▁B", -0.1)]))
    assert v.letter == "B"
    assert v.p_a < 0.5


def test_bpe_leading_space_is_stripped():
    # Ġ is the GPT-2/BPE marker Qwen returns; ▁ is SentencePiece (Gemma). Both are in
    # the eval matrix, so both need CI coverage.
    v = extract_verdict(_payload([("ĠA", -0.1), ("B", -3.0)]))
    assert v.letter == "A"
    assert v.p_a > 0.5


def test_lowercase_variants_count():
    v = extract_verdict(_payload([("a", -0.1), ("B", -3.0)]))
    assert v.letter == "A"
    assert v.p_a > 0.5


def test_variants_of_one_letter_are_summed_not_maxed():
    # Two B variants at -1.0 each sum to more mass than a single A at -0.8.
    v = extract_verdict(_payload([("A", -0.8), ("B", -1.0), (" B", -1.0)]))
    assert v.letter == "B"


def test_p_a_is_renormalized_over_a_and_b_only():
    # Half the mass sits on an irrelevant token; p_a must ignore it.
    v = extract_verdict(_payload([("A", math.log(0.25)), ("B", math.log(0.25)),
                                  ("Neither", math.log(0.5))]))
    assert v.p_a == pytest.approx(0.5)
    assert v.residual_mass == pytest.approx(0.5)


def test_argmax_always_agrees_with_p_a():
    for a_lp, b_lp in [(-0.1, -2.0), (-2.0, -0.1), (-0.69, -0.70)]:
        v = extract_verdict(_payload([("A", a_lp), ("B", b_lp)]))
        assert (v.letter == "A") == (v.p_a > 0.5)


def test_no_ab_token_is_a_hard_fail_not_a_coin_flip():
    with pytest.raises(HardFail):
        extract_verdict(_payload([("Neither", -0.1), ("\n", -2.0)]))


def test_trailing_newline_merged_into_verdict_token_is_stripped():
    # A chat template can merge a trailing newline into the verdict token itself
    # (e.g. "\nA"). Before stripping "\n", "\nA".strip(_STRIP) left the token
    # unclassified, so this would raise HardFail instead of picking A.
    v = extract_verdict(_payload([("\nA", -0.1), ("B", -3.0)]))
    assert v.letter == "A"
    assert v.p_a > 0.5


def test_bare_newline_token_is_still_not_a_verdict():
    # Stripping "\n" off a token must not turn a BARE newline into an empty string
    # that somehow reads as a letter -- confirm the existing hard-fail behaviour for a
    # pure "\n" token is unchanged now that "\n" is part of _STRIP.
    with pytest.raises(HardFail):
        extract_verdict(_payload([("Neither", -0.1), ("\n", -2.0)]))


def test_empty_payload_is_a_hard_fail():
    with pytest.raises(HardFail):
        extract_verdict([])
