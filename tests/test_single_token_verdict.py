"""Verdict + probability extraction from a one-token logprobs payload."""

import math

import pytest

from shared.single_token_verdict import MIN_AB_MASS, HardFail, extract_verdict

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
    assert v.ab_mass == pytest.approx(0.5)
    assert v.off_ab_mass == pytest.approx(0.5)
    assert v.total == pytest.approx(1.0)


def test_off_ab_mass_ignores_the_top_k_truncation():
    # Only 0.30 of the distribution is visible in the top-k; the rest fell off the end.
    # A top-k residual (total - ab) would report 0.10 here and 0.50 for the same 0.20 of
    # A/B mass in a payload that happened to show all of itself, i.e. it would read
    # QUIETER on the flatter, junkier row. off_ab_mass is 0.80 either way.
    v = extract_verdict(_payload([("A", math.log(0.15)), ("B", math.log(0.05)),
                                  ("Neither", math.log(0.10))]))
    assert v.total == pytest.approx(0.30)
    assert v.ab_mass == pytest.approx(0.20)
    assert v.off_ab_mass == pytest.approx(0.80)


def test_letter_follows_the_larger_letter_mass():
    # `letter` is DEFINED as p_a > 0.5, so comparing it against p_a proves nothing. The
    # property worth pinning is the letter against the RAW summed masses, recomputed here
    # from the payload rather than read back off the Verdict. Fixtures are deliberately
    # unambiguous (the two masses differ by more than a rounding step) so the quotient
    # p_a = mass_a / (mass_a + mass_b) cannot resolve the wrong side of 0.5 through
    # floating-point error.
    cases = [
        [("A", -0.1), ("B", -2.0)],
        [("A", -2.0), ("B", -0.1)],
        [("A", -0.8), ("B", -1.0), (" B", -1.0)],   # summed B variants outweigh A
        [("ĠA", -1.0), ("a", -1.0), ("B", -0.8)],   # summed A variants outweigh B
        [("A", math.log(0.30)), ("B", math.log(0.29))],
    ]
    for pairs in cases:
        mass_a = sum(math.exp(lp) for t, lp in pairs if t.strip(" Ġ").upper() == "A")
        mass_b = sum(math.exp(lp) for t, lp in pairs if t.strip(" Ġ").upper() == "B")
        assert mass_a != mass_b, pairs
        v = extract_verdict(_payload(pairs))
        assert v.letter == ("A" if mass_a > mass_b else "B"), pairs


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


# --- the absolute A/B mass floor -------------------------------------------------

def test_stray_indefinite_article_does_not_manufacture_a_verdict():
    # The exact payload that motivated the floor: the model emitted a think tag, and the
    # only "A/B" token in the top-k is " a", the English indefinite article, at 1e-9.
    # Renormalizing that against a zero B yields p_a == 1.0 -- a CERTAIN A off noise,
    # which brier then scores as a maximally confident prediction.
    with pytest.raises(HardFail):
        extract_verdict(_payload([("<think>", math.log(0.60)),
                                  ("Answer", math.log(0.399)),
                                  (" a", math.log(1e-9))]))


def test_mass_just_under_the_floor_hard_fails():
    with pytest.raises(HardFail):
        extract_verdict(_payload([("Neither", math.log(0.99)),
                                  ("A", math.log(MIN_AB_MASS * 0.99))]))


def test_mass_just_over_the_floor_is_accepted():
    v = extract_verdict(_payload([("Neither", math.log(0.99)),
                                  ("A", math.log(MIN_AB_MASS * 1.01))]))
    assert v.letter == "A"
    assert v.ab_mass == pytest.approx(MIN_AB_MASS * 1.01)


def test_a_genuine_verdict_split_across_variants_clears_the_floor():
    # No single variant reaches the floor; their sum does. The floor must gate the SUM,
    # not any individual token, or a tokenizer that spreads the verdict over "A"/" A"/"a"
    # would hard-fail on a perfectly good answer.
    v = extract_verdict(_payload([("A", math.log(0.004)), (" A", math.log(0.004)),
                                  ("a", math.log(0.004)), ("B", math.log(0.002)),
                                  ("<think>", math.log(0.98))]))
    assert v.letter == "A"
    assert v.ab_mass == pytest.approx(0.014)
    assert v.p_a == pytest.approx(0.012 / 0.014)


def test_the_floor_is_overridable_per_call_but_defaults_to_the_constant():
    payload = _payload([("Neither", math.log(0.99)), ("A", math.log(1e-3))])
    with pytest.raises(HardFail):
        extract_verdict(payload)
    v = extract_verdict(payload, min_ab_mass=1e-6)
    assert v.letter == "A"


# --- the structural check on the sampled token ------------------------------------

def test_sampled_token_that_is_not_a_verdict_hard_fails():
    # The top-k carries a clean, high-mass A -- but the model actually emitted "<think>",
    # so this position is not a verdict position at all and no mass threshold can tell.
    with pytest.raises(HardFail, match="not an A/B verdict"):
        extract_verdict(_payload([("A", -0.1), ("B", -2.0)]), sampled_token="<think>")


def test_sampled_token_that_is_a_verdict_variant_passes():
    v = extract_verdict(_payload([("A", -0.1), ("B", -2.0)]), sampled_token="ĠA")
    assert v.letter == "A"


def test_sampled_token_may_disagree_with_the_argmax():
    # Sampling can land on the minority letter. That is a real verdict, not a structural
    # failure -- the check is "is this a verdict position", not "did it pick the top token".
    v = extract_verdict(_payload([("A", -0.1), ("B", -2.0)]), sampled_token="B")
    assert v.letter == "A"


def test_omitting_the_sampled_token_skips_the_structural_check():
    v = extract_verdict(_payload([("A", -0.1), ("B", -2.0)]))
    assert v.letter == "A"
