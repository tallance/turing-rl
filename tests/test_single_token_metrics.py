"""Accuracy, degeneracy and calibration columns for single-token cells."""

import pytest

from scripts.single_token_metrics import summarize

def _rows(verdicts):
    """verdicts: [(pair_id, human_is_b, letter, p_a), ...]"""
    return [
        {"pair_id": p, "human_is_b": h, "letter": l, "p_a": pa}
        for p, h, l, pa in verdicts
    ]


PERFECT = _rows([
    ("p1", False, "A", 0.9), ("p1", True, "B", 0.1),
    ("p2", False, "A", 0.8), ("p2", True, "B", 0.2),
])

ALWAYS_A = _rows([
    ("p1", False, "A", 0.9), ("p1", True, "A", 0.9),
    ("p2", False, "A", 0.9), ("p2", True, "A", 0.9),
])

# 8 human-in-A, 2 human-in-B, judged perfectly. Raw a_rate lands at 0.8 -- outside the
# old fixed [0.3, 0.7] window -- even though nothing is wrong with the judge; only the
# sample is imbalanced.
IMBALANCED_PERFECT = _rows([
    ("p1", False, "A", 0.9), ("p2", False, "A", 0.9), ("p3", False, "A", 0.9),
    ("p4", False, "A", 0.9), ("p5", False, "A", 0.9), ("p6", False, "A", 0.9),
    ("p7", False, "A", 0.9), ("p8", False, "A", 0.9),
    ("p9", True, "B", 0.1), ("p10", True, "B", 0.1),
])

# No pair_id repeats -- every pair has exactly one scored presentation, so there are no
# complete pairs to measure order consistency from.
SINGLETONS = _rows([
    ("p1", False, "A", 0.9), ("p2", True, "B", 0.1), ("p3", False, "A", 0.7),
])

# 4 human-in-A rows, all judged correctly (letter "A") -> s = accuracy on human-in-A
# = 1.0. 4 human-in-B rows, half judged correctly (letter "B") and half wrongly (letter
# "A") -> t = accuracy on human-in-B = 0.5. accuracy = (s+t)/2 = 0.75,
# a_rate_excess = (s-t)/2 = 0.25 -- above the fixed 0.2 threshold (Fix 2) and below the
# accuracy-dependent ceiling of 1 - accuracy = 0.25 is exactly the boundary, so this
# cell sits at the ceiling itself: the largest excess reachable at 0.75 accuracy.
EXCESS_QUARTER = _rows([
    ("a0", False, "A", 0.9), ("a1", False, "A", 0.9),
    ("a2", False, "A", 0.9), ("a3", False, "A", 0.9),
    ("b0", True, "B", 0.1), ("b1", True, "B", 0.1),
    ("b2", True, "A", 0.9), ("b3", True, "A", 0.9),
])


def test_perfect_judge():
    s = summarize(PERFECT)
    assert s["accuracy"] == pytest.approx(1.0)
    assert s["order_consistency"] == pytest.approx(1.0)
    assert s["a_rate"] == pytest.approx(0.5)
    assert s["expected_a_rate"] == pytest.approx(0.5)
    assert s["a_rate_excess"] == pytest.approx(0.0)
    assert s["tie_rate"] == 0.0


def test_always_a_judge_is_flagged_degenerate_not_merely_chance():
    s = summarize(ALWAYS_A)
    assert s["accuracy"] == pytest.approx(0.5)   # looks like chance...
    assert s["a_rate"] == pytest.approx(1.0)     # ...but is not
    assert s["expected_a_rate"] == pytest.approx(0.5)  # sample is balanced...
    assert s["a_rate_excess"] == pytest.approx(0.5)    # ...so the excess is real bias
    assert s["order_consistency"] == pytest.approx(0.0)
    assert s["degenerate"] is True


def test_perfect_judge_is_not_flagged_degenerate():
    assert summarize(PERFECT)["degenerate"] is False


def test_imbalanced_but_correct_judge_is_not_flagged_degenerate():
    # Reviewer's counterexample: 80/20 sample imbalance, perfect judging. Raw a_rate
    # (0.8) falls outside the old fixed window and would have been wrongly flagged.
    s = summarize(IMBALANCED_PERFECT)
    assert s["accuracy"] == pytest.approx(1.0)
    assert s["a_rate"] == pytest.approx(0.8)
    assert s["expected_a_rate"] == pytest.approx(0.8)
    assert s["a_rate_excess"] == pytest.approx(0.0)
    assert s["degenerate"] is False


def test_excess_above_0_2_is_flagged_degenerate():
    # Pins the threshold itself: the old 0.3 value can never fire on any cell that
    # passes the accuracy gate (ceiling is 1 - accuracy, which is below 0.3 at every
    # accuracy the gate allows through), so a fixture at exactly 0.0 or 0.5 (the only
    # values the rest of this suite exercises) cannot distinguish 0.2 from 0.3. This
    # fixture sits at 0.25, which is degenerate under 0.2 but not under 0.3.
    s = summarize(EXCESS_QUARTER)
    assert s["accuracy"] == pytest.approx(0.75)
    assert s["a_rate_excess"] == pytest.approx(0.25)
    assert s["degenerate"] is True


def test_order_consistency_is_none_without_complete_pairs():
    # Unmeasurable is not the same as zero: no complete pairs means no evidence of
    # position-locking, not evidence of it.
    s = summarize(SINGLETONS)
    assert s["order_consistency"] is None
    assert s["degenerate"] is False


def test_incomplete_pair_counts_toward_accuracy_but_excluded_from_order_consistency():
    # p3 has one order hard-failed and one surviving. The surviving row is still scored
    # and contributes to accuracy; the pair itself has no partner to compare orders
    # against, so it must not enter the order_consistency denominator.
    rows = PERFECT + [
        {"pair_id": "p3", "human_is_b": False, "letter": "A", "p_a": 0.85},
        {"pair_id": "p3", "human_is_b": True, "letter": None, "p_a": None, "hard_fail": True},
    ]
    s = summarize(rows)
    assert s["scored"] == 5
    assert s["accuracy"] == pytest.approx(1.0)
    assert s["order_consistency"] == pytest.approx(1.0)  # unaffected: only p1, p2 complete


def test_brier_and_auc_on_a_known_case():
    # p_human == p_a when the human is in slot A, else 1 - p_a.
    # PERFECT: p_human = .9, .9, .8, .8 -> brier = mean((1-p)^2) = 0.025
    s = summarize(PERFECT)
    assert s["brier"] == pytest.approx(0.025)
    assert s["auc"] == pytest.approx(1.0)


def test_hard_fails_are_counted_and_excluded_from_accuracy():
    rows = PERFECT + [{"pair_id": "p3", "human_is_b": False,
                       "letter": None, "p_a": None, "hard_fail": True}]
    s = summarize(rows)
    assert s["hard_fail"] == pytest.approx(1 / 5)
    assert s["accuracy"] == pytest.approx(1.0)  # scored rows only
    assert s["scored"] == 4


def test_empty_rows_returns_zeros_without_crashing():
    s = summarize([])
    assert s["n"] == 0
    assert s["scored"] == 0
    assert s["hard_fail"] == 0.0
    assert s["accuracy"] == 0.0
    assert s["a_rate"] == 0.0
    assert s["expected_a_rate"] == 0.0
    assert s["a_rate_excess"] == 0.0
    assert s["order_consistency"] is None
    assert s["brier"] == 0.0
    assert s["auc"] == 0.0
    assert s["degenerate"] is False
