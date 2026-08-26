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


def test_perfect_judge():
    s = summarize(PERFECT)
    assert s["accuracy"] == pytest.approx(1.0)
    assert s["order_consistency"] == pytest.approx(1.0)
    assert s["a_rate"] == pytest.approx(0.5)
    assert s["tie_rate"] == 0.0


def test_always_a_judge_is_flagged_degenerate_not_merely_chance():
    s = summarize(ALWAYS_A)
    assert s["accuracy"] == pytest.approx(0.5)   # looks like chance...
    assert s["a_rate"] == pytest.approx(1.0)     # ...but is not
    assert s["order_consistency"] == pytest.approx(0.0)
    assert s["degenerate"] is True


def test_perfect_judge_is_not_flagged_degenerate():
    assert summarize(PERFECT)["degenerate"] is False


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
