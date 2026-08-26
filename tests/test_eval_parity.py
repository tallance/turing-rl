"""Parity tests for the RL-generator eval scorer.

The non-negotiable goal: ``scripts.eval_rl_generator.directional_accuracy`` must
reproduce the SAME "picked the true human" numbers the judge sweep computes on the
same pairs. The authoritative sweep scorer is ``scripts/analyze_judge_sweep.py``
(``per_call_features`` -> ``accuracy``), which encodes the real convention:

  * canonical rating = ``rating_gt_first`` if set else ``rating_gen_first``
    (analyze_judge_sweep.py:68-70)
  * rating scale is "1=strongly A ... 4=tie ... 7=strongly B"
    (analyze_judge_sweep.py:295); so among valid 1-7 non-tie ratings,
    ``rating < 4 -> judge picks side A``, ``rating > 4 -> picks side B``
    (analyze_judge_sweep.py:86-91)
  * ``human_side`` = row value, else ``"A" if generated_is_b else "B"``
    (analyze_judge_sweep.py:84); and by the reward-path coupling
    (training/grpo/reward.py:718-763) ``rating_gt_first`` is set IFF
    ``generated_is_b`` is True IFF the human/GT turn sits on side A.
  * ``picked_human = int(pick == human_side)``; ties (rating==4) and
    invalid/parse-fail ratings are excluded from the denominator.

So for a gt-first pair (human=A) the judge is correct iff rating < 4, and for a
gen-first pair (human=B) the judge is correct iff rating > 4 (inverted threshold).
"""
from __future__ import annotations

import pytest

from scripts.eval_rl_generator import directional_accuracy


def test_matches_sweep_convention():
    # rating_gt_first means the GT/human turn was presented FIRST (side A). On the
    # "1=strongly A ... 7=strongly B" scale: rating<4 => picks A (human) => correct;
    # rating>4 => picks B (generated) => wrong; rating==4 => tie (excluded).
    rows = [
        {"rating_gt_first": 2, "rating_gen_first": None},  # <4 -> picks A(human) -> correct
        {"rating_gt_first": 6, "rating_gen_first": None},  # >4 -> picks B(gen)  -> wrong
        {"rating_gt_first": 4, "rating_gen_first": None},  # tie -> excluded
    ]
    acc = directional_accuracy(rows)
    assert acc["n_total"] == 3
    assert acc["n_nontie"] == 2 and acc["correct"] == 1
    assert acc["accuracy"] == 0.5 and acc["gen_win_rate"] == 0.5


def test_gen_first_threshold_is_inverted():
    # rating_gen_first means the GENERATED turn was presented first (side A), so the
    # human is side B. Correct iff the judge picks B, i.e. rating > 4.
    rows = [
        {"rating_gt_first": None, "rating_gen_first": 6},  # >4 -> picks B(human) -> correct
        {"rating_gt_first": None, "rating_gen_first": 2},  # <4 -> picks A(gen)   -> wrong
        {"rating_gt_first": None, "rating_gen_first": 4},  # tie -> excluded
    ]
    acc = directional_accuracy(rows)
    assert acc["n_nontie"] == 2 and acc["correct"] == 1
    assert acc["accuracy"] == 0.5 and acc["gen_win_rate"] == 0.5


def test_ties_and_parse_failures_excluded_from_denominator():
    rows = [
        {"rating_gt_first": 1, "rating_gen_first": None},   # correct
        {"rating_gt_first": 4, "rating_gen_first": None},   # tie
        {"rating_gt_first": 0, "rating_gen_first": None},   # parse-fail sentinel (invalid)
        {"rating_gt_first": None, "rating_gen_first": None}, # nothing recoverable
    ]
    acc = directional_accuracy(rows)
    assert acc["n_total"] == 4
    assert acc["n_nontie"] == 1 and acc["correct"] == 1
    assert acc["accuracy"] == 1.0 and acc["gen_win_rate"] == 0.0


def test_empty_is_zero_accuracy():
    acc = directional_accuracy([])
    assert acc["n_total"] == 0 and acc["n_nontie"] == 0
    assert acc["accuracy"] == 0.0 and acc["gen_win_rate"] == 1.0


def test_parity_against_real_sweep_scorer():
    """Drift guard: on realistic reward-dump rows (with generated_is_b / human_side
    exactly as the reward path emits them), directional_accuracy must return the SAME
    accuracy/correct/n_nontie that scripts/analyze_judge_sweep.py produces."""
    from scripts.analyze_judge_sweep import accuracy as sweep_accuracy, per_call_features

    # Rows shaped like real reward dumps: gt-first pairs carry rating_gt_first +
    # generated_is_b=True + human_side="A"; gen-first pairs carry rating_gen_first +
    # generated_is_b=False + human_side="B". Include ties and a parse-fail sentinel.
    rows = [
        {"rating_gt_first": 1, "rating_gen_first": None, "generated_is_b": True,  "human_side": "A"},
        {"rating_gt_first": 3, "rating_gen_first": None, "generated_is_b": True,  "human_side": "A"},
        {"rating_gt_first": 5, "rating_gen_first": None, "generated_is_b": True,  "human_side": "A"},
        {"rating_gt_first": 7, "rating_gen_first": None, "generated_is_b": True,  "human_side": "A"},
        {"rating_gt_first": 4, "rating_gen_first": None, "generated_is_b": True,  "human_side": "A"},
        {"rating_gt_first": None, "rating_gen_first": 2, "generated_is_b": False, "human_side": "B"},
        {"rating_gt_first": None, "rating_gen_first": 6, "generated_is_b": False, "human_side": "B"},
        {"rating_gt_first": None, "rating_gen_first": 4, "generated_is_b": False, "human_side": "B"},
        {"rating_gt_first": None, "rating_gen_first": 3, "generated_is_b": False, "human_side": "B"},
        {"rating_gt_first": 0, "rating_gen_first": None, "generated_is_b": True,  "human_side": "A"},
    ]

    calls = [per_call_features(r) for r in rows]
    expected_acc = sweep_accuracy(calls)  # mean picked_human over non-None (or None)
    expected_correct = sum(c["picked_human"] for c in calls if c["picked_human"] is not None)
    expected_nontie = sum(1 for c in calls if c["picked_human"] is not None)

    out = directional_accuracy(rows)
    assert out["n_nontie"] == expected_nontie
    assert out["correct"] == expected_correct
    assert expected_acc is not None
    assert out["accuracy"] == pytest.approx(expected_acc)
    assert out["gen_win_rate"] == pytest.approx(1.0 - expected_acc)


def test_parity_minimal_rows_match_full_rows():
    """directional_accuracy must infer orientation from which rating field is set when
    generated_is_b/human_side are absent (the reward-path coupling), so minimal rows
    give the same numbers as the fully-annotated ones."""
    minimal = [
        {"rating_gt_first": 2, "rating_gen_first": None},
        {"rating_gt_first": None, "rating_gen_first": 6},
    ]
    full = [
        {"rating_gt_first": 2, "rating_gen_first": None, "generated_is_b": True,  "human_side": "A"},
        {"rating_gt_first": None, "rating_gen_first": 6, "generated_is_b": False, "human_side": "B"},
    ]
    assert directional_accuracy(minimal) == directional_accuracy(full)
