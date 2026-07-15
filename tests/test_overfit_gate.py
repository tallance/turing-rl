# tests/test_overfit_gate.py
from scripts.overfit_gate_check import win_rate_from_rows

def _row(gen_is_b, rating):
    # human on A when gen is B: rating_gt_first is the "gen is more human" scale.
    # We store the oriented Likert directly as `likert` for the test.
    return {"likert": rating}

def test_win_excludes_ties_and_counts_ge5():
    rows = [{"likert": r} for r in [7,6,5,5,4,4,3,2,5,6]]
    # non-tie = 8 (drop the two 4s); wins (>=5) = 6
    wr = win_rate_from_rows(rows)
    assert wr["n_nontie"] == 8
    assert wr["wins"] == 6
    assert wr["win_rate"] == 0.75
    assert wr["passed"] is False   # 6/8 < 8/10 target on 10-set -> see threshold note

def test_pass_at_8_of_10():
    rows = [{"likert": r} for r in [7,7,7,6,6,5,5,5,3,4]]
    wr = win_rate_from_rows(rows, pass_wins=8)
    assert wr["wins"] == 8 and wr["passed"] is True
