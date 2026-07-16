# tests/test_overfit_gate.py
from scripts.overfit_gate_check import win_rate_from_rows, prompt_level_gate

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


def _r(uid, ts, likert):
    return {"user_id": uid, "post_id": "p", "target_idx": 0, "ts": ts, "turing_judge_score_raw": likert}

def test_prompt_level_gate_final_epoch_majority():
    rows = []
    # 10 prompts; give each 4 "early" low rollouts (ts 0-3) then 4 "final" high rollouts (ts 10-13)
    for i in range(10):
        for t in range(4):
            rows.append(_r(f"u{i}", t, 2))            # early: judged human (low)
        for t in range(10, 14):
            rows.append(_r(f"u{i}", t, 6 if i < 9 else 2))  # final: 9 prompts win, 1 loses
    g = prompt_level_gate(rows, pass_prompts=8, group_size=4)
    assert g["n_prompts"] == 10
    assert g["won"] == 9              # final-epoch majority; early low rollouts ignored
    assert g["passed"] is True
    assert 0.0 <= g["final_win_rate"] <= 1.0

def test_prompt_level_gate_fails_below_threshold():
    rows = []
    for i in range(10):
        for t in range(10, 14):
            rows.append(_r(f"u{i}", t, 6 if i < 5 else 2))  # only 5 prompts win
    g = prompt_level_gate(rows, pass_prompts=8, group_size=4)
    assert g["won"] == 5 and g["passed"] is False

def test_prompt_level_gate_even_split_is_tie_not_win():
    # final 4 rollouts per prompt = [6,6,2,2]: 2 win / 2 lose -> frac==0.5 -> TIE, not a win (strict >0.5)
    rows = []
    for i in range(10):
        for t, lk in zip(range(10, 14), [6, 6, 2, 2]):
            rows.append(_r(f"u{i}", t, lk))
    g = prompt_level_gate(rows, pass_prompts=8, group_size=4)
    assert all(p["frac"] == 0.5 for p in g["per_prompt"])
    assert g["won"] == 0 and g["passed"] is False
