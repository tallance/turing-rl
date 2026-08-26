from scripts.analyze_judge_sweep import per_call_features, aggregate_cell


def _row(rating, gen_b=True, finish="stop"):
    return {"pair_id": "p1", "user_id": "u", "generated_is_b": gen_b,
            "judge_finish_reason": finish,
            "judge_raw_content": f'{{"rating": {rating}}}'}


def test_per_call_parses_rating():
    f = per_call_features(_row(3))
    assert f["rating"] == 3 and f["format_ok"] and not f["budget_hit"]


def test_budget_hit():
    assert per_call_features(_row(3, finish="length"))["budget_hit"]


def test_aggregate_accuracy_tie_excluded():
    # both orderings rating 4 -> abstain -> excluded from accuracy
    calls = [per_call_features(_row(4, gen_b=True)), per_call_features(_row(4, gen_b=False))]
    summ, _ = aggregate_cell("c", "off", calls)
    assert summ["n_calls"] == 2


def test_accuracy_picks_human():
    # generated_is_b=True -> human is A -> judge picks A when rating<4 -> correct
    from scripts.analyze_judge_sweep import accuracy
    assert accuracy([per_call_features(_row(1, gen_b=True))]) == 1.0   # picks A = human
    assert accuracy([per_call_features(_row(7, gen_b=True))]) == 0.0   # picks B = generator
