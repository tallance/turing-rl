"""Judge pair parquet -> lora_sft JSONL."""

import json
from collections import Counter

import pandas as pd

from scripts.build_judge_ce_dataset import build_ce_records, split_by_user


def _rows():
    # Realistic endings: a real rendered single-token judge prompt always ends with
    # "Your output:" (the model is asked to emit the bare verdict next), never with the
    # answer letter itself. Distinct bodies keep the two rows distinguishable.
    return pd.DataFrame([
        {"prompt": [{"role": "user",
                     "content": "history-1 context-1 response-1a response-1b\n\nYour output:"}],
         "reward_model": {"ground_truth": "A"},
         "extra_info": {"pair_id": "p1", "order": "human_a", "user_id": "u1"}},
        {"prompt": [{"role": "user",
                     "content": "history-2 context-2 response-2a response-2b\n\nYour output:"}],
         "reward_model": {"ground_truth": "B"},
         "extra_info": {"pair_id": "p1", "order": "human_b", "user_id": "u1"}},
    ])


def test_one_record_per_row_with_the_label_as_the_assistant_turn():
    recs = build_ce_records(_rows())
    assert len(recs) == 2
    assert recs[0]["messages"] == [
        {"role": "user",
         "content": "history-1 context-1 response-1a response-1b\n\nYour output:"},
        {"role": "assistant", "content": "A"},
    ]
    assert recs[1]["messages"][-1] == {"role": "assistant", "content": "B"}


def test_assistant_content_is_exactly_one_bare_letter():
    for rec in build_ce_records(_rows()):
        target = rec["messages"][-1]["content"]
        assert target in ("A", "B")
        assert target == target.strip()


def test_the_label_never_leaks_into_the_prompt():
    """If the answer were visible in the prompt, CE would learn a shortcut and the eval
    would collapse for reasons that look like a modelling result. Real rendered prompts
    end with "Your output:", never with the answer letter, so this checks the real
    invariant rather than an accident of fixture spelling."""
    for rec in build_ce_records(_rows()):
        user = rec["messages"][0]["content"]
        assert "ground_truth" not in user
        assert not user.rstrip().endswith(rec["messages"][-1]["content"])


def test_records_are_json_serializable():
    for rec in build_ce_records(_rows()):
        json.loads(json.dumps(rec))


def test_unexpected_label_raises():
    df = pd.DataFrame([
        {"prompt": [{"role": "user", "content": "prompt body\n\nYour output:"}],
         "reward_model": {"ground_truth": "C"},
         "extra_info": {"pair_id": "p2", "order": "human_a", "user_id": "u2"}},
    ])
    try:
        build_ce_records(df)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for a non A/B label")


def _multi_user_rows():
    """Ten users, each contributing both orders of one pair -- enough rows that a val_frac
    of 0.1 holds out exactly one whole user, so the by-user split is checkable precisely.

    Deliberately interleaved (all human_a rows first, then all human_b rows) rather than
    grouped per user: a naive positional/row split would then put a user's human_a in one
    half and its human_b in the other, so this layout is what actually exercises the
    by-user invariant instead of passing by accident of construction order."""
    rows = []
    for order, label in (("human_a", "A"), ("human_b", "B")):
        for i in range(10):
            user_id = f"u{i}"
            pair_id = f"p{i}"
            rows.append({
                "prompt": [{"role": "user",
                            "content": f"history-{i} response-{i}-{order}\n\nYour output:"}],
                "reward_model": {"ground_truth": label},
                "extra_info": {"pair_id": pair_id, "order": order, "user_id": user_id},
            })
    return pd.DataFrame(rows)


def test_split_by_user_holds_out_whole_users_not_rows():
    """Both presentation orders of a pair must land on the same side of the split, and
    every pair from a given user must land on the same side too -- otherwise the val
    score is contaminated by memorising the human/generated identity from the train side
    of the same pair, and a later task uses that score to decide when training converged."""
    records = build_ce_records(_multi_user_rows())
    train, val = split_by_user(records, val_frac=0.1)

    train_users = {r["user_id"] for r in train}
    val_users = {r["user_id"] for r in val}

    # No user straddles both splits.
    assert not (train_users & val_users)
    # 10 users, val_frac=0.1 -> exactly 1 user held out -> both its rows (both orders) in val.
    assert len(val_users) == 1
    assert len(val) == 2
    assert {r["order"] for r in val} == {"human_a", "human_b"}
    assert len(train) == 18


def test_split_by_user_never_splits_a_pair_across_sides():
    """A pair's two rows (both presentation orders) never land on opposite sides."""
    records = build_ce_records(_multi_user_rows())
    train, val = split_by_user(records, val_frac=0.3)

    train_pairs = {r["pair_id"] for r in train}
    val_pairs = {r["pair_id"] for r in val}
    assert not (train_pairs & val_pairs)

    train_counts = Counter(r["pair_id"] for r in train)
    val_counts = Counter(r["pair_id"] for r in val)
    assert all(c == 2 for c in train_counts.values())
    assert all(c == 2 for c in val_counts.values())
