"""Judge pair parquet -> lora_sft JSONL."""

import json
from collections import Counter

import pandas as pd
import pytest

from scripts.build_judge_ce_dataset import (
    build_ce_records,
    check_prompt_style,
    split_by_user,
)
from scripts.build_judge_train_pairs import render_turing_prompt


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


def test_split_by_user_raises_on_single_user():
    """A small --limit smoke run can plausibly hit exactly one user. Silently writing an
    empty train (or val) file is worse than a loud failure: it looks like a training job
    that ran and did nothing, rather than a config mistake caught at data-build time."""
    records = build_ce_records(_rows())  # both rows are user "u1"
    try:
        split_by_user(records, val_frac=0.1)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError when only one user is present")


def test_split_by_user_rejects_val_frac_out_of_range():
    records = build_ce_records(_multi_user_rows())
    for bad in (-0.1, 1.0, 1.5):
        try:
            split_by_user(records, val_frac=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for val_frac={bad!r}")


def test_split_by_user_is_deterministic():
    """A later task uses the val score to decide when training has converged; that is only
    meaningful if repeated data-build invocations hold out the same users every time."""
    records = build_ce_records(_multi_user_rows())
    train1, val1 = split_by_user(records, val_frac=0.3)
    train2, val2 = split_by_user(records, val_frac=0.3)
    assert [r["pair_id"] for r in train1] == [r["pair_id"] for r in train2]
    assert [r["pair_id"] for r in val1] == [r["pair_id"] for r in val2]


def test_records_survive_a_real_parquet_round_trip(tmp_path):
    """The in-memory fixtures above can never expose a numpy leak: an in-memory DataFrame
    keeps "prompt" as a plain Python list, so json.dumps would succeed on it even if the
    whole column leaked into a record by mistake. Verified by hand: writing the fixture
    through pd.DataFrame.to_parquet and reading it back makes "prompt" decode as a
    numpy.ndarray (pyarrow's real behavior for a nested list-of-struct column), and
    json.dumps on an ndarray raises -- so only a write-then-read of a real parquet file
    exercises the decode path this dataset is actually built from."""
    parquet_path = tmp_path / "pairs.parquet"
    _multi_user_rows().to_parquet(parquet_path)
    df = pd.read_parquet(parquet_path)

    recs = build_ce_records(df)
    assert len(recs) == 20
    for rec in recs:
        json.dumps(rec)


# --- prompt-style provenance ------------------------------------------------------
#
# The pair builder records prompt_style in its sidecar and, until now, nothing read it.
# Pointing --pairs at a full-schema parquet trains the discriminator on 20k-char rubric
# prompts that are then served against ~900-char single-token prompts: nothing crashes,
# the val curve looks sane, and the bad number reads as "the single-token protocol does
# not work".

_PROMPT_FIELDS = dict(
    user_history="[HUMAN]: earlier turn",
    context="[OTHER]: something happened",
    response_a="first candidate",
    response_b="second candidate",
)


def _styled_records(style):
    """CE records whose prompts are rendered by the real template, not hand-spelled.

    A hand-written fixture ending in "Your output:" would pass the text check whatever
    the source style actually was, which is precisely the fixture weakness that let this
    defect through.
    """
    prompt = render_turing_prompt(**_PROMPT_FIELDS, prompt_style=style)
    return [{"messages": [{"role": "user", "content": prompt},
                          {"role": "assistant", "content": "A"}],
             "pair_id": "p1", "order": "human_a", "user_id": "u1"}]


def _pairs_with_meta(tmp_path, meta):
    """A --pairs path whose sidecar carries `meta` (None writes no sidecar at all)."""
    pairs = tmp_path / "pairs.parquet"
    pairs.write_bytes(b"")  # only the path is read; the parquet itself is loaded upstream
    if meta is not None:
        (tmp_path / "pairs.meta.json").write_text(json.dumps(meta))
    return pairs


def test_matching_prompt_style_is_accepted(tmp_path):
    pairs = _pairs_with_meta(tmp_path, {"prompt_style": "single_token"})
    check_prompt_style(_styled_records("single_token"), pairs, expected="single_token")


def test_full_schema_pairs_are_rejected_when_single_token_is_expected(tmp_path):
    pairs = _pairs_with_meta(tmp_path, {"prompt_style": "full"})
    with pytest.raises(ValueError) as exc:
        check_prompt_style(_styled_records("full"), pairs, expected="single_token")
    # Naming both values is the point: the operator has to see which side is wrong.
    assert "'full'" in str(exc.value)
    assert "'single_token'" in str(exc.value)


def test_expecting_full_accepts_full_pairs(tmp_path):
    pairs = _pairs_with_meta(tmp_path, {"prompt_style": "full"})
    check_prompt_style(_styled_records("full"), pairs, expected="full")


def test_single_token_pairs_are_rejected_when_full_is_expected(tmp_path):
    pairs = _pairs_with_meta(tmp_path, {"prompt_style": "single_token"})
    with pytest.raises(ValueError, match="prompt style mismatch"):
        check_prompt_style(_styled_records("single_token"), pairs, expected="full")


def test_a_sidecar_predating_the_field_reads_as_full(tmp_path):
    pairs = _pairs_with_meta(tmp_path, {"n_rows": 4})
    with pytest.raises(ValueError, match="'full'"):
        check_prompt_style(_styled_records("full"), pairs, expected="single_token")


def test_missing_sidecar_still_catches_full_schema_prompts(tmp_path):
    """Belt and braces: a .meta.json can go missing, so the prompt text must also carry
    the evidence."""
    pairs = _pairs_with_meta(tmp_path, None)
    with pytest.raises(ValueError, match="does not end with the single-letter instruction"):
        check_prompt_style(_styled_records("full"), pairs, expected="single_token")


def test_missing_sidecar_passes_when_the_prompts_are_single_token(tmp_path, capsys):
    pairs = _pairs_with_meta(tmp_path, None)
    check_prompt_style(_styled_records("single_token"), pairs, expected="single_token")
    assert "WARNING" in capsys.readouterr().out


def test_a_lying_sidecar_does_not_excuse_full_schema_prompts(tmp_path):
    """The two checks are independent on purpose: a hand-edited or copied sidecar that
    claims single_token must still lose to what the prompts actually say."""
    pairs = _pairs_with_meta(tmp_path, {"prompt_style": "single_token"})
    with pytest.raises(ValueError, match="does not end with the single-letter instruction"):
        check_prompt_style(_styled_records("full"), pairs, expected="single_token")


def test_unknown_expected_style_is_rejected(tmp_path):
    pairs = _pairs_with_meta(tmp_path, {"prompt_style": "single_token"})
    with pytest.raises(ValueError, match="expect-prompt-style"):
        check_prompt_style(_styled_records("single_token"), pairs, expected="nonsense")


def _styled_parquet(tmp_path, style):
    """A --pairs parquet plus the sidecar the pair builder would have written."""
    prompt = render_turing_prompt(**_PROMPT_FIELDS, prompt_style=style)
    pd.DataFrame([
        {"prompt": [{"role": "user", "content": prompt}],
         "reward_model": {"ground_truth": "A"},
         "extra_info": {"pair_id": "p1", "order": "human_a", "user_id": "u1"}},
    ]).to_parquet(tmp_path / "pairs.parquet")
    (tmp_path / "pairs.meta.json").write_text(json.dumps({"prompt_style": style}))
    return tmp_path / "pairs.parquet"


def _run_cli(tmp_path, pairs, *extra):
    import subprocess
    import sys

    return subprocess.run(
        [sys.executable, "scripts/build_judge_ce_dataset.py",
         "--pairs", str(pairs), "--out", str(tmp_path / "out.jsonl"), *extra],
        capture_output=True, text=True)


def test_cli_defaults_to_expecting_single_token(tmp_path):
    """The default matters more than the flag: the mistake this guards against is an
    operator reusing a full-schema parquet without thinking about the flag at all."""
    r = _run_cli(tmp_path, _styled_parquet(tmp_path, "full"))
    assert r.returncode != 0
    assert "prompt style mismatch" in r.stderr
    assert not (tmp_path / "out.jsonl").exists()


def test_cli_accepts_single_token_pairs(tmp_path):
    r = _run_cli(tmp_path, _styled_parquet(tmp_path, "single_token"))
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "out.jsonl").exists()


def test_cli_can_be_pointed_at_full_schema_pairs_deliberately(tmp_path):
    r = _run_cli(tmp_path, _styled_parquet(tmp_path, "full"),
                 "--expect-prompt-style", "full")
    assert r.returncode == 0, r.stderr
