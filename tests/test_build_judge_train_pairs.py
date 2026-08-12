"""Unit tests for the judge-training pair builder."""

import pandas as pd
import pytest

from scripts.build_judge_train_pairs import (
    build_judge_rows,
    flatten_all_generations,
    render_turing_prompt,
)


def _inference(n_gens: int = 2):
    return {
        "u1": {
            "user_id": "u1",
            "test_targets": [
                {
                    "user_id": "u1",
                    "post_id": "p1",
                    "target_idx": 0,
                    "generations": [
                        {"raw_completion": f"<reasoning>r</reasoning>[HUMAN]: fake {i}"}
                        for i in range(n_gens)
                    ],
                }
            ],
        }
    }


def _source_df():
    return pd.DataFrame(
        [
            {
                "data_source": "prism",
                "prompt": [{"role": "user", "content": "ignored"}],
                "reward_model": {"ground_truth": "real human turn"},
                "extra_info": {
                    "user_id": "u1",
                    "post_id": "p1",
                    "target_idx": 0,
                    "user_history": "hist",
                    "context": "ctx",
                },
            }
        ]
    )


def test_flatten_keeps_every_generation():
    flat = flatten_all_generations(_inference(n_gens=3))
    assert flat[("u1", "p1", "0")] == ["fake 0", "fake 1", "fake 2"]


def test_render_places_each_response_in_its_slot():
    prompt = render_turing_prompt(
        user_history="hist", context="ctx", response_a="AAA", response_b="BBB"
    )
    assert prompt.index("AAA") < prompt.index("BBB")
    assert "<|Response A|>" in prompt and "<|Response B|>" in prompt


def test_two_rows_per_generation_one_per_order():
    df, meta = build_judge_rows(
        _source_df(), flatten_all_generations(_inference(n_gens=3)),
        lo=0.0, hi=1.0, limit=None, split="train",
    )
    assert len(df) == 6
    assert meta["n_contexts"] == 1
    assert meta["n_generations"] == 3


def test_human_side_is_exactly_balanced():
    df, _ = build_judge_rows(
        _source_df(), flatten_all_generations(_inference(n_gens=4)),
        lo=0.0, hi=1.0, limit=None, split="train",
    )
    human_is_b = [r["human_is_b"] for r in df["extra_info"]]
    assert sum(human_is_b) * 2 == len(human_is_b)


def test_ground_truth_names_the_slot_holding_the_human():
    df, _ = build_judge_rows(
        _source_df(), flatten_all_generations(_inference(n_gens=1)),
        lo=0.0, hi=1.0, limit=None, split="train",
    )
    for _, row in df.iterrows():
        human_is_b = row["extra_info"]["human_is_b"]
        assert row["reward_model"]["ground_truth"] == ("B" if human_is_b else "A")
        text = row["prompt"][0]["content"]
        a_start = text.index("<|Response A|>")
        b_start = text.index("<|Response B|>")
        human_at = text.index("real human turn")
        assert (human_at > b_start) == human_is_b
        assert (b_start > human_at > a_start) != human_is_b


def test_row_ids_are_unique():
    df, _ = build_judge_rows(
        _source_df(), flatten_all_generations(_inference(n_gens=4)),
        lo=0.0, hi=1.0, limit=None, split="train",
    )
    row_ids = [r["row_id"] for r in df["extra_info"]]
    assert len(set(row_ids)) == len(row_ids)


def test_split_tag_is_propagated():
    df, _ = build_judge_rows(
        _source_df(), flatten_all_generations(_inference(n_gens=1)),
        lo=0.0, hi=1.0, limit=None, split="val",
    )
    assert all(r["split"] == "val" for r in df["extra_info"])


def test_missing_generation_raises():
    with pytest.raises(AssertionError):
        build_judge_rows(_source_df(), {}, lo=0.0, hi=1.0, limit=None, split="train")
