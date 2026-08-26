"""Unit tests for the judge-training pair builder."""

import pandas as pd
import pytest

from scripts.build_judge_train_pairs import (
    CHARS_PER_TOKEN_ESTIMATE,
    build_judge_rows,
    flatten_all_generations,
    prompt_length_stats,
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


def test_render_strips_the_control_characters_the_reward_path_strips():
    """reward.py sanitizes the four content fields before formatting; so must this."""
    prompt = render_turing_prompt(
        user_history="hi\x0bstory", context="c\x00tx", response_a="A\x1fA", response_b="B\x08B"
    )
    assert "\x0b" not in prompt and "\x00" not in prompt
    assert "\x1f" not in prompt and "\x08" not in prompt
    assert "history" in prompt and "ctx" in prompt and "AA" in prompt and "BB" in prompt


def test_prompt_length_stats_report_percentiles_and_the_over_budget_count():
    stats = prompt_length_stats(["x" * 100, "x" * 200, "x" * 300, "x" * 400], budget_tokens=50)
    assert stats["prompt_chars_p50"] == 200
    assert stats["prompt_chars_p95"] == 400
    assert stats["prompt_chars_max"] == 400
    assert stats["prompt_tokens_est_max"] == pytest.approx(400 / CHARS_PER_TOKEN_ESTIMATE, abs=0.1)
    # budget 50 tokens ~= 195 chars, so 200/300/400 are over.
    assert stats["n_over_budget"] == 3
    assert stats["over_budget_rate"] == pytest.approx(0.75)
    assert stats["prompt_budget_tokens"] == 50


def test_prompt_length_stats_on_no_rows_do_not_crash():
    stats = prompt_length_stats([], budget_tokens=10240)
    assert stats["prompt_chars_max"] == 0
    assert stats["n_over_budget"] == 0
    assert stats["over_budget_rate"] == 0.0


def test_prompt_style_changes_only_the_prompt_text():
    full, _ = build_judge_rows(
        _source_df(), flatten_all_generations(_inference(n_gens=2)),
        lo=0.0, hi=1.0, limit=None, split="train", prompt_style="full",
    )
    single, _ = build_judge_rows(
        _source_df(), flatten_all_generations(_inference(n_gens=2)),
        lo=0.0, hi=1.0, limit=None, split="train", prompt_style="single_token",
    )

    assert len(full) == len(single)
    for f, s in zip(full.to_dict("records"), single.to_dict("records")):
        assert f["reward_model"] == s["reward_model"]
        assert f["extra_info"] == s["extra_info"]
        assert f["prompt"][0]["content"] != s["prompt"][0]["content"]


def test_the_two_orders_of_a_pair_carry_opposite_labels():
    """A systematic order/label swap is learnable, so training would 'succeed' and the
    eval would land near 0.25. Cheaper to assert than to diagnose."""
    rows, _ = build_judge_rows(
        _source_df(), flatten_all_generations(_inference(n_gens=3)),
        lo=0.0, hi=1.0, limit=None, split="train", prompt_style="single_token",
    )
    by_pair = {}
    for r in rows.to_dict("records"):
        by_pair.setdefault(r["extra_info"]["pair_id"], []).append(r)

    assert by_pair, "no pairs built"
    for pair_id, group in by_pair.items():
        labels = sorted(r["reward_model"]["ground_truth"] for r in group)
        assert labels == ["A", "B"], f"{pair_id} has labels {labels}"


def test_label_matches_which_slot_holds_the_human():
    rows, _ = build_judge_rows(
        _source_df(), flatten_all_generations(_inference(n_gens=3)),
        lo=0.0, hi=1.0, limit=None, split="train", prompt_style="single_token",
    )
    for r in rows.to_dict("records"):
        expected = "B" if r["extra_info"]["human_is_b"] else "A"
        assert r["reward_model"]["ground_truth"] == expected


def test_meta_records_the_prompt_style():
    _df, meta = build_judge_rows(
        _source_df(), flatten_all_generations(_inference(n_gens=1)),
        lo=0.0, hi=1.0, limit=None, split="train", prompt_style="single_token",
    )
    assert meta["prompt_style"] == "single_token"


def test_meta_carries_the_prompt_length_measurement():
    """max_prompt_length cannot be chosen without this; filter_overlong_prompts drops the rest."""
    _df, meta = build_judge_rows(
        _source_df(), flatten_all_generations(_inference(n_gens=2)),
        lo=0.0, hi=1.0, limit=None, split="train", prompt_budget_tokens=1,
    )
    for key in (
        "prompt_chars_p50", "prompt_chars_p95", "prompt_chars_max",
        "prompt_tokens_est_p50", "prompt_tokens_est_p95", "prompt_tokens_est_max",
        "n_over_budget", "prompt_budget_tokens", "chars_per_token_estimate",
    ):
        assert key in meta, key
    # The rendered rubric alone is thousands of characters, so a 1-token budget catches all 4.
    assert meta["n_over_budget"] == 4
    assert meta["prompt_chars_max"] >= meta["prompt_chars_p50"] > 0
