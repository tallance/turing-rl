"""Mixing sampling temperatures inside one judge-training pair set.

A judge trained only on turns sampled at the eval temperature learns a detector for that
temperature (the CE judge scored the same policy 0.179 at T=0.7 and 0.564 at T=1.0). Drawing
half the fake turns at the generator's TRAINING temperature is the fix, and these tests cover
the part of it that can silently produce a plausible-but-wrong pair set.
"""

import pandas as pd
import pytest

from scripts.build_judge_train_pairs import (
    build_judge_rows,
    flatten_all_generations,
    merge_generation_sets,
    render_turing_prompt,
)


def _inference(tag: str, n_gens: int = 2):
    """One context, ``n_gens`` generations whose text names the set they came from."""
    return {
        "u1": {
            "user_id": "u1",
            "test_targets": [
                {
                    "user_id": "u1",
                    "post_id": "p1",
                    "target_idx": 0,
                    "generations": [
                        {"raw_completion": f"<reasoning>r</reasoning>[HUMAN]: {tag} {i}"}
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


def _warm_and_hot():
    return (
        flatten_all_generations(_inference("warm")),
        flatten_all_generations(_inference("hot")),
    )


def _rows(gen_temperatures=None, generations=None):
    return build_judge_rows(
        _source_df(),
        generations,
        lo=0.0,
        hi=1.0,
        limit=None,
        split="train",
        gen_temperatures=gen_temperatures,
    )


# --- merging ------------------------------------------------------------------------------


def test_merge_concatenates_sets_in_order():
    warm, hot = _warm_and_hot()

    merged, temps = merge_generation_sets([warm, hot], [0.7, 1.0])

    assert merged[("u1", "p1", "0")] == ["warm 0", "warm 1", "hot 0", "hot 1"]
    assert temps[("u1", "p1", "0")] == [0.7, 0.7, 1.0, 1.0]


def test_merge_without_temperatures_returns_none():
    """The single-pickle path must stay exactly as it was, with no stamped column."""
    warm, _ = _warm_and_hot()

    merged, temps = merge_generation_sets([warm])

    assert merged == warm
    assert temps is None


def test_merge_rejects_sets_covering_different_contexts():
    """A short or stale second pickle would otherwise skew the mix for those contexts alone,
    with a plausible total row count and nothing raising."""
    warm, hot = _warm_and_hot()
    hot[("u2", "p9", "0")] = ["extra"]

    with pytest.raises(ValueError, match="different contexts"):
        merge_generation_sets([warm, hot], [0.7, 1.0])


def test_merge_rejects_a_temperature_count_mismatch():
    warm, _ = _warm_and_hot()

    with pytest.raises(ValueError, match="exactly one --gen_temperature"):
        merge_generation_sets([warm], [0.7, 1.0])


def test_merge_needs_at_least_one_set():
    with pytest.raises(ValueError, match="at least one"):
        merge_generation_sets([])


# --- what lands in the rows -----------------------------------------------------------------


def test_gen_idx_is_continuous_across_merged_sets():
    """``gen_idx`` must number the MERGED list rather than restarting per pickle.

    Restarting would give a context's 0.7 and 1.0 generations the same ``gen_idx``, hence the
    same ``pair_id`` -- two different fake turns sharing one identity, which nothing
    downstream checks.
    """
    warm, hot = _warm_and_hot()
    merged, temps = merge_generation_sets([warm, hot], [0.7, 1.0])

    df, _ = _rows(gen_temperatures=temps, generations=merged)

    by_pair = {row["pair_id"]: row for row in df["extra_info"]}
    assert len(by_pair) == 4, "four generations must yield four distinct pair_ids"
    assert sorted(row["gen_idx"] for row in by_pair.values()) == [0, 1, 2, 3]


def test_rows_are_evenly_split_and_still_label_balanced():
    warm, hot = _warm_and_hot()
    merged, temps = merge_generation_sets([warm, hot], [0.7, 1.0])

    df, meta = _rows(gen_temperatures=temps, generations=merged)

    # 4 generations x 2 A/B orders = 8 rows.
    assert len(df) == 8
    assert meta["rows_per_gen_temperature"] == {"0.7": 4, "1.0": 4}
    # Both orders survive the merge, so the label stays balanced by construction.
    assert meta["human_is_b_rate"] == 0.5


def test_temperature_stays_attached_to_its_own_generation():
    """Not merely the right counts -- the right pairing. A transposed stamp keeps every
    aggregate correct while inverting the per-temperature accuracy split it exists to
    measure."""
    warm, hot = _warm_and_hot()
    merged, temps = merge_generation_sets([warm, hot], [0.7, 1.0])

    df, _ = _rows(gen_temperatures=temps, generations=merged)

    for extra, prompt in zip(df["extra_info"], df["prompt"]):
        text = prompt[0]["content"]
        assert extra["gen_temperature"] == (0.7 if "warm" in text else 1.0)


def test_single_pickle_rows_carry_no_temperature():
    warm, _ = _warm_and_hot()

    df, meta = _rows(generations=warm)

    assert all(row["gen_temperature"] is None for row in df["extra_info"])
    assert "rows_per_gen_temperature" not in meta


# --- the rating_only template ----------------------------------------------------------------


def test_rating_only_style_renders_a_rating_prompt():
    prompt = render_turing_prompt(
        user_history="hist",
        context="ctx",
        response_a="AAA",
        response_b="BBB",
        prompt_style="rating_only",
    )

    assert '{"rating": <integer from 1 to 7>}' in prompt
    assert "immediate_target_score" not in prompt, "the rubric must not come along"
    assert prompt.index("AAA") < prompt.index("BBB")
