"""Reward-dir discovery must work under both judge_sweep_cell.sh layouts.

A single_token cell nests its dumps one level deeper than a full-schema one. Both the
summarizer and the completeness verifier used to glob only the shallow layout and read the
gen_key at a fixed depth, so a single-token sweep that had scored every pair reported "no
reward dirs" -- and the verifier, which exists to catch incomplete scoring, silently checked
nothing at all.
"""

from __future__ import annotations

import pytest

from scripts.eval_rl_generator import find_reward_dirs, gen_key_of


def _make(root, gen_key, cell, mode, style=None):
    d = root / "raw" / gen_key / "sweep" / cell / mode
    if style:
        d = d / style
    d = d / "reward"
    d.mkdir(parents=True)
    return d


def test_finds_the_flat_full_schema_layout(tmp_path):
    d = _make(tmp_path, "g1t07-step6", "gemma4-12b", "on")
    assert find_reward_dirs(tmp_path) == [d]
    assert gen_key_of(d) == "g1t07-step6"


def test_finds_the_nested_single_token_layout(tmp_path):
    d = _make(tmp_path, "g2t10-step18", "9b-ce3", "off", style="single_token")
    assert find_reward_dirs(tmp_path) == [d]
    # The extra style level must not shift the gen_key -- the bug this replaces read it by
    # a fixed parents[] index and would have returned "sweep" here.
    assert gen_key_of(d) == "g2t10-step18"


def test_finds_both_layouts_together_without_duplicates(tmp_path):
    flat = _make(tmp_path, "g0t07-step0", "gemma4-12b", "on")
    nested = _make(tmp_path, "g0t07-step0", "9b-ce", "off", style="single_token")
    found = find_reward_dirs(tmp_path)
    assert sorted(found) == sorted([flat, nested])
    assert len(found) == len(set(found)), "a dir was reported twice"
    assert {gen_key_of(p) for p in found} == {"g0t07-step0"}


def test_cell_and_mode_filters_apply_to_both_layouts(tmp_path):
    _make(tmp_path, "k", "9b-ce", "off", style="single_token")
    _make(tmp_path, "k", "9b-ce2", "off", style="single_token")
    _make(tmp_path, "k", "gemma4-12b", "on")
    assert len(find_reward_dirs(tmp_path, cell="9b-ce")) == 1
    assert len(find_reward_dirs(tmp_path, mode="on")) == 1
    assert len(find_reward_dirs(tmp_path)) == 3


def test_a_path_outside_raw_is_rejected_rather_than_guessed(tmp_path):
    # Returning a wrong-but-plausible key here would mislabel an entire results table.
    stray = tmp_path / "sweep" / "9b-ce" / "off" / "reward"
    stray.mkdir(parents=True)
    with pytest.raises(ValueError, match="not under a raw/"):
        gen_key_of(stray)
