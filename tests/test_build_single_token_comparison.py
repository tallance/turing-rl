"""The harmonised accuracy rule that makes the two judge arms comparable.

The arms disagree on what a non-answer is: full-schema emits rating 4 (a tie) or an
unparseable response, single-token emits a hard failure. Getting the denominator wrong
here silently rescales one whole arm against the other, which is exactly the comparison
the table exists to make.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_single_token_comparison import (  # noqa: E402
    discover,
    summarize_full_schema,
    summarize_single_token,
)


def test_single_token_counts_hard_fail_wrong_but_keeps_it_in_the_denominator() -> None:
    rows = [
        {"letter": "A", "human_is_b": False},              # correct
        {"letter": "B", "human_is_b": True},               # correct
        {"letter": "A", "human_is_b": True},               # wrong
        {"hard_fail": True, "letter": None, "human_is_b": False},
    ]
    got = summarize_single_token(rows)
    assert got["n_pairs"] == 4
    assert got["accuracy_half_tie"] == 2 / 4     # hard fail in the denominator, scores 0
    assert got["accuracy_parse_ok"] == 2 / 3     # hard fail excluded
    assert got["failure_rate"] == 1 / 4
    assert got["tie_rate"] == 0.0


def test_full_schema_counts_a_tie_as_half_right() -> None:
    # human_side is derived as "A" iff generated_is_b. rating<4 picks A, >4 picks B.
    rows = [
        {"rating_gt_first": 1, "generated_is_b": True},    # picks A, human A -> correct
        {"rating_gt_first": 7, "generated_is_b": False},   # picks B, human B -> correct
        {"rating_gt_first": 7, "generated_is_b": True},    # picks B, human A -> wrong
        {"rating_gt_first": 4, "generated_is_b": True},    # tie
    ]
    got = summarize_full_schema(rows)
    assert got["n_pairs"] == 4
    assert got["tie_rate"] == 1 / 4
    assert got["accuracy_half_tie"] == (2 + 0.5) / 4
    assert got["accuracy_parse_ok"] == 2 / 3   # tie excluded from this one


def test_full_schema_unparseable_response_scores_zero_and_is_not_a_tie() -> None:
    rows = [
        {"rating_gt_first": 1, "generated_is_b": True},    # correct
        {"judge_raw_content": "no rating here"},           # parse error
    ]
    got = summarize_full_schema(rows)
    assert got["failure_rate"] == 1 / 2
    assert got["tie_rate"] == 0.0
    assert got["accuracy_half_tie"] == 1 / 2   # parse error stays in the denominator
    assert got["accuracy_parse_ok"] == 1.0     # ...but not in this one


def _write(reward_dir: Path, rows: list[dict]) -> None:
    reward_dir.mkdir(parents=True)
    reward_dir.joinpath("reward-1.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows)
    )


def test_mixed_pair_sets_are_refused_by_default(tmp_path: Path, capsys) -> None:
    # A run can hold one sweep per generator checkpoint, each with its own 880-pair file.
    # Pointing at the wrong sibling yields a plausible-looking but incomparable table.
    from scripts.build_single_token_comparison import main

    sweep = tmp_path / "run" / "raw" / "sweep"
    for cell, pairs in (("a", "gen_step0_880.parquet"), ("b", "gen_step320_880.parquet")):
        reward = sweep / cell / "on" / "reward"
        _write(reward, [{"rating_gt_first": 1, "generated_is_b": True}])
        (reward.parent / "run_metadata.json").write_text(
            json.dumps({"thinking_mode": "on", "pair_source": pairs})
        )

    out = tmp_path / "t.csv"
    argv = ["prog", "--sweep-root", str(sweep), "--out", str(out)]
    sys.argv = argv
    assert main() == 2
    assert not out.exists()
    assert "more than one pair set" in capsys.readouterr().err

    sys.argv = argv + ["--allow-mixed-pairs"]
    assert main() == 0
    assert out.exists()


def test_discover_reads_both_layouts_and_trusts_metadata_for_style(tmp_path: Path) -> None:
    sweep = tmp_path / "run" / "raw" / "sweep"
    # full-schema layout: <cell>/<mode>/reward
    _write(sweep / "zs" / "on" / "reward", [{"rating_gt_first": 1, "generated_is_b": True}])
    # single-token layout: <cell>/<mode>/<style>/reward
    st_reward = sweep / "trained" / "off" / "single_token" / "reward"
    _write(st_reward, [{"letter": "A", "human_is_b": False}])
    (st_reward.parent / "run_metadata.json").write_text(
        json.dumps({"prompt_style": "single_token", "thinking_mode": "off"})
    )

    got = {r["cell"]: r for r in discover(sweep)}
    assert got["zs"]["prompt_style"] == "full"
    assert got["zs"]["thinking_mode"] == "on"
    assert got["trained"]["prompt_style"] == "single_token"
    assert got["trained"]["accuracy_half_tie"] == 1.0
