"""The JOIN: rows the real scorer emits -> the per-cell CSV.

Both halves of this path were tested in isolation and the join between them did not
exist. ``scripts/analyze_judge_sweep.py`` globs ``<cell>/<mode>/reward/*.jsonl`` one level
too shallow for the ``<style>`` segment the single-token writer adds, and skips any cell
absent from ``SIZE_MAP`` -- including the reference cell used below. ``summarize`` had no
caller at all. So nothing here hand-writes a dump row: every row is produced by
``eval.single_token_judge.score_single_token_with_info`` writing through its own dump
path into the real ``<cell>/<mode>/<style>/reward/`` tree, and the CSV is read back off
disk.
"""

from __future__ import annotations

import asyncio
import csv
import json
import math
from pathlib import Path

import pytest

from eval import single_token_judge as stj
from scripts.analyze_single_token_cells import build_table, main
from shared.judge_utils import _stable_turing_generated_is_b, sanitize_prompt_text

# judge-9b-graded-step52 is the reference cell of the whole comparison AND is absent from
# configs.judge_sweep_cells.SIZE_MAP -- the sweep analyzer skips it outright. Using it as
# the primary fixture cell keeps that regression impossible to reintroduce quietly.
REFERENCE_CELL = "judge-9b-graded-step52"
OTHER_CELL = "qwen35-4b"

_PAIR = dict(
    ground_truth="the real human turn",
    user_history="[HUMAN]: past turn",
    context="[OTHER]: hello",
)

# target_idx values chosen for the A/B slot they produce; asserted in the fixture builder
# so a change to the randomization fails loudly instead of silently re-labelling the rows.
HUMAN_IN_A = (4, 6)   # generated_is_b True  -> human sits on A
HUMAN_IN_B = (0, 1, 2)  # generated_is_b False -> human sits on B


def _human_is_b(target_idx: int) -> bool:
    response = sanitize_prompt_text(f"gen{target_idx}")
    return not _stable_turing_generated_is_b(
        response, user_id="u", post_id="p", target_idx=target_idx
    )


def _split_choice(p_a: float, sampled: str) -> dict:
    """A verdict position whose renormalized P(A) is exactly ``p_a``.

    A/B carry half the total mass, which clears MIN_AB_MASS by two orders of magnitude,
    so the fixture exercises the metrics rather than the floor.
    """
    return {
        "logprobs": {
            "content": [
                {
                    "token": sampled,
                    "top_logprobs": [
                        {"token": "A", "logprob": math.log(p_a * 0.5)},
                        {"token": "B", "logprob": math.log((1.0 - p_a) * 0.5)},
                        {"token": "<think>", "logprob": math.log(0.5)},
                    ],
                }
            ]
        }
    }


# The only payload in this file that hard-fails: a stray indefinite article at 1e-9 is the
# whole A/B mass, far under the floor.
_HARD_FAIL_CHOICE = {
    "logprobs": {
        "content": [
            {
                "token": " a",
                "top_logprobs": [
                    {"token": "<think>", "logprob": math.log(0.60)},
                    {"token": "Answer", "logprob": math.log(0.399)},
                    {"token": " a", "logprob": math.log(1e-9)},
                ],
            }
        ]
    }
}


def _score_into(monkeypatch, reward_dir: Path, *, target_idx: int, choice: dict) -> dict:
    """Run the real scorer once, letting it write its own dump row into ``reward_dir``."""

    async def fake_post(session, payload, *, semaphore, api_key=None, max_retries=None):
        return choice

    monkeypatch.setenv("JUDGE_MODEL", "Qwen/Qwen3.5-9B")
    monkeypatch.setenv("PERSONA_JUDGE_DUMP_RATE", "1.0")
    monkeypatch.setenv("PERSONA_REWARD_DUMP_DIR", str(reward_dir))
    monkeypatch.setattr(stj, "post_chat_choice_async", fake_post)

    return asyncio.run(
        stj.score_single_token_with_info(
            object(),
            "EMPTY",
            response=f"gen{target_idx}",
            **_PAIR,
            user_id="u",
            post_id="p",
            target_idx=target_idx,
            pair_id=f"pair-{target_idx}",
        )
    )


def _p_a_for(target_idx: int, p_human: float) -> float:
    """P(A) that assigns ``p_human`` to whichever slot actually holds the human."""
    return (1.0 - p_human) if _human_is_b(target_idx) else p_human


def _write_cell(monkeypatch, sweep_root: Path, cell: str, spec: list[tuple[int, float | None]]):
    """Score ``spec`` = [(target_idx, p_human or None for a hard fail), ...] into a cell."""
    reward_dir = sweep_root / cell / "off" / "single_token" / "reward"
    rows = []
    for target_idx, p_human in spec:
        if p_human is None:
            choice = _HARD_FAIL_CHOICE
        else:
            p_a = _p_a_for(target_idx, p_human)
            choice = _split_choice(p_a, sampled="A" if p_a > 0.5 else "B")
        rows.append(_score_into(monkeypatch, reward_dir, target_idx=target_idx, choice=choice))
    return rows


@pytest.fixture
def scored_tree(monkeypatch, tmp_path):
    """A two-cell sweep tree written by the scorer itself.

    Reference cell: 3 correct rows (p_human 0.9), 1 wrong row (p_human 0.2, human in B)
    and 1 hard fail. Second cell: 2 correct rows, one per slot.
    """
    sweep_root = tmp_path / "raw" / "sweep"
    # The placements the hand-computed expectations below depend on.
    assert all(not _human_is_b(i) for i in HUMAN_IN_A), "fixture: expected human in slot A"
    assert all(_human_is_b(i) for i in HUMAN_IN_B), "fixture: expected human in slot B"

    reference = _write_cell(
        monkeypatch,
        sweep_root,
        REFERENCE_CELL,
        [(4, 0.9), (6, 0.9), (0, 0.9), (1, 0.2), (2, None)],
    )
    other = _write_cell(monkeypatch, sweep_root, OTHER_CELL, [(4, 0.9), (0, 0.9)])
    return sweep_root, {REFERENCE_CELL: reference, OTHER_CELL: other}


def _csv_rows(path: Path) -> dict[str, dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["cell"]: row for row in csv.DictReader(handle)}


def test_the_csv_summarizes_the_rows_the_scorer_actually_wrote(scored_tree, tmp_path):
    sweep_root, _ = scored_tree
    out = tmp_path / "derived" / "single_token_cells.csv"

    main(["--sweep_root", str(sweep_root), "--out", str(out)])

    rows = _csv_rows(out)
    assert set(rows) == {REFERENCE_CELL, OTHER_CELL}

    ref = rows[REFERENCE_CELL]
    assert ref["prompt_style"] == "single_token"
    assert ref["thinking_mode"] == "off"
    assert int(ref["n"]) == 5
    assert int(ref["scored"]) == 4
    assert float(ref["hard_fail"]) == pytest.approx(0.2)
    # 3 of 4 scored rows name the human's slot.
    assert float(ref["accuracy"]) == pytest.approx(0.75)
    # letters: A, A (human in A, correct), B (human in B, correct), A (human in B, wrong).
    assert float(ref["a_rate"]) == pytest.approx(0.75)
    assert float(ref["expected_a_rate"]) == pytest.approx(0.5)
    assert float(ref["a_rate_excess"]) == pytest.approx(0.25)
    assert ref["degenerate"] == "True"
    # p_human = .9, .9, .9, .2 -> brier = (3*0.01 + 0.64) / 4
    assert float(ref["brier"]) == pytest.approx(0.1675)
    assert float(ref["auc"]) == pytest.approx(0.9375)
    assert float(ref["tie_rate"]) == 0.0
    # 5 distinct pair_ids, one presentation each: unmeasured, not zero.
    assert ref["order_consistency"] == ""

    second = rows[OTHER_CELL]
    assert int(second["n"]) == 2
    assert float(second["accuracy"]) == pytest.approx(1.0)
    assert float(second["hard_fail"]) == 0.0
    assert second["degenerate"] == "False"
    # Two rows, two DIFFERENT pairs. A pair key that collapsed them into one group would
    # report a measured order_consistency for a pairing that does not exist.
    assert second["order_consistency"] == ""


def test_the_csv_matches_the_jsonl_on_disk(scored_tree, tmp_path):
    """Independent recount straight off the artifacts: if the analyzer summarized a
    different (or empty) set of rows, these two disagree."""
    sweep_root, _ = scored_tree
    out = tmp_path / "cells.csv"
    main(["--sweep_root", str(sweep_root), "--out", str(out)])

    on_disk: dict[str, list[dict]] = {}
    for jsonl in sweep_root.rglob("reward/*.jsonl"):
        cell = jsonl.relative_to(sweep_root).parts[0]
        on_disk.setdefault(cell, []).extend(
            json.loads(line) for line in jsonl.read_text().splitlines() if line.strip()
        )

    assert on_disk, "the scorer wrote no dump rows; the fixture is not exercising anything"
    for cell, row in _csv_rows(out).items():
        written = on_disk[cell]
        scored = [r for r in written if not r["hard_fail"]]
        assert int(row["n"]) == len(written)
        assert int(row["scored"]) == len(scored)
        assert float(row["a_rate"]) == pytest.approx(
            sum(r["letter"] == "A" for r in scored) / len(scored)
        )
        assert float(row["expected_a_rate"]) == pytest.approx(
            sum(not r["human_is_b"] for r in scored) / len(scored)
        )


def test_the_style_segment_is_not_lost_from_the_cell_name(scored_tree, tmp_path):
    """The rows live at <cell>/off/single_token/reward/. A shallower walk would either
    find nothing or name the cell after the mode/style directory."""
    sweep_root, _ = scored_tree
    out = tmp_path / "cells.csv"
    main(["--sweep_root", str(sweep_root), "--out", str(out)])

    cells = set(_csv_rows(out))
    assert cells == {REFERENCE_CELL, OTHER_CELL}
    assert not cells & {"off", "single_token", "reward", "raw", "sweep"}


def test_an_empty_tree_is_an_error_not_an_empty_table(tmp_path):
    """The original failure mode reported a table of zero cells and exited 0."""
    empty = tmp_path / "raw" / "sweep"
    (empty / "qwen35-4b" / "off" / "single_token" / "reward").mkdir(parents=True)

    with pytest.raises(SystemExit, match="no reward"):
        main(["--sweep_root", str(empty), "--out", str(tmp_path / "cells.csv")])


def test_full_schema_rows_under_a_single_token_path_are_refused(tmp_path):
    """'The wrong protocol ran under the right label' is invisible in the numbers."""
    reward = tmp_path / "sweep" / "qwen35-4b" / "off" / "single_token" / "reward"
    reward.mkdir(parents=True)
    (reward / "reward-1-1.jsonl").write_text(
        json.dumps({"judge_prompt_style": "full", "rating_gt_first": 3}) + "\n"
    )

    # Matched on the message, not on "single_token": that substring also appears in the
    # path, so a laxer pattern would be satisfied by an unrelated error naming the file.
    with pytest.raises(ValueError, match="holds judge_prompt_style"):
        build_table(tmp_path / "sweep")


def test_the_run_root_is_refused_in_place_of_the_sweep_root(scored_tree, tmp_path):
    """rglob from the run root still finds the files; without a shape check every cell
    would be attributed to the directory literally named 'raw'."""
    sweep_root, _ = scored_tree
    run_root = sweep_root.parent.parent

    with pytest.raises(ValueError, match="path segments"):
        build_table(run_root)
