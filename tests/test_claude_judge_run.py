"""Tests for the frontier-judge scoring path (eval/claude_judge_run.py).

The expensive failure this guards against is the orientation flip. A frontier
judge writes rows into the same reward-dump layout the open-weight judges use;
if the flip is inverted, half the pairs are silently negated and every
downstream summary and plot still looks entirely reasonable. Nothing else in
the pipeline would catch it.

The strongest check here is the golden one: rebuild reward rows for the five
INCUMBENT judges from their own recorded ratings, using the same row builder
the frontier path uses, run the real scripts/summarize_test_eval.py over them,
and require that it reproduces the PUBLISHED step-320 summary CSVs exactly. It
exercises the row shape, field naming, the flip, the directory layout and the
summarize integration at once, against judges whose answers are already known.

The data files it needs are gitignored results, so those tests skip when the
results tree is absent.
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from eval.claude_judge_run import build_row, oriented_score, row_from_incumbent
from scripts.eval_rl_generator import directional_accuracy

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = Path.home() / "Projects/turing-rl/results"
SLIM = RESULTS / "2026-09-15-claude-judge-step320/pairs_step320_all880_slim.jsonl"
PUBLISHED = RESULTS / "2026-08-10-test-eval-9b-full5ep-full-schema"
GEN_KEY = "9b-full5ep-step320"
CELLS = ["qwen35-4b", "qwen35-9b", "qwen35-27b", "gemma4-12b", "gemma4-31b"]

needs_data = pytest.mark.skipif(
    not SLIM.is_file() or not PUBLISHED.is_dir(),
    reason="gitignored results tree not present",
)


# --------------------------------------------------------------------------
# The flip, in isolation
# --------------------------------------------------------------------------

def test_oriented_score_passes_through_when_generation_is_b():
    # Generation on side B: the 1..7 "A vs B" scale already points at it.
    assert oriented_score(1, True) == 1
    assert oriented_score(7, True) == 7


def test_oriented_score_inverts_when_generation_is_a():
    # Generation on side A: the scale must be reflected, or a judge that
    # strongly picked the generation would be recorded as rejecting it.
    assert oriented_score(1, False) == 7
    assert oriented_score(7, False) == 1


def test_oriented_score_keeps_a_tie_a_tie():
    assert oriented_score(4, True) == oriented_score(4, False) == 4


def test_oriented_score_of_unscorable_pair_is_zero():
    # 0 is what likerts() drops and what the reward path dumps on a parse fail.
    assert oriented_score(None, True) == 0
    assert oriented_score(None, False) == 0


# --------------------------------------------------------------------------
# Row shape: the rating_gt_first / rating_gen_first trap
# --------------------------------------------------------------------------

@pytest.mark.parametrize("gib", [True, False])
def test_build_row_sets_exactly_one_rating_field(gib):
    # _canonical_rating prefers rating_gt_first and falls back to
    # rating_gen_first, so setting both reads a plausible but wrong value.
    row = build_row({"user_id": "u", "post_id": "p", "target_idx": 0,
                     "generated_is_b": gib}, 6)
    set_fields = [f for f in ("rating_gt_first", "rating_gen_first")
                  if row[f] is not None]
    assert set_fields == (["rating_gt_first"] if gib else ["rating_gen_first"])
    # rating_gt_first is populated iff the human turn was shown first (side A),
    # which is iff the generation is B.
    assert row["human_side"] == ("A" if gib else "B")


@pytest.mark.parametrize("gib,rating,correct", [
    (True, 2, True),    # human on A, judge said A -> picked the human
    (True, 6, False),   # human on A, judge said B -> picked the generation
    (False, 6, True),   # human on B, judge said B -> picked the human
    (False, 2, False),  # human on B, judge said A -> picked the generation
])
def test_row_round_trips_through_directional_accuracy(gib, rating, correct):
    row = build_row({"user_id": "u", "post_id": "p", "target_idx": 0,
                     "generated_is_b": gib}, rating)
    acc = directional_accuracy([row])
    assert acc["n_nontie"] == 1
    assert acc["accuracy"] == (1.0 if correct else 0.0)
    assert acc["gen_win_rate"] == (0.0 if correct else 1.0)


def test_ties_and_parse_errors_leave_the_denominator():
    rec = {"user_id": "u", "post_id": "p", "target_idx": 0, "generated_is_b": True}
    acc = directional_accuracy([
        build_row(rec, 2),     # counted, correct
        build_row(rec, 4),     # tie
        build_row(rec, None),  # parse failure
    ])
    assert (acc["n_total"], acc["n_nontie"], acc["n_tie"], acc["n_parse_error"]) == (3, 1, 1, 1)
    assert acc["accuracy"] == 1.0


def test_parse_error_row_is_marked_and_scores_zero():
    row = build_row({"user_id": "u", "post_id": "p", "target_idx": 0,
                     "generated_is_b": False}, None)
    assert row["parse_error"] is True
    assert row["turing_judge_score_raw"] == 0
    assert row["rating_gt_first"] is None and row["rating_gen_first"] is None


# --------------------------------------------------------------------------
# Golden: reproduce the published step-320 summaries from incumbent ratings
# --------------------------------------------------------------------------

def _slim_rows():
    return [json.loads(line) for line in open(SLIM) if line.strip()]


def _canonical(inc):
    rating = inc.get("rating_gt_first")
    return inc.get("rating_gen_first") if rating is None else rating


def _published(cell):
    with open(PUBLISHED / f"summary_{cell}.csv") as fh:
        for row in csv.DictReader(fh):
            if row["checkpoint"] == GEN_KEY:
                return row
    raise AssertionError(f"no {GEN_KEY} row in summary_{cell}.csv")


@needs_data
@pytest.mark.parametrize("cell", CELLS)
def test_golden_reproduces_published_step320_summary(cell, tmp_path):
    pairs = _slim_rows()
    assert len(pairs) == 880

    out_dir = tmp_path / "raw" / GEN_KEY / "sweep" / cell / "on" / "reward"
    out_dir.mkdir(parents=True)
    with open(out_dir / "rows.jsonl", "w") as sink:
        for rec in pairs:
            row = build_row(rec, _canonical(rec["incumbent"][cell]),
                            judge_model=cell)
            sink.write(json.dumps(row) + "\n")

    csv_path = tmp_path / f"summary_{cell}.csv"
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "summarize_test_eval.py"),
         "--eval_root", str(tmp_path), "--cell", cell, "--mode", "on",
         "--expect_pairs", "880", "--max_missing_frac", "0",
         "--out_csv", str(csv_path)],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    assert proc.returncode == 0, proc.stderr

    got = next(r for r in csv.DictReader(open(csv_path)) if r["checkpoint"] == GEN_KEY)
    want = _published(cell)
    for field in ("n_scored", "n_unique_pairs", "n_likert", "likert_mean",
                  "win_rate_ge5", "pct_7", "judge_accuracy", "gen_win_rate",
                  "n_nontie", "n_tie", "n_parse_error"):
        assert got[field] == want[field], (
            f"{cell}.{field}: rebuilt {got[field]} != published {want[field]}")


@needs_data
def test_golden_pins_the_headline_number():
    # Guards the parametrized test above against a published CSV being swapped
    # out from under it: 0.1788 is the number the whole experiment reacts to.
    assert _published("qwen35-9b")["judge_accuracy"] == "0.1788"
    assert _published("qwen35-9b")["likert_mean"] == "4.9659"


@needs_data
def test_low_oriented_score_means_the_judge_picked_the_human():
    # The flip tests prove internal consistency; this proves the convention
    # points the way it is read in the write-up, over 880 real pairs: a LOW
    # turing_judge_score_raw must coincide with the judge picking the human.
    rows = [build_row(rec, _canonical(rec["incumbent"]["qwen35-9b"]))
            for rec in _slim_rows()]
    nontie = [r for r in rows if r["turing_judge_score_raw"] not in (0, 4)]
    assert len(nontie) == 867  # matches published n_nontie
    for row in nontie:
        picked_human = directional_accuracy([row])["accuracy"] == 1.0
        assert (row["turing_judge_score_raw"] < 4) == picked_human


@needs_data
@pytest.mark.parametrize("cell", CELLS)
def test_incumbent_row_builder_agrees_with_build_row(cell):
    # row_from_incumbent (used by the comparison table and the trajectory plot)
    # passes the stored orientation through; build_row (used for the frontier
    # judges) derives it. They must not drift, or frontier and incumbent numbers
    # stop being comparable while both still look fine.
    for rec in _slim_rows():
        got = row_from_incumbent(rec, cell)
        want = build_row(rec, _canonical(rec["incumbent"][cell]))
        for field in ("generated_is_b", "human_side", "rating_gt_first",
                      "rating_gen_first", "turing_judge_score_raw"):
            assert got[field] == want[field], (cell, field, rec["user_id"])


@needs_data
def test_both_orientations_are_represented_in_the_real_data():
    # A flip bug is invisible if every pair happens to sit on one side.
    gib = [rec["generated_is_b"] for rec in _slim_rows()]
    assert 0.4 < sum(gib) / len(gib) < 0.6


# --------------------------------------------------------------------------
# Plot: the point-judge channel must not have weakened the curve guards
# --------------------------------------------------------------------------

PLOT = REPO_ROOT / "scripts" / "plot_test_eval_judges.py"


def _run_plot(eval_root, out_dir, *extra):
    # No PYTHONPATH needed: plotstyle.py and judges.py are siblings of the
    # script, so running it puts scripts/ on sys.path[0] and both resolve.
    return subprocess.run(
        [sys.executable, str(PLOT), "--eval_root", str(eval_root),
         "--out_dir", str(out_dir), "--stem", "t", *extra],
        capture_output=True, text=True,
    )


@needs_data
def test_original_five_judge_figure_still_renders(tmp_path):
    proc = _run_plot(PUBLISHED, tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert (tmp_path / "t.png").is_file()


@needs_data
def test_pair_count_guard_still_fires_for_a_curve_judge(tmp_path):
    # The risk of adding point judges is weakening this guard so the new markers
    # fit, which would let a genuinely incomparable CURVE through later. Feed a
    # mismatched pair count through the normal curve path and require a failure.
    for cell in CELLS:
        rows = list(csv.DictReader(open(PUBLISHED / f"summary_{cell}.csv")))
        if cell == "gemma4-31b":
            rows[-1]["n_scored"] = "100"
        with open(tmp_path / f"summary_{cell}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    proc = _run_plot(tmp_path, tmp_path)
    assert proc.returncode != 0
    assert "different pair counts" in proc.stderr


@needs_data
def test_point_judge_is_exempt_and_disclosed(tmp_path):
    for cell in CELLS:
        (tmp_path / f"summary_{cell}.csv").write_text(
            (PUBLISHED / f"summary_{cell}.csv").read_text())
    # A single-checkpoint, 100-pair cell: rejected as a curve, fine as a point.
    (tmp_path / "summary_claude-opus-5.csv").write_text(
        "checkpoint,n_scored,likert_mean,win_rate_ge5\n"
        f"{GEN_KEY},100,3.2,0.27\n")
    proc = _run_plot(tmp_path, tmp_path, "--point_cell", "claude-opus-5")
    assert proc.returncode == 0, proc.stderr
    assert (tmp_path / "t.png").is_file()
