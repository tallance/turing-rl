#!/usr/bin/env python3
"""Score extracted Turing pairs with a frontier Claude judge.

Replays the judge_prompt the open-weight judges actually saw (see
scripts/extract_step320_subset.py) against Claude, parses the reply with the
production parser, and writes reward-dump-shaped rows so the existing
aggregators read them with no changes:

    scripts/summarize_test_eval.py --eval_root <out_root> --cell claude-opus-5

    python eval/claude_judge_run.py --pairs pairs.jsonl --out_root results/X \\
        --model opus --cell claude-opus-5

Appends as it goes and skips pairs already present in the output, so an
interrupted run resumes instead of re-buying answers it already has.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.claude_call import EFFORT_LEVELS, ask  # noqa: E402
from eval.metrics import _parse_turing_response  # noqa: E402

KEY_FIELDS = ("user_id", "post_id", "target_idx")
GEN_KEY = "9b-full5ep-step320"


def oriented_score(rating, generated_is_b):
    """Reward-dump ``turing_judge_score_raw``: higher = the GENERATION looked
    more human, whichever side it was shown on.

    The judge rates 1..7 on "A vs B" (1 = strongly A, 4 = tie, 7 = strongly B).
    When the generation is B the rating already points that way; when it is A
    the scale has to be reflected. Get this backwards and half the dataset is
    silently inverted while every downstream plot still looks reasonable, so it
    lives in one function with one test.

    Returns 0 for an unscorable pair, matching how the reward path dumps a
    parse failure (``likerts()`` drops 0, ``directional_accuracy`` counts it as
    a parse error).
    """
    if rating is None:
        return 0
    return int(rating) if generated_is_b else 8 - int(rating)


def build_row(rec, rating, *, verdict=None, envelope=None, judge_model="", effort=None):
    """Reward-dump-shaped row for one scored pair.

    Exactly one of rating_gt_first / rating_gen_first is set, mirroring the
    reward path: ``_canonical_rating`` prefers gt_first and falls back to
    gen_first, so setting both (or the wrong one) reads a plausible but wrong
    value. rating_gt_first is the one set iff the human/GT turn was shown first
    (side A), which is exactly when generated_is_b is True.
    """
    gib = bool(rec["generated_is_b"])
    verdict = verdict or {}
    envelope = envelope or {}
    return {
        "user_id": rec["user_id"],
        "post_id": rec["post_id"],
        "target_idx": rec["target_idx"],
        "generated_is_b": gib,
        "human_side": "A" if gib else "B",
        "rating_gt_first": rating if gib else None,
        "rating_gen_first": None if gib else rating,
        "turing_judge_score_raw": oriented_score(rating, gib),
        "judge_model": judge_model,
        "effort": effort,
        "score_gap": verdict.get("score_gap"),
        "judge_reasoning": verdict.get("reasoning"),
        "judge_raw_content": envelope.get("result"),
        "parse_error": rating is None,
        "cost_usd": envelope.get("total_cost_usd"),
        "judge_usage": envelope.get("usage"),
        "duration_api_ms": envelope.get("duration_api_ms"),
    }


def key_of(rec):
    return tuple(str(rec.get(f, "")) for f in KEY_FIELDS)


def check_flip_against_incumbents(pairs):
    """Replay the orientation rule over the incumbent columns.

    The frontier rows are only comparable to the published ones if they encode
    orientation the same way. Rather than trust that, recompute every incumbent
    turing_judge_score_raw from its own rating and require an exact match.
    """
    checked = skipped = 0
    for rec in pairs:
        gib = bool(rec["generated_is_b"])
        for cell, inc in rec.get("incumbent", {}).items():
            raw = inc.get("turing_judge_score_raw")
            rating = inc.get("rating_gt_first")
            if rating is None:
                rating = inc.get("rating_gen_first")
            if raw is None or rating is None or int(round(float(raw))) == 0:
                skipped += 1
                continue
            want = oriented_score(rating, gib)
            if int(round(float(raw))) != want:
                raise SystemExit(
                    "FAIL: orientation rule does not reproduce %s on pair %s: "
                    "stored raw=%s, rating=%s, generated_is_b=%s -> computed %s"
                    % (cell, key_of(rec), raw, rating, gib, want)
                )
            checked += 1
    print("orientation check OK: reproduced %d incumbent scores (%d unscorable "
          "rows skipped)" % (checked, skipped), file=sys.stderr)


def score_one(rec, model, effort, timeout):
    """One pair -> (row, cost). Retries once on a parse failure, as the
    production judge path does, then records the failure and moves on."""
    last_env, last_verdict = {}, {}
    for _ in range(2):
        env = ask(rec["judge_prompt"], model=model, effort=effort, full=True,
                  timeout=timeout)
        verdict = _parse_turing_response(env["result"])
        last_env, last_verdict = env, verdict
        if not verdict.get("parse_error") and verdict.get("rating") is not None:
            rating = int(verdict["rating"])
            if 1 <= rating <= 7:
                return build_row(rec, rating, verdict=verdict, envelope=env,
                                 judge_model=model, effort=effort), env.get("total_cost_usd") or 0.0
    return build_row(rec, None, verdict=last_verdict, envelope=last_env,
                     judge_model=model, effort=effort), last_env.get("total_cost_usd") or 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--model", default="opus")
    ap.add_argument("--cell", required=True, help="sweep cell dir, e.g. claude-opus-5")
    ap.add_argument("--gen_key", default=GEN_KEY)
    ap.add_argument("--mode", default="on")
    # Pinned, not left to the binary's default: the five incumbent cells all ran
    # thinking=on, and an unpinned effort makes two models incomparable.
    ap.add_argument("--effort", choices=EFFORT_LEVELS, default="high")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--check_only", action="store_true",
                    help="run the orientation check against the pairs file and exit")
    a = ap.parse_args()

    pairs = [json.loads(line) for line in open(a.pairs) if line.strip()]
    check_flip_against_incumbents(pairs)
    if a.check_only:
        return
    if a.limit:
        pairs = pairs[: a.limit]

    out_dir = Path(a.out_root) / "raw" / a.gen_key / "sweep" / a.cell / a.mode / "reward"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "scores.jsonl"

    done = set()
    if out_path.exists():
        with open(out_path) as fh:
            for line in fh:
                if line.strip():
                    done.add(key_of(json.loads(line)))
    todo = [r for r in pairs if key_of(r) not in done]
    print("%d pairs, %d already scored, %d to score -> %s"
          % (len(pairs), len(done), len(todo), out_path), file=sys.stderr)
    if not todo:
        return

    lock = threading.Lock()
    total_cost = 0.0
    n_err = 0
    started = time.time()
    with open(out_path, "a") as sink, ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures = {pool.submit(score_one, r, a.model, a.effort, a.timeout): r for r in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            rec = futures[fut]
            try:
                row, cost = fut.result()
            except Exception as exc:  # network/CLI failure: record, keep going
                row, cost = build_row(rec, None, judge_model=a.model, effort=a.effort), 0.0
                row["error"] = repr(exc)
            with lock:
                total_cost += cost
                n_err += int(bool(row.get("parse_error") or row.get("error")))
                sink.write(json.dumps(row) + "\n")
                sink.flush()
                print("[%3d/%3d] rating=%-4s raw=%-4s $%.4f  total $%.2f  %.0fs elapsed"
                      % (i, len(todo),
                         row["rating_gt_first"] if row["generated_is_b"] else row["rating_gen_first"],
                         row["turing_judge_score_raw"], cost, total_cost,
                         time.time() - started), file=sys.stderr)

    print("done: %d scored, %d unscorable, $%.2f, %.1f min"
          % (len(todo), n_err, total_cost, (time.time() - started) / 60), file=sys.stderr)


if __name__ == "__main__":
    main()
