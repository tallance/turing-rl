"""Compare every judge on the SAME 100 step-320 pairs, ranked by judge accuracy.

Judge accuracy -- how often the judge picks the actual human turn -- is the
headline. Every open-weight judge scores far below chance at step 320, which
says the GRPO generator learned to fool them rather than that it out-humans
humans. The question this table answers is whether a frontier judge resists it.

The incumbents are re-scored on the identical 100 keys the frontier judges saw,
not quoted from their published 880-pair numbers: mixing a judge effect with a
sampling effect would make the ranking meaningless. Their published 880 figures
are carried alongside purely as a sanity anchor.

    python scripts/compare_judges_step320.py --eval_root results/2026-09-15-... \\
        --pairs results/2026-09-15-.../pairs_step320_100.jsonl
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval_rl_generator import _canonical_rating, _picked_human  # noqa: E402
from scripts.eval_rl_generator import directional_accuracy  # noqa: E402
from scripts.summarize_test_eval import likerts  # noqa: E402

GEN_KEY = "9b-full5ep-step320"
INCUMBENTS = ["qwen35-4b", "qwen35-9b", "qwen35-27b", "gemma4-12b", "gemma4-31b"]
LABELS = {
    "qwen35-4b": "Qwen3.5 4B", "qwen35-9b": "Qwen3.5 9B *", "qwen35-27b": "Qwen3.5 27B",
    "gemma4-12b": "Gemma 4 12B", "gemma4-31b": "Gemma 4 31B",
    "claude-opus-5": "Claude Opus 5", "claude-sonnet-5": "Claude Sonnet 5",
}
KEY_FIELDS = ("user_id", "post_id", "target_idx")
Z = 1.959963985  # 95%


def key_of(rec):
    return tuple(str(rec.get(f, "")) for f in KEY_FIELDS)


def wilson(correct, n):
    """Wilson score interval. Closed form, so no scipy dependency."""
    if not n:
        return (0.0, 0.0)
    p = correct / n
    denom = 1 + Z * Z / n
    centre = (p + Z * Z / (2 * n)) / denom
    half = Z / denom * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n))
    return (max(0.0, centre - half), min(1.0, centre + half))


def summarise(rows):
    acc = directional_accuracy(rows)
    lk = likerts(rows)
    lo, hi = wilson(acc["correct"], acc["n_nontie"])
    return {
        "judge_accuracy": round(acc["accuracy"], 4),
        "acc_lo": round(lo, 4),
        "acc_hi": round(hi, 4),
        "gen_win_rate": round(acc["gen_win_rate"], 4),
        "likert_mean": round(statistics.mean(lk), 4) if lk else None,
        "win_rate_ge5": round(sum(1 for v in lk if v >= 5) / len(lk), 4) if lk else None,
        "n_nontie": acc["n_nontie"],
        "n_tie": acc["n_tie"],
        "n_parse_error": acc["n_parse_error"],
    }


def load_claude(root, cell):
    path = root / "raw" / GEN_KEY / "sweep" / cell / "on" / "reward" / "scores.jsonl"
    if not path.is_file():
        return None
    return [json.loads(line) for line in open(path) if line.strip()]


def published(pub_dir, cell):
    path = pub_dir / f"summary_{cell}.csv"
    if not path.is_file():
        return {}
    for row in csv.DictReader(open(path)):
        if row["checkpoint"] == GEN_KEY:
            return row
    return {}


def picked_human_map(rows):
    """key -> 1/0/None (None = tie or unparseable), for pairwise agreement."""
    return {key_of(r): _picked_human(r) for r in rows}


def rating_map(rows):
    return {key_of(r): _canonical_rating(r) for r in rows}


def pair_agreement(a_rows, b_rows, n_boot=10000, seed=0):
    """Opus vs Sonnet on the pairs both actually scored."""
    ra, rb = rating_map(a_rows), rating_map(b_rows)
    pa, pb = picked_human_map(a_rows), picked_human_map(b_rows)
    shared = sorted(set(ra) & set(rb))
    both_rated = [k for k in shared if ra[k] is not None and rb[k] is not None]
    both_called = [k for k in shared if pa[k] is not None and pb[k] is not None]

    exact = sum(1 for k in both_rated if ra[k] == rb[k])
    mad = (statistics.mean(abs(ra[k] - rb[k]) for k in both_rated)
           if both_rated else None)
    agree = sum(1 for k in both_called if pa[k] == pb[k])
    po = agree / len(both_called) if both_called else 0.0
    m_a = sum(pa[k] for k in both_called) / len(both_called) if both_called else 0.0
    m_b = sum(pb[k] for k in both_called) / len(both_called) if both_called else 0.0
    pe = m_a * m_b + (1 - m_a) * (1 - m_b)
    kappa = (po - pe) / (1 - pe) if pe < 1 else float("nan")

    # Paired bootstrap over pairs (not over judges): both judges saw identical
    # items, so resampling items keeps the pairing and removes item difficulty
    # from the difference.
    rng = random.Random(seed)
    diffs = []
    for _ in range(n_boot):
        draw = [shared[rng.randrange(len(shared))] for _ in shared]
        ca = [pa[k] for k in draw if pa[k] is not None]
        cb = [pb[k] for k in draw if pb[k] is not None]
        if ca and cb:
            diffs.append(sum(ca) / len(ca) - sum(cb) / len(cb))
    diffs.sort()
    ci = ((diffs[int(0.025 * len(diffs))], diffs[int(0.975 * len(diffs))])
          if diffs else (float("nan"), float("nan")))
    return {
        "n_shared": len(shared),
        "exact_match": exact / len(both_rated) if both_rated else None,
        "mean_abs_rating_diff": mad,
        "binary_agreement": po,
        "cohens_kappa": kappa,
        "acc_diff": m_a - m_b,
        "acc_diff_ci": ci,
    }


def cost_summary(rows):
    costs = [r["cost_usd"] for r in rows if r.get("cost_usd")]
    if not costs:
        return None
    return {"n": len(costs), "total": sum(costs), "mean": statistics.mean(costs)}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval_root", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--published", default=str(
        Path.home() / "Projects/turing-rl/results/2026-08-10-test-eval-9b-full5ep-full-schema"))
    ap.add_argument("--claude_cells", nargs="*", default=["claude-opus-5", "claude-sonnet-5"])
    ap.add_argument("--retest_of", default="claude-opus-5",
                    help="cell that --retest_cell is a second pass of")
    ap.add_argument("--retest_cell", default="claude-opus-5-retest",
                    help="second pass of the same pairs, for the noise floor")
    ap.add_argument("--out_stem", default="comparison_step320")
    a = ap.parse_args()

    root = Path(a.eval_root)
    pub_dir = Path(a.published)
    pairs = [json.loads(line) for line in open(a.pairs) if line.strip()]
    keys = {key_of(p) for p in pairs}

    table, claude_rows = [], {}

    # Incumbents, re-scored on the SAME 100 keys.
    for cell in INCUMBENTS:
        rows = []
        for rec in pairs:
            inc = rec["incumbent"][cell]
            rows.append({
                "user_id": rec["user_id"], "post_id": rec["post_id"],
                "target_idx": rec["target_idx"],
                "generated_is_b": rec["generated_is_b"],
                "human_side": "A" if rec["generated_is_b"] else "B",
                "rating_gt_first": inc.get("rating_gt_first"),
                "rating_gen_first": inc.get("rating_gen_first"),
                "turing_judge_score_raw": inc.get("turing_judge_score_raw"),
            })
        pub = published(pub_dir, cell)
        table.append({"judge": LABELS[cell], "cell": cell, "n_pairs": len(rows),
                      **summarise(rows),
                      "judge_accuracy_880": pub.get("judge_accuracy"),
                      "win_rate_ge5_880": pub.get("win_rate_ge5")})

    for cell in a.claude_cells:
        rows = load_claude(root, cell)
        if not rows:
            print(f"# skipping {cell}: no scores.jsonl", file=sys.stderr)
            continue
        rows = [r for r in rows if key_of(r) in keys]
        claude_rows[cell] = rows
        table.append({"judge": LABELS.get(cell, cell), "cell": cell, "n_pairs": len(rows),
                      **summarise(rows),
                      "judge_accuracy_880": "", "win_rate_ge5_880": ""})

    table.sort(key=lambda r: r["judge_accuracy"], reverse=True)

    cols = ["judge", "cell", "n_pairs", "judge_accuracy", "acc_lo", "acc_hi",
            "gen_win_rate", "likert_mean", "win_rate_ge5", "n_nontie", "n_tie",
            "n_parse_error", "judge_accuracy_880", "win_rate_ge5_880"]
    csv_path = root / f"{a.out_stem}.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(table)

    lines = [
        f"# Judge comparison on {len(pairs)} identical step-320 pairs",
        "",
        "`judge_accuracy` = fraction of non-tie pairs where the judge picked the "
        "REAL human turn. 0.5 = chance. Below 0.5 means the judge is reliably "
        "fooled: it prefers the generated turn. The step-0 anchor (same 9B judge, "
        "pre-RL init, 880 pairs) is 0.518 -- i.e. chance -- so the collapse at "
        "step 320 is produced by GRPO, not baked into the judge.",
        "",
        "Ranked by accuracy, best evaluator first. The `_880` columns are the "
        "published full-set numbers, for reference only.",
        "",
        "| " + " | ".join(cols) + " |",
        "|" + "|".join(["---"] * len(cols)) + "|",
    ]
    for r in table:
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")

    lines += ["", "## Is each judge distinguishable from chance?", ""]
    for r in table:
        verdict = ("BELOW chance" if r["acc_hi"] < 0.5 else
                   "ABOVE chance" if r["acc_lo"] > 0.5 else
                   "indistinguishable from chance")
        lines.append(f"- **{r['judge']}**: {r['judge_accuracy']:.3f} "
                     f"[{r['acc_lo']:.3f}, {r['acc_hi']:.3f}] -- {verdict}")

    if len(claude_rows) == 2:
        ca, cb = a.claude_cells[0], a.claude_cells[1]
        ag = pair_agreement(claude_rows[ca], claude_rows[cb])
        lines += [
            "", f"## {LABELS.get(ca, ca)} vs {LABELS.get(cb, cb)}", "",
            f"- pairs both scored: {ag['n_shared']}",
            f"- identical rating: {ag['exact_match']:.1%}",
            f"- mean |rating difference|: {ag['mean_abs_rating_diff']:.2f}",
            f"- agree on picked-the-human: {ag['binary_agreement']:.1%} "
            f"(Cohen's kappa {ag['cohens_kappa']:.3f})",
            f"- accuracy difference {ag['acc_diff']:+.3f}, paired bootstrap 95% CI "
            f"[{ag['acc_diff_ci'][0]:+.3f}, {ag['acc_diff_ci'][1]:+.3f}]"
            + ("  <- includes 0: not separated by 100 pairs"
               if ag['acc_diff_ci'][0] <= 0 <= ag['acc_diff_ci'][1] else ""),
        ]

    # Noise floor. Claude exposes no temperature or seed, so the same prompt can
    # come back with a different rating. Without this the Opus-Sonnet gap cannot
    # be read: a difference smaller than a judge's disagreement with ITSELF is
    # not a difference.
    retest = load_claude(root, a.retest_cell)
    base = claude_rows.get(a.retest_of)
    if retest and base:
        retest_keys = {key_of(r) for r in retest}
        ag = pair_agreement([r for r in base if key_of(r) in retest_keys], retest)
        lines += [
            "", f"## Noise floor: {LABELS.get(a.retest_of, a.retest_of)} scored twice", "",
            f"- pairs rescored: {ag['n_shared']}",
            f"- identical rating: {ag['exact_match']:.1%}",
            f"- mean |rating difference|: {ag['mean_abs_rating_diff']:.2f}",
            f"- agrees with itself on picked-the-human: {ag['binary_agreement']:.1%}",
            f"- accuracy difference between the two passes: {ag['acc_diff']:+.3f}",
            "",
            "Read the Opus-Sonnet gap against this: a between-model difference "
            "smaller than a model's disagreement with itself is not a result.",
        ]

    lines += ["", "## Measured cost", ""]
    for cell, rows in claude_rows.items():
        c = cost_summary(rows)
        if c:
            lines.append(f"- {LABELS.get(cell, cell)}: ${c['total']:.2f} over "
                         f"{c['n']} calls (${c['mean']:.4f}/call)")

    # What fools the frontier judge, in its own words -- the most actionable
    # output of the run if it turns out to be fooled too.
    for cell, rows in claude_rows.items():
        fooled = [r for r in rows if _picked_human(r) == 0 and r.get("judge_reasoning")]
        if not fooled:
            continue
        lines += ["", f"## {LABELS.get(cell, cell)}: 3 pairs where it picked the "
                      f"generation over the human", ""]
        for r in fooled[:3]:
            reason = " ".join(str(r["judge_reasoning"]).split())[:600]
            lines.append(f"- `{key_of(r)}` rating={_canonical_rating(r)} "
                         f"score_gap={r.get('score_gap')}\n  > {reason}")

    md_path = root / f"{a.out_stem}.md"
    md_path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {csv_path}\nwrote {md_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
