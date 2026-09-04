"""Verify a judge cell actually scored every held-out pair exactly once.

WHY A ROW COUNT IS NOT ENOUGH
-----------------------------
Two independent failure modes make "≈880 rows" meaningless:

1. Silent under-completion. ``run_judge_sweep_cell.py`` catches per-pair exceptions,
   increments an ``err`` counter, prints, and continues -- the shard still exits 0. A cell
   can score 800/880 and look successful to Slurm.
2. Stale accumulation. Reward dumps are per-worker ``reward-<jobid>-<pid>.jsonl`` files that
   ACCUMULATE in a reused directory. A re-run mixes old rows with new, so a directory can
   hold 880 rows that are really 700 fresh + 180 stale duplicates.

So we check the *set* of unique ``(user_id, post_id, target_idx)`` keys against the pair-set,
and require every dump file to come from a single Slurm job id (freshness).

Usage:
  python scripts/verify_judge_completeness.py --eval_root results/2026-08-03-test-eval-9b-half
  python scripts/verify_judge_completeness.py --reward_dir <...>/reward --pairs <...>_880.parquet
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

JOB_RE = re.compile(r"reward-(\d+)-\d+\.jsonl$")
KEY_FIELDS = ("user_id", "post_id", "target_idx")


from scripts.eval_rl_generator import find_reward_dirs, gen_key_of

def _key(row: dict) -> tuple:
    return tuple(str(row.get(f, "")) for f in KEY_FIELDS)


def load_rows(reward_dir: Path) -> tuple[list[dict], set[str]]:
    rows: list[dict] = []
    job_ids: set[str] = set()
    for jl in sorted(reward_dir.rglob("reward-*.jsonl")):
        m = JOB_RE.search(jl.name)
        if m:
            job_ids.add(m.group(1))
        for line in jl.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows, job_ids


def expected_keys(pairs_path: Path) -> set[tuple]:
    import pandas as pd

    df = pd.read_parquet(pairs_path)
    return {tuple(str(r[f]) for f in KEY_FIELDS) for _, r in df.iterrows()}


def check(reward_dir: Path, pairs_path: Path, allow_multi_job: bool,
          write_missing: Path | None = None, max_missing_frac: float = 0.0) -> list[str]:
    problems: list[str] = []
    if not reward_dir.is_dir():
        return [f"{reward_dir}: reward dir does not exist"]

    rows, job_ids = load_rows(reward_dir)
    want = expected_keys(pairs_path)
    got = Counter(_key(r) for r in rows)

    missing = want - set(got)
    extra = set(got) - want
    dupes = {k: n for k, n in got.items() if n > 1}

    label = f"{reward_dir.parent.parent.name}/{reward_dir.parent.name}"
    print(f"[{label}] rows={len(rows)} unique={len(got)} expected={len(want)} "
          f"missing={len(missing)} extra={len(extra)} duplicated={len(dupes)} job_ids={sorted(job_ids)}")

    if len(job_ids) > 1 and not allow_multi_job:
        problems.append(
            f"{label}: reward dumps span {len(job_ids)} Slurm jobs {sorted(job_ids)} -- stale rows from a "
            "previous run are mixed in. Use a fresh output dir (or pass --allow_multi_job if intended)."
        )
    if missing:
        frac = len(missing) / len(want) if want else 1.0
        msg = (f"{label}: {len(missing)} pairs never scored ({frac:.1%}), "
               f"e.g. {sorted(missing)[:3]}")
        if frac <= max_missing_frac:
            # Tolerated: judge timeouts run ~1-3% and blocking on every straggler stalls the
            # pipeline. Comparability is preserved downstream -- summarize_test_eval.py scores
            # every checkpoint on the pairs they ALL have.
            print(f"[{label}] WARN tolerated: {msg}")
        else:
            problems.append(msg + f" -- above --max_missing_frac={max_missing_frac:.1%}")
        if write_missing is not None:
            import pandas as pd

            df = pd.read_parquet(pairs_path)
            keep = df.apply(lambda r: tuple(str(r[f]) for f in KEY_FIELDS) in missing, axis=1)
            out = write_missing / f"{pairs_path.stem}_missing.parquet"
            out.parent.mkdir(parents=True, exist_ok=True)
            df[keep].to_parquet(out)
            print(f"[{label}] wrote {int(keep.sum())} missing pairs -> {out}")
    if extra:
        problems.append(f"{label}: {len(extra)} scored rows are not in the pair-set, e.g. {sorted(extra)[:3]}")
    if dupes:
        problems.append(f"{label}: {len(dupes)} pairs scored more than once, e.g. {list(dupes)[:3]}")

    # Judge-side quality signal (not fatal on its own, but surfaced).
    parse_fail = sum(
        1 for r in rows
        if r.get("rating_gt_first") is None and r.get("rating_gen_first") is None
    )
    if parse_fail:
        print(f"[{label}] NOTE: {parse_fail} rows have no valid rating (parse failures)")
    return problems


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval_root", default=None, help="Scan <root>/raw/<gen>/sweep/<cell>/<mode>/reward")
    ap.add_argument("--reward_dir", default=None, help="Check a single reward dir")
    ap.add_argument("--pairs", default=None, help="Pair-set parquet (required with --reward_dir)")
    ap.add_argument("--expect_pairs", type=int, default=880)
    ap.add_argument("--pairs_tag", default="880",
                    help="Row-count tag in the pair-set filename (gen_<key>_<tag>.parquet). Must "
                         "match what launch_test_eval.sh used, or --eval_root finds no pair-sets.")
    ap.add_argument("--allow_multi_job", action="store_true",
                    help="Permit dumps spanning several jobs (legitimate after a targeted re-run "
                         "of timed-out pairs; the unique-key check still guards correctness)")
    ap.add_argument("--write_missing", default=None,
                    help="Directory to write <pairs>_missing.parquet subsets for a targeted re-judge")
    ap.add_argument("--max_missing_frac", type=float, default=0.0,
                    help="Tolerate this fraction of unscored pairs per cell (e.g. 0.03). Judge "
                         "timeouts run ~1-3%%; summarize_test_eval.py then scores every checkpoint "
                         "on the pairs they ALL have, so comparability is preserved.")
    a = ap.parse_args()

    checks: list[tuple[Path, Path]] = []
    unpaired: list[str] = []
    if a.reward_dir:
        if not a.pairs:
            ap.error("--pairs is required with --reward_dir")
        checks.append((Path(a.reward_dir), Path(a.pairs)))
    elif a.eval_root:
        root = Path(a.eval_root)
        for reward_dir in find_reward_dirs(root):
            gen_key = gen_key_of(reward_dir)
            pairs = root / "raw" / "pairs" / f"gen_{gen_key}_{a.pairs_tag}.parquet"
            if not pairs.exists():
                # NOT a silent skip. A scored cell with no matching pair-set means either the
                # wrong --pairs_tag (so this run verified a subset of what exists and would still
                # print PASS) or a half-built tree. Both must fail loudly -- silently checking
                # fewer cells than exist is the exact blind spot this script was written for.
                unpaired.append(f"{reward_dir}: no pair-set at {pairs}")
                continue
            checks.append((reward_dir, pairs))
    else:
        ap.error("provide --eval_root or --reward_dir")

    if not checks and not unpaired:
        raise SystemExit("FAIL: nothing to verify (no reward dirs found)")

    problems: list[str] = list(unpaired)
    for reward_dir, pairs in checks:
        import pandas as pd

        n_pairs = len(pd.read_parquet(pairs))
        if n_pairs != a.expect_pairs:
            problems.append(f"{pairs}: pair-set has {n_pairs} rows, expected {a.expect_pairs}")
        problems.extend(check(reward_dir, pairs, a.allow_multi_job,
                              Path(a.write_missing) if a.write_missing else None,
                              a.max_missing_frac))

    if problems:
        print(f"\nFAILED ({len(problems)}):", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        raise SystemExit(1)
    print(f"\nPASS: {len(checks)} cell(s) scored the expected pairs, no duplicates or strays"
          f"{' (tolerating <=' + format(a.max_missing_frac, '.1%') + ' unscored)' if a.max_missing_frac else ''}.")


if __name__ == "__main__":
    main()
