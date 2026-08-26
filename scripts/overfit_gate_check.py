# scripts/overfit_gate_check.py
"""Overfit-gate metric: win-rate of the generated turn on the (few) training prompts.

`likert` per row is the judge's 1-7 rating oriented so higher = judge thinks the GENERATED
turn is more human (reward.py's `turing_judge_score_raw`). pick=gen if likert>=5; ties (4) excluded.
Gate passes when wins >= pass_wins (default 8, i.e. 8/10)."""
from __future__ import annotations
import argparse, glob, json, os

def win_rate_from_rows(rows, pass_wins: int = 8) -> dict:
    # Tolerate both {"likert": x} rows and full reward-dump rows ({"turing_judge_score_raw": x}).
    def _lk(r):
        return r["likert"] if r.get("likert") is not None else r.get("turing_judge_score_raw")
    likerts = [float(_lk(r)) for r in rows if _lk(r) is not None]
    nontie = [x for x in likerts if int(round(x)) != 4]
    wins = sum(1 for x in nontie if x >= 5)
    n = len(nontie)
    return {"n_total": len(likerts), "n_nontie": n, "wins": wins,
            "win_rate": (wins / n) if n else 0.0, "passed": wins >= pass_wins}

def prompt_key(row) -> tuple:
    # stable per-prompt identity; fall back through available fields
    return (row.get("user_id"), row.get("post_id"), row.get("target_idx"))

def prompt_level_gate(rows, pass_prompts: int = 8, group_size: int = 4) -> dict:
    """Per-prompt overfit gate. Groups rows by prompt_key; for each prompt takes its LATEST
    `group_size` rollouts by ts (final-epoch proxy); the prompt 'wins' if a STRICT majority of those
    non-tie rollouts have Likert>=5 (frac>0.5; an even split frac==0.5 is a TIE, not a win). ties=4
    and parse-fails (0/None) are excluded from the fraction. Gate passes when #winning prompts >= pass_prompts.
    Returns: {n_prompts, won, pass_prompts, passed, final_win_rate (overall over the selected
    final rollouts), per_prompt: [{key, n, wins, frac, won}]}.
    """
    groups: dict = {}
    for r in rows:
        groups.setdefault(prompt_key(r), []).append(r)

    per_prompt = []
    won_count = 0
    total_wins = 0
    total_nontie = 0
    for key, grp in groups.items():
        # sort by ts ascending; take last `group_size` as the "final epoch" rollouts
        grp_sorted = sorted(grp, key=lambda r: (r.get("ts") is None, r.get("ts")))
        final = grp_sorted[-group_size:] if group_size and group_size > 0 else grp_sorted
        wins = 0
        n_nontie = 0
        for r in final:
            lk = r.get("turing_judge_score_raw")
            if lk is None:
                continue
            lk_i = int(round(float(lk)))
            if lk_i == 0 or lk_i == 4:  # parse-fail sentinel / tie -> excluded
                continue
            n_nontie += 1
            if lk >= 5:
                wins += 1
        frac = (wins / n_nontie) if n_nontie else 0.0
        # strict majority: an even split among non-tie rollouts (frac==0.5) is a TIE, not a win.
        won = (n_nontie > 0) and (frac > 0.5)
        if won:
            won_count += 1
        total_wins += wins
        total_nontie += n_nontie
        per_prompt.append({"key": key, "n": n_nontie, "wins": wins, "frac": frac, "won": won})

    return {
        "n_prompts": len(groups),
        "won": won_count,
        "pass_prompts": pass_prompts,
        "passed": won_count >= pass_prompts,
        "final_win_rate": (total_wins / total_nontie) if total_nontie else 0.0,
        "per_prompt": per_prompt,
    }

def _load_reward_dump(dump_dir: str) -> list[dict]:
    rows = []
    for f in glob.glob(os.path.join(dump_dir, "reward-*.jsonl")):
        for line in open(f):
            try: d = json.loads(line)
            except Exception: continue
            # keep FULL row so prompt_key/ts are available; oriented Likert is turing_judge_score_raw
            if d.get("turing_judge_score_raw") is not None:
                rows.append(d)
    return rows

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump_dir", required=True)
    ap.add_argument("--pass_wins", type=int, default=8)
    ap.add_argument("--pass_prompts", type=int, default=8)
    ap.add_argument("--group_size", type=int, default=4)
    a = ap.parse_args()
    rows = _load_reward_dump(a.dump_dir)
    overall = win_rate_from_rows(rows, pass_wins=a.pass_wins)
    gate = prompt_level_gate(rows, pass_prompts=a.pass_prompts, group_size=a.group_size)
    print(json.dumps({"overall": overall, "prompt_level_gate": gate}, indent=2))
    # authoritative gate is the per-prompt final-epoch gate
    raise SystemExit(0 if gate["passed"] else 1)
