# scripts/overfit_gate_check.py
"""Overfit-gate metric: win-rate of the generated turn on the (few) training prompts.

`likert` per row is the judge's 1-7 rating oriented so higher = judge thinks the GENERATED
turn is more human (reward.py's `turing_judge_score_raw`). pick=gen if likert>=5; ties (4) excluded.
Gate passes when wins >= pass_wins (default 8, i.e. 8/10)."""
from __future__ import annotations
import argparse, glob, json, os

def win_rate_from_rows(rows, pass_wins: int = 8) -> dict:
    likerts = [float(r["likert"]) for r in rows if r.get("likert") is not None]
    nontie = [x for x in likerts if int(round(x)) != 4]
    wins = sum(1 for x in nontie if x >= 5)
    n = len(nontie)
    return {"n_total": len(likerts), "n_nontie": n, "wins": wins,
            "win_rate": (wins / n) if n else 0.0, "passed": wins >= pass_wins}

def _load_reward_dump(dump_dir: str) -> list[dict]:
    rows = []
    for f in glob.glob(os.path.join(dump_dir, "reward-*.jsonl")):
        for line in open(f):
            try: d = json.loads(line)
            except Exception: continue
            # oriented Likert: reward.py dumps turing_judge_score_raw
            lk = d.get("turing_judge_score_raw")
            if lk is not None:
                rows.append({"likert": lk})
    return rows

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump_dir", required=True)
    ap.add_argument("--pass_wins", type=int, default=8)
    a = ap.parse_args()
    res = win_rate_from_rows(_load_reward_dump(a.dump_dir), pass_wins=a.pass_wins)
    print(json.dumps(res, indent=2))
    raise SystemExit(0 if res["passed"] else 1)
