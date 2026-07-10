"""Aggregate 50-pair calibration cells into a per-cell throughput report."""
import json
from pathlib import Path
ROOT = Path("/storage/home/lancewicki/projects/turing-rl/results/2026-07-08-judge-sweep")

def extrapolate_wall_hours(n_calls, wall_s, total_calls=1760):
    return (total_calls / (n_calls / wall_s)) / 3600 if wall_s > 0 and n_calls else float("inf")

def main():
    rows = []
    for meta_path in (ROOT / "raw" / "sweep").glob("*/*/run_metadata.json"):
        m = json.loads(meta_path.read_text())
        cell, mode = meta_path.parent.parent.name, meta_path.parent.name
        n = m.get("n_pairs", 0); wall = m.get("wall_seconds", 0.0); calls = n * 2
        req_s = calls / wall if wall > 0 else 0.0
        rows.append({"cell": cell, "mode": mode, "n_pairs": n, "wall_s": wall,
                     "req_per_s": req_s, "proj_1760_h": extrapolate_wall_hours(calls, wall)})
    (ROOT / "raw" / "calibration").mkdir(parents=True, exist_ok=True)
    (ROOT / "raw" / "calibration" / "calibration_metadata.json").write_text(json.dumps(rows, indent=2))
    out = ROOT / "derived" / "calibration_report.md"; out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        f.write("# Per-cell throughput calibration\n\n")
        f.write("_Precision caveat: 100 calls/cell → extrapolations are ±30%; use only for the >4h gate._\n\n")
        f.write("| Cell | Mode | Pairs | Wall(s) | Req/s | Proj 1760-call | >4h? |\n|---|---|---|---|---|---|---|\n")
        for r in sorted(rows, key=lambda x: (x["cell"], x["mode"])):
            gate = "**YES**" if r["proj_1760_h"] > 4 else "no"
            f.write(f"| {r['cell']} | {r['mode']} | {r['n_pairs']} | {r['wall_s']:.0f} "
                    f"| {r['req_per_s']:.2f} | {r['proj_1760_h']:.1f}h | {gate} |\n")
    print("wrote", out)

if __name__ == "__main__":
    main()
