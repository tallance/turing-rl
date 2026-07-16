"""Generator-sweep analyzer.

(A) Per generator, run the existing analyzers on that generator's standard sweep subtree
    (raw/<gen>/sweep/<judgecell>/<mode>) -> the full per-model plot set in derived/<gen>/.
(B) Read each derived/<gen>/summary.parquet and draw cross-generator comparison plots
    (one line per generator) in derived/compare/plots/.
"""
from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO_ROOT))
import pandas as pd
from configs.judge_sweep_cells import SIZE_MAP

PY = sys.executable
GEN_LABELS = {
    "qwen3-8b-base": "qwen3-8B base", "qwen3-8b-sft": "qwen3-8B SFT",
    "qwen35-9b-base": "qwen3.5-9B base", "qwen35-9b-sft": "qwen3.5-9B SFT",
}
GEN_ORDER = ["qwen3-8b-base", "qwen3-8b-sft", "qwen35-9b-base", "qwen35-9b-sft"]
# Comparison metrics (must be columns in analyze_judge_sweep's summary.parquet).
CMP_METRICS = [
    ("accuracy", "accuracy | parse ok (picks true human)", (0.45, 0.85), 0.5),
    ("accuracy_penalized", "accuracy (parse-fail counted wrong)", (0.45, 0.85), 0.5),
    ("parse_error_rate", "parse-error rate", None, None),
    ("tie_rate", "tie rate (rating==4)", None, None),
]


def discover_generators(raw_root: Path) -> list[str]:
    """Sub-dirs of raw_root that contain a sweep/ dir (i.e. a generator subtree)."""
    return sorted(p.name for p in raw_root.iterdir()
                  if p.is_dir() and (p / "sweep").is_dir())


def comparison_rows(derived_root: Path, generators: list[str]) -> list[dict]:
    """Flatten each generator's summary.parquet into rows tagged by generator + judge."""
    rows: list[dict] = []
    for gen in generators:
        pq = derived_root / gen / "summary.parquet"
        if not pq.exists():
            print(f"[gen-analyzer] no summary for {gen} (skipping in comparison)", flush=True)
            continue
        for rec in pd.read_parquet(pq).to_dict("records"):
            rec = dict(rec); rec["generator"] = gen; rec["judge"] = rec.get("cell")
            rows.append(rec)
    return rows


def run_per_generator_analyzers(raw_root: Path, derived_root: Path, gen: str) -> None:
    """Full per-model plot set for one generator via the existing two analyzers."""
    graw, gderived = raw_root / gen, derived_root / gen
    subprocess.run([PY, "scripts/analyze_judge_sweep.py",
                    "--raw_root", str(graw), "--derived_root", str(gderived)],
                   cwd=REPO_ROOT, check=True)
    subprocess.run([PY, "scripts/plot_field_compliance.py",
                    "--raw_root", str(graw), "--out_dir", str(gderived / "plots")],
                   cwd=REPO_ROOT, check=True)


def write_comparison_plots(rows: list[dict], out_dir: Path) -> None:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    out_dir.mkdir(parents=True, exist_ok=True)
    by = {(r["generator"], r["judge"], r["mode"]): r for r in rows}
    for mode in ("off", "on"):
        for metric, ylab, ylim, ref in CMP_METRICS:
            fig, ax = plt.subplots(figsize=(7, 5))
            for gen in GEN_ORDER:
                pts = [(SIZE_MAP[j], by[(gen, j, mode)].get(metric))
                       for (g, j, m) in by if g == gen and m == mode
                       and j in SIZE_MAP and by[(gen, j, mode)].get(metric) is not None]
                if not pts:
                    continue
                pts.sort(); xs, ys = zip(*pts)
                ax.plot(xs, ys, marker="o", label=GEN_LABELS.get(gen, gen))
            if ref is not None:
                ax.axhline(ref, ls="--", c="gray", lw=1)
            if ylim:
                ax.set_ylim(*ylim)
            ax.set_xscale("log"); ax.set_xlabel("judge active-params (B)")
            ax.set_ylabel(ylab); ax.set_title(f"{metric} — thinking {mode} (per generator)")
            ax.legend(); fig.tight_layout()
            fig.savefig(out_dir / f"cmp_{metric}_{mode}.png", dpi=130); plt.close(fig)


def main() -> None:
    base = REPO_ROOT / "results" / "2026-07-15-generator-sweep"
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_root", type=Path, default=base / "raw")
    ap.add_argument("--derived_root", type=Path, default=base / "derived")
    args = ap.parse_args()

    gens = discover_generators(args.raw_root)
    print(f"[gen-analyzer] generators: {gens}", flush=True)
    for gen in gens:                              # (A) full per-model plot set
        run_per_generator_analyzers(args.raw_root, args.derived_root, gen)
    rows = comparison_rows(args.derived_root, gens)   # (B) cross-generator comparison
    write_comparison_plots(rows, args.derived_root / "compare" / "plots")
    pd.DataFrame(rows).to_parquet(args.derived_root / "compare" / "comparison_summary.parquet",
                                  index=False)
    print(f"[gen-analyzer] per-gen plots in derived/<gen>/plots; comparison in "
          f"derived/compare/plots ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
