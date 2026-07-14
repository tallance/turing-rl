"""Per-field format-compliance of the Turing judge's JSON output, from raw dumps.

For every judge call (raw judge_raw_content) parse the JSON and check, per field,
whether it was produced in a valid format/range (scores/penalties in [0,1], base &
response in [0,3], score_gap in [-3,3], rating int 1-7, reasoning nonempty).
Produces two figures:
  plots/field_compliance.png        valid-format % per field, pooled over all cells
  plots/rubric_complete_by_model.png  per model (off vs on): all-6-scores valid % + rating valid %

Note: the reward's own extractor is more tolerant than this strict parse, so these
rates are a LOWER BOUND on effective field capture (rating_gt_first was ~100% non-null).
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO_ROOT))
import numpy as np
from configs.judge_sweep_cells import SIZE_MAP

PLOT_ORDER = ["qwen35-4b", "qwen35-9b", "qwen35-27b", "qwen35-35b-a3b",
              "qwen35-122b", "qwen35-397b", "qwen3-8b"]
LABELS = {"qwen35-4b": "3.5-4B", "qwen35-9b": "3.5-9B", "qwen35-27b": "3.5-27B",
          "qwen35-35b-a3b": "3.5-35B-A3B", "qwen35-122b": "3.5-122B", "qwen35-397b": "3.5-397B",
          "qwen3-8b": "qwen3-8B"}

DIM = ["immediate_target_score", "human_goal_score", "communication_style_score"]
PEN = ["source_copy_penalty", "assistant_like_penalty", "wrong_target_or_role_penalty",
       "unsupported_adversarial_reframing_penalty"]
# field -> (min, max) numeric range; rating handled specially
UNIT = [f"{d}_{s}" for d in DIM for s in ("a", "b")] + [f"{p}_{s}" for p in PEN for s in ("a", "b")]
FIELD_RANGE = {**{f: (0.0, 1.0) for f in UNIT},
               "base_score_a": (0.0, 3.0), "base_score_b": (0.0, 3.0),
               "response_a_score": (0.0, 3.0), "response_b_score": (0.0, 3.0),
               "score_gap": (-3.0, 3.0)}
FIELD_ORDER = UNIT + ["base_score_a", "base_score_b", "response_a_score", "response_b_score",
                      "score_gap", "rating", "reasoning"]


def parse_json(text: str):
    if not text:
        return None
    s = text.strip()
    a, b = s.find("{"), s.rfind("}")
    if a < 0 or b <= a:
        return None
    for cand in (s, s[a:b + 1]):
        try:
            d = json.loads(cand)
            return d if isinstance(d, dict) else None
        except json.JSONDecodeError:
            continue
    return None


def _num_ok(v, lo, hi):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and lo <= float(v) <= hi


def field_valid(data: dict, field: str) -> bool:
    if data is None or field not in data:
        return False
    if field == "rating":
        v = data["rating"]
        return _num_ok(v, 1, 7) and float(v) == int(v)
    if field == "reasoning":
        return isinstance(data["reasoning"], str) and len(data["reasoning"].strip()) > 0
    lo, hi = FIELD_RANGE[field]
    return _num_ok(data[field], lo, hi)


def main() -> None:
    ap = argparse.ArgumentParser()
    base = REPO_ROOT / "results" / "2026-07-08-judge-sweep"
    ap.add_argument("--raw_root", type=Path, default=base / "raw")
    ap.add_argument("--out_dir", type=Path, default=base / "derived" / "plots")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # counts[field] -> [valid, total]; per_cell[(cell,mode)] -> dict
    pooled = {f: [0, 0] for f in FIELD_ORDER}
    per_cell = {}
    for cell in PLOT_ORDER:
        for mode in ("off", "on"):
            mdir = args.raw_root / "sweep" / cell / mode / "reward"
            if not mdir.is_dir():
                continue
            v6 = vr = n = 0
            cf = {f: [0, 0] for f in FIELD_ORDER}  # per-(cell,mode) per-field valid/total
            for jl in sorted(mdir.glob("*.jsonl")):
                for line in jl.read_text().splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    n += 1
                    d = parse_json(json.loads(line).get("judge_raw_content") or "")
                    for f in FIELD_ORDER:
                        pooled[f][1] += 1
                        cf[f][1] += 1
                        if field_valid(d, f):
                            pooled[f][0] += 1
                            cf[f][0] += 1
                    if all(field_valid(d, f"{dd}_{s}") for dd in DIM for s in ("a", "b")):
                        v6 += 1
                    if field_valid(d, "rating"):
                        vr += 1
            if n:
                per_cell[(cell, mode)] = {"n": n, "all6": v6 / n, "rating": vr / n, "cf": cf}

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Figure 1: per-field valid % pooled
    fields = [f for f in FIELD_ORDER if pooled[f][1]]
    rates = [100 * pooled[f][0] / pooled[f][1] for f in fields]
    fig, ax = plt.subplots(figsize=(9, 8))
    y = np.arange(len(fields))
    ax.barh(y, rates, color="#4C78A8")
    for i, r in enumerate(rates):
        ax.text(r + 0.5, i, f"{r:.0f}", va="center", fontsize=8)
    ax.set_yticks(y); ax.set_yticklabels(fields, fontsize=8); ax.invert_yaxis()
    ax.set_xlabel("valid-format rate (%) — pooled over all 7 models x 2 modes")
    ax.set_xlim(0, 105); ax.set_title("Judge output: per-field valid-format rate (strict raw parse)")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout(); fig.savefig(args.out_dir / "field_compliance.png", dpi=130); plt.close(fig)

    # Figure 2: per-model all-6-scores valid % + rating valid %, off vs on
    cells = [c for c in PLOT_ORDER if any(k[0] == c for k in per_cell)]
    x = np.arange(len(cells)); w = 0.2
    fig, ax = plt.subplots(figsize=(11, 5.5))
    series = [("off", "all6", -1.5 * w, "#4C78A8", "6 scores valid (off)"),
              ("on", "all6", -0.5 * w, "#9ecae1", "6 scores valid (on)"),
              ("off", "rating", 0.5 * w, "#F58518", "rating valid (off)"),
              ("on", "rating", 1.5 * w, "#fdd0a2", "rating valid (on)")]
    for mode, key, off, color, label in series:
        vals = [100 * per_cell.get((c, mode), {}).get(key, np.nan) for c in cells]
        ax.bar(x + off, vals, w, color=color, label=label)
    ax.set_xticks(x); ax.set_xticklabels([LABELS.get(c, c) for c in cells], rotation=20, ha="right")
    ax.set_ylabel("valid-format rate (%)"); ax.set_ylim(0, 105)
    ax.set_title("Rubric (6 scores) vs rating validity by model (strict raw parse; lower bound)")
    ax.legend(fontsize=8, ncol=2); ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(args.out_dir / "rubric_complete_by_model.png", dpi=130); plt.close(fig)

    # Figure 3: one subplot per field, each a grouped bar (model on x, off vs on)
    fields = [f for f in FIELD_ORDER if pooled[f][1]]
    ncol = 4; nrow = (len(fields) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 2.6 * nrow), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).flatten()
    xb = np.arange(len(cells)); wb = 0.38

    def _rate(cell, mode, field):
        cf = per_cell.get((cell, mode), {}).get("cf")
        if not cf or cf[field][1] == 0:
            return np.nan
        return 100 * cf[field][0] / cf[field][1]

    for i, f in enumerate(fields):
        ax = axes[i]
        off = [_rate(c, "off", f) for c in cells]
        on = [_rate(c, "on", f) for c in cells]
        ax.bar(xb - wb / 2, off, wb, color="#4C78A8", label="off")
        ax.bar(xb + wb / 2, on, wb, color="#F58518", label="on")
        ax.set_title(f, fontsize=8)
        ax.set_ylim(0, 105); ax.grid(True, axis="y", alpha=0.3)
        ax.set_xticks(xb); ax.set_xticklabels([LABELS.get(c, c) for c in cells], rotation=90, fontsize=6)
    for j in range(len(fields), len(axes)):
        axes[j].axis("off")
    axes[0].legend(fontsize=7)
    fig.suptitle("Per-field valid-format rate by model — thinking off vs on (strict raw parse; lower bound)")
    fig.supylabel("valid-format rate (%)")
    fig.tight_layout(); fig.savefig(args.out_dir / "field_compliance_by_model.png", dpi=130); plt.close(fig)
    print("wrote field_compliance.png + rubric_complete_by_model.png + field_compliance_by_model.png")


if __name__ == "__main__":
    main()
