"""Compute derived metrics from raw reward dumps. Idempotent: delete derived/ and re-run.
Reads raw/sweep/<cell>/<mode>/reward/*.jsonl -> derived/{summary.md,summary.parquet,per_pair_metrics.parquet,plots/}.

Reward-dump schema reality (see post-plans): the reward path scores ONE randomized
ordering per pair (position randomized ACROSS pairs, not both orderings per pair). So
each jsonl row = one pair = one judge call, with exactly one of rating_gt_first /
rating_gen_first set and `human_side` marking which side (A/B) is the real human.
Because each cell randomizes independently, raw "picks A/B" is NOT comparable across
cells; the order-invariant decision is "picked_human" (judge chose the true human).
Accuracy and kappa are both built on picked_human.
"""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO_ROOT))
import numpy as np, pandas as pd
from configs.judge_sweep_cells import SIZE_MAP, ANCHOR_CELL   # single source of truth

RATING_RE = re.compile(r'"rating"\s*:\s*(\d+)')
METRICS_FOR_PLOT = ("accuracy", "format_ok_rate", "budget_hit_rate", "position_bias_delta")


def _parse_rating_from_text(text: str) -> int | None:
    if not text:
        return None
    m = RATING_RE.search(text)
    if not m:
        return None
    r = int(m.group(1))
    return r if 1 <= r <= 7 else None


def load_cell_rows(mode_dir: Path) -> list[dict]:
    rows = []
    for jl in sorted((mode_dir / "reward").glob("*.jsonl")):
        for line in jl.read_text().splitlines():
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def per_call_features(row: dict) -> dict:
    # Canonical rating: the pre-parsed field for whichever ordering ran; fall back
    # to parsing the raw judge text (unit-test path / older dumps).
    rating = row.get("rating_gt_first")
    if rating is None:
        rating = row.get("rating_gen_first")
    from_text = False
    if rating is None:
        rating = _parse_rating_from_text(row.get("judge_raw_content") or "")
        from_text = rating is not None
    rating = int(rating) if isinstance(rating, (int, float)) and rating is not None else None

    generated_is_b = bool(row.get("generated_is_b"))
    # human_side and generated_is_b are complementary: human is A iff generator is B.
    human_side = row.get("human_side") or ("A" if generated_is_b else "B")

    # judge's order-invariant choice: rating<4 -> picks A, >4 -> picks B, ==4 -> tie
    if rating is None or rating == 4:
        picked_human = None  # abstain / unparseable -> excluded from acc & kappa
    else:
        pick = "A" if rating < 4 else "B"
        picked_human = int(pick == human_side)

    return {
        "pair_id": row.get("pair_id") or f'{row.get("user_id")}::{row.get("post_id")}::{row.get("target_idx")}',
        "user_id": row.get("user_id"),
        "generated_is_b": generated_is_b,
        "human_side": human_side,
        "randomized_order": row.get("randomized_order"),
        "rating": rating,
        "format_ok": rating is not None,
        "rating_recovered_from_text": from_text,
        "picked_human": picked_human,
        "budget_hit": (row.get("judge_finish_reason") or "") == "length",
    }


def accuracy(calls: list[dict]) -> float | None:
    """Fraction of non-tie, parsed calls where the judge picked the true human side."""
    vals = [c["picked_human"] for c in calls if c["picked_human"] is not None]
    return float(np.mean(vals)) if vals else None


def _acc_subset(df: pd.DataFrame) -> float | None:
    v = df["picked_human"].dropna()
    return float(v.mean()) if len(v) else None


def aggregate_cell(cell: str, mode: str, calls: list[dict]):
    df = pd.DataFrame(calls)
    if df.empty:
        return {"cell": cell, "mode": mode, "n_calls": 0}, pd.DataFrame()
    ratings = df["rating"].dropna().astype(int).tolist()
    hist = {str(k): ratings.count(k) for k in range(1, 8)}
    acc = _acc_subset(df)
    # Position bias: does accuracy depend on which side the human is on? (A vs B)
    acc_a = _acc_subset(df[df["human_side"] == "A"])
    acc_b = _acc_subset(df[df["human_side"] == "B"])
    pos_bias = abs(acc_a - acc_b) if (acc_a is not None and acc_b is not None) else float("nan")
    summ = {
        "cell": cell, "mode": mode, "n_calls": len(df),
        "format_ok_rate": float(df["format_ok"].mean()),
        "rating_recovery_rate": float(df["rating_recovered_from_text"].mean()),
        "budget_hit_rate": float(df["budget_hit"].mean()),
        "tie_rate": float((df["rating"] == 4).mean()),
        "n_scored": int(df["picked_human"].notna().sum()),
        "accuracy": acc if acc is not None else 0.0,
        "position_bias_delta": pos_bias,
        "rating_mean": float(np.mean(ratings)) if ratings else 0.0,
        "rating_mode": int(max(hist, key=lambda k: hist[k])) if ratings else 0,
        "rating_histogram": hist,
    }
    # per-pair frame for kappa: one picked_human per pair (drop ties/unparsed)
    pdf = df[["pair_id", "picked_human"]].copy()
    pdf["cell"] = cell
    pdf["mode"] = mode
    return summ, pdf


def cohen_kappa_binary(xs: list[int], ys: list[int]) -> float:
    """Cohen's kappa for two binary raters. No sklearn dependency."""
    n = len(xs)
    if n == 0:
        return float("nan")
    xs = np.asarray(xs); ys = np.asarray(ys)
    po = float(np.mean(xs == ys))
    px1 = float(np.mean(xs)); py1 = float(np.mean(ys))
    pe = px1 * py1 + (1 - px1) * (1 - py1)
    return (po - pe) / (1 - pe) if pe != 1.0 else float("nan")


def compute_kappa_vs_anchor(pair_dfs, anchor_cell):
    out = {}
    for mode in ("off", "on"):
        adf = pair_dfs.get((anchor_cell, mode))
        if adf is None or adf.empty:
            continue
        amap = {r["pair_id"]: int(r["picked_human"])
                for _, r in adf.iterrows() if pd.notna(r["picked_human"])}
        for (cell, m), df in pair_dfs.items():
            if m != mode or cell == anchor_cell or df.empty:
                continue
            xs, ys = [], []
            for _, r in df.iterrows():
                if pd.notna(r["picked_human"]) and r["pair_id"] in amap:
                    xs.append(int(r["picked_human"])); ys.append(amap[r["pair_id"]])
            out[(cell, mode)] = cohen_kappa_binary(xs, ys) if len(xs) >= 30 else float("nan")
    return out


def write_summary(rows, kappas, out_md, out_pq):
    s = [{"cell": r["cell"], "mode": r["mode"], "n_calls": r["n_calls"],
          "n_scored": r.get("n_scored"), "format_ok": r.get("format_ok_rate"),
          "tie_rate": r.get("tie_rate"), "budget_hit": r.get("budget_hit_rate"),
          "accuracy": r.get("accuracy"), "pos_bias": r.get("position_bias_delta"),
          "rating_mean": r.get("rating_mean"),
          "kappa_vs_anchor": kappas.get((r["cell"], r["mode"]), float("nan"))}
         for r in rows if r["n_calls"]]
    df = pd.DataFrame(s).sort_values(["mode", "cell"])
    df.to_parquet(out_pq, index=False)
    with out_md.open("w") as f:
        f.write("# Judge sweep summary\n\n")
        f.write("_accuracy = judge picks the true human (ties excluded); kappa_vs_anchor = "
                "Cohen's kappa on picked_human vs the 397B anchor, same thinking-mode, shared pairs._\n\n")
        f.write("_Caveat: accuracies are vs ONE stochastic generator draw (1 sample/row, T=0.6). "
                "Judge-vs-judge gaps <5pp are within generator sampling noise._\n\n")
        f.write(df.to_markdown(index=False)); f.write("\n")


def write_plots(rows, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out_dir.mkdir(parents=True, exist_ok=True)
    for metric in METRICS_FOR_PLOT:
        fig, ax = plt.subplots(figsize=(6, 4))
        for mode, mk in (("off", "o"), ("on", "s")):
            pts = sorted(((SIZE_MAP[r["cell"]], r[metric]) for r in rows
                          if r["mode"] == mode and r["n_calls"] and r["cell"] in SIZE_MAP
                          and isinstance(r.get(metric), (int, float)) and not np.isnan(r[metric])),
                         key=lambda t: t[0])
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts], marker=mk, label=f"thinking={mode}")
        ax.set_xscale("log"); ax.set_xlabel("judge size (B; active-params for MoE)")
        ax.set_ylabel(metric); ax.set_title(f"{metric} vs size"); ax.legend(); ax.grid(True, alpha=0.3)
        fig.tight_layout(); fig.savefig(out_dir / f"{metric}.png"); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    base = REPO_ROOT / "results" / "2026-07-08-judge-sweep"
    ap.add_argument("--raw_root", type=Path, default=base / "raw")
    ap.add_argument("--derived_root", type=Path, default=base / "derived")
    args = ap.parse_args()
    args.derived_root.mkdir(parents=True, exist_ok=True)
    pair_dfs, summaries = {}, []
    for cell_dir in sorted((args.raw_root / "sweep").iterdir()):
        if not cell_dir.is_dir():
            continue
        if cell_dir.name not in SIZE_MAP:  # skip smoke/exploratory dirs (e.g. fam_*, *-fp8)
            print(f"[analyzer] skip {cell_dir.name} (not in SIZE_MAP)", flush=True)
            continue
        for mode_dir in sorted(cell_dir.iterdir()):
            if not mode_dir.is_dir():
                continue
            calls = [per_call_features(r) for r in load_cell_rows(mode_dir)]
            summ, pdf = aggregate_cell(cell_dir.name, mode_dir.name, calls)
            summaries.append(summ); pair_dfs[(cell_dir.name, mode_dir.name)] = pdf
            print(f"[analyzer] {cell_dir.name}/{mode_dir.name}: n={summ['n_calls']} "
                  f"acc={summ.get('accuracy', 0):.3f} ties={summ.get('tie_rate', 0):.2f}", flush=True)
    kappas = compute_kappa_vs_anchor(pair_dfs, ANCHOR_CELL)
    write_summary(summaries, kappas, args.derived_root / "summary.md", args.derived_root / "summary.parquet")
    nonempty = [d for d in pair_dfs.values() if not d.empty]
    (pd.concat(nonempty, ignore_index=True) if nonempty else pd.DataFrame()).to_parquet(
        args.derived_root / "per_pair_metrics.parquet", index=False)
    write_plots(summaries, args.derived_root / "plots")
    print("[analyzer] done", flush=True)


if __name__ == "__main__":
    main()
