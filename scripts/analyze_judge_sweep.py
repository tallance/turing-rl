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

# Plot order (qwen3.5 dense by size, then MoE by size, then cross-family bonus) + labels.
PLOT_ORDER = ["qwen35-4b", "qwen35-9b", "qwen35-27b", "qwen35-35b-a3b",
              "qwen35-122b", "qwen35-397b", "qwen3-8b"]
PLOT_LABELS = {"qwen35-4b": "3.5-4B", "qwen35-9b": "3.5-9B", "qwen35-27b": "3.5-27B",
               "qwen35-35b-a3b": "3.5-35B-A3B", "qwen35-122b": "3.5-122B(Int4)",
               "qwen35-397b": "3.5-397B(Int4)", "qwen3-8b": "qwen3-8B"}
# (metric key, y-label, optional (ymin,ymax), optional reference line)
PLOT_METRICS = [
    ("accuracy", "accuracy | parse ok (picks true human)", (0.45, 0.85), 0.5),
    ("accuracy_penalized", "accuracy (parse-fail counted wrong)", (0.45, 0.85), 0.5),
    ("kappa_vs_anchor", "Cohen's kappa vs 397B anchor", (0.0, 0.7), None),
    ("tie_rate", "tie rate (rating==4)", None, None),
    ("budget_hit_rate", "budget-hit rate (finish=length)", None, None),
    ("parse_error_rate", "parse-error rate (no valid 1-7 rating recovered)", None, None),
    ("position_bias_delta", "position bias |acc(humanA)-acc(humanB)|", None, None),
    ("position_bias_signed", "fraction A - fraction B", (-0.3, 0.3), 0.0),
]


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
    # parse error = no valid 1-7 rating recovered. rating==0 is the reward's parse-failure
    # sentinel (empty judge content, esp. thinking-on); None = nothing recoverable. rating==4
    # is a valid tie, NOT an error.
    valid_rating = rating is not None and 1 <= rating <= 7
    parse_error = not valid_rating

    generated_is_b = bool(row.get("generated_is_b"))
    # human_side and generated_is_b are complementary: human is A iff generator is B.
    human_side = row.get("human_side") or ("A" if generated_is_b else "B")

    # judge's order-invariant choice: rating<4 -> picks A, >4 -> picks B, ==4 -> tie.
    # Only a VALID 1-7, non-tie rating is a real pick; rating==0 sentinel / None -> abstain
    # (previously rating==0 leaked in as "picks A" since 0<4 — fixed here).
    if valid_rating and rating != 4:
        pick = "A" if rating < 4 else "B"
        picked_human = int(pick == human_side)
    else:
        picked_human = None

    return {
        "pair_id": row.get("pair_id") or f'{row.get("user_id")}::{row.get("post_id")}::{row.get("target_idx")}',
        "user_id": row.get("user_id"),
        "generated_is_b": generated_is_b,
        "human_side": human_side,
        "randomized_order": row.get("randomized_order"),
        "rating": rating,
        "format_ok": rating is not None,
        "parse_error": parse_error,
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
    # Directional position bias: P(judge picks position A) among non-tie calls. Human is
    # ~50/50 A/B, so pick_a_rate>0.5 = leans first/A, <0.5 = leans second/B.
    # valid, non-tie calls only (rating 1-7, !=4); rating==0 sentinel excluded.
    valid_nontie = df[df["rating"].notna() & df["rating"].between(1, 7) & (df["rating"] != 4)]
    pick_a_rate = float((valid_nontie["rating"].astype(int) < 4).mean()) if len(valid_nontie) else float("nan")
    # fraction picks A - fraction picks B  (= 2*pick_a_rate - 1); + = leans first/A.
    position_bias_signed = (2 * pick_a_rate - 1) if not np.isnan(pick_a_rate) else float("nan")
    # Two accuracy definitions (user request):
    #   accuracy            = accuracy | parse ok  (denominator = valid non-tie calls)
    #   accuracy_penalized  = parse failures counted WRONG (denominator adds parse-error calls)
    # Ties (rating==4) are legitimate abstentions, excluded from both.
    n_correct = int(df["picked_human"].sum()) if df["picked_human"].notna().any() else 0
    n_scored = int(df["picked_human"].notna().sum())        # valid, non-tie
    n_parse_error = int(df["parse_error"].sum())
    acc_parse_ok = (n_correct / n_scored) if n_scored else 0.0
    acc_penalized = (n_correct / (n_scored + n_parse_error)) if (n_scored + n_parse_error) else 0.0
    summ = {
        "cell": cell, "mode": mode, "n_calls": len(df),
        "format_ok_rate": float(df["format_ok"].mean()),
        "parse_error_rate": float(df["parse_error"].mean()),
        "rating_recovery_rate": float(df["rating_recovered_from_text"].mean()),
        "budget_hit_rate": float(df["budget_hit"].mean()),
        "tie_rate": float((df["rating"] == 4).mean()),
        "n_scored": n_scored,
        "n_parse_error": n_parse_error,
        "accuracy": acc_parse_ok,
        "accuracy_penalized": acc_penalized,
        "position_bias_delta": pos_bias,
        "pick_a_rate": pick_a_rate,
        "position_bias_signed": position_bias_signed,
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
          "n_scored": r.get("n_scored"), "parse_error": r.get("parse_error_rate"),
          "tie_rate": r.get("tie_rate"), "budget_hit": r.get("budget_hit_rate"),
          "acc_parse_ok": r.get("accuracy"), "acc_penalized": r.get("accuracy_penalized"),
          "pos_bias": r.get("position_bias_delta"), "rating_mean": r.get("rating_mean"),
          "kappa_vs_anchor": kappas.get((r["cell"], r["mode"]), float("nan"))}
         for r in rows if r["n_calls"]]
    df = pd.DataFrame(s).sort_values(["mode", "cell"])
    df.to_parquet(out_pq, index=False)
    with out_md.open("w") as f:
        f.write("# Judge sweep summary\n\n")
        f.write("_acc_parse_ok = judge picks the true human among valid non-tie calls (ties & parse "
                "failures excluded); acc_penalized = same but parse failures (rating=0/none) counted "
                "WRONG. kappa_vs_anchor = Cohen's kappa on picked_human vs the 397B anchor, same "
                "thinking-mode, shared pairs._\n\n")
        f.write("_Caveat: accuracies are vs ONE stochastic generator draw (1 sample/row, T=0.6). "
                "Judge-vs-judge gaps <5pp are within generator sampling noise._\n\n")
        f.write(df.to_markdown(index=False)); f.write("\n")


def write_plots(rows, out_dir):
    """One grouped bar chart per metric: model on x, thinking off vs on side by side.
    All models treated equally (anchor is just another bar)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out_dir.mkdir(parents=True, exist_ok=True)
    by = {(r["cell"], r["mode"]): r for r in rows if r["n_calls"]}
    cells = [c for c in PLOT_ORDER if any(k[0] == c for k in by)]
    if not cells:
        return

    def _val(cell, mode, metric):
        r = by.get((cell, mode))
        v = r.get(metric) if r else None
        return v if isinstance(v, (int, float)) and not (isinstance(v, float) and np.isnan(v)) else np.nan

    x = np.arange(len(cells)); w = 0.38
    for metric, ylabel, ylim, ref in PLOT_METRICS:
        off = [_val(c, "off", metric) for c in cells]
        on = [_val(c, "on", metric) for c in cells]
        fig, ax = plt.subplots(figsize=(11, 5.5))
        b1 = ax.bar(x - w / 2, off, w, label="thinking off", color="#4C78A8")
        b2 = ax.bar(x + w / 2, on, w, label="thinking on", color="#F58518")
        for bars in (b1, b2):
            for r in bars:
                h = r.get_height()
                if not np.isnan(h):
                    ax.text(r.get_x() + r.get_width() / 2, h, f"{h:.2f}",
                            ha="center", va="bottom", fontsize=8)
        if ref is not None:
            ax.axhline(ref, ls="--", c="gray", lw=1, alpha=0.7)
        ax.set_xticks(x); ax.set_xticklabels([PLOT_LABELS.get(c, c) for c in cells], rotation=20, ha="right")
        ax.set_ylabel(ylabel); ax.set_title(f"{metric} by model — thinking off vs on")
        if ylim:
            ax.set_ylim(*ylim)
        ax.legend(); ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout(); fig.savefig(out_dir / f"{metric}.png", dpi=130); plt.close(fig)


def write_rating_dist(rows, out_dir):
    """Small-multiples: rating (1-7) distribution per model, thinking off vs on, one figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out_dir.mkdir(parents=True, exist_ok=True)
    by = {(r["cell"], r["mode"]): r for r in rows if r["n_calls"]}
    cells = [c for c in PLOT_ORDER if any(k[0] == c for k in by)]
    if not cells:
        return
    ratings = list(range(1, 8)); xr = np.arange(7); w = 0.38
    ncol = 4; nrow = (len(cells) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 2.8 * nrow), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).flatten()
    for i, c in enumerate(cells):
        ax = axes[i]
        for mode, off, color in (("off", -w / 2, "#4C78A8"), ("on", w / 2, "#F58518")):
            r = by.get((c, mode))
            if not r:
                continue
            h = r.get("rating_histogram", {}) or {}
            tot = sum(h.values()) or 1
            frac = [h.get(str(k), 0) / tot for k in ratings]
            ax.bar(xr + off, frac, w, color=color, label=f"thinking {mode}")
        ax.set_title(PLOT_LABELS.get(c, c), fontsize=9)
        ax.set_xticks(xr); ax.set_xticklabels(ratings); ax.grid(True, axis="y", alpha=0.3)
    for j in range(len(cells), len(axes)):
        axes[j].axis("off")
    axes[0].legend(fontsize=8)
    fig.suptitle("Rating distribution (1-7) by model — thinking off vs on")
    fig.supxlabel("rating  (1=strongly A ... 4=tie ... 7=strongly B)"); fig.supylabel("fraction of calls")
    fig.tight_layout(); fig.savefig(out_dir / "rating_distribution.png", dpi=130); plt.close(fig)


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
    for r in summaries:  # fold kappa into rows so the bar plots can use it
        r["kappa_vs_anchor"] = kappas.get((r["cell"], r["mode"]), float("nan"))
    write_summary(summaries, kappas, args.derived_root / "summary.md", args.derived_root / "summary.parquet")
    nonempty = [d for d in pair_dfs.values() if not d.empty]
    (pd.concat(nonempty, ignore_index=True) if nonempty else pd.DataFrame()).to_parquet(
        args.derived_root / "per_pair_metrics.parquet", index=False)
    write_plots(summaries, args.derived_root / "plots")
    write_rating_dist(summaries, args.derived_root / "plots")
    print("[analyzer] done", flush=True)


if __name__ == "__main__":
    main()
