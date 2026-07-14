"""Compute derived metrics from raw reward dumps. Idempotent: delete derived/ and re-run.
Reads raw/sweep/<cell>/<mode>/reward/*.jsonl -> derived/{summary.md,summary.parquet,per_pair_metrics.parquet,plots/}."""
from __future__ import annotations
import argparse, json, re, sys
from collections import defaultdict
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO_ROOT))
import numpy as np, pandas as pd
from configs.judge_sweep_cells import SIZE_MAP, ANCHOR_CELL   # single source of truth

RUBRIC_FIELDS = ["immediate_target_score_a","immediate_target_score_b","human_goal_score_a",
    "human_goal_score_b","communication_style_score_a","communication_style_score_b",
    "response_a_source_copy","response_b_source_copy","source_copy_penalty_a","source_copy_penalty_b",
    "response_a_wrong_target_or_role","response_b_wrong_target_or_role","wrong_target_or_role_penalty_a",
    "wrong_target_or_role_penalty_b","response_a_unsupported_adversarial_reframing",
    "response_b_unsupported_adversarial_reframing","unsupported_adversarial_reframing_penalty_a",
    "unsupported_adversarial_reframing_penalty_b","response_a_assistant_like","response_b_assistant_like",
    "assistant_like_penalty_a","assistant_like_penalty_b","base_score_a","base_score_b",
    "response_a_score","response_b_score","score_gap","reasoning","rating"]
RATING_RE = re.compile(r'"rating"\s*:\s*(\d+)')

def try_parse_json(text: str) -> dict | None:
    if not text: return None
    s = text.strip()
    if not s.startswith("{"):
        i = s.find("{");  s = s[i:] if i >= 0 else s
        if i < 0: return None
    try: return json.loads(s)
    except json.JSONDecodeError: pass
    for end in range(len(s), 0, -1):
        try: return json.loads(s[:end])
        except json.JSONDecodeError: continue
    return None

def recover_rating_from_text(text: str) -> int | None:
    if not text: return None
    m = RATING_RE.search(text)
    if not m: return None
    r = int(m.group(1)); return r if 1 <= r <= 7 else None

def load_cell_rows(mode_dir: Path) -> list[dict]:
    rows = []
    for jl in sorted((mode_dir / "reward").glob("*.jsonl")):
        for line in jl.read_text().splitlines():
            line = line.strip()
            if line:
                try: rows.append(json.loads(line))
                except json.JSONDecodeError: pass
    return rows

def per_call_features(row: dict) -> dict:
    text = row.get("judge_raw_content") or ""
    parsed = try_parse_json(text)
    parsed_rating = int(parsed["rating"]) if parsed and isinstance(parsed.get("rating"), int) else None
    recovered = recover_rating_from_text(text) if parsed_rating is None else None
    rating = parsed_rating if parsed_rating is not None else recovered
    usage = row.get("judge_usage") or {}
    return {"pair_id": row.get("pair_id") or f'{row.get("user_id")}::{row.get("post_id")}::{row.get("target_idx")}',
        "user_id": row.get("user_id"), "generated_is_b": bool(row.get("generated_is_b")),
        "rating": rating, "format_ok": parsed is not None and parsed_rating is not None,
        "rating_recovered_from_text": recovered is not None and parsed_rating is None,
        "budget_hit": (row.get("judge_finish_reason") or "") == "length",
        "length_chars": len(text),
        "completion_tokens": usage.get("completion_tokens") if isinstance(usage, dict) else None,
        **{f"has_{f}": (parsed is not None and f in parsed) for f in RUBRIC_FIELDS}}

def accuracy(calls: list[dict]) -> float | None:
    """rating<4 -> judge picks A; >4 -> picks B; ==4 -> tie (excluded). Compare to human side."""
    num = den = 0
    for c in calls:
        r = c["rating"]
        if r is None or r == 4: continue
        judge_picks_b = r > 4
        correct = (judge_picks_b == (not c["generated_is_b"]))  # picks human?
        num += int(correct); den += 1
    return num / den if den else None

def budget_hit_rate(calls): return float(np.mean([c["budget_hit"] for c in calls])) if calls else 0.0

def aggregate_cell(cell: str, mode: str, calls: list[dict]):
    df = pd.DataFrame(calls)
    if df.empty: return {"cell": cell, "mode": mode, "n_calls": 0}, pd.DataFrame()
    per_pair = defaultdict(dict)
    for _, r in df.iterrows():
        side = "b" if r["generated_is_b"] else "a"
        per_pair[r["pair_id"]][f"rating_{side}"] = r["rating"]
        per_pair[r["pair_id"]][f"fmt_{side}"] = r["format_ok"]
    pair_rows = []
    for pid, d in per_pair.items():
        ra, rb = d.get("rating_a"), d.get("rating_b")
        pair_rows.append({"pair_id": pid, "cell": cell, "mode": mode, "rating_a": ra, "rating_b": rb,
            "picks_human_gen_b": (rb is not None and rb < 4),   # generated=B -> human=A -> pick A
            "picks_human_gen_a": (ra is not None and ra > 4),   # generated=A -> human=B -> pick B
            "both_parsed": bool(d.get("fmt_a") and d.get("fmt_b"))})
    pdf = pd.DataFrame(pair_rows)
    n1 = int(pdf["rating_b"].notna().sum()); n2 = int(pdf["rating_a"].notna().sum())
    acc1 = pdf["picks_human_gen_b"].sum() / max(n1, 1); acc2 = pdf["picks_human_gen_a"].sum() / max(n2, 1)
    both = pdf["both_parsed"]; nb = int(both.sum())
    acc_both = ((pdf.loc[both, "picks_human_gen_b"].sum() + pdf.loc[both, "picks_human_gen_a"].sum()) / (2 * nb)) if nb else 0.0
    ratings = df["rating"].dropna().astype(int).tolist()
    hist = {str(k): ratings.count(k) for k in range(1, 8)}
    summ = {"cell": cell, "mode": mode, "n_calls": len(df),
        "format_ok_rate": float(df["format_ok"].mean()),
        "rating_recovery_rate": float(df["rating_recovered_from_text"].mean()),
        "budget_hit_rate": float(df["budget_hit"].mean()),
        "accuracy_both_orderings": acc_both, "position_bias_delta": abs(acc1 - acc2),
        "rating_mean": float(np.mean(ratings)) if ratings else 0.0,
        "rating_mode": int(max(hist, key=lambda k: hist[k])) if ratings else 0,
        "field_presence": {f.replace("has_", ""): float(df[f].mean()) for f in df.columns if f.startswith("has_")},
        "rating_histogram": hist}
    return summ, pdf

def compute_kappa_vs_anchor(pair_dfs, anchor_cell):
    from sklearn.metrics import cohen_kappa_score
    out = {}
    for mode in ("off", "on"):
        adf = pair_dfs.get((anchor_cell, mode))
        if adf is None: continue
        amap = {r["pair_id"]: int(r["picks_human_gen_b"]) + int(r["picks_human_gen_a"])
                for _, r in adf.iterrows() if r["both_parsed"]}
        for (cell, m), df in pair_dfs.items():
            if m != mode or cell == anchor_cell: continue
            xs, ys = [], []
            for _, r in df.iterrows():
                if r["both_parsed"] and r["pair_id"] in amap:
                    xs.append(int(r["picks_human_gen_b"]) + int(r["picks_human_gen_a"])); ys.append(amap[r["pair_id"]])
            out[(cell, mode)] = cohen_kappa_score(xs, ys) if len(xs) >= 30 else float("nan")
    return out

def write_summary(rows, kappas, out_md, out_pq):
    s = [{"cell": r["cell"], "mode": r["mode"], "n_calls": r["n_calls"],
          "format_ok": r.get("format_ok_rate"), "rating_recovery": r.get("rating_recovery_rate"),
          "budget_hit": r.get("budget_hit_rate"), "accuracy": r.get("accuracy_both_orderings"),
          "pos_bias": r.get("position_bias_delta"), "rating_mean": r.get("rating_mean"),
          "kappa_vs_anchor": kappas.get((r["cell"], r["mode"]), float("nan"))} for r in rows if r["n_calls"]]
    df = pd.DataFrame(s).sort_values(["mode", "cell"]); df.to_parquet(out_pq, index=False)
    with out_md.open("w") as f:
        f.write("# Judge sweep summary\n\n")
        f.write("_Caveat: accuracies are vs ONE stochastic generator draw (1 sample/row, T=0.6). "
                "Judge-vs-judge gaps <5pp are within generator sampling noise._\n\n")
        f.write(df.to_markdown(index=False)); f.write("\n")

def write_plots(rows, out_dir):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    out_dir.mkdir(parents=True, exist_ok=True)
    for metric in ("accuracy_both_orderings", "format_ok_rate", "budget_hit_rate", "position_bias_delta"):
        fig, ax = plt.subplots(figsize=(6, 4))
        for mode, mk in (("off", "o"), ("on", "s")):
            pts = sorted(((SIZE_MAP[r["cell"]], r[metric]) for r in rows
                          if r["mode"] == mode and r["n_calls"] and r["cell"] in SIZE_MAP), key=lambda t: t[0])
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
    args = ap.parse_args(); args.derived_root.mkdir(parents=True, exist_ok=True)
    pair_dfs, summaries = {}, []
    for cell_dir in sorted((args.raw_root / "sweep").iterdir()):
        if not cell_dir.is_dir(): continue
        for mode_dir in sorted(cell_dir.iterdir()):
            if not mode_dir.is_dir(): continue
            calls = [per_call_features(r) for r in load_cell_rows(mode_dir)]
            summ, pdf = aggregate_cell(cell_dir.name, mode_dir.name, calls)
            summaries.append(summ); pair_dfs[(cell_dir.name, mode_dir.name)] = pdf
            print(f"[analyzer] {cell_dir.name}/{mode_dir.name}: n={summ['n_calls']} "
                  f"acc={summ.get('accuracy_both_orderings', 0):.3f}", flush=True)
    kappas = compute_kappa_vs_anchor(pair_dfs, ANCHOR_CELL)
    write_summary(summaries, kappas, args.derived_root / "summary.md", args.derived_root / "summary.parquet")
    nonempty = [d for d in pair_dfs.values() if not d.empty]
    (pd.concat(nonempty, ignore_index=True) if nonempty else pd.DataFrame()).to_parquet(
        args.derived_root / "per_pair_metrics.parquet", index=False)
    write_plots(summaries, args.derived_root / "plots")
    print("[analyzer] done", flush=True)

if __name__ == "__main__":
    main()
