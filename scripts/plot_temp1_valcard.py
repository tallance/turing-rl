"""Temp-1.0 / model-card-val overfit probe: train-vs-inference analysis + plots.

For the `*_temp1_valcard` runs, training rollouts sample at temperature 1.0 (full-distribution
exploration) while a validation pass samples at the Qwen3-8B non-thinking model card
(temp 0.7, top_p 0.8, top_k 20) on the SAME 10 overfit turns. The reward dump tags each row
`split` = "train" | "val" (reward.py), so this script separates the two and asks: does temp-1.0
TRAIN exploration surface Likert-7 (judge fully fooled) more often than temp-0.7 INFERENCE?

Per run it produces:
  <label>_train_val_scatter.png  - side-by-side scatter: TRAIN (temp 1.0) | INFERENCE (temp 0.7),
                                   pooled judge Likert vs epoch (per-rollout scatter + epoch-mean
                                   line; green=win Likert>=5, purple=max Likert=7). Panel titles
                                   carry the Likert-7 frequency.
Across runs:
  winrate_train_val.png          - one subplot per run, train win-rate + inference win-rate vs epoch
                                   (win=Likert>=5, ties=4 / parse-fails=0/None excluded; gray=0.5).
  temp1_valcard_summary.txt      - Likert-7 frequency (overall + last-10-epoch) and final win-rate
                                   for train vs inference, per run.

Epoch reconstruction:
  - TRAIN: exactly G rollouts/example/epoch -> per-example ts-sort, chunk by G (epoch 1..E).
  - VAL:   n=1/example/pass but occasional double-scores (randomized gt/gen ordering) make per-example
           gap-splitting overcount. Val passes are GLOBAL (all 10 examples scored back-to-back, then a
           full training step elapses), so we split the globally ts-sorted val rows into bursts on a
           large gap (`--val_gap` s). Burst 0 = val_before_train baseline. Recovered pass count is
           printed for sanity (expect ~epochs+1).

Usage (cluster; env turing-rl-train has matplotlib):
  python scripts/plot_temp1_valcard.py \
    --run "lr=1e-4:results/grpo/rl-generator/8b_proper_kl1e4_lr1e4_temp1_valcard/reward_dump" \
    --run "lr=1e-5:results/grpo/rl-generator/8b_proper_kl1e4_lr1e5_temp1_valcard/reward_dump" \
    --outdir results/2026-07-24-reward-hack-proper-checkpoint/temp1-valcard \
    [--group_size 4] [--val_gap 200]
"""
from __future__ import annotations
import argparse, glob, json, os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load(dump_dir: str) -> list[dict]:
    rows = []
    for f in glob.glob(os.path.join(dump_dir, "reward-*.jsonl")):
        for line in open(f):
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def _key(r: dict) -> tuple:
    return (r.get("user_id"), r.get("post_id"), r.get("target_idx"))


def _valid(v) -> bool:
    return v is not None and int(round(float(v))) != 0


def train_epochs(rows: list[dict], G: int) -> list[list[dict]]:
    """List over epochs; each entry is the pooled list of rollout rows in that epoch
    (all examples). Per-example ts-sort then chunk by G; epoch index aligns across examples."""
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault(_key(r), []).append(r)
    for k in groups:
        groups[k].sort(key=lambda r: r.get("ts") or 0)
    max_e = max(((len(v) + G - 1) // G for v in groups.values()), default=0)
    epochs = []
    for e in range(max_e):
        pooled = []
        for v in groups.values():
            pooled.extend(v[e * G:(e + 1) * G])
        epochs.append(pooled)
    return epochs


def val_passes(rows: list[dict], gap: float) -> list[list[dict]]:
    """Split globally ts-sorted val rows into bursts separated by > gap seconds. Each burst = one
    validation pass (all examples scored back-to-back). Returns list over passes of pooled rows."""
    rs = sorted(rows, key=lambda r: r.get("ts") or 0.0)
    passes, cur, prev = [], [], None
    for r in rs:
        t = r.get("ts") or 0.0
        if prev is not None and (t - prev) > gap and cur:
            passes.append(cur); cur = []
        cur.append(r); prev = t
    if cur:
        passes.append(cur)
    return passes


def _likert7_freq(epochs: list[list[dict]], last_k: int | None = None):
    """Fraction of VALID rollouts (Likert 1-7, parse-fails excluded) that scored exactly 7."""
    chunks = epochs[-last_k:] if last_k else epochs
    vals = [r.get("turing_judge_score_raw") for ch in chunks for r in ch]
    vals = [float(v) for v in vals if _valid(v)]
    if not vals:
        return 0.0, 0
    n7 = sum(1 for v in vals if int(round(v)) == 7)
    return n7 / len(vals), len(vals)


def _winrate_series(epochs: list[list[dict]], x0: int):
    """Per-epoch win-rate = wins(Likert>=5)/nontie; ties(4)/parse-fails excluded. x starts at x0."""
    xs, wr = [], []
    for i, ch in enumerate(epochs):
        wins = nontie = 0
        for r in ch:
            v = r.get("turing_judge_score_raw")
            if not _valid(v):
                continue
            vi = int(round(float(v)))
            if vi == 4:
                continue
            nontie += 1
            if float(v) >= 5:
                wins += 1
        if nontie:
            xs.append(x0 + i); wr.append(wins / nontie)
    return xs, wr


def _scatter_panel(ax, epochs, x0, title, jitter=0.10):
    sx, sy, fx, fy, mx, my = [], [], [], [], [], []
    for i, ch in enumerate(epochs):
        x = x0 + i
        valid = []
        n = len(ch)
        for j, r in enumerate(ch):
            v = r.get("turing_judge_score_raw")
            jx = x + (jitter * (2.0 * (j / (n - 1)) - 1.0) if n > 1 else 0.0)
            if not _valid(v):
                fx.append(jx); fy.append(0.7); continue
            sx.append(jx); sy.append(float(v)); valid.append(float(v))
        if valid:
            mx.append(x); my.append(sum(valid) / len(valid))
    ax.scatter(sx, sy, s=14, alpha=0.30, color="tab:blue", zorder=2, label="per-rollout")
    if fx:
        ax.scatter(fx, fy, s=16, marker="x", color="red", alpha=0.55, zorder=2, label="parse-fail")
    ax.plot(mx, my, color="tab:red", lw=1.5, zorder=3, label="epoch mean")
    ax.axhline(7, ls="-", color="tab:purple", lw=0.8, alpha=0.7)   # max Likert (judge fully fooled)
    ax.axhline(5, ls="--", color="green", lw=0.8)                  # win threshold
    ax.axhline(4, ls=":", color="gray", lw=0.8)                    # tie
    ax.set_ylim(0.4, 7.4)
    ax.grid(True, alpha=0.2)
    ax.set_title(title, fontsize=10)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help="label:dump_dir (repeatable)")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--group_size", type=int, default=4, help="GRPO G (train rollouts/example/epoch)")
    ap.add_argument("--val_gap", type=float, default=200.0, help="seconds; gap> => new val pass")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    runs = []
    for spec in a.run:
        label, _, d = spec.partition(":")
        runs.append((label, d))

    summary = ["Temp-1.0 / model-card-val overfit probe: TRAIN (temp 1.0) vs INFERENCE (temp 0.7)",
               "=" * 84, ""]

    parsed = []  # (label, train_epochs, val_passes)
    for label, d in runs:
        rows = _load(d)
        tr = [r for r in rows if r.get("split") == "train"]
        va = [r for r in rows if r.get("split") == "val"]
        te = train_epochs(tr, a.group_size)
        vp = val_passes(va, a.val_gap)
        parsed.append((label, te, vp))

        f7_tr, n_tr = _likert7_freq(te)
        f7_tr_last, _ = _likert7_freq(te, last_k=10)
        f7_va, n_va = _likert7_freq(vp)
        f7_va_last, _ = _likert7_freq(vp, last_k=10)
        _, wr_tr = _winrate_series(te, 1)
        _, wr_va = _winrate_series(vp, 0)
        summary += [
            f"[{label}]  train rows={len(tr)}  val rows={len(va)}",
            f"  epochs: train={len(te)}  val passes={len(vp)} (expect ~train+1; val_gap={a.val_gap:g}s)",
            f"  Likert-7 freq  TRAIN(temp1.0)={f7_tr:.3f} (last10={f7_tr_last:.3f}, n={n_tr})",
            f"  Likert-7 freq  INFER(temp0.7)={f7_va:.3f} (last10={f7_va_last:.3f}, n={n_va})",
            f"  final win-rate TRAIN={wr_tr[-1] if wr_tr else float('nan'):.3f}  "
            f"INFER={wr_va[-1] if wr_va else float('nan'):.3f}",
            "",
        ]

        # ---- side-by-side scatter: TRAIN | INFERENCE ----
        fig, (axl, axr) = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
        _scatter_panel(axl, te, 1, f"TRAIN  temp=1.0   Likert-7 freq={f7_tr:.2f} (last10 {f7_tr_last:.2f})")
        _scatter_panel(axr, vp, 0, f"INFERENCE  temp=0.7   Likert-7 freq={f7_va:.2f} (last10 {f7_va_last:.2f})")
        axl.legend(fontsize=8, loc="lower right")
        axl.set_xlabel("overfit epoch (G rollouts/example, ts-chunked)")
        axr.set_xlabel("validation pass (0 = before training)")
        axl.set_ylabel("judge Likert (1-7)")
        fig.suptitle(f"{label}: temp-1.0 train vs temp-0.7 inference on the same 10 turns  "
                     f"(purple=max 7, green=win>=5, gray=tie 4)", fontsize=11)
        fig.tight_layout(rect=[0, 0.02, 1, 0.96])
        out = os.path.join(a.outdir, f"{_safe(label)}_train_val_scatter.png")
        fig.savefig(out, dpi=120); plt.close(fig)
        print(f"wrote {out}")

    # ---- winrate: one subplot per run, train + inference ----
    n = len(parsed)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 4.5), sharey=True, squeeze=False)
    axes = axes[0]
    for ax, (label, te, vp) in zip(axes, parsed):
        xt, wt = _winrate_series(te, 1)
        xv, wv = _winrate_series(vp, 0)
        ax.plot(xt, wt, color="tab:blue", lw=1.4, marker="o", ms=3, alpha=0.85, label="TRAIN temp=1.0")
        ax.plot(xv, wv, color="tab:orange", lw=1.4, marker="s", ms=3, alpha=0.85, label="INFER temp=0.7")
        ax.axhline(0.5, ls="--", color="gray", lw=0.9)
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(f"{label}   final: train={wt[-1] if wt else float('nan'):.2f} "
                     f"infer={wv[-1] if wv else float('nan'):.2f}", fontsize=10)
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("overfit epoch / val pass")
    axes[0].legend(fontsize=8, loc="lower right")
    axes[0].set_ylabel("overall win-rate (Likert>=5)")
    fig.suptitle("Temp-1.0/val-card overfit: overall win-rate vs epoch (train temp 1.0 vs infer temp 0.7; "
                 "10 turns pooled; ties/parse-fails excluded)", fontsize=11)
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    out = os.path.join(a.outdir, "winrate_train_val.png")
    fig.savefig(out, dpi=120); plt.close(fig)
    print(f"wrote {out}")

    txt = os.path.join(a.outdir, "temp1_valcard_summary.txt")
    with open(txt, "w") as fh:
        fh.write("\n".join(summary) + "\n")
    print(f"wrote {txt}")
    print("\n".join(summary))


def _safe(s: str) -> str:
    return s.replace("=", "").replace("/", "_").replace(" ", "_")


if __name__ == "__main__":
    main()
