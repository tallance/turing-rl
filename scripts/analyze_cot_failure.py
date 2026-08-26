"""Diagnose the thinking-ON parse-failure mode: repetition vs generation length.

Reads the per-call HTTP dumps (raw/sweep/<cell>/<mode>/http/*.jsonl), which store the
FULL vLLM response including the raw <think> chain-of-thought at
  response.choices[0].message.reasoning         (older vLLM key; "reasoning_content" absent)
For each judge call we compute a repetition metric on the raw CoT and correlate it with
generation length (completion tokens) and outcome (ok / cap_runaway / stop_malformed /
timeout). This tests the hypothesis that the long thinking-ON generations that fail to
emit parseable JSON do so because the model falls into a runaway repetition loop.

Failure cases come from the `*-diag` cells (plan smooth-strolling-lake: previously-timed-out
pairs replayed with a 1800s timeout so they complete and get dumped); successful baseline
points come from the original `<model>/on` cells. stdlib + numpy only (no sklearn/scipy),
per repo precedent.

Usage (cluster, env turing-rl-train):
  python scripts/analyze_cot_failure.py
"""
from __future__ import annotations
import argparse, json, re, sys, zlib
from collections import Counter
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO_ROOT))
import numpy as np

RATING_RE = re.compile(r'"rating"\s*:\s*(\d+)')
WORD_RE = re.compile(r"\w+")

# (cell dir, model label) pairs: diag = failure cases, base = successful baseline.
DEFAULT_SOURCES = [
    ("qwen35-397b-diag", "397B", "diag"),
    ("qwen35-397b", "397B", "base"),
    ("qwen35-9b-diag", "9B", "diag"),
    ("qwen35-9b", "9B", "base"),
]
MODEL_MARKER = {"397B": "o", "9B": "^"}
OUTCOME_COLOR = {"ok": "#4C78A8", "cap_runaway": "#E4572E",
                 "stop_malformed": "#F58518", "timeout": "#8C8C8C"}
OUTCOME_ORDER = ["ok", "cap_runaway", "stop_malformed", "timeout"]


# ---- pure, unit-tested helpers ------------------------------------------------
def repetition_metrics(text: str) -> dict:
    """Quantify self-repetition of a text. Higher zlib_ratio / max_line_repeat and
    LOWER distinct_3gram all indicate more repetition."""
    text = text or ""
    b = text.encode("utf-8", "ignore")
    zlib_ratio = (len(b) / len(zlib.compress(b, 6))) if b else 1.0

    words = WORD_RE.findall(text.lower())
    tris = list(zip(words, words[1:], words[2:]))
    distinct_3gram = (len(set(tris)) / len(tris)) if tris else 1.0

    # longest run of an identical repeated non-empty line (literal loop detector)
    max_line_repeat = 1 if text.strip() else 0
    run = 0
    prev = None
    for ln in text.splitlines():
        s = ln.strip()
        if not s:
            continue
        run = run + 1 if s == prev else 1
        prev = s
        max_line_repeat = max(max_line_repeat, run)
    return {"zlib_ratio": zlib_ratio, "distinct_3gram": distinct_3gram,
            "max_line_repeat": max_line_repeat}


def _valid_rating(content: str) -> bool:
    m = RATING_RE.search(content or "")
    return bool(m) and 1 <= int(m.group(1)) <= 7


def classify_outcome(finish_reason, content: str) -> str:
    """ok = parseable 1-7 rating in content; else bucket by why it failed."""
    if _valid_rating(content):
        return "ok"
    if finish_reason == "length":
        return "cap_runaway"
    if finish_reason == "stop":
        return "stop_malformed"
    return "timeout"  # finish_reason None/missing -> did not complete normally


# ---- IO / feature extraction --------------------------------------------------
def _iter_http_rows(cell_dir: Path, mode: str):
    hdir = cell_dir / mode / "http"
    if not hdir.is_dir():
        return
    for jl in sorted(hdir.glob("*.jsonl")):
        for line in jl.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                pass


def extract_call(row: dict) -> dict | None:
    resp = row.get("response") or {}
    choices = resp.get("choices") or []
    if not choices:
        return None
    ch = choices[0]
    msg = ch.get("message") or {}
    reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""
    content = msg.get("content") or ""
    finish = ch.get("finish_reason")
    usage = resp.get("usage") or {}
    ctok = usage.get("completion_tokens")
    rep = repetition_metrics(reasoning)
    return {
        "completion_tokens": ctok,
        "reasoning_chars": len(reasoning),
        "content_chars": len(content),
        "finish_reason": finish,
        "outcome": classify_outcome(finish, content),
        "reasoning": reasoning,
        **rep,
    }


def collect(raw_root: Path, sources) -> list[dict]:
    out = []
    seen = Counter()
    for cell, model, src in sources:
        cell_dir = raw_root / "sweep" / cell
        n = 0
        for row in _iter_http_rows(cell_dir, "on"):
            c = extract_call(row)
            if c is None or c["completion_tokens"] is None:
                continue
            c["model"] = model
            c["source"] = src
            c["cell"] = cell
            out.append(c)
            n += 1
        seen[(cell, model, src)] = n
    for (cell, model, src), n in seen.items():
        print(f"  {cell:22s} [{model} {src}] rows={n}")
    return out


# ---- plots --------------------------------------------------------------------
def write_plots(rows: list[dict], out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out_dir.mkdir(parents=True, exist_ok=True)
    models = sorted({r["model"] for r in rows})

    # 1) scatter: length vs repetition, color=outcome, marker=model
    fig, ax = plt.subplots(figsize=(10, 6.5))
    for model in models:
        mk = MODEL_MARKER.get(model, "s")
        for outcome in OUTCOME_ORDER:
            pts = [r for r in rows if r["model"] == model and r["outcome"] == outcome]
            if not pts:
                continue
            ax.scatter([p["completion_tokens"] for p in pts], [p["zlib_ratio"] for p in pts],
                       s=16, alpha=0.45, marker=mk, color=OUTCOME_COLOR[outcome],
                       label=f"{model} · {outcome} (n={len(pts)})")
    ax.axvline(8192, ls="--", color="red", lw=1)
    ax.text(8192, ax.get_ylim()[1], " max_tokens=8192", color="red", fontsize=8, va="top")
    ax.set_xlabel("generation length = judge completion tokens")
    ax.set_ylabel("repetition = zlib compression ratio (higher = more repetitive)")
    ax.set_title("Judge CoT: repetition vs generation length (thinking ON)\n"
                 "marker = model, color = outcome")
    ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(out_dir / "cot_repetition_vs_length.png", dpi=130); plt.close(fig)

    # 2) grouped bar: outcome fraction per model
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(OUTCOME_ORDER)); w = 0.8 / max(1, len(models))
    for i, model in enumerate(models):
        mrows = [r for r in rows if r["model"] == model]
        tot = len(mrows) or 1
        fracs = [sum(r["outcome"] == o for r in mrows) / tot for o in OUTCOME_ORDER]
        bars = ax.bar(x + (i - (len(models) - 1) / 2) * w, fracs, w, label=f"{model} (n={len(mrows)})")
        for b, f in zip(bars, fracs):
            if f > 0:
                ax.text(b.get_x() + b.get_width() / 2, f, f"{f:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(OUTCOME_ORDER)
    ax.set_ylabel("fraction of judge calls"); ax.set_title("CoT outcome mix by model (thinking ON)")
    ax.legend(fontsize=8); ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(out_dir / "cot_failure_modes.png", dpi=130); plt.close(fig)

    # 3) repetition distribution: ok vs failed (pooled)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ok = [r["zlib_ratio"] for r in rows if r["outcome"] == "ok"]
    bad = [r["zlib_ratio"] for r in rows if r["outcome"] != "ok"]
    lo = min([r["zlib_ratio"] for r in rows], default=1.0)
    hi = max([r["zlib_ratio"] for r in rows], default=2.0)
    bins = np.linspace(lo, hi, 40)
    if ok:
        ax.hist(ok, bins=bins, alpha=0.6, color="#4C78A8", label=f"ok (n={len(ok)})", density=True)
    if bad:
        ax.hist(bad, bins=bins, alpha=0.6, color="#E4572E", label=f"failed (n={len(bad)})", density=True)
    ax.set_xlabel("repetition = zlib compression ratio"); ax.set_ylabel("density")
    ax.set_title("CoT repetition: parseable vs failed calls (thinking ON, pooled)")
    ax.legend(fontsize=8); ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(out_dir / "cot_repetition_dist.png", dpi=130); plt.close(fig)


def write_worst_examples(rows: list[dict], out_path: Path, k: int = 5) -> None:
    failed = [r for r in rows if r["outcome"] != "ok" and r["reasoning"]]
    failed.sort(key=lambda r: r["zlib_ratio"], reverse=True)
    with out_path.open("w") as fo:
        fo.write("Most-repetitive FAILED judge CoTs (thinking ON), sorted by zlib ratio desc.\n")
        fo.write("Source: raw/sweep/<cell>/on/http/*.jsonl  message.reasoning\n" + "=" * 100 + "\n\n")
        for r in failed[:k]:
            fo.write(f"### model={r['model']} outcome={r['outcome']} finish={r['finish_reason']} "
                     f"compl_tok={r['completion_tokens']} zlib_ratio={r['zlib_ratio']:.2f} "
                     f"distinct_3gram={r['distinct_3gram']:.3f} max_line_repeat={r['max_line_repeat']}\n")
            fo.write("-" * 100 + "\n")
            fo.write((r["reasoning"] or "<empty>")[:8000] + "\n...[truncated]...\n\n" + "=" * 100 + "\n\n")


def write_summary(rows: list[dict], out_path: Path) -> None:
    lines = ["# CoT failure diagnostic\n",
             "_repetition (zlib compression ratio of the raw <think>) vs generation length._\n"]
    for model in sorted({r["model"] for r in rows}):
        mrows = [r for r in rows if r["model"] == model]
        tot = len(mrows) or 1
        lines.append(f"\n## {model}  (n={len(mrows)})\n")
        cnt = Counter(r["outcome"] for r in mrows)
        for o in OUTCOME_ORDER:
            lines.append(f"- {o}: {cnt.get(o,0)} ({cnt.get(o,0)/tot:.1%})")
        ok = np.array([r["zlib_ratio"] for r in mrows if r["outcome"] == "ok"])
        bad = np.array([r["zlib_ratio"] for r in mrows if r["outcome"] != "ok"])
        if len(ok):
            lines.append(f"- median zlib_ratio (ok):     {np.median(ok):.2f}")
        if len(bad):
            lines.append(f"- median zlib_ratio (failed): {np.median(bad):.2f}")
        L = np.array([r["completion_tokens"] for r in mrows], float)
        Z = np.array([r["zlib_ratio"] for r in mrows], float)
        if len(L) > 2 and L.std() > 0 and Z.std() > 0:
            lines.append(f"- corr(length, repetition):   {np.corrcoef(L, Z)[0,1]:+.2f}")
    out_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    base = REPO_ROOT / "results" / "2026-07-08-judge-sweep"
    ap.add_argument("--raw_root", type=Path, default=base / "raw")
    ap.add_argument("--out_dir", type=Path, default=base / "derived" / "cot_failure")
    args = ap.parse_args()
    print("[cot-diag] collecting http dumps ...")
    rows = collect(args.raw_root, DEFAULT_SOURCES)
    if not rows:
        raise SystemExit("no http rows found; did the diag jobs run? (raw/sweep/*-diag/on/http)")
    print(f"[cot-diag] {len(rows)} calls total")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_plots(rows, args.out_dir / "plots")
    write_worst_examples(rows, args.out_dir / "cot_worst_examples.txt")
    write_summary(rows, args.out_dir / "summary.md")
    print(f"[cot-diag] wrote plots + summary.md + cot_worst_examples.txt to {args.out_dir}")


if __name__ == "__main__":
    main()
