"""CoT fidelity check: self-hosted (thinking-off) vs OpenRouter Qwen3-8B.

Validates the self-hosting substitution used by ``scripts/generate_cot_served.py``
against the original reference the paper actually used (OpenRouter Qwen3-8B via
the upstream ``data/sft/generate_cot.py`` payload). For a fixed sample of ~20 SFT
rows (seed 42) it reconstructs the SAME rationalize messages for both backends,
generates a CoT trace each way, then compares distributions (completion length,
first-/third-person perspective, leakage rate) and writes a soft PASS/FAIL
``cot_fidelity.md`` report with a side-by-side of 5 rows.

This is a *fidelity report*, not a hard gate: large divergence prompts a
sampling/template re-check, it does not fail a pipeline.

Design: the pure comparison helpers (``perspective``, ``summarize``,
``leakage_flags``, ``write_report``) are network-free and unit tested in
``tests/test_cot_fidelity_check.py``. The two generation backends
(``gen_openrouter``, ``gen_selfhosted``) run in different places — the Mac runs
``--mode openrouter`` ($10 OpenRouter account), the cluster runs
``--mode selfhosted`` against a served Qwen3-8B. Each side dumps its 20 raw
texts to a JSON next to ``--out``; whichever side runs second (or ``--mode both``)
merges the two dumps and writes the report.

Importing this module performs NO network I/O; all requests live inside
functions invoked only from ``main``.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Reuse the exact served payload builder and the confirmed upstream rationalize
# business logic. Do NOT reimplement any of these.
from scripts.generate_cot_served import build_cot_payload  # noqa: E402
from data.sft.generate_cot import (  # noqa: E402
    RATIONALIZE_SYSTEM_PROMPT,
    RATIONALIZE_USER_TEMPLATE,
    _as_text,
    _row_context,
    reasoning_leaks_reply,
)

DEFAULT_INPUT = "data/prism/full_s42_history_sft40_grpo60_test10/sft/train.parquet"
DEFAULT_OUT = "results/2026-07-08-judge-sweep/derived/cot_fidelity.md"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = "qwen/qwen3-8b"
SELFHOSTED_MODEL = "Qwen/Qwen3-8B"
MAX_COMPLETION_TOKENS = 4096

# Word/phrase markers for the perspective heuristic. Whichever family's earliest
# marker appears first in the text decides the label (mirrors the intuition of
# the reward-side ``wrong_perspective`` signal, which is judge-based; there is no
# reusable text heuristic in the repo, so this is a deliberately simple one).
_FIRST_RE = re.compile(r"\b(i|i'm|i'll|i've|i'd|my|me|mine|myself|we|our|ours|us)\b")
_THIRD_RE = re.compile(
    r"\b(the user|the person|the reddit user|the poster|the commenter|"
    r"they|them|their|theirs|he|she|his|her|hers)\b"
)


# --------------------------------------------------------------------------- #
# Pure, network-free comparison helpers (unit tested).
# --------------------------------------------------------------------------- #
def perspective(text: str) -> str:
    """Classify narrative perspective as ``"first"``, ``"third"`` or ``"other"``.

    First-person ("I/my/me/we...") vs third-person ("the user/they/he...").
    Whichever family's earliest marker occurs first in the text wins; ties break
    to first-person; no marker at all -> ``"other"``.
    """
    if not text or not text.strip():
        return "other"
    low = text.lower()
    first_m = _FIRST_RE.search(low)
    third_m = _THIRD_RE.search(low)
    if first_m and third_m:
        return "first" if first_m.start() <= third_m.start() else "third"
    if first_m:
        return "first"
    if third_m:
        return "third"
    return "other"


def _percentile(sorted_vals: list[float], q: float) -> float:
    """Linear-interpolation percentile (matches numpy's default ``linear``)."""
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    rank = (q / 100.0) * (len(sorted_vals) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = rank - lo
    return float(sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * frac)


def summarize(texts: list[str]) -> dict[str, Any]:
    """Summarize a list of CoT texts.

    Returns ``n``, ``empty`` (whitespace-only count), character-length quartiles
    ``len_p25/len_p50/len_p75``, and ``perspective_counts`` over all texts.
    """
    n = len(texts)
    empty = sum(1 for t in texts if not (t or "").strip())
    lengths = sorted(len(t or "") for t in texts)
    counts = {"first": 0, "third": 0, "other": 0}
    for t in texts:
        counts[perspective(t or "")] += 1
    return {
        "n": n,
        "empty": empty,
        "len_p25": _percentile(lengths, 25),
        "len_p50": _percentile(lengths, 50),
        "len_p75": _percentile(lengths, 75),
        "perspective_counts": counts,
    }


def leakage_flags(rows: list[dict[str, Any]], texts: list[str]) -> list[bool]:
    """Per-row leakage flags via the upstream ``reasoning_leaks_reply`` guard."""
    flags: list[bool] = []
    for row, text in zip(rows, texts):
        reward_model = dict(row.get("reward_model") or {})
        ground_truth = _as_text(reward_model.get("ground_truth"))
        flags.append(reasoning_leaks_reply((text or "").strip(), ground_truth))
    return flags


def _rate(flags: list[bool]) -> float:
    return (sum(1 for f in flags if f) / len(flags)) if flags else 0.0


def _first_frac(counts: dict[str, int]) -> float:
    total = sum(counts.values())
    return (counts.get("first", 0) / total) if total else 0.0


def _snip(text: str, limit: int = 300) -> str:
    """One-line snippet for the markdown table cell."""
    flat = " ".join((text or "").split())
    return flat if len(flat) <= limit else flat[:limit].rstrip() + "..."


def _verdict(
    self_summ: dict[str, Any],
    or_summ: dict[str, Any],
    self_leak: list[bool],
    or_leak: list[bool],
) -> tuple[str, list[str]]:
    """Soft PASS/FAIL: flags large divergence between the two backends.

    Not a hard gate — a FAIL means "re-check sampling/template", not "abort".
    """
    notes: list[str] = []
    verdict = "PASS"

    s_med = self_summ["len_p50"]
    o_med = or_summ["len_p50"]
    denom = max(s_med, o_med, 1.0)
    len_rel = abs(s_med - o_med) / denom
    notes.append(f"median-length relative diff = {len_rel:.2f} (self={s_med:.0f}, or={o_med:.0f})")
    if len_rel > 0.5:
        verdict = "FAIL"

    leak_diff = abs(_rate(self_leak) - _rate(or_leak))
    notes.append(
        f"leakage-rate diff = {leak_diff:.2f} "
        f"(self={_rate(self_leak):.2f}, or={_rate(or_leak):.2f})"
    )
    if leak_diff > 0.20:
        verdict = "FAIL"

    persp_diff = abs(_first_frac(self_summ["perspective_counts"]) - _first_frac(or_summ["perspective_counts"]))
    notes.append(
        f"first-person fraction diff = {persp_diff:.2f} "
        f"(self={_first_frac(self_summ['perspective_counts']):.2f}, "
        f"or={_first_frac(or_summ['perspective_counts']):.2f})"
    )
    if persp_diff > 0.30:
        verdict = "FAIL"

    empty_diff = abs(self_summ["empty"] - or_summ["empty"])
    if empty_diff > max(1, self_summ["n"] // 5):
        verdict = "FAIL"
        notes.append(f"empty-count diff = {empty_diff} (self={self_summ['empty']}, or={or_summ['empty']})")

    return verdict, notes


def write_report(
    self_summ: dict[str, Any],
    or_summ: dict[str, Any],
    self_leak: list[bool],
    or_leak: list[bool],
    samples: list[dict[str, str]],
    out_path: str | Path,
) -> str:
    """Write the fidelity markdown report and return the verdict string."""
    out_path = Path(out_path)
    verdict, notes = _verdict(self_summ, or_summ, self_leak, or_leak)

    def _pcounts(c: dict[str, int]) -> str:
        return f"first={c['first']}, third={c['third']}, other={c['other']}"

    lines: list[str] = []
    lines.append("# CoT fidelity: self-hosted (thinking-off) vs OpenRouter Qwen3-8B")
    lines.append("")
    lines.append(f"_Generated {_dt.datetime.now().isoformat(timespec='seconds')}._")
    lines.append("")
    lines.append(
        "Soft fidelity report (not a hard gate). Compares the self-hosted served "
        "CoT against the original OpenRouter reference on a fixed seed-42 sample. "
        "A FAIL means re-check the sampling params / chat template, not that a "
        "pipeline should abort."
    )
    lines.append("")
    lines.append(f"## Verdict: {verdict}")
    lines.append("")
    for note in notes:
        lines.append(f"- {note}")
    lines.append("")
    lines.append("## Distributions")
    lines.append("")
    lines.append("| metric | self-hosted | openrouter |")
    lines.append("| --- | --- | --- |")
    lines.append(f"| n | {self_summ['n']} | {or_summ['n']} |")
    lines.append(f"| empty | {self_summ['empty']} | {or_summ['empty']} |")
    lines.append(f"| len p25 | {self_summ['len_p25']:.0f} | {or_summ['len_p25']:.0f} |")
    lines.append(f"| len p50 | {self_summ['len_p50']:.0f} | {or_summ['len_p50']:.0f} |")
    lines.append(f"| len p75 | {self_summ['len_p75']:.0f} | {or_summ['len_p75']:.0f} |")
    lines.append(f"| perspective | {_pcounts(self_summ['perspective_counts'])} | {_pcounts(or_summ['perspective_counts'])} |")
    lines.append(f"| leakage rate | {_rate(self_leak):.2f} | {_rate(or_leak):.2f} |")
    lines.append("")
    lines.append("## Side-by-side (first 5 rows)")
    lines.append("")
    for i, s in enumerate(samples[:5]):
        lines.append(f"### Row {i}")
        lines.append("")
        lines.append(f"- **ground truth reply**: {_snip(s.get('ground_truth', ''))}")
        lines.append(f"- **self-hosted CoT**: {_snip(s.get('self', ''))}")
        lines.append(f"- **openrouter CoT**: {_snip(s.get('openrouter', ''))}")
        lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return verdict


# --------------------------------------------------------------------------- #
# Row loading + message building (shared by both backends).
# --------------------------------------------------------------------------- #
def sample_rows(input_path: str | Path, n: int, seed: int) -> list[dict[str, Any]]:
    """Deterministic seed-``seed`` sample of ``n`` rows from the SFT parquet."""
    import pandas as pd

    df = pd.read_parquet(input_path)
    take = min(n, len(df))
    sample = df.sample(n=take, random_state=seed).reset_index(drop=True)
    return [dict(r) for r in sample.to_dict(orient="records")]


def build_messages(row: dict[str, Any]) -> tuple[list[dict], str]:
    """Build the identical rationalize messages both backends receive.

    Mirrors ``generate_reasoning_for_row``'s first-attempt (no regen nudge)
    message list, using the shared prompt constants + ``_row_context``.
    """
    extra_info = dict(row.get("extra_info") or {})
    reward_model = dict(row.get("reward_model") or {})
    ground_truth = _as_text(reward_model.get("ground_truth"))
    messages = [
        {"role": "system", "content": RATIONALIZE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": RATIONALIZE_USER_TEMPLATE.format(
                context=_row_context(extra_info), ground_truth=ground_truth
            ),
        },
    ]
    return messages, ground_truth


# --------------------------------------------------------------------------- #
# Generation backends (network; only invoked from main).
# --------------------------------------------------------------------------- #
def gen_openrouter(messages: list[dict]) -> str:
    """Generate one CoT via OpenRouter with the upstream generate_cot payload.

    Model ``qwen/qwen3-8b``, Morph provider, reasoning OFF, no sampling,
    ``max_completion_tokens=4096``. Uses ``OPENROUTER_API_KEY`` from env.
    """
    from shared.api_client import post_chat_sync

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set in the environment")
    payload = {
        "model": OPENROUTER_MODEL,
        "messages": messages,
        "max_completion_tokens": MAX_COMPLETION_TOKENS,
        "provider": {"order": ["Morph"], "allow_fallbacks": False},
    }
    api_base = OPENROUTER_URL.rsplit("/chat/completions", 1)[0]
    return post_chat_sync(payload, api_base=api_base, api_key=api_key) or ""


def gen_selfhosted(messages: list[dict], endpoint: str) -> str:
    """Generate one CoT via a self-hosted vLLM replica (thinking-off).

    POSTs ``build_cot_payload("Qwen/Qwen3-8B", messages, max_completion_tokens=4096)``
    to ``<endpoint>/chat/completions``.
    """
    from shared.api_client import post_chat_sync

    payload = build_cot_payload(
        SELFHOSTED_MODEL, messages, max_completion_tokens=MAX_COMPLETION_TOKENS
    )
    api_base = endpoint.rstrip("/")
    api_key = os.environ.get("OPENAI_API_KEY", "dummy-self-hosted")
    return post_chat_sync(payload, api_base=api_base, api_key=api_key) or ""


# --------------------------------------------------------------------------- #
# JSON dump/merge plumbing so the two halves run independently.
# --------------------------------------------------------------------------- #
def _dump_path(out: str | Path, side: str) -> Path:
    return Path(out).parent / f"cot_fidelity_{side}.json"


def _write_dump(out: str | Path, side: str, texts: list[str], ground_truths: list[str], *, n: int, seed: int) -> Path:
    path = _dump_path(out, side)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"side": side, "n": n, "seed": seed, "texts": texts, "ground_truths": ground_truths},
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _load_dump(out: str | Path, side: str) -> list[str] | None:
    path = _dump_path(out, side)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return list(data.get("texts") or [])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CoT fidelity check: self-hosted (thinking-off) vs OpenRouter Qwen3-8B."
    )
    parser.add_argument("--n", type=int, default=20, help="Number of fixed sample rows.")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed.")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Source SFT parquet.")
    parser.add_argument("--served_url", default=None, help="Self-hosted endpoint base URL (omit for openrouter-only).")
    parser.add_argument("--out", default=DEFAULT_OUT, help="Report markdown path.")
    parser.add_argument(
        "--mode",
        choices=["both", "openrouter", "selfhosted"],
        default="both",
        help="Which backend(s) to run in this invocation.",
    )
    args = parser.parse_args()

    rows = sample_rows(args.input, args.n, args.seed)
    built = [build_messages(r) for r in rows]
    messages_list = [m for m, _ in built]
    ground_truths = [g for _, g in built]

    self_texts: list[str] | None = None
    or_texts: list[str] | None = None

    if args.mode in ("both", "openrouter"):
        print(f"[cot_fidelity] generating {len(messages_list)} OpenRouter traces...", flush=True)
        or_texts = [gen_openrouter(m) for m in messages_list]
        path = _write_dump(args.out, "openrouter", or_texts, ground_truths, n=args.n, seed=args.seed)
        print(f"[cot_fidelity] wrote OpenRouter dump -> {path}", flush=True)

    if args.mode in ("both", "selfhosted"):
        if not args.served_url:
            raise SystemExit("--served_url is required for mode 'selfhosted'/'both'")
        print(f"[cot_fidelity] generating {len(messages_list)} self-hosted traces...", flush=True)
        self_texts = [gen_selfhosted(m, args.served_url) for m in messages_list]
        path = _write_dump(args.out, "selfhosted", self_texts, ground_truths, n=args.n, seed=args.seed)
        print(f"[cot_fidelity] wrote self-hosted dump -> {path}", flush=True)

    # Merge in the other half from a prior run's dump if we only ran one side.
    if self_texts is None:
        self_texts = _load_dump(args.out, "selfhosted")
    if or_texts is None:
        or_texts = _load_dump(args.out, "openrouter")

    if self_texts is None or or_texts is None:
        missing = "self-hosted" if self_texts is None else "openrouter"
        print(
            f"[cot_fidelity] only one side present ({missing} dump missing); "
            f"wrote this side's JSON. Run the other side to produce {args.out}.",
            flush=True,
        )
        return

    if len(self_texts) != len(or_texts):
        raise SystemExit(
            f"dump length mismatch: self-hosted={len(self_texts)} openrouter={len(or_texts)} "
            "(were both run with the same --n/--seed/--input?)"
        )

    self_summ = summarize(self_texts)
    or_summ = summarize(or_texts)
    self_leak = leakage_flags(rows, self_texts)
    or_leak = leakage_flags(rows, or_texts)
    samples = [
        {"ground_truth": ground_truths[i], "self": self_texts[i], "openrouter": or_texts[i]}
        for i in range(min(5, len(self_texts)))
    ]
    verdict = write_report(self_summ, or_summ, self_leak, or_leak, samples, args.out)
    print(json.dumps({"out": args.out, "verdict": verdict, "self": self_summ, "openrouter": or_summ}, ensure_ascii=False))


if __name__ == "__main__":
    main()
