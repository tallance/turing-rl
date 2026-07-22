"""Plot raw Turing-judge rubric fields over SFT epochs.

Unlike the normalized reward dump, this script reads ``judge_raw_content`` directly. Missing
or invalid primitive fields are never converted to zero. Mean trajectories use paired common
support: for a given model, judge, and field, only pair IDs with a valid raw value at every
epoch are averaged. Field-error trajectories use every call and report the fraction for which
that exact raw field is missing or invalid.

Usage:
  python scripts/plot_rubric_trajectory.py \
      [--raw_root results/2026-07-15-generator-sweep/raw] \
      [--out_dir results/2026-07-21-sft-checkpoint-trajectory] \
      [--mode on] [--judges qwen35-4b qwen35-9b qwen35-27b]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


RAW_FIELDS = {
    "immediate_target_score": "context targeting score",
    "human_goal_score": "goal plausibility score",
    "communication_style_score": "communication style score",
    "source_copy_penalty": "source-copy penalty",
    "assistant_like_penalty": "assistant-like penalty",
    "wrong_target_or_role_penalty": "wrong-target/role penalty",
    "unsupported_adversarial_reframing_penalty": "unsupported-reframing penalty",
}
SCORE_FIELDS = {
    "immediate_target_score",
    "human_goal_score",
    "communication_style_score",
}
MODELS = [
    ("qwen35-9b", "qwen3.5-9B"),
    ("qwen3-8b", "qwen3-8B"),
]
GENERATORS = {
    "qwen35-9b": {
        0: "qwen35-9b-base",
        1: "qwen35-9b-sft-ep1",
        2: "qwen35-9b-sft-ep2",
        3: "qwen35-9b-sft-ep3",
    },
    "qwen3-8b": {
        0: "qwen3-8b-base",
        1: "qwen3-8b-sft-ep1",
        2: "qwen3-8b-sft-ep2",
        3: "qwen3-8b-sft-ep3",
    },
}
DEFAULT_JUDGES = ["qwen35-4b", "qwen35-9b", "qwen35-27b"]
JUDGE_COLORS = {
    "qwen35-4b": "tab:blue",
    "qwen35-9b": "tab:orange",
    "qwen35-27b": "tab:green",
    "qwen35-397b": "tab:red",
    "qwen3-8b": "tab:purple",
}


def parse_json_object(text: Any) -> dict[str, Any] | None:
    """Parse a raw judge JSON object, tolerating surrounding text or fences."""
    if not isinstance(text, str) or not text.strip():
        return None
    stripped = text.strip()
    start, end = stripped.find("{"), stripped.rfind("}")
    candidates = [stripped]
    if start >= 0 and end > start:
        candidates.append(stripped[start : end + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _pair_id(row: dict[str, Any]) -> str:
    return str(
        row.get("pair_id")
        or f'{row.get("user_id")}::{row.get("post_id")}::{row.get("target_idx")}'
    )


def raw_generated_field(row: dict[str, Any], field: str) -> float | None:
    """Return a valid raw field for the generated A/B side, without fallback coercion."""
    parsed = parse_json_object(row.get("judge_raw_content"))
    if parsed is None:
        return None
    side = "b" if bool(row.get("generated_is_b")) else "a"
    value = parsed.get(f"{field}_{side}")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if 0.0 <= value <= 1.0 else None


def read_cell_frame(mode_dir: Path) -> pd.DataFrame:
    """Read one generator/judge/mode cell into one row per pair and raw primitive field."""
    records: list[dict[str, Any]] = []
    reward_dir = mode_dir / "reward"
    for jsonl_path in sorted(reward_dir.glob("*.jsonl")):
        for line in jsonl_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            record: dict[str, Any] = {"pair_id": _pair_id(row)}
            for field in RAW_FIELDS:
                record[field] = raw_generated_field(row, field)
            records.append(record)
    if not records:
        return pd.DataFrame(columns=["pair_id", *RAW_FIELDS])
    # Targeted reruns may append a newer result for a pair. The last sorted dump wins.
    return pd.DataFrame(records).drop_duplicates("pair_id", keep="last").reset_index(drop=True)


def summarize_trajectories(
    cells: dict[tuple[str, int, str], pd.DataFrame],
    *,
    judges: list[str],
) -> pd.DataFrame:
    """Build available-case/error metrics and paired-common-support means."""
    records: list[dict[str, Any]] = []
    for model_key, _ in MODELS:
        epochs = sorted(GENERATORS[model_key])
        for judge in judges:
            for field in RAW_FIELDS:
                frames = {epoch: cells.get((model_key, epoch, judge)) for epoch in epochs}
                complete = all(frame is not None and not frame.empty for frame in frames.values())
                common_ids: set[str] = set()
                if complete:
                    valid_sets = {
                        epoch: set(frame.loc[frame[field].notna(), "pair_id"].astype(str))
                        for epoch, frame in frames.items()
                        if frame is not None
                    }
                    common_ids = set.intersection(*valid_sets.values()) if valid_sets else set()

                for epoch in epochs:
                    frame = frames[epoch]
                    if frame is None or frame.empty:
                        continue
                    valid = frame[field].dropna()
                    paired = frame.loc[frame["pair_id"].astype(str).isin(common_ids), field]
                    records.append(
                        {
                            "model_key": model_key,
                            "epoch": epoch,
                            "generator": GENERATORS[model_key][epoch],
                            "judge": judge,
                            "field": field,
                            "n_calls": len(frame),
                            "n_valid": len(valid),
                            "n_field_error": len(frame) - len(valid),
                            "field_error_fraction": 1.0 - len(valid) / len(frame),
                            "available_mean": float(valid.mean()) if len(valid) else float("nan"),
                            "paired_n": len(paired),
                            "paired_mean": float(paired.mean()) if len(paired) else float("nan"),
                        }
                    )
    return pd.DataFrame(records)


def collect_summary(raw_root: Path, *, mode: str, judges: list[str]) -> pd.DataFrame:
    cells: dict[tuple[str, int, str], pd.DataFrame] = {}
    for model_key, epochs in GENERATORS.items():
        for epoch, generator in epochs.items():
            for judge in judges:
                mode_dir = raw_root / generator / "sweep" / judge / mode
                if mode_dir.is_dir():
                    cells[(model_key, epoch, judge)] = read_cell_frame(mode_dir)
    return summarize_trajectories(cells, judges=judges)


def _dynamic_upper(values: pd.Series, *, floor: float) -> float:
    finite = values.dropna()
    if finite.empty:
        return floor
    return min(1.0, max(floor, float(finite.max()) * 1.2))


def plot_field(
    summary: pd.DataFrame,
    *,
    field: str,
    label: str,
    mode: str,
    judges: list[str],
    out_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    field_df = summary[summary["field"] == field]
    fig, axes = plt.subplots(2, len(MODELS), figsize=(6.4 * len(MODELS), 8.0), squeeze=False)
    mean_upper = 1.0 if field in SCORE_FIELDS else _dynamic_upper(
        field_df["paired_mean"], floor=0.1
    )
    error_upper = _dynamic_upper(field_df["field_error_fraction"], floor=0.1)

    for col, (model_key, model_title) in enumerate(MODELS):
        model_df = field_df[field_df["model_key"] == model_key]
        mean_ax, error_ax = axes[0, col], axes[1, col]
        for judge in judges:
            judge_df = model_df[model_df["judge"] == judge].sort_values("epoch")
            if judge_df.empty:
                continue
            color = JUDGE_COLORS.get(judge)
            paired_n = int(judge_df["paired_n"].min())
            mean_ax.plot(
                judge_df["epoch"],
                judge_df["paired_mean"],
                marker="o",
                color=color,
                label=f"{judge} (paired n={paired_n})",
            )
            mean_ax.plot(
                judge_df["epoch"],
                judge_df["available_mean"],
                marker="x",
                linestyle="--",
                linewidth=1.2,
                color=color,
                label="_nolegend_",
            )
            error_ax.plot(
                judge_df["epoch"],
                judge_df["field_error_fraction"],
                marker="o",
                color=color,
                label=judge,
            )

        mean_ax.set_title(f"{model_title}: paired raw mean")
        mean_ax.set_ylim(0.0, mean_upper)
        error_ax.set_title(f"{model_title}: raw-field error fraction")
        error_ax.set_ylim(0.0, error_upper)
        for ax in (mean_ax, error_ax):
            ax.set_xticks([0, 1, 2, 3])
            ax.set_xlabel("SFT epoch (0 = base)")
            ax.grid(alpha=0.25)
        judge_legend = mean_ax.legend(title="judge", fontsize=7, loc="best")
        mean_ax.add_artist(judge_legend)
        mean_ax.legend(
            handles=[
                Line2D([0], [0], color="black", marker="o", label="paired common support"),
                Line2D(
                    [0], [0], color="black", marker="x", linestyle="--",
                    label="unpaired available cases",
                ),
            ],
            title="support",
            fontsize=7,
            loc="lower right",
        )
        error_ax.legend(title="judge", fontsize=7)
    axes[0, 0].set_ylabel(label)
    axes[1, 0].set_ylabel("missing or invalid field / all calls")
    fig.suptitle(
        f"Raw judge field: {field} — thinking {mode}\n"
        "Means use pair IDs valid at all epochs; errors are field-specific",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_overview(
    summary: pd.DataFrame,
    *,
    metric: str,
    ylabel: str,
    mode: str,
    judges: list[str],
    out_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fields = list(RAW_FIELDS)
    fig, axes = plt.subplots(
        len(fields), len(MODELS), figsize=(6.2 * len(MODELS), 2.7 * len(fields)), squeeze=False
    )
    for row, field in enumerate(fields):
        field_df = summary[summary["field"] == field]
        upper = 1.0
        if metric == "paired_mean" and field not in SCORE_FIELDS:
            upper = _dynamic_upper(field_df[metric], floor=0.1)
        elif metric == "field_error_fraction":
            upper = _dynamic_upper(field_df[metric], floor=0.1)
        for col, (model_key, model_title) in enumerate(MODELS):
            ax = axes[row, col]
            model_df = field_df[field_df["model_key"] == model_key]
            for judge in judges:
                judge_df = model_df[model_df["judge"] == judge].sort_values("epoch")
                if judge_df.empty:
                    continue
                ax.plot(
                    judge_df["epoch"], judge_df[metric], marker="o",
                    color=JUDGE_COLORS.get(judge), label=judge,
                )
            ax.set_ylim(0.0, upper)
            ax.set_xticks([0, 1, 2, 3])
            ax.grid(alpha=0.25)
            ax.set_title(f"{model_title}: {RAW_FIELDS[field]}", fontsize=9)
            if row == len(fields) - 1:
                ax.set_xlabel("SFT epoch (0 = base)")
            if col == 0:
                ax.set_ylabel(ylabel, fontsize=8)
            if row == 0:
                ax.legend(title="judge", fontsize=7)
    fig.suptitle(f"Raw rubric trajectory — thinking {mode}", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--raw_root", type=Path,
        default=repo / "results" / "2026-07-15-generator-sweep" / "raw",
    )
    ap.add_argument(
        "--out_dir", type=Path,
        default=repo / "results" / "2026-07-21-sft-checkpoint-trajectory",
    )
    ap.add_argument("--mode", default="on")
    ap.add_argument("--judges", nargs="+", default=DEFAULT_JUDGES)
    args = ap.parse_args()

    summary = collect_summary(args.raw_root, mode=args.mode, judges=args.judges)
    if summary.empty:
        raise SystemExit(f"no rubric cells found under {args.raw_root}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = args.out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / f"rubric_trajectory_{args.mode}.csv"
    summary.sort_values(["field", "model_key", "judge", "epoch"]).to_csv(csv_path, index=False)

    for field, label in RAW_FIELDS.items():
        out_path = plots_dir / f"traj_raw_{field}_{args.mode}.png"
        plot_field(
            summary, field=field, label=label, mode=args.mode,
            judges=args.judges, out_path=out_path,
        )
        print(f"[rubric-traj] wrote {out_path}", flush=True)

    plot_overview(
        summary,
        metric="paired_mean",
        ylabel="paired raw mean",
        mode=args.mode,
        judges=args.judges,
        out_path=plots_dir / f"traj_raw_rubric_overview_{args.mode}.png",
    )
    plot_overview(
        summary,
        metric="available_mean",
        ylabel="unpaired available-case mean",
        mode=args.mode,
        judges=args.judges,
        out_path=plots_dir / f"traj_raw_rubric_available_overview_{args.mode}.png",
    )
    plot_overview(
        summary,
        metric="field_error_fraction",
        ylabel="field error fraction",
        mode=args.mode,
        judges=args.judges,
        out_path=plots_dir / f"traj_raw_rubric_field_error_{args.mode}.png",
    )
    print(f"[rubric-traj] wrote {csv_path} + {len(RAW_FIELDS) + 3} plots", flush=True)


if __name__ == "__main__":
    main()
