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
import math
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
COMBINED_METRICS = {
    "immediate_target_score": "context targeting score",
    "human_goal_score": "goal plausibility score",
    "communication_style_score": "communication style score",
    "normalized_generated_rating": "normalized effective rating (higher = generated preferred)",
    "accuracy": "accuracy (judge picks human)",
}
DIFFERENCE_METRICS = {
    "immediate_target_score_diff": "context targeting difference (generated - human)",
    "human_goal_score_diff": "goal plausibility difference (generated - human)",
    "communication_style_score_diff": "communication style difference (generated - human)",
    "normalized_generated_rating": "normalized effective rating (higher = generated preferred)",
    "accuracy": "accuracy (judge picks human)",
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


def raw_field_difference(row: dict[str, Any], field: str) -> float | None:
    """Return generated minus human for a valid raw side-specific primitive field."""
    parsed = parse_json_object(row.get("judge_raw_content"))
    if parsed is None:
        return None
    generated_side = "b" if bool(row.get("generated_is_b")) else "a"
    human_side = "a" if generated_side == "b" else "b"
    values = []
    for side in (generated_side, human_side):
        value = parsed.get(f"{field}_{side}")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        value = float(value)
        if not 0.0 <= value <= 1.0:
            return None
        values.append(value)
    return values[0] - values[1]


def effective_oriented_rating(row: dict[str, Any]) -> float | None:
    """Return the pipeline's effective rating, generated-oriented and normalized to 0-1."""
    rating = row.get("rating_gt_first")
    if rating is None:
        rating = row.get("rating_gen_first")
    if isinstance(rating, bool) or not isinstance(rating, (int, float)):
        return None
    if float(rating) != int(rating) or not 1 <= int(rating) <= 7:
        return None
    generated_rating = int(rating) if bool(row.get("generated_is_b")) else 8 - int(rating)
    return (generated_rating - 1) / 6.0


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
            for field in SCORE_FIELDS:
                record[f"{field}_diff"] = raw_field_difference(row, field)
            normalized_rating = effective_oriented_rating(row)
            record["normalized_generated_rating"] = normalized_rating
            if normalized_rating is None or normalized_rating == 0.5:
                record["picked_human"] = None
            else:
                record["picked_human"] = float(normalized_rating < 0.5)
            records.append(record)
    if not records:
        return pd.DataFrame(
            columns=["pair_id", *RAW_FIELDS, "normalized_generated_rating", "picked_human"]
        )
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
    cells = collect_cells(raw_root, mode=mode, judges=judges)
    return summarize_trajectories(cells, judges=judges)


def collect_cells(
    raw_root: Path, *, mode: str, judges: list[str]
) -> dict[tuple[str, int, str], pd.DataFrame]:
    cells: dict[tuple[str, int, str], pd.DataFrame] = {}
    for model_key, epochs in GENERATORS.items():
        for epoch, generator in epochs.items():
            for judge in judges:
                mode_dir = raw_root / generator / "sweep" / judge / mode
                if mode_dir.is_dir():
                    cells[(model_key, epoch, judge)] = read_cell_frame(mode_dir)
    return cells


def summarize_combined_metrics(
    cells: dict[tuple[str, int, str], pd.DataFrame], *, judges: list[str]
) -> pd.DataFrame:
    """Summarize available-case scores, raw normalized ratings, and human-pick accuracy."""
    records: list[dict[str, Any]] = []
    source_columns = {
        "immediate_target_score": "immediate_target_score",
        "human_goal_score": "human_goal_score",
        "communication_style_score": "communication_style_score",
        "normalized_generated_rating": "normalized_generated_rating",
        "accuracy": "picked_human",
    }
    for model_key, _ in MODELS:
        for epoch, generator in GENERATORS[model_key].items():
            for judge in judges:
                frame = cells.get((model_key, epoch, judge))
                if frame is None or frame.empty:
                    continue
                for metric, column in source_columns.items():
                    values = frame[column].dropna().astype(float)
                    records.append(
                        {
                            "model_key": model_key,
                            "epoch": epoch,
                            "generator": generator,
                            "judge": judge,
                            "metric": metric,
                            "n": len(values),
                            "mean": float(values.mean()) if len(values) else float("nan"),
                            "std": (
                                float(values.std(ddof=1))
                                if metric != "accuracy" and len(values) > 1
                                else float("nan")
                            ),
                        }
                    )
    return pd.DataFrame(records)


def summarize_difference_metrics(
    cells: dict[tuple[str, int, str], pd.DataFrame], *, judges: list[str]
) -> pd.DataFrame:
    """Summarize raw score differences plus effective rating and accuracy."""
    source_columns = {
        "immediate_target_score_diff": "immediate_target_score_diff",
        "human_goal_score_diff": "human_goal_score_diff",
        "communication_style_score_diff": "communication_style_score_diff",
        "normalized_generated_rating": "normalized_generated_rating",
        "accuracy": "picked_human",
    }
    records: list[dict[str, Any]] = []
    for model_key, _ in MODELS:
        for epoch, generator in GENERATORS[model_key].items():
            for judge in judges:
                frame = cells.get((model_key, epoch, judge))
                if frame is None or frame.empty:
                    continue
                for metric, column in source_columns.items():
                    values = frame[column].dropna().astype(float)
                    records.append(
                        {
                            "model_key": model_key,
                            "epoch": epoch,
                            "generator": generator,
                            "judge": judge,
                            "metric": metric,
                            "n": len(values),
                            "mean": float(values.mean()) if len(values) else float("nan"),
                            "std": (
                                float(values.std(ddof=1))
                                if metric != "accuracy" and len(values) > 1
                                else float("nan")
                            ),
                        }
                    )
    return pd.DataFrame(records)


def _dynamic_upper(values: pd.Series, *, floor: float) -> float:
    finite = values.dropna()
    if finite.empty:
        return floor
    return min(1.0, max(floor, float(finite.max()) * 1.2))


def common_score_ylim(summary: pd.DataFrame, metric: str) -> tuple[float, float]:
    """Return one tight, outward-rounded y-range spanning all three score fields."""
    values = summary.loc[summary["field"].isin(SCORE_FIELDS), metric].dropna()
    if values.empty:
        return 0.0, 1.0
    low, high = float(values.min()), float(values.max())
    padding = max((high - low) * 0.1, 0.025)
    step = 0.05
    lower = max(0.0, math.floor((low - padding) / step) * step)
    upper = min(1.0, math.ceil((high + padding) / step) * step)
    return lower, upper


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

        mean_ax.set_title(f"{model_title}: raw mean (paired solid, unpaired dashed)")
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
        "Solid = pair IDs valid at all epochs; dashed = available cases; errors are field-specific",
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
    score_ylim = (
        common_score_ylim(summary, metric)
        if metric in {"paired_mean", "available_mean"}
        else None
    )
    fig, axes = plt.subplots(
        len(fields), len(MODELS), figsize=(6.2 * len(MODELS), 2.7 * len(fields)), squeeze=False
    )
    for row, field in enumerate(fields):
        field_df = summary[summary["field"] == field]
        upper = 1.0
        if metric in {"paired_mean", "available_mean"} and field not in SCORE_FIELDS:
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
            ax.set_ylim(*(score_ylim if field in SCORE_FIELDS and score_ylim else (0.0, upper)))
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


def plot_scores_rating_accuracy(
    summary: pd.DataFrame, *, mode: str, judges: list[str], out_path: Path
) -> None:
    """Plot the three scores, oriented normalized rating, and human-pick accuracy."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = list(COMBINED_METRICS)
    fig, axes = plt.subplots(
        len(metrics), len(MODELS), figsize=(6.4 * len(MODELS), 2.9 * len(metrics)), squeeze=False
    )
    for row, metric in enumerate(metrics):
        metric_df = summary[summary["metric"] == metric]
        for col, (model_key, model_title) in enumerate(MODELS):
            ax = axes[row, col]
            model_df = metric_df[metric_df["model_key"] == model_key]
            for judge in judges:
                judge_df = model_df[model_df["judge"] == judge].sort_values("epoch")
                if judge_df.empty:
                    continue
                x = judge_df["epoch"].to_numpy(dtype=float)
                mean = judge_df["mean"].to_numpy(dtype=float)
                color = JUDGE_COLORS.get(judge)
                ax.plot(x, mean, marker="o", color=color, label=judge)
                if metric != "accuracy":
                    std = judge_df["std"].fillna(0.0).to_numpy(dtype=float)
                    ax.fill_between(
                        x,
                        (mean - std).clip(0.0, 1.0),
                        (mean + std).clip(0.0, 1.0),
                        color=color,
                        alpha=0.12,
                        linewidth=0,
                    )
            if metric == "normalized_generated_rating":
                ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
            elif metric == "accuracy":
                ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
            ax.set_ylim(0.0, 1.0)
            ax.set_xticks([0, 1, 2, 3])
            ax.grid(alpha=0.25)
            ax.set_title(f"{model_title}: {COMBINED_METRICS[metric]}", fontsize=9)
            if row == len(metrics) - 1:
                ax.set_xlabel("SFT epoch (0 = base)")
            if col == 0:
                ax.set_ylabel("mean" if metric != "accuracy" else "fraction", fontsize=8)
            if row == 0:
                ax.legend(title="judge", fontsize=7)
    fig.suptitle(
        f"Scores, normalized generated rating, and accuracy — thinking {mode}\n"
        "Shading = ±1 SD for scores/rating; accuracy excludes ties",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_differences_rating_accuracy(
    summary: pd.DataFrame, *, mode: str, judges: list[str], out_path: Path
) -> None:
    """Plot generated-minus-human raw score differences, rating, and accuracy."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = list(DIFFERENCE_METRICS)
    fig, axes = plt.subplots(
        len(metrics), len(MODELS), figsize=(6.4 * len(MODELS), 2.9 * len(metrics)), squeeze=False
    )
    for row, metric in enumerate(metrics):
        metric_df = summary[summary["metric"] == metric]
        for col, (model_key, model_title) in enumerate(MODELS):
            ax = axes[row, col]
            model_df = metric_df[metric_df["model_key"] == model_key]
            for judge in judges:
                judge_df = model_df[model_df["judge"] == judge].sort_values("epoch")
                if judge_df.empty:
                    continue
                x = judge_df["epoch"].to_numpy(dtype=float)
                mean = judge_df["mean"].to_numpy(dtype=float)
                color = JUDGE_COLORS.get(judge)
                ax.plot(x, mean, marker="o", color=color, label=judge)
                if metric != "accuracy":
                    std = judge_df["std"].fillna(0.0).to_numpy(dtype=float)
                    lower, upper = mean - std, mean + std
                    if metric == "normalized_generated_rating":
                        lower, upper = lower.clip(0.0, 1.0), upper.clip(0.0, 1.0)
                    else:
                        lower, upper = lower.clip(-1.0, 1.0), upper.clip(-1.0, 1.0)
                    ax.fill_between(x, lower, upper, color=color, alpha=0.12, linewidth=0)
            if metric.endswith("_diff"):
                ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
                ax.set_ylim(-1.0, 1.0)
            else:
                ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
                ax.set_ylim(0.0, 1.0)
            ax.set_xticks([0, 1, 2, 3])
            ax.grid(alpha=0.25)
            ax.set_title(f"{model_title}: {DIFFERENCE_METRICS[metric]}", fontsize=9)
            if row == len(metrics) - 1:
                ax.set_xlabel("SFT epoch (0 = base)")
            if col == 0:
                ax.set_ylabel("mean difference" if metric.endswith("_diff") else "mean", fontsize=8)
            if row == 0:
                ax.legend(title="judge", fontsize=7)
    fig.suptitle(
        f"Generated-vs-human score differences, normalized rating, and accuracy — thinking {mode}\n"
        "Shading = ±1 SD for score differences/rating; accuracy excludes ties",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
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

    cells = collect_cells(args.raw_root, mode=args.mode, judges=args.judges)
    summary = summarize_trajectories(cells, judges=args.judges)
    if summary.empty:
        raise SystemExit(f"no rubric cells found under {args.raw_root}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = args.out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / f"rubric_trajectory_{args.mode}.csv"
    summary.sort_values(["field", "model_key", "judge", "epoch"]).to_csv(csv_path, index=False)
    combined = summarize_combined_metrics(cells, judges=args.judges)
    combined_csv_path = args.out_dir / f"scores_rating_accuracy_trajectory_{args.mode}.csv"
    combined.sort_values(["metric", "model_key", "judge", "epoch"]).to_csv(
        combined_csv_path, index=False
    )
    differences = summarize_difference_metrics(cells, judges=args.judges)
    differences_csv_path = args.out_dir / f"score_differences_rating_accuracy_{args.mode}.csv"
    differences.sort_values(["metric", "model_key", "judge", "epoch"]).to_csv(
        differences_csv_path, index=False
    )

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
    combined_plot = plots_dir / f"traj_scores_rating_accuracy_{args.mode}.png"
    plot_scores_rating_accuracy(
        combined, mode=args.mode, judges=args.judges, out_path=combined_plot
    )
    differences_plot = plots_dir / f"traj_score_differences_rating_accuracy_{args.mode}.png"
    plot_differences_rating_accuracy(
        differences, mode=args.mode, judges=args.judges, out_path=differences_plot
    )
    print(
        f"[rubric-traj] wrote {csv_path}, {combined_csv_path}, {differences_csv_path} "
        f"+ {len(RAW_FIELDS) + 5} plots",
        flush=True,
    )


if __name__ == "__main__":
    main()
