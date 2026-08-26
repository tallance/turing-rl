"""Decide whether the judge overfit run actually learned.

Checks that the tail of the training-accuracy series has saturated. A flat series near 0.5
means the loop is wired but learning nothing; a flat series near the tie payoff means the
model found the hedge instead of the signal.

WHERE THE SERIES COMES FROM
---------------------------
veRL writes no metrics file. Its ``trainer.logger`` list selects tracking *backends*
(wandb, console, mlflow, tensorboard, ...), none of which produce a per-step JSONL on disk.
``training/grpo/configs/qwen35_judge_grpo.yaml`` therefore enables the ``console`` backend
alongside wandb, which prints every logged metric to stdout — i.e. into the job's Slurm
``.out`` file. Point ``--metrics_file`` at that log:

    python scripts/judge_overfit_gate.py --metrics_file logs/judge_grpo-<jobid>.out

Two line formats are accepted, so a hand-exported wandb JSONL works too:

  * JSON object per line — ``{"reward/judge_acc/mean": 0.97, ...}``
  * veRL console line   — ``step:12 - reward/judge_acc/mean:0.970 - ...``

The console backend formats floats to three decimals, which is ample for a 0.95 gate but
is why the reported means are not bit-exact against wandb.

The default key is the name the metric actually carries once
``training/grpo/verl_metric_patch.py`` has aggregated it: the reward function emits
``judge_acc`` per sample and the patch logs it as ``reward/judge_acc/mean``.
"""

from __future__ import annotations

import argparse
import json

DEFAULT_METRIC_KEY = "reward/judge_acc/mean"


def parse_console_line(line: str, key: str) -> float | None:
    """Pull ``key`` out of a veRL console line: ``step:N - a/b:0.1 - c/d:0.2``."""
    for field in line.split(" - "):
        name, sep, value = field.strip().rpartition(":")
        if not sep or name != key:
            continue
        try:
            return float(value)
        except ValueError:
            return None
    return None


def read_metric_series(path: str, key: str) -> list[float]:
    """Pull one metric out of a veRL metrics log, skipping lines that lack it."""
    series: list[float] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                value = parse_console_line(line, key)
                if value is not None:
                    series.append(value)
                continue
            if isinstance(record, dict) and key in record:
                series.append(float(record[key]))
    return series


def gate_verdict(series: list[float], *, threshold: float, tail: int) -> dict:
    """Pass when the mean of the last ``tail`` points clears ``threshold``."""
    if not series:
        return {"passed": False, "reason": "no metric points found", "tail_mean": 0.0, "n": 0}
    window = series[-tail:] if tail > 0 else series
    tail_mean = sum(window) / len(window)
    passed = tail_mean >= threshold
    return {
        "passed": passed,
        "reason": "saturated" if passed else f"tail_mean {tail_mean:.4f} < {threshold}",
        "tail_mean": tail_mean,
        "n": len(series),
        "first": series[0],
        "last": series[-1],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Judge overfit gate check")
    parser.add_argument(
        "--metrics_file", "--metrics_jsonl", dest="metrics_file", required=True,
        help="veRL console log (the job's Slurm .out) or a JSONL of per-step metric dicts",
    )
    parser.add_argument("--key", default=DEFAULT_METRIC_KEY)
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument("--tail", type=int, default=3)
    args = parser.parse_args()

    verdict = gate_verdict(
        read_metric_series(args.metrics_file, args.key),
        threshold=args.threshold,
        tail=args.tail,
    )
    print(json.dumps(verdict, indent=2, sort_keys=True))
    return 0 if verdict["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
