"""Decide whether the judge overfit run actually learned.

Reads a veRL metrics JSONL and checks that the tail of the training-accuracy series has
saturated. A flat series near 0.5 means the loop is wired but learning nothing; a flat
series near the tie payoff means the model found the hedge instead of the signal.
"""

from __future__ import annotations

import argparse
import json


def read_metric_series(jsonl_path: str, key: str) -> list[float]:
    """Pull one metric out of a veRL metrics JSONL, skipping rows that lack it."""
    series: list[float] = []
    with open(jsonl_path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
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
    parser.add_argument("--metrics_jsonl", required=True)
    parser.add_argument("--key", default="judge_acc")
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument("--tail", type=int, default=3)
    args = parser.parse_args()

    verdict = gate_verdict(
        read_metric_series(args.metrics_jsonl, args.key),
        threshold=args.threshold,
        tail=args.tail,
    )
    print(json.dumps(verdict, indent=2, sort_keys=True))
    return 0 if verdict["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
