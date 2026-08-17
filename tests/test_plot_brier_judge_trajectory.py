from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "plot_brier_judge_trajectory.py"
BASE_CELLS = ("qwen35-4b", "qwen35-9b", "qwen35-27b", "gemma4-12b", "gemma4-31b")
FIELDS = ("checkpoint", "n_scored", "likert_mean", "win_rate_ge5")


def _write_summary(path: Path, steps: tuple[int, ...], n: int = 880) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for i, step in enumerate(steps):
            writer.writerow(
                {
                    "checkpoint": f"9b-full5ep-step{step}",
                    "n_scored": n,
                    "likert_mean": 4.0 + i / 10,
                    "win_rate_ge5": 0.5 + i / 100,
                }
            )


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    base = tmp_path / "base"
    full_steps = tuple(range(0, 321, 32))
    for cell in BASE_CELLS:
        _write_summary(base / f"summary_{cell}.csv", full_steps)
    brier = tmp_path / "summary_judge-9b-brier.csv"
    _write_summary(brier, (0, 64, 128, 192, 256, 320), n=872)
    return base, brier, tmp_path / "overlay.png"


def test_renders_five_dense_curves_and_sparse_brier_trajectory(tmp_path: Path) -> None:
    base, brier, out = _fixture(tmp_path)
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--base-eval-root",
            str(base),
            "--brier-summary",
            str(brier),
            "--out",
            str(out),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert out.is_file() and out.stat().st_size > 10_000
    with Image.open(out) as image:
        assert image.width > 2000
        assert image.height > 700


def test_rejects_a_non_epoch_brier_checkpoint(tmp_path: Path) -> None:
    base, brier, out = _fixture(tmp_path)
    _write_summary(brier, (0, 32, 64, 128, 192, 256, 320), n=872)

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--base-eval-root",
            str(base),
            "--brier-summary",
            str(brier),
            "--out",
            str(out),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "expected Brier steps [0, 64, 128, 192, 256, 320]" in result.stderr
