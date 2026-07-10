"""Unit tests for the per-cell calibration extrapolation.

Only the pure extrapolation function is exercised here; the aggregator/main
reads per-cell run_metadata.json on the cluster and is validated there.
"""

from scripts.calibration_report import extrapolate_wall_hours


def test_extrapolate():
    assert round(extrapolate_wall_hours(100, 300, 1760), 2) == 1.47


def test_gate_over_4h():
    assert extrapolate_wall_hours(100, 3600, 1760) > 4
