from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/slurm/judge_sweep_cell.sh"


class JudgeSweepTimingMarkersTest(unittest.TestCase):
    def test_writes_machine_readable_startup_and_scoring_timings(self) -> None:
        source = SCRIPT.read_text()

        required_fragments = (
            "TIMING_JOB_STARTED_EPOCH",
            "TIMING_SERVERS_READY_EPOCH",
            "TIMING_CLIENTS_FINISHED_EPOCH",
            'MODE_DIR/timing.json',
            '"model_startup_seconds"',
            '"scoring_seconds"',
            '"instrumented_total_seconds"',
            '"client_exit_code"',
        )
        for fragment in required_fragments:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, source)

        self.assertLess(
            source.index("TIMING_SERVERS_READY_EPOCH"),
            source.index("CLIENT_PIDS=()"),
        )
        self.assertLess(
            source.index("TIMING_CLIENTS_FINISHED_EPOCH"),
            source.index('"scoring_seconds"'),
        )


if __name__ == "__main__":
    unittest.main()
