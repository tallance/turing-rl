from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAJECTORY_LAUNCHER = ROOT / "scripts/launch_qwen08b_trained_frac10_eval.sh"


class Qwen08BTrainedFrac10EvalTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)

    def test_dedicated_launcher_pins_run_steps_order_and_modes(self) -> None:
        eval_root = self.root / "trajectory"
        source = self.root / "source.parquet"
        source.touch()
        env = {
            **os.environ,
            "DRY": "1",
            "PY": sys.executable,
            "TURING_RL_WORK_ROOT": str(ROOT),
            "TURING_RL_CODE_ROOT": str(ROOT),
            "TURING_RL_INPUT_DATA_ROOT": str(self.root),
            "TURING_RL_GENERATED_DATA_ROOT": str(self.root),
            "TURING_RL_RUN_ROOT": str(eval_root),
            "EVAL_ROOT": str(eval_root),
            "SOURCE_EVAL_PARQUET": str(source),
            "EVAL_PARQUET": str(self.root / "test_seed42_n440.parquet"),
            "REUSE_STEP0": "0",
        }
        result = subprocess.run(
            ["bash", str(TRAJECTORY_LAUNCHER)],
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        source_text = TRAJECTORY_LAUNCHER.read_text()
        for fragment in (
            "9b_frac10_20ep_qwen08b_nothink_kl1e4_lr1e4_temp1",
            'STEPS="0 12 24 36 48 60 72 84 96 108 120"',
            'MERGE_STEPS="12 24 36 48 60 72 84 96 108 120"',
            'JUDGES="qwen35-0.8b gemma4-12b gemma4-31b qwen35-9b"',
            'JUDGE_MODES="off on on on"',
            'REUSED_STEP0_CELLS="gemma4-12b gemma4-31b qwen35-9b"',
            "9b-qwen08btrain-step",
        ):
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, source_text)
        self.assertIn("[DRY] reuse verified step 0", result.stderr)


if __name__ == "__main__":
    unittest.main()
