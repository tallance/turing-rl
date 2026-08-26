from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from configs.judge_sweep_cells import resolve_cell


ROOT = Path(__file__).resolve().parents[1]
MATRIX_LAUNCHER = ROOT / "scripts/launch_full_schema_eval.sh"


class Qwen08BEvalSupportTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)

    def matrix_env(self, *, steps: tuple[int, ...] = (0, 12)) -> dict[str, str]:
        eval_root = self.root / "matrix"
        pairs = eval_root / "raw/pairs"
        pairs.mkdir(parents=True)
        for step in steps:
            (pairs / f"gen_9b-qwen08btrain-step{step}_440.parquet").touch()
        return {
            **os.environ,
            "DRY": "1",
            "SKIP_SPLIT_GUARD": "1",
            "REPO": str(ROOT),
            "PY": sys.executable,
            "EVAL_ROOT": str(eval_root),
            "STEPS": " ".join(str(step) for step in steps),
            "JUDGES": "qwen35-0.8b gemma4-12b",
            "JUDGE_MODES": "off on",
            "GEN_KEY_PREFIX": "9b-qwen08btrain-step",
            "PAIRS_TAG": "440",
            "JOB_PREFIX": "te_q08t50",
            "BATCH_SIZE": "8",
        }

    def test_qwen08b_is_an_opt_in_tp1_eight_replica_cell(self) -> None:
        cell = resolve_cell("qwen35-0.8b")

        self.assertEqual(cell["model_id"], "Qwen/Qwen3.5-0.8B")
        self.assertEqual(cell["tp"], 1)
        self.assertEqual(cell["replicas"], 8)
        self.assertEqual(cell["concurrency"], 32)

    def test_matrix_applies_one_thinking_mode_per_judge(self) -> None:
        result = subprocess.run(
            ["bash", str(MATRIX_LAUNCHER)],
            env=self.matrix_env(),
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        planned = [line for line in result.stderr.splitlines() if line.startswith("[DRY] sbatch")]
        self.assertEqual(len(planned), 4)
        self.assertTrue(all("qwen35-0.8b" in line for line in planned[:2]))
        self.assertTrue(all("THINKING_MODE=off" in line for line in planned[:2]))
        self.assertTrue(all("gemma4-12b" in line for line in planned[2:]))
        self.assertTrue(all("THINKING_MODE=on" in line for line in planned[2:]))

    def test_matrix_reuses_only_authorized_completed_step0_cell(self) -> None:
        env = self.matrix_env(steps=(0,))
        env.update(
            {
                "JUDGES": "gemma4-12b qwen35-0.8b",
                "JUDGE_MODES": "on off",
                "REUSED_STEP0_CELLS": "gemma4-12b",
            }
        )
        reward = (
            Path(env["EVAL_ROOT"])
            / "raw/9b-qwen08btrain-step0/sweep/gemma4-12b/on/reward"
        )
        reward.mkdir(parents=True)
        (reward / "reward-123-1.jsonl").write_text("{}\n")
        manifest = Path(env["EVAL_ROOT"]) / "provenance/step0_reuse.json"
        manifest.parent.mkdir(parents=True)
        manifest.write_text("{}\n")

        result = subprocess.run(
            ["bash", str(MATRIX_LAUNCHER)],
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        planned = [line for line in result.stderr.splitlines() if line.startswith("[DRY] sbatch")]
        self.assertEqual(len(planned), 1)
        self.assertIn("qwen35-0.8b", planned[0])
        self.assertIn("THINKING_MODE=off", planned[0])
        self.assertIn("reusing verified step-0 cell gemma4-12b/on", result.stderr)

    def test_skip_only_batch_chains_continuation_after_current_controller(self) -> None:
        env = self.matrix_env(steps=(0, 12))
        env.update(
            {
                "JUDGES": "gemma4-12b",
                "JUDGE_MODES": "on",
                "REUSED_STEP0_CELLS": "gemma4-12b",
                "BATCH_SIZE": "1",
                "OFFSET": "0",
                "SLURM_JOB_ID": "777",
            }
        )
        reward = (
            Path(env["EVAL_ROOT"])
            / "raw/9b-qwen08btrain-step0/sweep/gemma4-12b/on/reward"
        )
        reward.mkdir(parents=True)
        (reward / "reward-123-1.jsonl").write_text("{}\n")
        manifest = Path(env["EVAL_ROOT"]) / "provenance/step0_reuse.json"
        manifest.parent.mkdir(parents=True)
        manifest.write_text("{}\n")

        result = subprocess.run(
            ["bash", str(MATRIX_LAUNCHER)],
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        planned = [line for line in result.stderr.splitlines() if line.startswith("[DRY] sbatch")]
        self.assertEqual(len(planned), 1)
        self.assertIn("--dependency=afterok:777", planned[0])
        self.assertNotIn("--dependency=afterok: ", planned[0])


if __name__ == "__main__":
    unittest.main()
