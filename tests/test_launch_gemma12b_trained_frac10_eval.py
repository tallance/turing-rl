from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts/launch_gemma12b_trained_frac10_eval.sh"


class LaunchGemma12BTrainedFrac10EvalTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        root = Path(self.temp_dir.name)
        self.source = root / "source-eval"
        self.destination = root / "gemma12b-trained-eval"
        source_key = "9b-train10pct-step0"
        pair_dir = self.source / "raw/pairs"
        pair_dir.mkdir(parents=True)
        pair_path = pair_dir / f"gen_{source_key}_2.parquet"
        pair_path.write_bytes(b"pair bytes\n")
        keys = [("u1", "p1", 0), ("u2", "p2", 1)]
        for index, cell in enumerate(("gemma4-12b", "qwen35-9b"), start=1):
            mode = self.source / "raw" / source_key / "sweep" / cell / "on"
            reward = mode / "reward"
            reward.mkdir(parents=True)
            with (reward / f"reward-{index}.jsonl").open("w") as handle:
                for user_id, post_id, target_idx in keys:
                    handle.write(
                        json.dumps(
                            {"user_id": user_id, "post_id": post_id, "target_idx": target_idx}
                        )
                        + "\n"
                    )
            (mode / "run_metadata.json").write_text(
                json.dumps(
                    {
                        "cell_name": cell,
                        "thinking_mode": "on",
                        "slurm_job_id": str(100 + index),
                        "pair_source": str(pair_path),
                        "n_pairs_total": 2,
                    }
                )
            )

    def test_reuses_step0_then_submits_six_epoch_boundary_merges(self) -> None:
        env = {
            **os.environ,
            "DRY": "1",
            "PY": sys.executable,
            "TURING_RL_WORK_ROOT": str(ROOT),
            "TURING_RL_CODE_ROOT": str(ROOT),
            "TURING_RL_RUN_ROOT": str(self.destination),
            "EVAL_ROOT": str(self.destination),
            "STEP0_SOURCE_ROOT": str(self.source),
            "EVAL_ROWS": "2",
            "PAIRS_TAG": "2",
        }
        result = subprocess.run(
            ["bash", str(LAUNCHER)],
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        planned = [line for line in result.stderr.splitlines() if line.startswith("[DRY] sbatch")]
        merges = [line for line in planned if "_merge_" in line]
        self.assertEqual(len(merges), 6)
        for step, line in zip((12, 24, 36, 48, 60, 72), merges):
            self.assertIn(f"STEP={step}", line)
        self.assertEqual(sum("_continue" in line for line in planned), 1)

        manifest = json.loads((self.destination / "provenance/step0_reuse.json").read_text())
        self.assertEqual(manifest["cells"], ["gemma4-12b", "qwen35-9b"])
        self.assertTrue(
            (
                self.destination
                / "raw/pairs/gen_9b-gemma12btrain-step0_2.parquet"
            ).is_file()
        )

    def test_pins_distinct_run_and_requested_judges(self) -> None:
        source = LAUNCHER.read_text()
        for fragment in (
            "9b_frac10_20ep_gemma12b_kl1e4_lr1e4_temp1",
            'STEPS="12 24 36 48 60 72"',
            'MERGE_STEPS="12 24 36 48 60 72"',
            'JUDGES="gemma4-12b qwen35-9b"',
            "9b-gemma12btrain-step",
            "test_seed42_n440.parquet",
            "PHASE=merge",
        ):
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, source)


if __name__ == "__main__":
    unittest.main()
