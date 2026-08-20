from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.summarize_eval_timings import summarize_timings


class SummarizeEvalTimingsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.sacct = self.root / "sacct.psv"
        fieldnames = [
            "JobIDRaw",
            "JobName",
            "State",
            "ExitCode",
            "Submit",
            "Eligible",
            "Start",
            "End",
            "Elapsed",
            "ElapsedRaw",
            "ReqTRES",
            "AllocTRES",
        ]
        rows = [
            {
                "JobIDRaw": "1",
                "JobName": "te_g12t50_merge_12",
                "State": "COMPLETED",
                "ExitCode": "0:0",
                "Submit": "2026-08-20T10:00:00",
                "Eligible": "2026-08-20T10:00:00",
                "Start": "2026-08-20T10:00:10",
                "End": "2026-08-20T10:03:10",
                "Elapsed": "00:03:00",
                "ElapsedRaw": "180",
                "ReqTRES": "billing=16,cpu=16,mem=256G,node=1",
                "AllocTRES": "billing=16,cpu=16,mem=256G,node=1",
            },
            {
                "JobIDRaw": "2",
                "JobName": "te_g12t50_gen_12",
                "State": "COMPLETED",
                "ExitCode": "0:0",
                "Submit": "2026-08-20T10:03:10",
                "Eligible": "2026-08-20T10:03:20",
                "Start": "2026-08-20T10:04:20",
                "End": "2026-08-20T10:12:20",
                "Elapsed": "00:08:00",
                "ElapsedRaw": "480",
                "ReqTRES": "billing=8,cpu=8,gres/gpu=1,mem=64G,node=1",
                "AllocTRES": "billing=8,cpu=8,gres/gpu=1,mem=64G,node=1",
            },
            {
                "JobIDRaw": "3",
                "JobName": "te_g12t50_gemma4-12b_12",
                "State": "COMPLETED",
                "ExitCode": "0:0",
                "Submit": "2026-08-20T10:12:20",
                "Eligible": "2026-08-20T10:12:20",
                "Start": "2026-08-20T10:13:20",
                "End": "2026-08-20T10:43:20",
                "Elapsed": "00:30:00",
                "ElapsedRaw": "1800",
                "ReqTRES": "billing=64,cpu=64,gres/gpu=8,mem=512G,node=1",
                "AllocTRES": "billing=64,cpu=64,gres/gpu=8,mem=512G,node=1",
            },
        ]
        with self.sacct.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="|", lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)

        timing_dir = self.root / "raw/gemma-trained-step12/sweep/gemma4-12b/on"
        timing_dir.mkdir(parents=True)
        (timing_dir / "timing.json").write_text(
            json.dumps(
                {
                    "slurm_job_id": "3",
                    "cell_name": "gemma4-12b",
                    "model_startup_seconds": 120.0,
                    "scoring_seconds": 1680.0,
                    "instrumented_total_seconds": 1800.0,
                    "client_exit_code": 0,
                }
            )
        )

    def test_generates_job_table_and_stage_summary(self) -> None:
        jobs_out = self.root / "pipeline_jobs.csv"
        summary_out = self.root / "timing_summary.json"
        summary = summarize_timings(
            sacct_path=self.sacct,
            eval_root=self.root,
            jobs_out=jobs_out,
            summary_out=summary_out,
        )

        with jobs_out.open(newline="") as handle:
            jobs = list(csv.DictReader(handle))
        self.assertEqual([row["stage"] for row in jobs], ["merge", "generation", "judge"])
        self.assertEqual(jobs[1]["checkpoint"], "12")
        self.assertEqual(jobs[1]["gpus"], "1")
        self.assertEqual(jobs[1]["queue_wait_seconds"], "60.0")
        self.assertEqual(jobs[2]["judge"], "gemma4-12b")
        self.assertEqual(jobs[2]["model_startup_seconds"], "120.0")
        self.assertEqual(jobs[2]["scoring_seconds"], "1680.0")

        self.assertEqual(summary["job_count"], 3)
        self.assertEqual(summary["completed_job_count"], 3)
        self.assertEqual(summary["critical_path_active_seconds"], 2460.0)
        self.assertEqual(summary["stages"]["generation"]["median_active_seconds"], 480.0)
        self.assertEqual(summary["judges"]["gemma4-12b"]["total_model_startup_seconds"], 120.0)
        self.assertEqual(summary["judges"]["gemma4-12b"]["total_scoring_seconds"], 1680.0)
        self.assertEqual(json.loads(summary_out.read_text()), summary)


if __name__ == "__main__":
    unittest.main()
