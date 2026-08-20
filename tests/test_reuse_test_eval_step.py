from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.reuse_test_eval_step import reuse_step


CELLS = ("gemma4-12b", "qwen35-9b")
MODELS = {"gemma4-12b": "google/gemma-4-12B-it", "qwen35-9b": "Qwen/Qwen3.5-9B"}
CONCURRENCY = {"gemma4-12b": 4, "qwen35-9b": 32}


class ReuseTestEvalStepTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        root = Path(self.temp_dir.name)
        self.source = root / "source"
        self.destination = root / "destination"
        self.source_key = "old-step0"
        self.destination_key = "gemma-trained-step0"
        pair_dir = self.source / "raw/pairs"
        pair_dir.mkdir(parents=True)
        self.pair_path = pair_dir / f"gen_{self.source_key}_2.parquet"
        self.pair_path.write_bytes(b"canonical parquet bytes\n")
        self.keys = {("u1", "p1", 0), ("u2", "p2", 1)}
        self.eval_parquet = root / "test_seed42_n2.parquet"
        self.eval_parquet.write_bytes(b"eval parquet bytes\n")
        generator = self.source / "raw/generator" / self.source_key
        generator.mkdir(parents=True)
        (generator / "gen_metadata.json").write_text(
            json.dumps(
                {
                    "gen_key": self.source_key,
                    "model_id": "/runtime/checkpoints/sft/qwen35_9b_prism_full_s42_bf16_fsdp_nopack_epochsave/merged_ep3",
                    "checkpoint_dir": "",
                    "base_model": True,
                    "test_parquet": str(self.eval_parquet),
                    "eval_expect": "heldout",
                    "gen_num": 1,
                    "backend": "vllm",
                    "sampling_overrides": "--temperature 0.7 --top_p 0.8 --top_k 20 --max_tokens 1024 --vllm_truncate_prompt_tokens 12500 --vllm_max_model_len 13524",
                    "slurm_job_id": "12344",
                }
            )
        )
        for cell in CELLS:
            mode = self.source / "raw" / self.source_key / "sweep" / cell / "on"
            reward = mode / "reward"
            reward.mkdir(parents=True)
            with (reward / f"reward-{cell}.jsonl").open("w") as handle:
                for user_id, post_id, target_idx in sorted(self.keys):
                    handle.write(
                        json.dumps(
                            {
                                "user_id": user_id,
                                "post_id": post_id,
                                "target_idx": target_idx,
                                "rating_gen_first": 5,
                            }
                        )
                        + "\n"
                    )
            (mode / "run_metadata.json").write_text(
                json.dumps(
                    {
                        "cell_name": cell,
                        "model": MODELS[cell],
                        "thinking_mode": "on",
                        "num_endpoints": 8,
                        "concurrency_per_endpoint": CONCURRENCY[cell],
                        "sampling": '{"repetition_penalty":1.1,"temperature":0.6}',
                        "slurm_job_id": "12345" if cell == CELLS[0] else "12346",
                        "pair_source": str(self.pair_path),
                        "n_pairs_total": 2,
                        "json_schema": "1",
                        "enable_thinking": "1",
                        "max_completion_tokens": "8192",
                        "disable_openrouter_extras": "1",
                    }
                )
            )
            (mode / "http").mkdir()
            (mode / "http/diagnostic.jsonl").write_text("not required for reuse\n")

    def reuse(self) -> dict[str, object]:
        return reuse_step(
            source_root=self.source,
            destination_root=self.destination,
            source_gen_key=self.source_key,
            destination_gen_key=self.destination_key,
            pairs_tag=2,
            cells=CELLS,
            expect_pairs=2,
            expected_eval_parquet=self.eval_parquet,
            pair_key_reader=lambda _path, _expect: self.keys,
        )

    def test_copies_requested_cells_and_records_verified_hashes(self) -> None:
        manifest = self.reuse()

        copied_pair = self.destination / "raw/pairs" / f"gen_{self.destination_key}_2.parquet"
        self.assertEqual(copied_pair.read_bytes(), self.pair_path.read_bytes())
        self.assertEqual(manifest["pair_sha256"], manifest["destination_pair_sha256"])
        self.assertEqual(manifest["expected_pairs"], 2)
        self.assertEqual(manifest["source_job_ids"], {"gemma4-12b": "12345", "qwen35-9b": "12346"})
        for cell in CELLS:
            copied = self.destination / "raw" / self.destination_key / "sweep" / cell / "on"
            self.assertTrue((copied / "run_metadata.json").is_file())
            self.assertFalse((copied / "http").exists())
            self.assertEqual(manifest["cell_tree_sha256"][cell], manifest["destination_cell_tree_sha256"][cell])
        stored = json.loads(
            (self.destination / "provenance/step0_reuse.json").read_text()
        )
        self.assertEqual(stored, manifest)

    def test_refuses_an_existing_destination_artifact(self) -> None:
        pair_dir = self.destination / "raw/pairs"
        pair_dir.mkdir(parents=True)
        (pair_dir / f"gen_{self.destination_key}_2.parquet").write_bytes(b"stale")

        with self.assertRaisesRegex(ValueError, "existing evaluation payload"):
            self.reuse()

    def test_refuses_other_existing_evaluation_payload(self) -> None:
        stale = self.destination / "models/step12"
        stale.mkdir(parents=True)

        with self.assertRaisesRegex(ValueError, "existing evaluation payload"):
            self.reuse()

    def test_refuses_incomplete_reward_keys(self) -> None:
        reward = (
            self.source
            / "raw"
            / self.source_key
            / "sweep"
            / CELLS[0]
            / "on/reward"
            / f"reward-{CELLS[0]}.jsonl"
        )
        reward.write_text(json.dumps({"user_id": "u1", "post_id": "p1", "target_idx": 0}) + "\n")

        with self.assertRaisesRegex(ValueError, "expected 2 unique keys"):
            self.reuse()

    def test_refuses_key_mismatch_between_judges(self) -> None:
        reward = (
            self.source
            / "raw"
            / self.source_key
            / "sweep"
            / CELLS[1]
            / "on/reward"
            / f"reward-{CELLS[1]}.jsonl"
        )
        reward.write_text(
            json.dumps({"user_id": "u1", "post_id": "p1", "target_idx": 0})
            + "\n"
            + json.dumps({"user_id": "different", "post_id": "p2", "target_idx": 1})
            + "\n"
        )

        with self.assertRaisesRegex(ValueError, "key set differs"):
            self.reuse()

    def test_refuses_missing_source_job_id(self) -> None:
        metadata = (
            self.source
            / "raw"
            / self.source_key
            / "sweep"
            / CELLS[0]
            / "on/run_metadata.json"
        )
        data = json.loads(metadata.read_text())
        data.pop("slurm_job_id")
        metadata.write_text(json.dumps(data))

        with self.assertRaisesRegex(ValueError, "slurm_job_id"):
            self.reuse()

    def test_refuses_wrong_judge_configuration(self) -> None:
        metadata = (
            self.source
            / "raw"
            / self.source_key
            / "sweep"
            / CELLS[0]
            / "on/run_metadata.json"
        )
        data = json.loads(metadata.read_text())
        data["json_schema"] = "0"
        metadata.write_text(json.dumps(data))

        with self.assertRaisesRegex(ValueError, "json_schema"):
            self.reuse()

    def test_refuses_reward_keys_that_do_not_match_pair_parquet(self) -> None:
        with self.assertRaisesRegex(ValueError, "pair parquet key set differs"):
            reuse_step(
                source_root=self.source,
                destination_root=self.destination,
                source_gen_key=self.source_key,
                destination_gen_key=self.destination_key,
                pairs_tag=2,
                cells=CELLS,
                expect_pairs=2,
                expected_eval_parquet=self.eval_parquet,
                pair_key_reader=lambda _path, _expect: {
                    ("other", "pair", 9),
                    ("u2", "p2", 1),
                },
            )


if __name__ == "__main__":
    unittest.main()
