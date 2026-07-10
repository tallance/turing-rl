from scripts.run_judge_sweep_cell import cell_env, shard_indices, cell_output_dirs


def test_cell_env_locks_config():
    env = cell_env(model_id="Qwen/Qwen3-8B", mode="off", sampling={"temperature": 0.7},
                   out_dir="/tmp/x")
    assert env["PERSONA_JUDGE_JSON_SCHEMA"] == "1"
    assert env["PERSONA_JUDGE_DUMP_RATE"] == "1.0"
    assert env["PERSONA_JUDGE_ENABLE_THINKING"] == "0"
    assert env["PERSONA_DISABLE_OPENROUTER_EXTRAS"] == "1"
    assert env["JUDGE_MODEL"] == "Qwen/Qwen3-8B"
    assert env["PERSONA_JUDGE_MAX_COMPLETION_TOKENS"] == "8192"


def test_cell_env_thinking_on():
    env = cell_env(model_id="Qwen/Qwen3-8B", mode="on", out_dir="/tmp/x")
    assert env["PERSONA_JUDGE_ENABLE_THINKING"] == "1"


def test_cell_env_does_not_set_sampling():
    # Task-1 froze the policy: no wire override for sampling; vLLM uses each
    # model's generation_config.json defaults. cell_env must NOT emit it.
    env = cell_env(model_id="Qwen/Qwen3-8B", mode="off", sampling={"temperature": 0.7},
                   out_dir="/tmp/x")
    assert "PERSONA_JUDGE_SAMPLING" not in env


def test_cell_env_dump_dirs():
    env = cell_env(model_id="Qwen/Qwen3-8B", mode="off", out_dir="/tmp/x")
    assert env["PERSONA_JUDGE_DUMP_DIR"].endswith("/http")
    assert env["PERSONA_REWARD_DUMP_DIR"].endswith("/reward")


def test_shard():
    assert shard_indices(list(range(10)), endpoint_index=0, num_endpoints=2) == [0, 2, 4, 6, 8]
    assert shard_indices(list(range(10)), endpoint_index=1, num_endpoints=2) == [1, 3, 5, 7, 9]


def test_output_dirs(tmp_path):
    d = cell_output_dirs(str(tmp_path), "qwen3-8b", "off")
    assert d["reward"].endswith("qwen3-8b/off/reward") and d["http"].endswith("qwen3-8b/off/http")
