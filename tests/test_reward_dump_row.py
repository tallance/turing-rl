from training.grpo.reward import _build_reward_dump_row

REQUIRED = {"generated_is_b", "human_side", "rating_gt_first", "rating_gen_first", "randomized_order",
    "response", "ground_truth", "context", "user_history", "judge_response", "judge_prompt",
    "judge_raw_content", "judge_reasoning", "judge_latency_ms", "judge_finish_reason", "judge_model",
    "judge_usage", "final_reward", "turing_judge_score_raw", "turing_judge_score_clipped",
    "source_copy_penalty", "assistant_like_penalty", "wrong_target_or_role_penalty",
    "unsupported_adversarial_reframing_penalty", "call_id", "user_id", "post_id", "target_idx",
    "persona", "ts", "worker_pid"}
KW = dict(response="g", ground_truth="h", context="c", user_history="hist", human_side="A",
    generated_is_b=True, randomized_order="gt_first", rating_gt_first=3, rating_gen_first=None,
    judge_response={"rating": 3, "reasoning": "..."}, judge_prompt="P", judge_raw_content="{...}",
    judge_reasoning="<think></think>", judge_latency_ms=1, judge_finish_reason="stop",
    judge_model="qwen3-8b", judge_usage={"completion_tokens": 9}, final_reward=0.3,
    turing_judge_score_raw=3.0, turing_judge_score_clipped=3.0, source_copy_penalty=0.0,
    assistant_like_penalty=0.0, wrong_target_or_role_penalty=0.0,
    unsupported_adversarial_reframing_penalty=0.0, call_id="c1", user_id="u", post_id="p",
    target_idx=0, persona="", ts=1.0, worker_pid=42)


def test_has_all_viewer_keys():
    assert REQUIRED <= set(_build_reward_dump_row(**KW))


def test_generated_is_b_present_when_false():
    assert "generated_is_b" in _build_reward_dump_row(**{**KW, "generated_is_b": False})


def test_rating_not_stored():
    assert "rating" not in _build_reward_dump_row(**KW)  # viewer derives it
