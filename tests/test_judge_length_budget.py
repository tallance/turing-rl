"""The response budget must be declared once, and must fit inside the context window.

veRL holds the response budget in two places: `data.max_response_length` sizes the training
batch, while `actor_rollout_ref.rollout.response_length` is what vLLM receives as max_new_tokens.
The judge config carried both as literals, so raising only the former produced a run that looked
correctly configured, reported `data.max_response_length: 10752`, and still generated at most
7680 tokens -- answering the wrong question (job 18583, 100% unclosed_thinking at step 0).
"""

from pathlib import Path

import pytest

import yaml

CONFIG_DIR = Path(__file__).resolve().parents[1] / "training" / "grpo" / "configs"
JUDGE_CONFIG = CONFIG_DIR / "qwen35_judge_grpo.yaml"


def _raw(path):
    return path.read_text()


def _loaded(path):
    # Hydra/OmegaConf interpolations are not resolvable by plain yaml, so read them as strings.
    return yaml.safe_load(_raw(path))


def test_rollout_response_length_is_interpolated_not_duplicated():
    """One source of truth, so a caller override cannot raise one and leave the other behind."""
    rollout = _loaded(JUDGE_CONFIG)["actor_rollout_ref"]["rollout"]

    assert rollout["response_length"] == "${data.max_response_length}", (
        "rollout.response_length must interpolate data.max_response_length; a second literal is "
        "how an override silently applied to the batch but not to generation"
    )


def test_prompt_plus_response_fits_the_context_window():
    """max_prompt_length + max_response_length must not exceed max_model_len.

    max_model_len 22016 was measured on a 40GB A100 (capacity probe, job 15926), so exceeding it
    is an OOM or a vLLM refusal to allocate KV cache, not a soft limit.
    """
    config = _loaded(JUDGE_CONFIG)
    data = config["data"]
    max_model_len = config["actor_rollout_ref"]["rollout"]["max_model_len"]

    total = data["max_prompt_length"] + data["max_response_length"]
    assert total <= max_model_len, f"{total} exceeds max_model_len {max_model_len}"


def test_prompt_allowance_covers_the_measured_corpus():
    """Measured val prompts: p50 6774, p99 9594, max 10535 tokens (Qwen3.5 tokenizer).

    The allowance must clear the longest real prompt, or prompts are left-truncated and the judge
    silently scores a conversation it only partly saw.

    The allowance is also bounded above, now that the trade-off is measured rather than assumed.
    Prompt and response share max_model_len, so every token of unused prompt allowance is taken
    out of generation. At 14336 the response budget was squeezed to 7680, where 12.5% of 9B
    rollouts never closed <think> and step-0 accuracy fell below chance.
    """
    data = _loaded(JUDGE_CONFIG)["data"]
    measured_max_prompt_tokens = 10535

    assert data["max_prompt_length"] > measured_max_prompt_tokens, "prompts would be truncated"
    assert data["max_prompt_length"] <= measured_max_prompt_tokens + 1024, (
        "prompt allowance exceeds the measured corpus by more than a safety margin; that surplus "
        "comes out of the response budget, since both share max_model_len"
    )


def test_response_budget_is_the_measured_trainable_value():
    """9216, not 7680 and not 10752.

    Measured on the 9B, thinking ON, step-0 validation over 200 held-out rows: 7680 leaves 12.5%
    of rollouts unclosed and accuracy below chance; 10752 is marginally better than 9216 but OOMs
    in update_actor's log_softmax at micro_batch 1. 9216 trains and reaches 0.945 coverage.
    """
    config = _loaded(JUDGE_CONFIG)
    data = config["data"]
    max_model_len = config["actor_rollout_ref"]["rollout"]["max_model_len"]

    assert data["max_response_length"] == 9216
    # Slack under the context window is deliberate: 10752 saturates it exactly and OOMs.
    assert data["max_prompt_length"] + data["max_response_length"] < max_model_len
