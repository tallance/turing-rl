"""Thinking-mode must travel from config to the rendered rollout prompt.

Regression cover for a silent defect: `qwen35_judge_grpo.yaml` set
`data.apply_chat_template_kwargs.enable_thinking: true`, but the patched agent loop resolved the
mode from PERSONA_ENABLE_THINKING alone -- a variable no launcher set. Every judge rollout
rendered with thinking OFF (an empty `<think></think>` block) for four ~9h runs, with nothing in
the log to say so. The job script exported PERSONA_JUDGE_ENABLE_THINKING, a different name that
governs only the served-judge path.
"""

import os
from types import SimpleNamespace

import pytest

from shared.prompt_utils import (
    ENABLE_THINKING_OVERRIDE_ENV,
    get_chat_template_kwargs_for_prompt_mode,
)
from training.grpo.verl_runtime_patch import (
    _assert_enable_thinking_propagated,
    _merge_propagated_runtime_env_vars,
    _render_text_prompt_ids,
    _resolve_config_enable_thinking,
    _seed_enable_thinking_env_from_config,
)


def _config(enable_thinking):
    """A veRL-shaped config; SimpleNamespace matches how _config_get walks attributes."""
    return SimpleNamespace(
        data=SimpleNamespace(apply_chat_template_kwargs={"enable_thinking": enable_thinking})
    )


@pytest.fixture(autouse=True)
def _clear_thinking_env(monkeypatch):
    monkeypatch.delenv(ENABLE_THINKING_OVERRIDE_ENV, raising=False)


class _FakeTokenizer:
    """Emulates the Qwen chat template's two thinking renderings, recording the kwargs it got."""

    def __init__(self):
        self.seen_kwargs = None
        self.tokenized_text = None

    def apply_chat_template(self, messages, **kwargs):
        self.seen_kwargs = kwargs
        # Qwen closes the block immediately when thinking is disabled; leaves it open otherwise.
        return "PROMPT<think>\n" if kwargs.get("enable_thinking") else "PROMPT<think>\n\n</think>\n\n"

    def __call__(self, text, add_special_tokens=False):
        self.tokenized_text = text
        return {"input_ids": [1, 2, 3]}


def test_config_true_reaches_the_render_kwargs():
    _seed_enable_thinking_env_from_config(_config(True))

    assert os.environ[ENABLE_THINKING_OVERRIDE_ENV] == "1"
    assert get_chat_template_kwargs_for_prompt_mode("reasoning")["enable_thinking"] is True


def test_config_false_is_honoured_too():
    # Generator configs legitimately want thinking off; the fix must not force it on.
    _seed_enable_thinking_env_from_config(_config(False))

    assert os.environ[ENABLE_THINKING_OVERRIDE_ENV] == "0"
    assert get_chat_template_kwargs_for_prompt_mode("reasoning")["enable_thinking"] is False


def test_explicit_env_override_beats_the_config(monkeypatch):
    monkeypatch.setenv(ENABLE_THINKING_OVERRIDE_ENV, "0")

    _seed_enable_thinking_env_from_config(_config(True))

    assert os.environ[ENABLE_THINKING_OVERRIDE_ENV] == "0"
    assert get_chat_template_kwargs_for_prompt_mode("reasoning")["enable_thinking"] is False


@pytest.mark.parametrize(
    "config",
    [
        None,
        SimpleNamespace(),
        SimpleNamespace(data=SimpleNamespace()),
        SimpleNamespace(data=SimpleNamespace(apply_chat_template_kwargs={})),
        SimpleNamespace(data=SimpleNamespace(apply_chat_template_kwargs={"enable_thinking": "banana"})),
    ],
    ids=["no-config", "no-data", "no-kwargs", "no-key", "malformed"],
)
def test_unresolvable_thinking_fails_closed(config):
    """Never default to False: that is the defect, and it costs ~9h per run to discover."""
    with pytest.raises(RuntimeError):
        _resolve_config_enable_thinking(config)


def test_malformed_explicit_override_also_fails_closed(monkeypatch):
    monkeypatch.setenv(ENABLE_THINKING_OVERRIDE_ENV, "yes-please")

    with pytest.raises(RuntimeError):
        _seed_enable_thinking_env_from_config(_config(True))


def test_thinking_mode_is_propagated_to_ray_workers(monkeypatch):
    monkeypatch.setenv(ENABLE_THINKING_OVERRIDE_ENV, "1")
    # Both propagation feature flags off: thinking must travel regardless, because a worker that
    # renders the wrong mode yields a healthy-looking run that measures the wrong model.
    monkeypatch.setenv("PERSONA_ENABLE_RUNTIME_ENV_PROPAGATION", "0")
    monkeypatch.setenv("PERSONA_ENABLE_WORKER_PROCESS_SETUP_HOOK", "0")

    merged = _merge_propagated_runtime_env_vars({})

    assert merged["env_vars"][ENABLE_THINKING_OVERRIDE_ENV] == "1"


def test_worker_without_the_mode_aborts_instead_of_defaulting_off():
    with pytest.raises(RuntimeError, match="unset in this rollout worker"):
        _assert_enable_thinking_propagated()


def test_worker_assertion_returns_the_resolved_mode(monkeypatch):
    monkeypatch.setenv(ENABLE_THINKING_OVERRIDE_ENV, "1")

    assert _assert_enable_thinking_propagated() is True


def test_rendered_prompt_keeps_the_thinking_block_open(monkeypatch):
    """The direct observable for the bug: an open `<think>` vs a pre-closed empty one."""
    monkeypatch.setenv(ENABLE_THINKING_OVERRIDE_ENV, "1")
    tokenizer = _FakeTokenizer()

    _render_text_prompt_ids(tokenizer, [{"role": "user", "content": "hi"}], prompt_mode="reasoning")

    assert tokenizer.seen_kwargs["enable_thinking"] is True
    # The prompt the model actually sees must leave the block open for it to reason into.
    assert tokenizer.tokenized_text.endswith("<think>\n")
    assert "</think>" not in tokenizer.tokenized_text


def test_thinking_off_renders_the_empty_block(monkeypatch):
    monkeypatch.setenv(ENABLE_THINKING_OVERRIDE_ENV, "0")
    tokenizer = _FakeTokenizer()

    _render_text_prompt_ids(tokenizer, [{"role": "user", "content": "hi"}], prompt_mode="reasoning")

    assert tokenizer.seen_kwargs["enable_thinking"] is False
    # This pre-closed empty block is what all four retained judge runs actually trained against.
    assert tokenizer.tokenized_text.endswith("<think>\n\n</think>\n\n")
