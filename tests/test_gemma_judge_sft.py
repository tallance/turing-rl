"""Pin the Gemma judge's supervised span, and the reason it needs its own builder.

Gemma 4's chat template is asymmetric in a way Qwen's is not. Under
add_generation_prompt=True with enable_thinking=False it appends an empty, already-closed
thought channel, so the served prompt ends:

    <|turn>model\\n<|channel>thought\\n<channel|>

but rendering a COMPLETED assistant turn omits that channel entirely:

    <|turn>model\\nB<turn|>\\n

So the full render does not start with the prompt render, and the diff-based builder cannot
locate a target span at all. Verified against the real tokenizer on the cluster, including
that no template kwarg restores it (preserve_thinking, enable_thinking, reasoning="" and
reasoning=" " were all tried). Real Gemma tokenizers cannot be loaded locally, so these
tests use a fake that reproduces the asymmetry.

The consequence that matters: training on the completed render would teach the model
P(label | ...model\\n) while the judge is actually asked P(label | ...<channel|>). The
prefix builder removes that skew by construction.
"""

import subprocess
from pathlib import Path

import pytest
import yaml

from training.sft.lora_sft import (
    build_chat_template_sft_features,
    build_completion_mask_mapper,
    build_generation_prefix_sft_features,
    get_lora_targets,
    uses_generation_prefix_masking,
    MODEL_MAP,
)


ROOT = Path(__file__).resolve().parents[1]
THOUGHT_CHANNEL = "<|channel>thought\n<channel|>"
TURN_END = "<turn|>"

# Multi-character units the real Gemma tokenizer keeps whole.
ATOMS = ("<|turn>", "<turn|>", "<|channel>", "<channel|>", "<bos>", "thought", "\n")


class FakeGemma4Tokenizer:
    """Reproduces Gemma 4's generation-prompt-only thought channel."""

    def __init__(self):
        self._vocab: dict[str, int] = {}
        self.render_kwargs: list[dict] = []

    def _atomize(self, text: str) -> list[tuple[str, int]]:
        pieces: list[tuple[str, int]] = []
        cursor = 0
        while cursor < len(text):
            for atom in ATOMS:
                if text.startswith(atom, cursor):
                    pieces.append((atom, cursor))
                    cursor += len(atom)
                    break
            else:
                pieces.append((text[cursor], cursor))
                cursor += 1
        return pieces

    def _token_id(self, piece: str) -> int:
        return self._vocab.setdefault(piece, 2000 + len(self._vocab))

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        pieces = self._atomize(text)
        out = {"input_ids": [self._token_id(piece) for piece, _ in pieces]}
        if return_offsets_mapping:
            out["offset_mapping"] = [(start, start + len(p)) for p, start in pieces]
        return out

    def decode(self, token_ids) -> str:
        reverse = {tid: piece for piece, tid in self._vocab.items()}
        return "".join(reverse[tid] for tid in token_ids)

    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=True,
        **kwargs,
    ) -> str:
        assert tokenize is False, "the SFT mask path renders to text, not ids"
        self.render_kwargs.append(
            {"add_generation_prompt": add_generation_prompt, "enable_thinking": enable_thinking}
        )
        parts = ["<bos>"]
        for message in messages:
            role = "model" if message["role"] == "assistant" else message["role"]
            # NOTE: no thought channel here even for an assistant turn — this asymmetry
            # against the add_generation_prompt branch below is the whole point.
            parts.append(f"<|turn>{role}\n{message['content']}{TURN_END}\n")
        if add_generation_prompt:
            parts.append("<|turn>model\n")
            if not enable_thinking:
                parts.append(THOUGHT_CHANNEL)
        return "".join(parts)


JUDGE_MESSAGES = [
    {"role": "user", "content": "Turn A: ... Turn B: ..."},
    {"role": "assistant", "content": "B"},
]


def supervised_text(tokenizer, features) -> str:
    return tokenizer.decode(
        [
            token_id
            for token_id, keep in zip(features["input_ids"], features["completion_mask"])
            if keep
        ]
    )


def load_config(alias: str) -> dict:
    with open(ROOT / f"training/sft/configs/{alias}_lora.yaml") as handle:
        return yaml.safe_load(handle)


# --- fixture faithfulness -------------------------------------------------------------


def test_fake_template_reproduces_the_prefix_asymmetry():
    """If this ever passes trivially, every test below is measuring nothing."""
    tokenizer = FakeGemma4Tokenizer()
    served = tokenizer.apply_chat_template(
        JUDGE_MESSAGES[:1], add_generation_prompt=True, enable_thinking=False
    )
    full = tokenizer.apply_chat_template(JUDGE_MESSAGES)
    assert served.endswith(THOUGHT_CHANNEL)
    assert THOUGHT_CHANNEL not in full
    assert not full.startswith(served)


# --- why the Qwen builder cannot be reused ----------------------------------------------


def test_diff_builder_refuses_gemma_rather_than_mistraining():
    tokenizer = FakeGemma4Tokenizer()
    with pytest.raises(ValueError, match="does not start with the rendered prompt prefix"):
        build_chat_template_sft_features(
            tokenizer, JUDGE_MESSAGES, supervise_think_prefill=False
        )


# --- the prefix builder -----------------------------------------------------------------


def test_prefix_builder_supervises_exactly_the_label_token():
    tokenizer = FakeGemma4Tokenizer()
    features = build_generation_prefix_sft_features(tokenizer, JUDGE_MESSAGES)
    assert sum(features["completion_mask"]) == 1
    assert supervised_text(tokenizer, features) == "B"


def test_prefix_builder_keeps_the_thought_channel_in_the_masked_context():
    """The channel must be context, not target: it is already in the prompt at eval."""
    tokenizer = FakeGemma4Tokenizer()
    features = build_generation_prefix_sft_features(tokenizer, JUDGE_MESSAGES)
    masked = tokenizer.decode(
        [
            token_id
            for token_id, keep in zip(features["input_ids"], features["completion_mask"])
            if not keep
        ]
    )
    assert masked.endswith(THOUGHT_CHANNEL)


def test_prefix_builder_excludes_the_turn_closer():
    """<turn|> is in eos_token_id, but the judge decodes with max_tokens=1 and is never
    asked to terminate, so supervising it would make the target two tokens."""
    tokenizer = FakeGemma4Tokenizer()
    features = build_generation_prefix_sft_features(tokenizer, JUDGE_MESSAGES)
    assert TURN_END not in supervised_text(tokenizer, features)


def test_prefix_builder_renders_the_served_prompt_shape():
    tokenizer = FakeGemma4Tokenizer()
    build_generation_prefix_sft_features(tokenizer, JUDGE_MESSAGES)
    assert tokenizer.render_kwargs == [
        {"add_generation_prompt": True, "enable_thinking": False}
    ]


def test_prefix_builder_rejects_a_non_assistant_last_message():
    tokenizer = FakeGemma4Tokenizer()
    with pytest.raises(ValueError, match="last SFT message must be the assistant target"):
        build_generation_prefix_sft_features(tokenizer, JUDGE_MESSAGES[:1])


def test_prefix_builder_rejects_an_empty_target():
    tokenizer = FakeGemma4Tokenizer()
    messages = [JUDGE_MESSAGES[0], {"role": "assistant", "content": ""}]
    with pytest.raises(ValueError, match="assistant target is empty"):
        build_generation_prefix_sft_features(tokenizer, messages)


# --- dispatch ----------------------------------------------------------------------------


def test_only_gemma_routes_to_the_prefix_builder():
    assert uses_generation_prefix_masking("gemma4-12b-judge") is True
    for alias in ("qwen35-4b-judge", "qwen35-9b-judge", "qwen35-9b", "qwen3-8b"):
        assert uses_generation_prefix_masking(alias) is False
    assert uses_generation_prefix_masking(None) is False


def test_mapper_uses_the_prefix_builder_for_the_gemma_alias():
    tokenizer = FakeGemma4Tokenizer()
    mapper = build_completion_mask_mapper(
        tokenizer, load_config("gemma4_12b_judge"), "gemma4-12b-judge"
    )
    features = mapper({"messages": JUDGE_MESSAGES})
    assert supervised_text(tokenizer, features) == "B"
    assert sum(features["completion_mask"]) == 1


def test_mapper_without_a_model_keeps_the_diff_builder():
    """Omitting the model must not silently switch existing callers onto the new path."""
    tokenizer = FakeGemma4Tokenizer()
    mapper = build_completion_mask_mapper(tokenizer, load_config("gemma4_12b_judge"))
    with pytest.raises(ValueError, match="does not start with the rendered prompt prefix"):
        mapper({"messages": JUDGE_MESSAGES})


# --- model registry ----------------------------------------------------------------------


def test_gemma_alias_maps_to_the_instruction_tuned_12b():
    assert MODEL_MAP["gemma4-12b-judge"] == "google/gemma-4-12B-it"


def test_gemma_lora_targets_are_the_seven_text_projections():
    targets = get_lora_targets("gemma4-12b-judge")
    assert targets == [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    ]


# --- config ------------------------------------------------------------------------------


def test_gemma_judge_config_exists_and_keeps_the_target_at_one_token():
    config = load_config("gemma4_12b_judge")
    assert config["supervise_stop_token"] is False
    assert config["lora_r"] == 64
    assert config["lora_alpha"] == 128
    assert config["use_qlora"] is False


def test_gemma_judge_config_omits_the_diff_builder_knob():
    """supervise_think_prefill only affects the builder Gemma does not use; setting it
    would read as if it had an effect."""
    assert "supervise_think_prefill" not in load_config("gemma4_12b_judge")


# --- launcher ------------------------------------------------------------------------------


def test_ce_launcher_accepts_the_gemma_alias():
    script = (ROOT / "scripts/launch_judge_ce_train.sh").read_text()
    assert "gemma4-12b-judge" in script


def test_ce_launcher_still_rejects_a_non_judge_alias(tmp_path):
    """Widening the allowlist must not turn it into a pass-through — 31B is deliberately
    NOT trainable yet (different architecture, 410 targets, vision-tower name collisions)."""
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/launch_judge_ce_train.sh")],
        env={
            "PATH": "/usr/bin:/bin",
            "TURING_RL_WORK_ROOT": str(tmp_path),
            "MODEL": "gemma4-31b-judge",
        },
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    assert "must be a judge alias" in result.stderr
