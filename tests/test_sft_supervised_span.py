"""Pin the exact token span that SFT supervises, per training config.

Qwen3.5's chat template does not omit the think block under enable_thinking=False: it
PREFILLS a closed, empty one into the assistant turn, so the generation prompt ends with
"<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n" (verified on the cluster against the
real tokenizer; for the target "A" the five supervised ids were 248068, 271, 248069, 271,
32). Stripping that prefill off the masked prefix moves the mask boundary earlier and pulls
the prefill into the loss — right for the generator (served with thinking ON, so it must
emit the block) and wrong for the judge (served with thinking OFF, so the prefill is
already in the prompt and only the verdict token is in question).

Real Qwen3.5 tokenizers cannot be loaded locally, so these tests use a fake tokenizer that
reproduces that template and segments the prefill into the same four atoms.
"""

import yaml

from training.sft.lora_sft import (
    build_chat_template_sft_features,
    build_completion_mask_mapper,
    resolve_mask_options,
)


PREFILL = "<think>\n\n</think>\n\n"
GENERATION_PROMPT = "<|im_start|>assistant\n"

# Multi-character units the real Qwen tokenizer keeps whole; everything else falls back to
# one token per character, which is enough to pin span boundaries.
ATOMS = ("<|im_start|>", "<|im_end|>", "<think>", "</think>", "\n\n")


class FakeQwen35Tokenizer:
    """Chat template + offset-mapping tokenizer mimicking Qwen3.5's empty-think prefill."""

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
        return self._vocab.setdefault(piece, 1000 + len(self._vocab))

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
        parts = []
        for message in messages:
            parts.append(f"<|im_start|>{message['role']}\n")
            if message["role"] == "assistant" and not enable_thinking:
                parts.append(PREFILL)
            parts.append(f"{message['content']}<|im_end|>\n")
        if add_generation_prompt:
            parts.append(GENERATION_PROMPT)
            if not enable_thinking:
                parts.append(PREFILL)
        return "".join(parts)


JUDGE_MESSAGES = [
    {"role": "system", "content": "Which turn is human?"},
    {"role": "user", "content": "Turn A: ... Turn B: ..."},
    {"role": "assistant", "content": "A"},
]

GENERATOR_MESSAGES = [
    {"role": "system", "content": "You are simulating a user."},
    {"role": "user", "content": "How can I help?"},
    {"role": "assistant", "content": "that works for me"},
]


def supervised_text(tokenizer, features) -> str:
    return tokenizer.decode(
        [
            token_id
            for token_id, keep in zip(features["input_ids"], features["completion_mask"])
            if keep
        ]
    )


def supervised_count(features) -> int:
    return sum(features["completion_mask"])


def load_config(alias: str) -> dict:
    with open(f"training/sft/configs/{alias}_lora.yaml") as handle:
        return yaml.safe_load(handle)


# --- fixture faithfulness -------------------------------------------------------------


def test_fake_template_prefills_an_empty_think_block_like_qwen35():
    tokenizer = FakeQwen35Tokenizer()
    prompt = tokenizer.apply_chat_template(
        JUDGE_MESSAGES[:-1], tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    full = tokenizer.apply_chat_template(
        JUDGE_MESSAGES, tokenize=False, add_generation_prompt=False, enable_thinking=False
    )
    assert prompt.endswith(GENERATION_PROMPT + PREFILL)
    assert full == prompt + "A<|im_end|>\n"


def test_fake_tokenizer_segments_the_prefill_into_four_atoms_plus_the_answer():
    tokenizer = FakeQwen35Tokenizer()
    ids = tokenizer(PREFILL + "A")["input_ids"]
    assert len(ids) == 5
    assert [tokenizer.decode([i]) for i in ids] == ["<think>", "\n\n", "</think>", "\n\n", "A"]


def test_mask_path_always_renders_with_thinking_disabled():
    """The span arithmetic assumes the prefilled rendering; both renders must request it."""
    tokenizer = FakeQwen35Tokenizer()
    build_chat_template_sft_features(tokenizer, JUDGE_MESSAGES)
    assert [kw["enable_thinking"] for kw in tokenizer.render_kwargs] == [False, False]
    assert [kw["add_generation_prompt"] for kw in tokenizer.render_kwargs] == [True, False]


# --- the supervised span, per option --------------------------------------------------


def test_default_supervises_the_think_prefill_and_the_answer():
    """Today's behaviour, unchanged: 5 tokens, 4 of them the fixed prefill."""
    tokenizer = FakeQwen35Tokenizer()
    features = build_chat_template_sft_features(
        tokenizer, JUDGE_MESSAGES, supervise_think_prefill=True
    )
    assert supervised_text(tokenizer, features) == PREFILL + "A"
    assert supervised_count(features) == 5


def test_omitting_the_option_keeps_the_default():
    tokenizer = FakeQwen35Tokenizer()
    assert build_chat_template_sft_features(tokenizer, JUDGE_MESSAGES) == (
        build_chat_template_sft_features(
            tokenizer, JUDGE_MESSAGES, supervise_think_prefill=True
        )
    )


def test_disabled_supervises_exactly_the_answer_token():
    tokenizer = FakeQwen35Tokenizer()
    features = build_chat_template_sft_features(
        tokenizer, JUDGE_MESSAGES, supervise_think_prefill=False
    )
    assert supervised_text(tokenizer, features) == "A"
    assert supervised_count(features) == 1


def test_disabled_leaves_the_prefill_in_the_masked_prompt():
    """The prefill must still be present as context — it is in the prompt at eval time."""
    tokenizer = FakeQwen35Tokenizer()
    features = build_chat_template_sft_features(
        tokenizer, JUDGE_MESSAGES, supervise_think_prefill=False
    )
    first_supervised = features["completion_mask"].index(1)
    prompt_ids = features["input_ids"][:first_supervised]
    assert tokenizer.decode(prompt_ids).endswith(GENERATION_PROMPT + PREFILL)


def test_option_is_independent_of_stop_token_supervision():
    tokenizer = FakeQwen35Tokenizer()
    features = build_chat_template_sft_features(
        tokenizer,
        JUDGE_MESSAGES,
        supervise_stop_token=True,
        supervise_think_prefill=False,
    )
    assert supervised_text(tokenizer, features) == "A<|im_end|>"


# --- config wiring, end to end through the dataset mapper -----------------------------


def test_resolve_mask_options_defaults_to_the_generator_behaviour():
    assert resolve_mask_options({}) == {
        "supervise_stop_token": False,
        "supervise_think_prefill": True,
    }


def test_judge_configs_disable_prefill_supervision():
    for alias in ("qwen35_4b_judge", "qwen35_9b_judge"):
        assert load_config(alias)["supervise_think_prefill"] is False
        assert resolve_mask_options(load_config(alias))["supervise_think_prefill"] is False


def test_generator_configs_still_resolve_to_prefill_supervision():
    for alias in ("qwen35_9b", "qwen3_8b"):
        config = load_config(alias)
        assert "supervise_think_prefill" not in config
        assert resolve_mask_options(config)["supervise_think_prefill"] is True


def test_judge_config_mapper_supervises_one_token():
    for alias in ("qwen35_4b_judge", "qwen35_9b_judge"):
        tokenizer = FakeQwen35Tokenizer()
        mapper = build_completion_mask_mapper(tokenizer, load_config(alias))
        features = mapper({"messages": JUDGE_MESSAGES})
        assert supervised_text(tokenizer, features) == "A"
        assert supervised_count(features) == 1


def test_generator_config_mapper_span_is_unchanged():
    """Prefill + full turn + <|im_end|>: byte-identical to what the generator trained on."""
    for alias in ("qwen35_9b", "qwen3_8b"):
        tokenizer = FakeQwen35Tokenizer()
        mapper = build_completion_mask_mapper(tokenizer, load_config(alias))
        features = mapper({"messages": GENERATOR_MESSAGES})
        assert supervised_text(tokenizer, features) == PREFILL + "that works for me<|im_end|>"
