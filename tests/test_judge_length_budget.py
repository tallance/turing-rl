"""The response budget must be declared once, and must fit inside the context window.

veRL holds the response budget in two places: `data.max_response_length` sizes the training
batch, while `actor_rollout_ref.rollout.response_length` is what vLLM receives as max_new_tokens.
The judge config carried both as literals, so raising only the former produced a run that looked
correctly configured, reported `data.max_response_length: 10752`, and still generated at most
7680 tokens -- answering the wrong question (job 18583, 100% unclosed_thinking at step 0).
"""

import tempfile
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
    """8192: the eval's completion budget, and still under what OOMs.

    Measured on the 9B, thinking ON, step-0 validation over 200 held-out rows: 10752 OOMs in
    update_actor's log_softmax at micro_batch 1, so the budget has a hard ceiling well below the
    context window. 7680 trained on the 4B but is not what the 880-pair eval serves; 8192 matches
    PERSONA_JUDGE_MAX_COMPLETION_TOKENS (docs/default-params.md) so a checkpoint is validated
    under the budget it is scored with. Worst case through log_softmax: 10,049 + 8,192 = 18,241.
    """
    config = _loaded(JUDGE_CONFIG)
    data = config["data"]
    max_model_len = config["actor_rollout_ref"]["rollout"]["max_model_len"]

    assert data["max_response_length"] == 8192
    # Slack under the context window is deliberate: 10752 saturates it exactly and OOMs.
    assert data["max_prompt_length"] + data["max_response_length"] < max_model_len


def test_valsmoke_uses_the_longest_prompts_not_the_first():
    """A smoke that stands in for a full run must carry the corpus length tail.

    Peak training memory is set by the longest sequence, not a typical one. The leading 8 pairs
    top out at 7,083 prompt tokens against the full train set's 10,049; a 'first' smoke therefore
    omits ~3,000 tokens of the distribution -- exactly the margin that decides whether
    log_softmax fits. Three full arms were launched on a budget a 'first' smoke had passed and
    all three OOMed at that site.
    """
    launcher = (Path(__file__).resolve().parents[1] / "scripts" / "launch_judge_train.sh").read_text()
    valsmoke = launcher.split('if [ "$MODE" = valsmoke ]; then', 1)[1]
    code = "\n".join(l for l in valsmoke.splitlines() if not l.strip().startswith("#"))

    assert "--select longest" in code, "valsmoke must sample the length tail"


def test_overfit_gate_still_uses_deterministic_leading_pairs():
    """R0 asks only whether the loop learns, so its subset stays stable across runs."""
    launcher = (Path(__file__).resolve().parents[1] / "scripts" / "launch_judge_train.sh").read_text()
    overfit = launcher.split('if [ "$MODE" = overfit ]; then', 1)[1].split('if [ "$MODE" = valsmoke ]', 1)[0]

    assert "--select" not in overfit, "the overfit gate should keep the default 'first' selection"


def test_longest_selection_actually_picks_the_longest_pairs():
    """Behavioural, not a string match on the launcher.

    The previous test only asserted that '--select longest' appears in launch_judge_train.sh,
    which would still pass if the flag were parsed and ignored. This exercises the selection.
    """
    import pandas as pd

    from scripts.build_judge_overfit import build_judge_overfit

    rows = []
    for pair, size in enumerate([10, 500, 30, 900, 70]):  # deliberately unordered
        for human_is_b in (False, True):
            rows.append(
                {
                    "data_source": "prism_judge",
                    "prompt": [{"role": "user", "content": "x" * size}],
                    "reward_model": {"style": "rule"},
                    "extra_info": {"pair_id": f"p{pair}", "human_is_b": human_is_b},
                }
            )
    src = Path(tempfile.mkdtemp()) / "src.parquet"
    pd.DataFrame(rows).to_parquet(src, index=False)

    longest = build_judge_overfit(str(src), str(src.parent / "longest.parquet"), 2, "longest")
    first = build_judge_overfit(str(src), str(src.parent / "first.parquet"), 2, "first")

    got = {info["pair_id"] for info in longest["extra_info"]}
    assert got == {"p3", "p1"}, f"expected the 900- and 500-char pairs, got {got}"
    assert {info["pair_id"] for info in first["extra_info"]} == {"p0", "p1"}
    # Both orders of each pair survive, so human_is_b stays balanced.
    assert len(longest) == 4
    assert sum(info["human_is_b"] for info in longest["extra_info"]) * 2 == len(longest)


def test_longest_selection_rejects_an_unknown_mode():
    import pandas as pd

    from scripts.build_judge_overfit import build_judge_overfit

    src = Path(tempfile.mkdtemp()) / "s.parquet"
    pd.DataFrame(
        [{"data_source": "d", "prompt": [{"role": "user", "content": "x"}],
          "reward_model": {"style": "rule"}, "extra_info": {"pair_id": "p0", "human_is_b": False}}]
    ).to_parquet(src, index=False)

    with pytest.raises(ValueError, match="select must be"):
        build_judge_overfit(str(src), str(src.parent / "o.parquet"), 1, "middle")
