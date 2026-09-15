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
RATING_CONFIG = CONFIG_DIR / "qwen35_judge_rating_grpo.yaml"


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


def test_response_budget_uses_the_measured_fused_recipe():
    """The long response budget is safe only with the fused token-loss path.

    Job 18920 completed actor updates on the full 9B corpus with a 10,752-token response cap
    after enabling fused kernels. Earlier unfused attempts at shorter caps OOMed in
    update_actor's vocabulary-sized log_softmax, so the budget and fused path are one recipe.
    """
    config = _loaded(JUDGE_CONFIG)
    data = config["data"]
    model = config["actor_rollout_ref"]["model"]
    max_model_len = config["actor_rollout_ref"]["rollout"]["max_model_len"]

    assert data["max_response_length"] == 10752
    assert model["use_fused_kernels"] is True
    assert data["max_prompt_length"] + data["max_response_length"] == max_model_len


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


# --- the rating_only child config ---------------------------------------------------------


def _deep_merge(base: dict, override: dict) -> dict:
    """Hydra's defaults-list composition, for the one case this file needs: child over parent."""
    merged = dict(base)
    for key, value in override.items():
        if key == "defaults":
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _flatten(config: dict, prefix: str = "") -> dict:
    flat = {}
    for key, value in config.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(_flatten(value, f"{path}."))
        else:
            flat[path] = value
    return flat


def _composed_rating_config() -> dict:
    return _deep_merge(_loaded(JUDGE_CONFIG), _loaded(RATING_CONFIG))


def test_rating_config_composes_from_the_judge_config():
    """It must be a child, not a fork. A standalone copy would drift from the recipe the parent
    documents -- fused kernels, merged-LoRA rollout sync, use_v1 false, the nested reward block --
    each of which has its own incident behind it."""
    assert _loaded(RATING_CONFIG)["defaults"] == ["qwen35_judge_grpo", "_self_"]


def test_rating_config_changes_only_the_two_length_keys():
    """Minimal delta, enforced rather than trusted.

    Everything that makes a judge run work is inherited. Any new key appearing here is either a
    copy of a parent value (which will go stale silently) or an unreviewed change of recipe, and
    both read identically in a diff.
    """
    parent = _flatten(_loaded(JUDGE_CONFIG))
    child = _flatten(_composed_rating_config())

    changed = {key for key in child if parent.get(key) != child[key]}

    assert changed == {
        "data.max_prompt_length",
        "actor_rollout_ref.rollout.max_model_len",
    }, f"unexpected overrides: {sorted(changed)}"


def test_rating_config_prompt_plus_response_fits_its_context_window():
    config = _composed_rating_config()
    data = config["data"]
    max_model_len = config["actor_rollout_ref"]["rollout"]["max_model_len"]

    assert data["max_prompt_length"] + data["max_response_length"] == max_model_len


def test_rating_config_keeps_the_parent_response_budget():
    """The shorter prompt must not be spent shrinking generation: a rating_only answer is ~10
    tokens, so the whole budget is thinking room, and thinking is the point of this arm."""
    assert _composed_rating_config()["data"]["max_response_length"] == 10752


def test_rating_prompt_allowance_covers_the_measured_corpus():
    """Real Qwen3.5 tokenizer maxima on rating_iter1 (jobs 23068/23069): train 5375, val 5866.

    val is the binding one and is loaded by the same config as train, so the allowance must
    clear 5866 -- filter_overlong_prompts drops over-budget rows from both splits silently.

    Bounded above as well, on the parent's reasoning: prompt and response share max_model_len,
    so unused prompt allowance is taken out of generation.

    This replaced a chars/3.9 PROJECTION, which put the maximum at ~5598 and was wrong in both
    directions -- the same estimator called the real 5866 maximum 7807.
    """
    measured_combined_max = 5866

    allowance = _composed_rating_config()["data"]["max_prompt_length"]

    assert allowance > measured_combined_max, "prompts would be truncated"
    assert allowance <= measured_combined_max + 1024, (
        "allowance exceeds the measured corpus by more than the parent's safety margin"
    )


def test_the_trainer_can_select_the_rating_config():
    launcher = (
        Path(__file__).resolve().parents[1] / "scripts" / "slurm" / "judge_grpo_train.sh"
    ).read_text()

    assert "JUDGE_CONFIG_NAME" in launcher
    assert '--config-name "$JUDGE_CONFIG_NAME"' in launcher
    # Both names are real files, so a typo cannot reach Hydra as a missing-config crash.
    for name in ("qwen35_judge_grpo", "qwen35_judge_rating_grpo"):
        assert (CONFIG_DIR / f"{name}.yaml").is_file()
        assert name in launcher


def test_the_submit_launcher_forwards_the_config_name_explicitly():
    """Not left to --export=ALL. A dropped JUDGE_CONFIG_NAME silently falls back to the
    full-schema length profile, and that run does not fail -- an allowance that is too LARGE
    truncates nothing -- so it would train under the wrong context window while reading clean."""
    submit = (
        Path(__file__).resolve().parents[1] / "scripts" / "launch_judge_train.sh"
    ).read_text()

    assert "EXPORTS=$EXPORTS,JUDGE_CONFIG_NAME=$JUDGE_CONFIG_NAME" in submit.replace('"', "")


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
