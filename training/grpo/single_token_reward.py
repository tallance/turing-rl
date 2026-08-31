"""Single-token A/B judge scoring for the GENERATOR reward path.

The judge sees the same pairwise header as the full-schema protocol but answers with one
letter, and the verdict is read from the logprobs at that single position rather than from
a 37-field JSON body. The reward is ``p_human``: the judge's renormalized probability that
the GENERATED turn is the real human. Higher means the generator fooled the judge harder.

``p_human`` is already in [0, 1], so it substitutes for the full arm's ``(rating - 1) / 6``
and every downstream term (raw-reward scale, format bonus, length penalty) is unchanged.
A continuous score rather than a 0/1 from the letter because GRPO takes advantages WITHIN a
rollout group: at n=4 a binary reward yields zero advantage, and therefore no gradient,
whenever all four rollouts land on the same side.

Why this is not ``eval/single_token_judge.py``: that module is the eval-side measurement and
deliberately does not import the reward path. Importing it from here would invert that
dependency. Everything semantically load-bearing -- the prompt, the A/B side hash, the
verdict rule -- is imported from ``shared/`` by both, so only mechanical glue is repeated;
``tests/test_single_token_reward.py`` pins the pieces that must not drift.
"""

from __future__ import annotations

import json
import os
from typing import Any

import aiohttp

from shared.api_client import (
    build_chat_payload,
    get_judge_call_meta,
    post_chat_choice_async,
)
from shared.judge_prompts import TURING_SINGLE_TOKEN_PROMPT
from shared.judge_utils import (
    _stable_turing_generated_is_b,
    build_source_copy_warning,
    format_source_copy_watchlist,
    sanitize_prompt_text,
)
from shared.single_token_verdict import HardFail, extract_verdict, is_ab_token

# vLLM's ceiling for an OpenAI-compatible request. A/B mass is spread across tokenizer
# variants ("A", " A", "▁A", "a"), so a smaller k can truncate real verdict mass and push
# an honest answer under the MIN_AB_MASS floor. Must match eval/single_token_judge.py.
TOP_LOGPROBS = 20

# 1 == "definitely A", 7 == "definitely B" on the Likert scale the dump viewer and
# scripts/analyze_judge_sweep.py already speak. A single token cannot tie, so 4 never
# appears -- which is what lets those tools read a single-token run unchanged.
RATING_FOR_A = 1
RATING_FOR_B = 7

# The judge answered nothing usable. Neutral rather than 0: the generator earns neither
# credit nor punishment for a judge-side non-answer, so no gradient comes out of noise.
HARD_FAIL_P_HUMAN = 0.5


def _sampling_from_env() -> dict | None:
    raw = os.environ.get("PERSONA_JUDGE_SAMPLING")
    return json.loads(raw) if raw else None


def _read_verdict_position(choice: dict) -> dict:
    """Return the single decoded position, or raise if it carries no usable top-k.

    Deliberately NOT a HardFail. A HardFail is a property of one input and is recorded as
    an abstention; no top_logprobs at all means the request or the server is misconfigured,
    which as a hard fail would read as a 100%-abstention run that still trains happily on
    a constant 0.5 reward.
    """
    logprobs = (choice or {}).get("logprobs")
    content = (logprobs or {}).get("content")
    if not content:
        raise RuntimeError(
            "judge response carries no logprobs content; single-token scoring requires "
            f"logprobs=True and a server that honours it (got choice keys: "
            f"{sorted((choice or {}).keys())})"
        )
    position = content[0] or {}
    if not position.get("top_logprobs"):
        raise RuntimeError(
            f"judge response position carries no top_logprobs; single-token scoring "
            f"requires top_logprobs={TOP_LOGPROBS} and a server that honours it "
            f"(vLLM rejects or truncates above its --max-logprobs, default 20). "
            f"Got position keys: {sorted(position)}"
        )
    return position


async def score_turing_single_token_with_info(
    session: aiohttp.ClientSession,
    api_key: str,
    response: str,
    ground_truth: str,
    user_history: str,
    context: str,
    *,
    semaphore: Any = None,
    user_id: Any = "",
    post_id: Any = "",
    target_idx: Any = "",
    randomization_seed_material: str = "",
) -> dict[str, Any]:
    """Score one generated turn against the human turn with the single-token judge.

    Returns the keys the full-schema scorer returns (so the reward path and the dump keep
    one shape) plus the single-token extras.
    """
    response = sanitize_prompt_text(response)
    ground_truth = sanitize_prompt_text(ground_truth)
    user_history = sanitize_prompt_text(user_history)
    context = sanitize_prompt_text(context)

    response_warning = build_source_copy_warning(
        response, user_history=user_history, thread_context=context
    )
    ground_truth_warning = build_source_copy_warning(
        ground_truth, user_history=user_history, thread_context=context
    )

    # Same function and same seed material as the full arm, so a given pair lands in the
    # same A/B slot in every arm and a protocol comparison is not confounded by a
    # different randomization.
    generated_is_b = _stable_turing_generated_is_b(
        response,
        user_id=user_id,
        post_id=post_id,
        target_idx=target_idx,
        seed_material=randomization_seed_material,
    )
    if generated_is_b:
        response_a, response_b = ground_truth, response
        warning_a, warning_b = ground_truth_warning, response_warning
        randomized_order = "gt_first"
    else:
        response_a, response_b = response, ground_truth
        warning_a, warning_b = response_warning, ground_truth_warning
        randomized_order = "gen_first"

    prompt = TURING_SINGLE_TOKEN_PROMPT.format(
        user_history=user_history,
        context=context,
        response_a=response_a,
        response_b=response_b,
        source_copy_watchlist=format_source_copy_watchlist(
            [warning_a, warning_b],
            item_label="Response",
            labels=["Response A", "Response B"],
        ),
    )

    payload = build_chat_payload(
        model=os.environ.get("JUDGE_MODEL", ""),
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=1,
        response_format=None,
        reasoning=False,
        sampling=_sampling_from_env(),
        # Pinned, NOT read from PERSONA_JUDGE_ENABLE_THINKING (the generator run scripts
        # export =1): with a one-token budget a thinking-on judge spends that token on the
        # think opener and the verdict never exists. Silent, and it looks like a hard-fail
        # cliff rather than a config error.
        chat_template_kwargs={"enable_thinking": False},
    )
    payload["logprobs"] = True
    payload["top_logprobs"] = TOP_LOGPROBS

    # One call, deliberately no parse-retry loop (the full arm retries on malformed JSON).
    # A hard fail is a property of the input; retrying would hide it from the hard-fail
    # rate, which is the signal telling us whether the judge is holding up under attack.
    choice = await post_chat_choice_async(
        session, payload, semaphore=semaphore, api_key=api_key
    )
    judge_meta = get_judge_call_meta() or {}

    position = _read_verdict_position(choice)
    sampled_token = position.get("token")
    # Recorded, never gated on: the judge samples at its generation_config defaults, so
    # this token is a DRAW, not the argmax. The A/B mass floor owns abstention.
    sampled_token_is_ab = is_ab_token(sampled_token)

    verdict = None
    hard_fail_reason = None
    try:
        verdict = extract_verdict(position.get("top_logprobs") or [])
    except HardFail as exc:
        hard_fail_reason = str(exc)

    if verdict is None:
        p_human = HARD_FAIL_P_HUMAN
        letter = None
        rating = None
        likert_score = 0.0
    else:
        # p_a is P(the judge says the HUMAN is A). The generated turn is B exactly when
        # generated_is_b, so its probability of being called human is the other side.
        p_human = (1.0 - verdict.p_a) if generated_is_b else verdict.p_a
        letter = verdict.letter
        rating = RATING_FOR_A if letter == "A" else RATING_FOR_B
        # The full arm's Likert orientation: high == "the generated turn looks human".
        likert_score = float(rating) if generated_is_b else float(8 - rating)

    return {
        # --- keys the full-schema scorer also returns ---
        "score": likert_score,
        "rating_gt_first": rating if generated_is_b else None,
        "rating_gen_first": None if generated_is_b else rating,
        "rating_randomized": rating,
        "generated_is_b": generated_is_b,
        "randomized_order": randomized_order,
        "judge_prompt": prompt,
        "judge_raw_content": sampled_token,
        "judge_latency_ms": judge_meta.get("latency_ms"),
        "judge_finish_reason": judge_meta.get("finish_reason"),
        "judge_usage": judge_meta.get("usage") or {},
        # --- single-token extras ---
        "judge_prompt_style": "single_token",
        "p_human": p_human,
        "p_a": verdict.p_a if verdict else None,
        "ab_mass": verdict.ab_mass if verdict else None,
        "off_ab_mass": verdict.off_ab_mass if verdict else None,
        "letter": letter,
        "human_is_b": not generated_is_b,
        "hard_fail": verdict is None,
        "hard_fail_reason": hard_fail_reason,
        "sampled_token": sampled_token,
        "sampled_token_is_ab": sampled_token_is_ab,
        "enable_thinking": False,
    }
