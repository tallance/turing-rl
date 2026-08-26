# Single-Token Judge Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Measure whether a thinking-off, schema-free judge that emits a single `A`/`B` token matches the current 37-field JSON judge, zero-shot and trained, on the frozen 880-pair set.

**Architecture:** The existing `TURING_PROMPT` is split at the `## Evaluation Procedure` boundary into a header (task + all inputs, kept byte-identical) and a rubric/output tail. The single-token prompt is `header + one answer instruction`. Scoring uses `max_tokens=1` plus logprobs, taking the verdict as argmax over the A/B token variants and `p_a` as their renormalized probability. The trained arm is LoRA cross-entropy through the existing `training/sft/lora_sft.py`, not GRPO.

**Tech Stack:** Python 3.12, pytest, pandas, HF transformers/TRL/PEFT, vLLM (OpenAI-compatible server), Slurm on the RFAI V3 cluster.

**Spec:** `docs/superpowers/specs/2026-08-26-single-token-judge-design.md`

---

## Before you start

**Branch state.** This worktree is `worktree-single-token-judge`, already merged with `lancewicki/main` at `94bcee7` (which contains the judge-only-RLVR merge). Everything this plan touches exists. If `git log --oneline -1 lancewicki/main` has moved on, merge trunk in before starting — several tasks touch `scripts/build_judge_train_pairs.py` and `tests/`, which arrived in that merge.

**Run tests from the repo root**, which is how the existing suite resolves `from scripts...` imports:

```bash
cd /Users/lancewicki/Projects/turing-rl/.claude/worktrees/single-token-judge
python -m pytest tests/ -q
```

Confirm the suite is green before Task 1. If it is not, stop and report — you cannot tell your regressions from pre-existing ones otherwise.

**Phases A (Tasks 1-12) are local and need no GPU.** Phase B (Tasks 13-18) runs on the cluster and must go through `scripts/cluster_launch.sh`; never call `sbatch` directly, and run the `preflight-job-check` skill before each submit.

---

## Task 1: Split TURING_PROMPT into header and tail

The single-token prompt must share the *same* input text as the full prompt, not a copy that can drift. Splitting the existing constant is what makes that structural.

**Files:**
- Modify: `shared/judge_prompts.py:399` (the `TURING_PROMPT` assignment)
- Test: `tests/test_single_token_prompt.py` (create)

**Step 1: Record the current text so the refactor can be proven lossless**

Run this and keep the digest — it goes into the test:

```bash
python -c "
import hashlib, re
src = open('shared/judge_prompts.py').read()
p = re.search(r'TURING_PROMPT = \"\"\"(.*?)\"\"\"', src, re.S).group(1)
print(hashlib.sha256(p.encode()).hexdigest())
print('chars', len(p))
"
```

Expected: a 64-char digest and `chars 20525`. If the char count differs, the template has changed since the spec was written — stop and re-measure the section table in the spec before continuing.

**Step 2: Write the failing test**

Create `tests/test_single_token_prompt.py`:

```python
"""The single-token judge prompt shares its inputs with TURING_PROMPT by construction."""

import hashlib

from shared.judge_prompts import (
    TURING_PROMPT,
    TURING_PROMPT_HEADER,
    TURING_SINGLE_TOKEN_PROMPT,
)

# sha256 of TURING_PROMPT as it stood at 2026-08-26, before the header/tail split.
# Pins the refactor as text-preserving: the full-schema arm supplies the reference cell
# for the switch decision, so it must not move by a single character.
TURING_PROMPT_SHA256 = "PASTE_DIGEST_FROM_STEP_1"


def test_refactor_preserves_turing_prompt_exactly():
    assert hashlib.sha256(TURING_PROMPT.encode()).hexdigest() == TURING_PROMPT_SHA256


def test_both_templates_start_with_the_shared_header():
    assert TURING_PROMPT.startswith(TURING_PROMPT_HEADER)
    assert TURING_SINGLE_TOKEN_PROMPT.startswith(TURING_PROMPT_HEADER)


def test_header_ends_at_the_evaluation_procedure_boundary():
    assert TURING_PROMPT_HEADER.rstrip().endswith("<|End Source-Copy Watchlist|>")
    assert "## Evaluation Procedure" not in TURING_PROMPT_HEADER


def test_header_carries_every_input_placeholder():
    for field in ("user_history", "context", "response_a", "response_b",
                  "source_copy_watchlist"):
        assert "{" + field + "}" in TURING_PROMPT_HEADER
```

Paste the Step 1 digest over `PASTE_DIGEST_FROM_STEP_1`.

**Step 3: Run it and watch it fail**

```bash
python -m pytest tests/test_single_token_prompt.py -q
```

Expected: `ImportError: cannot import name 'TURING_PROMPT_HEADER'`.

**Step 4: Implement the split**

In `shared/judge_prompts.py`, replace the single `TURING_PROMPT = """..."""` assignment with three. Cut the existing body at the blank lines immediately before `## Evaluation Procedure` — do not retype any of it, move it:

```python
TURING_PROMPT_HEADER = """## Task
... everything through ...
<|End Source-Copy Watchlist|>

"""

_TURING_PROMPT_TAIL = """
## Evaluation Procedure
... everything from here to the end, unchanged, including 'Your output:' ...
"""

TURING_PROMPT = TURING_PROMPT_HEADER + _TURING_PROMPT_TAIL

_SINGLE_TOKEN_TAIL = """
## Output Format

Answer with a single letter, A or B, and nothing else.
A means Response A was written by the real [HUMAN]. B means Response B was.

Your output:"""

TURING_SINGLE_TOKEN_PROMPT = TURING_PROMPT_HEADER + _SINGLE_TOKEN_TAIL
```

The `test_refactor_preserves_turing_prompt_exactly` test is what tells you the seam is in the right place. If it fails, you have gained or lost a newline at the boundary — adjust the header's trailing blank lines, not the digest.

**Step 5: Run the full suite**

```bash
python -m pytest tests/ -q
```

Expected: all pass. Other tests import `TURING_PROMPT`, and it is unchanged, so nothing else should move.

**Step 6: Commit**

```bash
git add shared/judge_prompts.py tests/test_single_token_prompt.py
git commit -m "judge: split the Turing prompt into a shared header and a verdict tail"
```

---

## Task 2: Render the single-token prompt

**Files:**
- Modify: `scripts/build_judge_train_pairs.py:151` (`render_turing_prompt`)
- Test: `tests/test_single_token_prompt.py`

`render_turing_prompt` already sanitizes the four content fields and builds the source-copy watchlist. The single-token renderer must do all of that identically — only the template differs — so add a parameter rather than a second function.

**Step 1: Write the failing test**

Append to `tests/test_single_token_prompt.py`:

```python
from scripts.build_judge_train_pairs import render_turing_prompt

_FIELDS = dict(
    user_history="[HUMAN]: earlier turn",
    context="[OTHER]: something happened",
    response_a="first candidate",
    response_b="second candidate",
)


def test_single_token_render_shares_the_full_prompt_header():
    full = render_turing_prompt(**_FIELDS)
    single = render_turing_prompt(**_FIELDS, prompt_style="single_token")
    # Everything the model reads before the verdict instruction is byte-identical.
    head = full.split("## Evaluation Procedure")[0]
    assert single.startswith(head)


def test_single_token_render_drops_the_rubric_and_schema():
    single = render_turing_prompt(**_FIELDS, prompt_style="single_token")
    for marker in ("score_gap", "immediate_target_score_a", "## Criteria",
                   "## Penalty Checks", "rating"):
        assert marker not in single
    assert single.rstrip().endswith("Your output:")
    assert "Answer with a single letter, A or B" in single


def test_single_token_render_keeps_the_watchlist_block():
    single = render_turing_prompt(**_FIELDS, prompt_style="single_token")
    assert "<|Source-Copy Watchlist|>" in single


def test_unknown_prompt_style_is_rejected():
    import pytest
    with pytest.raises(ValueError, match="prompt_style"):
        render_turing_prompt(**_FIELDS, prompt_style="nonsense")
```

**Step 2: Run it and watch it fail**

```bash
python -m pytest tests/test_single_token_prompt.py -q
```

Expected: `TypeError: render_turing_prompt() got an unexpected keyword argument 'prompt_style'`.

**Step 3: Implement**

In `scripts/build_judge_train_pairs.py`, import `TURING_SINGLE_TOKEN_PROMPT` alongside `TURING_PROMPT`, then:

```python
_PROMPT_TEMPLATES = {
    "full": TURING_PROMPT,
    "single_token": TURING_SINGLE_TOKEN_PROMPT,
}


def render_turing_prompt(
    *, user_history: str, context: str, response_a: str, response_b: str,
    prompt_style: str = "full",
) -> str:
    """Render the judge prompt exactly as the reward path does at eval time."""
    try:
        template = _PROMPT_TEMPLATES[prompt_style]
    except KeyError:
        raise ValueError(
            f"prompt_style must be one of {sorted(_PROMPT_TEMPLATES)}, got {prompt_style!r}"
        ) from None
    # ... existing sanitize + watchlist body unchanged ...
    return template.format(
        persona="",
        user_history=user_history,
        # ... rest unchanged ...
    )
```

Keep `prompt_style="full"` as the default: every existing caller must be unaffected.

**Step 4: Run tests**

```bash
python -m pytest tests/test_single_token_prompt.py tests/test_build_judge_train_pairs.py -q
```

Expected: all pass.

**Step 5: Commit**

```bash
git add scripts/build_judge_train_pairs.py tests/test_single_token_prompt.py
git commit -m "judge: render the single-token prompt through the shared renderer"
```

---

## Task 3: Extract the verdict and p_a from logprobs

This is the core new logic. It is pure and needs no network, so it gets thorough tests.

**Files:**
- Create: `shared/single_token_verdict.py`
- Test: `tests/test_single_token_verdict.py` (create)

**Step 1: Write the failing test**

```python
"""Verdict + probability extraction from a one-token logprobs payload."""

import math

import pytest

from shared.single_token_verdict import HardFail, extract_verdict

def _payload(pairs):
    """OpenAI-shaped top_logprobs: [(token, logprob), ...] for the single position."""
    return [{"token": t, "logprob": lp} for t, lp in pairs]


def test_picks_a_when_a_is_more_probable():
    v = extract_verdict(_payload([("A", -0.1), ("B", -2.3)]))
    assert v.letter == "A"
    assert v.p_a > 0.5


def test_picks_b_when_b_is_more_probable():
    v = extract_verdict(_payload([("A", -3.0), ("B", -0.05)]))
    assert v.letter == "B"
    assert v.p_a < 0.5


def test_leading_space_and_case_variants_count():
    # Qwen and Gemma differ here; both must resolve to the same two classes.
    v = extract_verdict(_payload([(" A", -0.1), ("b", -2.0), ("▁B", -3.0)]))
    assert v.letter == "A"


def test_variants_of_one_letter_are_summed_not_maxed():
    # Two B variants at -1.0 each sum to more mass than a single A at -0.8.
    v = extract_verdict(_payload([("A", -0.8), ("B", -1.0), (" B", -1.0)]))
    assert v.letter == "B"


def test_p_a_is_renormalized_over_a_and_b_only():
    # Half the mass sits on an irrelevant token; p_a must ignore it.
    v = extract_verdict(_payload([("A", math.log(0.25)), ("B", math.log(0.25)),
                                  ("Neither", math.log(0.5))]))
    assert v.p_a == pytest.approx(0.5)
    assert v.residual_mass == pytest.approx(0.5)


def test_argmax_always_agrees_with_p_a():
    for a_lp, b_lp in [(-0.1, -2.0), (-2.0, -0.1), (-0.69, -0.70)]:
        v = extract_verdict(_payload([("A", a_lp), ("B", b_lp)]))
        assert (v.letter == "A") == (v.p_a > 0.5)


def test_no_ab_token_is_a_hard_fail_not_a_coin_flip():
    with pytest.raises(HardFail):
        extract_verdict(_payload([("Neither", -0.1), ("\n", -2.0)]))


def test_empty_payload_is_a_hard_fail():
    with pytest.raises(HardFail):
        extract_verdict([])
```

**Step 2: Run it and watch it fail**

```bash
python -m pytest tests/test_single_token_verdict.py -q
```

Expected: `ModuleNotFoundError: No module named 'shared.single_token_verdict'`.

**Step 3: Implement**

Create `shared/single_token_verdict.py`:

```python
"""Turn a one-token logprobs payload into an A/B verdict and a calibrated p_a.

Kept separate from eval/metrics.py so it can be unit-tested without a server, and so the
training-side and eval-side paths cannot diverge on what "the judge said A" means.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


class HardFail(Exception):
    """Neither an A nor a B token appeared. Never silently coin-flip this.

    These failures concentrate on the longest inputs (measured on the 0.8B judge), so
    treating them as chance would bias the result rather than merely thin it.
    """


@dataclass(frozen=True)
class Verdict:
    letter: str          # "A" or "B"
    p_a: float           # P(A) renormalized over A and B only
    residual_mass: float # probability that landed on neither, before renormalization


# SentencePiece renders a leading space as U+2581; BPE tokenizers use a literal space.
_STRIP = " \t▁Ġ"


def _classify(token: str) -> str | None:
    """Map a raw token to "A", "B", or None. Case- and leading-space-insensitive."""
    cleaned = token.strip(_STRIP).upper()
    return cleaned if cleaned in ("A", "B") else None


def extract_verdict(top_logprobs: list[dict]) -> Verdict:
    """Sum probability mass per letter across tokenizer variants, then renormalize."""
    mass = {"A": 0.0, "B": 0.0}
    total = 0.0
    for entry in top_logprobs:
        p = math.exp(entry["logprob"])
        total += p
        letter = _classify(entry["token"])
        if letter is not None:
            mass[letter] += p

    ab = mass["A"] + mass["B"]
    if ab <= 0.0:
        raise HardFail(
            f"no A/B token in top_logprobs; saw {[e['token'] for e in top_logprobs]!r}"
        )

    p_a = mass["A"] / ab
    residual = max(0.0, total - ab)
    # Ties go to B so the mapping stays a total function; p_a == 0.5 exactly is
    # vanishingly rare and is visible in the recorded p_a either way.
    return Verdict(letter="A" if p_a > 0.5 else "B", p_a=p_a, residual_mass=residual)
```

**Step 4: Run tests**

```bash
python -m pytest tests/test_single_token_verdict.py -q
```

Expected: 8 passed.

**Step 5: Commit**

```bash
git add shared/single_token_verdict.py tests/test_single_token_verdict.py
git commit -m "judge: extract A/B verdict and p_a from one-token logprobs"
```

---

## Task 4: Branch the eval scorer on JUDGE_PROMPT_STYLE

> **SUPERSEDED.** `eval/metrics.py` is not on either eval pipeline's path; both launchers
> reach `scripts/run_judge_sweep_cell.py`, which called the full-schema reward scorer
> regardless of the style. The branch below was implemented, then deleted. The shipped
> design is a standalone `eval/single_token_judge.py` dispatched from
> `run_judge_sweep_cell.py`; see `tests/test_single_token_judge.py`. The rest of this
> section is kept for the request-shape and hard-fail requirements it states, which the
> new module still honours.

**Files:**
- Modify: `eval/metrics.py:569-628` (`_turing_api_call`)
- Test: `tests/test_single_token_scoring.py` (create)

The default path must stay byte-identical. That is the point of the first test below — the full-schema arm supplies the reference number the switch decision is measured against.

**Step 1: Write the failing test**

```python
"""JUDGE_PROMPT_STYLE selects the judge protocol; unset must change nothing."""

from unittest.mock import patch

import pytest

from eval import metrics

_ARGS = dict(
    context="[OTHER]: hello",
    response_a="candidate a",
    response_b="candidate b",
    user_history="[HUMAN]: past turn",
)


def _choice(pairs):
    """A minimal OpenAI choice carrying one position's top_logprobs."""
    return {"logprobs": {"content": [
        {"top_logprobs": [{"token": t, "logprob": lp} for t, lp in pairs]}
    ]}}


def _capture(monkeypatch, env, *, text_reply=None, choice_reply=None):
    """Run one scoring call, returning the kwargs the HTTP layer was handed.

    Both transports are patched because the two paths use different ones: the
    full-schema path needs response text, the single-token path needs the choice
    object so it can read logprobs.
    """
    seen = {}

    def fake_text(kwargs):
        seen.update(kwargs)
        return text_reply

    def fake_choice(kwargs):
        seen.update(kwargs)
        return choice_reply

    for k, v in env.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(metrics, "post_chat_sync", fake_text)
    monkeypatch.setattr(metrics, "post_chat_choice_sync", fake_choice, raising=False)
    return seen


def test_default_path_is_unchanged(monkeypatch):
    seen = _capture(monkeypatch, {}, text_reply='{"rating": 5, "score_gap": 0.5}')
    metrics._turing_api_call(**_ARGS, max_tokens=2048)
    assert seen["response_format"] == {"type": "json_object"}
    assert seen["max_completion_tokens"] == 2048
    assert "logprobs" not in seen
    assert "## Criteria" in seen["messages"][0]["content"]


def test_single_token_request_shape(monkeypatch):
    seen = _capture(monkeypatch, {"JUDGE_PROMPT_STYLE": "single_token"},
                    choice_reply=_choice([("A", -0.1), ("B", -2.0)]))
    metrics._turing_api_call(**_ARGS, return_details=True)
    assert seen["max_completion_tokens"] == 1
    assert "response_format" not in seen
    assert seen["logprobs"] is True
    assert seen["top_logprobs"] == 20
    assert "## Criteria" not in seen["messages"][0]["content"]


def test_single_token_result_carries_letter_and_p_a(monkeypatch):
    _capture(monkeypatch, {"JUDGE_PROMPT_STYLE": "single_token"},
             choice_reply=_choice([("A", -0.1), ("B", -2.0)]))
    out = metrics._turing_api_call(**_ARGS, return_details=True)
    assert out["letter"] == "A"
    assert out["p_a"] > 0.5
    assert out["rating"] == 1          # 1 == "definitely A" on the existing scale
    assert out["parse_error"] is None


def test_single_token_maps_b_to_rating_seven(monkeypatch):
    _capture(monkeypatch, {"JUDGE_PROMPT_STYLE": "single_token"},
             choice_reply=_choice([("A", -2.0), ("B", -0.1)]))
    out = metrics._turing_api_call(**_ARGS, return_details=True)
    assert out["letter"] == "B"
    assert out["rating"] == 7


def test_hard_fail_propagates_and_is_not_retried(monkeypatch):
    """A hard fail is a property of the input, not a transient. Retrying would hide it
    from the hard_fail column and bias the accuracy toward the shorter inputs."""
    from shared.single_token_verdict import HardFail

    calls = []

    def fake_choice(kwargs):
        calls.append(kwargs)
        return _choice([("Neither", -0.1)])

    monkeypatch.setenv("JUDGE_PROMPT_STYLE", "single_token")
    monkeypatch.setattr(metrics, "post_chat_choice_sync", fake_choice, raising=False)
    with pytest.raises(HardFail):
        metrics._turing_api_call(**_ARGS, return_details=True)
    assert len(calls) == 1


def test_unknown_style_is_rejected(monkeypatch):
    monkeypatch.setenv("JUDGE_PROMPT_STYLE", "freeform")
    with pytest.raises(ValueError, match="full|single_token"):
        metrics._turing_api_call(**_ARGS)
```

**Step 2: Run it and watch it fail**

```bash
python -m pytest tests/test_single_token_scoring.py -q
```

Expected: `test_single_token_request_shape` fails — `max_completion_tokens` is 2048 and `response_format` is present.

**Step 3: Implement**

`post_chat_sync` currently returns response *text*. The single-token path needs the logprobs object, so add a sibling named **`post_chat_choice_sync`** that returns the full choice dict, rather than overloading the existing helper — changing `post_chat_sync`'s return type would touch every caller. Put it beside `post_chat_sync` in `shared/api_client.py` and import it into `eval/metrics.py` under that name (the tests patch `metrics.post_chat_choice_sync`).

**Do not copy the retry body.** `post_chat_sync` (`shared/api_client.py:267-290`) ends by returning `_extract_chat_content(parsed)`; everything above that — key resolution, URL, headers, retry count, sleep, timeout, the request loop — is transport that both callers need. Extract that body once into a private `_post_chat_json(payload, ...) -> dict` returning the parsed response, then:

```python
def post_chat_sync(payload, **kw) -> str:
    return _extract_chat_content(_post_chat_json(payload, **kw))


def post_chat_choice_sync(payload, **kw) -> dict:
    """Full first choice, so callers can read logprobs. Same transport and retries."""
    return _post_chat_json(payload, **kw)["choices"][0]
```

`post_chat_sync`'s signature and return type must not change — every existing caller depends on both.

In `eval/metrics.py`:

```python
def _judge_prompt_style() -> str:
    style = os.environ.get("JUDGE_PROMPT_STYLE", "full")
    if style not in ("full", "single_token"):
        raise ValueError(f"JUDGE_PROMPT_STYLE must be full|single_token, got {style!r}")
    return style
```

Inside `_turing_api_call`, after the watchlist is built and `user_history` validated, branch before the retry loop:

```python
    if _judge_prompt_style() == "single_token":
        prompt = TURING_SINGLE_TOKEN_PROMPT.format(
            user_history=user_history,
            context=context,
            response_a=response_a,
            response_b=response_b,
            source_copy_watchlist=source_copy_watchlist,
        )
        kwargs = {
            "model": get_judge_model(),
            "max_completion_tokens": 1,
            "messages": [{"role": "user", "content": prompt}],
            "logprobs": True,
            "top_logprobs": 20,
        }
        choice = post_chat_choice_sync(kwargs)
        verdict = extract_verdict(choice["logprobs"]["content"][0]["top_logprobs"])
        # rating maps onto the existing 1-7 scale so downstream accuracy code is
        # untouched: 7 == "definitely B", 1 == "definitely A". No tie is possible.
        parsed = {
            "rating": 7 if verdict.letter == "B" else 1,
            "letter": verdict.letter,
            "p_a": verdict.p_a,
            "residual_mass": verdict.residual_mass,
            "parse_error": None,
        }
        if return_details:
            parsed["source_copy_warning_a"] = source_copy_warning_a
            parsed["source_copy_warning_b"] = source_copy_warning_b
            return parsed
        return parsed["rating"]
```

Do not retry on `HardFail`: a hard fail is a property of the input, not a transient, and retrying would hide it from the `hard_fail` column. Let it propagate; the caller records it.

Mapping the letter onto ratings 1 and 7 is deliberate — every existing accuracy, tie and CSV path keeps working with no changes, and `p_a` rides alongside for the new columns.

**Step 4: Run tests**

```bash
python -m pytest tests/test_single_token_scoring.py tests/ -q
```

Expected: all pass, including the pre-existing eval tests.

**Step 5: Commit**

```bash
git add eval/metrics.py tests/test_single_token_scoring.py
git commit -m "eval: add the single-token judge scoring path behind JUDGE_PROMPT_STYLE"
```

---

## Task 5: Add --prompt-style to the pair builder, with label-alignment coverage

**Files:**
- Modify: `scripts/build_judge_train_pairs.py` (arg parsing + `build_judge_rows`)
- Test: `tests/test_build_judge_train_pairs.py` (extend)

**Step 1: Write the failing tests**

Append to `tests/test_build_judge_train_pairs.py`:

```python
def test_prompt_style_changes_only_the_prompt_text():
    full = build_judge_rows(_source_df(), _inference(), prompt_style="full")
    single = build_judge_rows(_source_df(), _inference(), prompt_style="single_token")

    assert len(full) == len(single)
    for f, s in zip(full.to_dict("records"), single.to_dict("records")):
        assert f["reward_model"] == s["reward_model"]
        assert f["extra_info"] == s["extra_info"]
        assert f["prompt"][0]["content"] != s["prompt"][0]["content"]


def test_the_two_orders_of_a_pair_carry_opposite_labels():
    """A systematic order/label swap is learnable, so training would 'succeed' and the
    eval would land near 0.25. Cheaper to assert than to diagnose."""
    rows = build_judge_rows(_source_df(), _inference(), prompt_style="single_token")
    by_pair = {}
    for r in rows.to_dict("records"):
        by_pair.setdefault(r["extra_info"]["pair_id"], []).append(r)

    assert by_pair, "no pairs built"
    for pair_id, group in by_pair.items():
        labels = sorted(r["reward_model"]["ground_truth"] for r in group)
        assert labels == ["A", "B"], f"{pair_id} has labels {labels}"


def test_label_matches_which_slot_holds_the_human():
    rows = build_judge_rows(_source_df(), _inference(), prompt_style="single_token")
    for r in rows.to_dict("records"):
        expected = "B" if r["extra_info"]["human_is_b"] else "A"
        assert r["reward_model"]["ground_truth"] == expected
```

**Step 2: Run and watch fail**

```bash
python -m pytest tests/test_build_judge_train_pairs.py -q
```

Expected: `TypeError: build_judge_rows() got an unexpected keyword argument 'prompt_style'`.

**Step 3: Implement**

Thread `prompt_style` from `argparse` through `build_judge_rows` into the `render_turing_prompt` call. Default `"full"` everywhere. Add to the argument parser:

```python
    ap.add_argument(
        "--prompt-style", choices=["full", "single_token"], default="full",
        help="Judge prompt template. single_token drops the rubric and asks for one letter.",
    )
```

Record it in the `.meta.json` next to the parquet — the prompt-length distribution written there is style-dependent, and a meta file that does not say which style it describes is a trap.

**Step 4: Run tests**

```bash
python -m pytest tests/test_build_judge_train_pairs.py -q
```

Expected: all pass.

**Step 5: Commit**

```bash
git add scripts/build_judge_train_pairs.py tests/test_build_judge_train_pairs.py
git commit -m "judge-pairs: add --prompt-style and pin label/order alignment"
```

---

## Task 6: Convert judge pairs to SFT JSONL

**Files:**
- Create: `scripts/build_judge_ce_dataset.py`
- Test: `tests/test_build_judge_ce_dataset.py` (create)

`lora_sft.py` reads a JSONL of `{"messages": [...]}` where the last message is the assistant target. One row per judge pair row.

**Step 1: Write the failing test**

```python
"""Judge pair parquet -> lora_sft JSONL."""

import json

import pandas as pd

from scripts.build_judge_ce_dataset import build_ce_records


def _rows():
    return pd.DataFrame([
        {"prompt": [{"role": "user", "content": "PROMPT-A"}],
         "reward_model": {"ground_truth": "A"},
         "extra_info": {"pair_id": "p1", "order": "human_a", "user_id": "u1"}},
        {"prompt": [{"role": "user", "content": "PROMPT-B"}],
         "reward_model": {"ground_truth": "B"},
         "extra_info": {"pair_id": "p1", "order": "human_b", "user_id": "u1"}},
    ])


def test_one_record_per_row_with_the_label_as_the_assistant_turn():
    recs = build_ce_records(_rows())
    assert len(recs) == 2
    assert recs[0]["messages"] == [
        {"role": "user", "content": "PROMPT-A"},
        {"role": "assistant", "content": "A"},
    ]
    assert recs[1]["messages"][-1] == {"role": "assistant", "content": "B"}


def test_assistant_content_is_exactly_one_bare_letter():
    for rec in build_ce_records(_rows()):
        target = rec["messages"][-1]["content"]
        assert target in ("A", "B")
        assert target == target.strip()


def test_the_label_never_leaks_into_the_prompt():
    """If the answer were visible in the prompt, CE would learn a shortcut and the eval
    would collapse for reasons that look like a modelling result."""
    for rec in build_ce_records(_rows()):
        user = rec["messages"][0]["content"]
        assert "ground_truth" not in user
        assert not user.rstrip().endswith(rec["messages"][-1]["content"])


def test_records_are_json_serializable():
    for rec in build_ce_records(_rows()):
        json.loads(json.dumps(rec))
```

**Step 2: Run and watch fail**

Expected: `ModuleNotFoundError`.

**Step 3: Implement**

```python
#!/usr/bin/env python
"""Convert a judge-training pair parquet into the JSONL lora_sft.py consumes.

The judge parquet already holds the rendered prompt and an "A"/"B" label, so this is a
reshape, not a transformation. Keeping it a separate script (rather than a flag on the
pair builder) means the veRL-shaped parquet stays the single source of pairs for both
the GRPO arms and the CE arm.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def build_ce_records(df: pd.DataFrame) -> list[dict]:
    records = []
    for row in df.to_dict("records"):
        label = row["reward_model"]["ground_truth"]
        if label not in ("A", "B"):
            raise ValueError(f"unexpected label {label!r} in {row['extra_info']}")
        records.append({
            "messages": [
                {"role": "user", "content": row["prompt"][0]["content"]},
                {"role": "assistant", "content": label},
            ],
            "pair_id": row["extra_info"]["pair_id"],
            "order": row["extra_info"]["order"],
            "user_id": row["extra_info"]["user_id"],
        })
    return records


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--val-out", default=None,
                    help="If set, hold out whole users into this file for early stopping.")
    ap.add_argument("--val-frac", type=float, default=0.1)
    args = ap.parse_args()

    records = build_ce_records(pd.read_parquet(args.pairs))

    if args.val_out:
        # Split by USER, not by row: both orders of a pair, and every pair from one user,
        # must land on the same side or the val score is contaminated.
        users = sorted({r["user_id"] for r in records})
        n_val = max(1, int(len(users) * args.val_frac))
        val_users = set(users[:n_val])
        train = [r for r in records if r["user_id"] not in val_users]
        val = [r for r in records if r["user_id"] in val_users]
        Path(args.val_out).write_text("".join(json.dumps(r) + "\n" for r in val))
        print(f"val: {len(val)} rows / {len(val_users)} users -> {args.val_out}")
    else:
        train = records

    Path(args.out).write_text("".join(json.dumps(r) + "\n" for r in train))
    print(f"train: {len(train)} rows -> {args.out}")


if __name__ == "__main__":
    main()
```

**Step 4: Run tests, then commit**

```bash
python -m pytest tests/test_build_judge_ce_dataset.py -q
git add scripts/build_judge_ce_dataset.py tests/test_build_judge_ce_dataset.py
git commit -m "judge-ce: convert judge pairs into lora_sft JSONL"
```

---

## Task 7: Single-token metrics

**Files:**
- Create: `scripts/single_token_metrics.py`
- Test: `tests/test_single_token_metrics.py` (create)

**Step 1: Write the failing test**

Three cases, not one. The degenerate case alone does not show that a *correct* judge scores correctly.

```python
"""Accuracy, degeneracy and calibration columns for single-token cells."""

import pytest

from scripts.single_token_metrics import summarize

def _rows(verdicts):
    """verdicts: [(pair_id, human_is_b, letter, p_a), ...]"""
    return [
        {"pair_id": p, "human_is_b": h, "letter": l, "p_a": pa}
        for p, h, l, pa in verdicts
    ]


PERFECT = _rows([
    ("p1", False, "A", 0.9), ("p1", True, "B", 0.1),
    ("p2", False, "A", 0.8), ("p2", True, "B", 0.2),
])

ALWAYS_A = _rows([
    ("p1", False, "A", 0.9), ("p1", True, "A", 0.9),
    ("p2", False, "A", 0.9), ("p2", True, "A", 0.9),
])


def test_perfect_judge():
    s = summarize(PERFECT)
    assert s["accuracy"] == pytest.approx(1.0)
    assert s["order_consistency"] == pytest.approx(1.0)
    assert s["a_rate"] == pytest.approx(0.5)
    assert s["tie_rate"] == 0.0


def test_always_a_judge_is_flagged_degenerate_not_merely_chance():
    s = summarize(ALWAYS_A)
    assert s["accuracy"] == pytest.approx(0.5)   # looks like chance...
    assert s["a_rate"] == pytest.approx(1.0)     # ...but is not
    assert s["order_consistency"] == pytest.approx(0.0)
    assert s["degenerate"] is True


def test_perfect_judge_is_not_flagged_degenerate():
    assert summarize(PERFECT)["degenerate"] is False


def test_brier_and_auc_on_a_known_case():
    # p_human == p_a when the human is in slot A, else 1 - p_a.
    # PERFECT: p_human = .9, .9, .8, .8 -> brier = mean((1-p)^2) = 0.025
    s = summarize(PERFECT)
    assert s["brier"] == pytest.approx(0.025)
    assert s["auc"] == pytest.approx(1.0)


def test_hard_fails_are_counted_and_excluded_from_accuracy():
    rows = PERFECT + [{"pair_id": "p3", "human_is_b": False,
                       "letter": None, "p_a": None, "hard_fail": True}]
    s = summarize(rows)
    assert s["hard_fail"] == pytest.approx(1 / 5)
    assert s["accuracy"] == pytest.approx(1.0)  # scored rows only
    assert s["scored"] == 4
```

**Step 2: Run and watch fail. Step 3: Implement**

```python
"""Per-cell metrics for single-token judge runs.

`accuracy` keeps the existing definition (1 correct / 0 wrong / 0.5 tie) so these cells
drop straight into the tables from the earlier judge evals. The extra columns exist to
distinguish a judge that is genuinely uncertain from one that is answering the same
letter every time — both score accuracy 0.5.
"""

from __future__ import annotations


def _p_human(row: dict) -> float:
    """Probability the judge assigned to the slot that actually holds the human."""
    return (1.0 - row["p_a"]) if row["human_is_b"] else row["p_a"]


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    scored = [r for r in rows if not r.get("hard_fail")]

    correct = [
        1.0 if (r["letter"] == ("B" if r["human_is_b"] else "A")) else 0.0
        for r in scored
    ]
    accuracy = sum(correct) / len(scored) if scored else 0.0
    a_rate = sum(r["letter"] == "A" for r in scored) / len(scored) if scored else 0.0

    # A pair is order-consistent when its two presentations name the same underlying
    # response. Exactly one of the two orders is correct for a position-locked judge,
    # so it lands at 0.0 while a consistent judge lands at 1.0.
    by_pair: dict[str, list[dict]] = {}
    for r in scored:
        by_pair.setdefault(r["pair_id"], []).append(r)
    complete = [g for g in by_pair.values() if len(g) == 2]
    consistent = sum(
        1
        for g in complete
        if (g[0]["letter"] == ("B" if g[0]["human_is_b"] else "A"))
        == (g[1]["letter"] == ("B" if g[1]["human_is_b"] else "A"))
    )
    order_consistency = consistent / len(complete) if complete else 0.0

    ph = [_p_human(r) for r in scored]
    brier = sum((1.0 - p) ** 2 for p in ph) / len(ph) if ph else 0.0

    return {
        "n": n,
        "scored": len(scored),
        "hard_fail": (n - len(scored)) / n if n else 0.0,
        "tie_rate": 0.0,  # structurally impossible for a single token
        "accuracy": accuracy,
        "a_rate": a_rate,
        "order_consistency": order_consistency,
        "brier": brier,
        "auc": _auc(ph),
        "degenerate": not (0.3 <= a_rate <= 0.7) or order_consistency < 0.3,
    }


def _auc(p_human: list[float]) -> float:
    """P(the human slot outscores the generated slot), over all cross pairs.

    Each row contributes p_human for the human slot and 1 - p_human for the other, so
    this is a Mann-Whitney statistic over those two score sets, ties counted as 0.5.
    """
    pos, neg = p_human, [1.0 - p for p in p_human]
    if not pos or not neg:
        return 0.0
    wins = sum((a > b) + 0.5 * (a == b) for a in pos for b in neg)
    return wins / (len(pos) * len(neg))
```

**Step 4: Run tests, then commit**

```bash
python -m pytest tests/test_single_token_metrics.py -q
git add scripts/single_token_metrics.py tests/test_single_token_metrics.py
git commit -m "judge-eval: single-token metrics with degeneracy detection"
```

---

## Task 8: Paired comparison statistics

The decision rule *is* a paired test, so an error here is an error in the switch/keep verdict.

**Files:**
- Create: `scripts/paired_judge_stats.py`
- Test: `tests/test_paired_judge_stats.py` (create)

**Step 1: Write the failing test**

```python
import pytest

from scripts.paired_judge_stats import clustered_ci, mcnemar


def test_mcnemar_against_a_hand_computed_table():
    # arm A right / arm B wrong: 20.  arm A wrong / arm B right: 5.  Concordant: 100.
    a = [1] * 20 + [0] * 5 + [1] * 100
    b = [0] * 20 + [1] * 5 + [1] * 100
    r = mcnemar(a, b)
    assert r["n_discordant"] == 25
    assert r["b01"] == 20 and r["b10"] == 5
    # chi2 with continuity correction = (|20-5|-1)^2 / 25 = 7.84
    assert r["chi2"] == pytest.approx(7.84, abs=1e-2)
    assert r["p_value"] < 0.01


def test_mcnemar_is_symmetric_in_its_verdict():
    a, b = [1, 0, 1, 1], [0, 1, 1, 1]
    assert mcnemar(a, b)["p_value"] == pytest.approx(mcnemar(b, a)["p_value"])


def test_no_discordant_pairs_is_not_significant():
    assert mcnemar([1, 1, 0], [1, 1, 0])["p_value"] == 1.0


def test_clustering_widens_the_interval_when_orders_agree():
    """Both orders of each pair identical -> effective n is halved, so the clustered
    interval must be materially wider. This widening is the whole reason the column
    exists."""
    correct = [1, 1, 0, 0] * 55          # 220 pairs x 2 orders = 440 rows
    pair_ids = [f"p{i // 2}" for i in range(len(correct))]
    naive = clustered_ci(correct, pair_ids=list(range(len(correct))), seed=0)
    clustered = clustered_ci(correct, pair_ids=pair_ids, seed=0)
    assert clustered["width"] > naive["width"] * 1.2
```

**Step 2: Run and watch fail. Step 3: Implement**

```python
"""Paired statistics for arm-vs-arm judge comparisons.

Two arms score the SAME 880 rows, so comparing their marginal confidence intervals
throws away the pairing and badly understates power. And those 880 rows are 440 pairs
seen in two presentation orders, so a row-level interval understates the width.
"""

from __future__ import annotations

import math
import random


def mcnemar(a_correct: list[int], b_correct: list[int]) -> dict:
    """McNemar with continuity correction over paired per-row correctness."""
    if len(a_correct) != len(b_correct):
        raise ValueError("arms must score the same rows")
    b01 = sum(1 for a, b in zip(a_correct, b_correct) if a == 1 and b == 0)
    b10 = sum(1 for a, b in zip(a_correct, b_correct) if a == 0 and b == 1)
    n = b01 + b10
    if n == 0:
        return {"n_discordant": 0, "b01": 0, "b10": 0, "chi2": 0.0, "p_value": 1.0}
    chi2 = (abs(b01 - b10) - 1) ** 2 / n
    p = math.erfc(math.sqrt(chi2 / 2.0))  # 1 - chi2_1.cdf(x) == erfc(sqrt(x/2))
    return {"n_discordant": n, "b01": b01, "b10": b10, "chi2": chi2, "p_value": p}


def clustered_ci(
    correct: list[float], pair_ids: list, *, seed: int = 0, iters: int = 2000,
    alpha: float = 0.05,
) -> dict:
    """Bootstrap CI resampling whole pair_id clusters, not rows."""
    clusters: dict = {}
    for c, pid in zip(correct, pair_ids):
        clusters.setdefault(pid, []).append(c)
    keys = list(clusters)
    rng = random.Random(seed)

    means = []
    for _ in range(iters):
        drawn = [v for _ in keys for v in clusters[rng.choice(keys)]]
        means.append(sum(drawn) / len(drawn))
    means.sort()
    lo = means[int(alpha / 2 * iters)]
    hi = means[int((1 - alpha / 2) * iters) - 1]
    return {
        "mean": sum(correct) / len(correct),
        "lo": lo, "hi": hi, "width": hi - lo, "n_clusters": len(keys),
    }
```

**Step 4: Run tests, then commit**

```bash
python -m pytest tests/test_paired_judge_stats.py -q
git add scripts/paired_judge_stats.py tests/test_paired_judge_stats.py
git commit -m "judge-eval: paired McNemar and pair-clustered bootstrap CI"
```

---

## Task 9: Leakage and pair-set gates

These run before any GPU work. The leakage gate is the one that can make you switch protocol on a bad number.

**Files:**
- Create: `scripts/judge_ce_guards.py`
- Test: `tests/test_judge_ce_guards.py` (create)

**Step 1: Write the failing test**

```python
import pytest

from scripts.judge_ce_guards import LeakageError, check_no_user_overlap, check_sha256


def test_disjoint_users_pass():
    assert check_no_user_overlap({"u1", "u2"}, {"u3"})["overlap"] == []


def test_any_shared_user_raises():
    with pytest.raises(LeakageError, match="u2"):
        check_no_user_overlap({"u1", "u2"}, {"u2", "u3"})


def test_checksum_mismatch_raises(tmp_path):
    f = tmp_path / "pairs.parquet"
    f.write_bytes(b"not the pair set")
    with pytest.raises(ValueError, match="checksum"):
        check_sha256(f, "0" * 64)


def test_checksum_match_passes(tmp_path):
    import hashlib
    f = tmp_path / "pairs.parquet"
    f.write_bytes(b"payload")
    check_sha256(f, hashlib.sha256(b"payload").hexdigest())
```

**Step 2: Run and watch fail. Step 3: Implement**

```python
"""Pre-run gates for the single-token judge comparison.

Both guard against a result that looks fine and is wrong: training on the users you
evaluate on, and evaluating on a pair set other than the one the reused cells used.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


class LeakageError(Exception):
    """CE training users overlap the eval users."""


def check_no_user_overlap(train_users: set, eval_users: set) -> dict:
    overlap = sorted(str(u) for u in (set(train_users) & set(eval_users)))
    if overlap:
        raise LeakageError(
            f"{len(overlap)} user(s) appear in both CE training and eval: "
            f"{overlap[:10]}{'...' if len(overlap) > 10 else ''}"
        )
    return {
        "n_train_users": len(train_users),
        "n_eval_users": len(eval_users),
        "overlap": overlap,
    }


def check_sha256(path: Path, expected: str) -> str:
    actual = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(f"checksum mismatch for {path}: {actual} != {expected}")
    return actual


def write_split_guard(out: Path, payload: dict) -> None:
    Path(out).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
```

Add a `main()` that takes `--train-pairs`, `--eval-pairs`, `--expected-sha256` and `--out`, reads the `user_id` sets from both parquets, runs both checks and writes `split_guard.json`. It must exit non-zero on failure so a launcher can gate on it.

**Step 4: Run tests, then commit**

```bash
python -m pytest tests/test_judge_ce_guards.py -q
git add scripts/judge_ce_guards.py tests/test_judge_ce_guards.py
git commit -m "judge-ce: leakage and pair-set identity gates"
```

---

## Task 10: Launcher support for JUDGE_PROMPT_STYLE

**Files:**
- Modify: `scripts/launch_judge_eval_matrix.sh`
- Test: `tests/test_judge_launchers.py` (extend)

Mirror the `THINKING_MODE` guard exactly — read lines 24-40 of the launcher first and follow that shape.

**Step 1: Write the failing test**

Follow the existing harness in `tests/test_judge_launchers.py` (it shells the script with `DRY=1`). Add:

```python
def test_single_token_style_requires_a_matching_eval_root():
    """Stops a single-token run being written into a full-schema results tree, where it
    would silently corrupt a comparison."""
    rc, out = run_launcher(JUDGE_PROMPT_STYLE="single_token",
                           EVAL_ROOT="/abs/path/without-the-style-name")
    assert rc != 0
    assert "single_token" in out


def test_single_token_style_accepts_a_matching_eval_root():
    rc, _ = run_launcher(JUDGE_PROMPT_STYLE="single_token",
                         EVAL_ROOT="/abs/path/2026-08-26-single-token-judge", DRY="1")
    assert rc == 0


def test_unknown_style_is_rejected():
    rc, out = run_launcher(JUDGE_PROMPT_STYLE="freeform", EVAL_ROOT="/abs/x")
    assert rc != 0
    assert "full|single_token" in out


def test_default_style_is_full():
    rc, out = run_launcher(EVAL_ROOT="/abs/x", DRY="1")
    assert rc == 0
    assert "single_token" not in out
```

**Step 2: Run and watch fail. Step 3: Implement**

In the launcher, after the `THINKING_MODE` case block:

```bash
JUDGE_PROMPT_STYLE=${JUDGE_PROMPT_STYLE:-full}
case "$JUDGE_PROMPT_STYLE" in
  full) ;;
  single_token)
    case "$EVAL_ROOT" in
      *single*token*|*single-token*) ;;
      *) echo "FATAL: a single_token EVAL_ROOT must name the style: $EVAL_ROOT" >&2; exit 2 ;;
    esac
    ;;
  *) echo "FATAL: JUDGE_PROMPT_STYLE must be full|single_token, got '$JUDGE_PROMPT_STYLE'" >&2; exit 2 ;;
esac
export JUDGE_PROMPT_STYLE
```

Also fold the style into the per-cell reward directory so the two styles cannot collide under one `EVAL_ROOT`:

```bash
reward_dir=$SWEEP_ROOT/$cell/$THINKING_MODE/$JUDGE_PROMPT_STYLE/reward
```

**Step 4: Run tests, then commit**

```bash
python -m pytest tests/test_judge_launchers.py -q
git add scripts/launch_judge_eval_matrix.sh tests/test_judge_launchers.py
git commit -m "judge-eval: launch matrix cells under JUDGE_PROMPT_STYLE"
```

---

## Task 11: Judge-CE model aliases and configs

**Files:**
- Modify: `training/sft/lora_sft.py:14-17` (`MODEL_MAP`)
- Create: `training/sft/configs/qwen35_4b_judge_lora.yaml`
- Create: `training/sft/configs/qwen35_9b_judge_lora.yaml`
- Test: `tests/test_judge_ce_config.py` (create)

`lora_sft.py:364` derives the config filename from the `--model` alias (`-` → `_`, plus `_lora.yaml`). Reusing `qwen35-9b` would pick up the *generator* SFT config, so judge CE gets its own aliases. `get_lora_targets` matches on the `qwen35` substring, so the new aliases still select the attention+MLP-only target list — do not rename around that.

**Step 1: Write the failing test**

```python
import yaml

from training.sft.lora_sft import MODEL_MAP, get_lora_targets


def test_judge_aliases_exist_and_point_at_the_right_bases():
    assert MODEL_MAP["qwen35-4b-judge"] == "Qwen/Qwen3.5-4B"
    assert MODEL_MAP["qwen35-9b-judge"] == "Qwen/Qwen3.5-9B"


def test_judge_aliases_get_the_hybrid_safe_lora_targets():
    """LoRA on the Gated-DeltaNet backbone is destructive on Qwen3.5; the alias must not
    fall through to the qwen3 target list."""
    targets = get_lora_targets("qwen35-9b-judge")
    assert "in_proj_qkv" not in targets
    assert "q_proj" in targets and "gate_proj" in targets


def test_judge_configs_disable_packing_and_stop_token_supervision():
    for alias in ("qwen35_4b_judge", "qwen35_9b_judge"):
        cfg = yaml.safe_load(open(f"training/sft/configs/{alias}_lora.yaml"))
        # Packing concatenates examples and destroys the one-token target boundary.
        assert cfg["packing"] is False
        # Supervising <|im_end|> would make the target two tokens, not one.
        assert cfg["supervise_stop_token"] is False


def test_generator_sft_config_is_untouched():
    cfg = yaml.safe_load(open("training/sft/configs/qwen35_9b_lora.yaml"))
    assert cfg["supervise_stop_token"] is True
```

**Step 2: Run and watch fail. Step 3: Implement**

Add to `MODEL_MAP`:

```python
    "qwen35-4b-judge": "Qwen/Qwen3.5-4B",
    "qwen35-9b-judge": "Qwen/Qwen3.5-9B",
```

Create both configs (4B shown; 9B identical but `batch_size: 1`):

```yaml
# Judge discriminator CE. Distinct from the generator SFT configs: the target is a single
# A/B token, so packing is off (it would merge examples across the target boundary) and
# the stop token is NOT supervised (that would make the target two tokens). The eval path
# decodes with max_tokens=1, so the model is never asked to emit a terminator.
lora_r: 64
lora_alpha: 128
lora_dropout: 0.05
use_qlora: false

packing: false
supervise_stop_token: false

num_epochs: 3
batch_size: 2
gradient_accumulation_steps: 8
learning_rate: 1e-4
lr_scheduler: cosine
warmup_ratio: 0.05
weight_decay: 0.01

gradient_checkpointing: true
logging_steps: 10
report_to: none
save_strategy: epoch
```

Confirm `lora_sft.py` honours a `packing: false` config key; if it only reads the `--no_packing` flag, pass that flag in the launcher and keep the config key as documentation of intent.

**Step 4: Run tests, then commit**

```bash
python -m pytest tests/test_judge_ce_config.py -q
git add training/sft/lora_sft.py training/sft/configs/qwen35_4b_judge_lora.yaml training/sft/configs/qwen35_9b_judge_lora.yaml tests/test_judge_ce_config.py
git commit -m "judge-ce: add judge model aliases and single-token CE configs"
```

---

## Task 12: Merge new cells into the comparison table

**Files:**
- Create: `scripts/merge_judge_comparison.py`
- Test: `tests/test_merge_judge_comparison.py` (create)

**Step 1: Write the failing test**

```python
import pandas as pd
import pytest

from scripts.merge_judge_comparison import merge_cells


def _existing():
    return pd.DataFrame([
        {"model": "qwen35-9b", "kind": "zero-shot", "thinking_mode": "on", "accuracy": 0.5182},
        {"model": "qwen35-9b", "kind": "zero-shot", "thinking_mode": "off", "accuracy": 0.4477},
    ])


def _new():
    return pd.DataFrame([
        {"model": "qwen35-9b", "kind": "zero-shot", "thinking_mode": "off",
         "prompt_style": "single_token", "accuracy": 0.61},
    ])


def test_every_row_survives_the_merge():
    out = merge_cells([_existing(), _new()])
    assert len(out) == 3


def test_existing_rows_default_to_the_full_prompt_style():
    out = merge_cells([_existing(), _new()])
    full = out[out["prompt_style"] == "full"]
    assert len(full) == 2


def test_duplicate_cell_keys_are_refused():
    """A duplicate key silently overwrites a published number in the final table."""
    with pytest.raises(ValueError, match="duplicate"):
        merge_cells([_new(), _new()])
```

**Step 2: Run and watch fail. Step 3: Implement**

`merge_cells` concatenates, fills a missing `prompt_style` with `"full"`, then raises if `(model, kind, thinking_mode, prompt_style)` is not unique. `main()` takes `--csv` repeatedly plus `--out`.

**Step 4: Run the whole suite, then commit**

```bash
python -m pytest tests/ -q
git add scripts/merge_judge_comparison.py tests/test_merge_judge_comparison.py
git commit -m "judge-eval: merge single-token cells into the comparison table"
```

**Phase A is complete.** Every local test must be green before any cluster work.

---

# Phase B — cluster

**Rewritten 2026-08-26, after Phase A.** The original Phase B was prose, and prose let five
assumptions through unchecked — every one of the form *"the capability exists, therefore
the cluster can reach it"*. All five are now fixed or tracked below. Each task here names
the exact command that carries each flag from launcher to Python process, because that is
the step prose kept skipping.

Read `docs/cluster-workflow.md` first. Publish with `scripts/cluster_launch.sh`; never call
`sbatch` directly. Run the `preflight-job-check` skill before each submit, and keep
concurrent jobs under ~10.

**The runner lies here too.** The Bash tool replaces pytest's stdout with
`Pytest: No tests collected` and returns rc=0 regardless of reality. Any local check in
these tasks must go through a subprocess wrapper.

### Naming constraints, now enforced at submit time

A single-token eval run must satisfy **both** guards, so its `EVAL_ROOT` must contain
`thinking-off` **and** `single-token`:

```
/home/lancewicki/projects/turing-rl/results/2026-08-26-single-token-judge-thinking-off
```

The original Task 17 command used `.../2026-08-26-single-token-judge`, which lacks
`thinking-off`. That was a latent submit-time failure *before* the coupling guard existed,
because the thinking-off guard has always required the substring.

Pair-build output is nested by style, and single-token requires the style in the path:

```
full          <GENERATED>/prism/judge/iter1                (unchanged)
single_token  <GENERATED>/prism/judge/iter1/single_token
```

---

## Task B0 (PREREQUISITE): let the eval matrix reference the CE-trained judges

**This blocks Task 17 and is the last known gap.** `scripts/launch_judge_eval_matrix.sh`
hardcodes four `JUDGE_*_MODEL` env vars for the *GRPO-trained* judges (lines ~19-22), a
fixed nine-row `MATRIX` heredoc, and a model-existence loop over those same four. There is
no hook for the CE-trained models this experiment produces, and a `single_token` run would
still demand the four GRPO models exist even though they belong to the other arm.

**Files:** `scripts/launch_judge_eval_matrix.sh`, `tests/test_launch_judge_eval_matrix.py`

Make the matrix and the existence check **style-dependent**:

- `full` → today's nine rows and today's four-model check, byte-identical. This is the
  regression guard that matters; an existing full-schema submission must not change.
- `single_token` → the seven cells from the spec: five zero-shot (`qwen35-4b`, `qwen35-9b`,
  `qwen35-27b`, `gemma4-12b`, `gemma4-31b`) plus `judge-4b-ce-st` and `judge-9b-ce-st`,
  with `JUDGE_4B_CE_MODEL` / `JUDGE_9B_CE_MODEL` env vars following the existing
  `JUDGE_*_MODEL` naming, and the existence check applied to those two rather than the four.

Cell names must differ from the full-schema arm's, or the merged table cannot attribute a
row to an arm by name alone — `prompt_style` is a column, not part of the cell name.

**Guard-test paths must be inert in naming AND writable.** Three tautology mechanisms have
already produced negative tests that could not be negative on this branch: fixture row
ordering; `tmp_path` embedding the test function name into a path the guard matches; and an
unwritable synthetic path supplying a substitute non-zero exit that impersonates the guard
firing.

---

## Task 13: Build the single-token pair sets and run the gates

**Step 1 — build the CE training pairs.** `PROMPT_STYLE` now reaches the builder
(`launch_judge_pairs.sh` → `judge_train_gen.sh` → `build_judge_train_pairs.py
--prompt-style`). The nested `OUT_DIR` default applies, and the style must appear in the
path:

```bash
scripts/cluster_launch.sh --dependency-profile data \
  --run-root <ABS_RUN_ROOT> \
  --env PROMPT_STYLE=single_token \
  --env SPLITS="train val" \
  scripts/launch_judge_pairs.sh
```

Source the pairs from the **same inference pickle and slice** the GRPO judge arms used —
see the MODELS/DATA sections of `results/2026-08-12-judge-only-rlvr/README.txt`. Different
source data would confound the trained comparison.

**Step 2 — read the `.meta.json` beside each parquet.** It records `prompt_style`, and the
single-token prompt-length distribution should sit ~5k tokens below the full-schema one. If
it does not, the wrong template was rendered — stop.

**Step 3 — run the leakage and pair-set gates.** Both parquet shapes are supported (the
flat evaluation shape and the nested `extra_info` training shape):

```bash
python scripts/judge_ce_guards.py \
  --train-pairs <OUT_DIR>/train.parquet \
  --eval-pairs  /home/lancewicki/projects/turing-rl/results/2026-08-10-test-eval-9b-full5ep-full-schema/raw/pairs/gen_9b-full5ep-step0_880.parquet \
  --expected-sha256 95f48a9c52d85a6f6c49fd3387e60efe0e1ee5e436bd961f1884750ecfcf7783 \
  --out <RUN_ROOT>/split_guard.json
```

Exit 0 and `overlap: []` required. **A non-empty overlap stops the plan.** Do not work
around it: the trained cell would inflate and the comparison would be invalid.

**Step 4 — convert to the CE JSONL.** `--expect-prompt-style` defaults to `single_token`
and is checked against the sibling `.meta.json` *and* the rendered prompt text, so a
full-schema parquet fails loudly here rather than training a judge on the wrong prompt:

```bash
python scripts/build_judge_ce_dataset.py \
  --pairs <OUT_DIR>/train.parquet \
  --out <RUN_ROOT>/ce_train.jsonl \
  --val-out <RUN_ROOT>/ce_val.jsonl --val-frac 0.1 \
  --expect-prompt-style single_token
```

The val split holds out whole **users**, not rows — both orders of a pair and all of a
user's pairs land on one side, or the val score measures memorisation.

**Step 5 — record provenance:** parquet SHA-256s, row counts, source pickle path.

---

## Task 14: Target length, and the CE overfit gate

`scripts/judge_overfit_gate.py` does **not** apply — it parses veRL console logs for
`reward/judge_acc/mean`, which CE training never emits.

**Step 1 — measure the supervised span** under each tokenizer in the matrix, using
`build_chat_template_sft_features` with `supervise_stop_token=False`. Expect 1. If any
model gives 2, the chat template folds a terminator into the span: acceptable, but record
the real number in the run README and the spec, because "single-token CE" then means
something slightly different.

**Step 2 — overfit gate.** Train on ~16 pairs for many epochs, then score those same 16
through the single-token eval path with the trained adapter. Accuracy must reach ~1.0.
**A flat 0.5 means the label is not wired to the supervised position** — stop; nothing
downstream is meaningful until it passes.

---

## Task 15: One-cell serving smoke

Serve zero-shot `Qwen/Qwen3.5-4B` and score ~20 pairs with `JUDGE_PROMPT_STYLE=single_token`.

Assert before spending seven cells:

- `hard_fail == 0`.
- Every response carries `logprobs.content[0].top_logprobs`.
- Both `A` and `B` appear across the 20.
- **`choice.logprobs.content[0].token` is present.** The structural sampled-token check is
  optional by design, so if vLLM omits this field it degrades silently to a no-op and only
  the mass floor protects the rows. Decide here whether to make it required.
- **Record the `ab_mass` distribution.** `MIN_AB_MASS = 0.01` is a reasoned argument, not a
  measurement against any real judge. This is the first point real logprobs exist; confirm
  the floor sits in an empty region between genuine verdicts and noise, and adjust with
  evidence if it does not.
- Revisit the deferred `_STRIP` question with real tokens: does any judge emit a verdict
  token this branch currently rejects (`"A."`, `"A)"`, `"**A"`)?

If `top_logprobs` is absent, serving was misconfigured — fix serving, not the parser.

---

## Task 16: CE training runs

Use the existing launcher. **Do not write a new one** and **do not invoke
`training.sft.lora_sft` directly** — its own `--max_seq_length` default is 5120, judge
prompts run ~5k+, and TRL truncates from the **right**, where the `A`/`B` target sits. The
launcher hardcodes 8192.

```bash
scripts/cluster_launch.sh --dependency-profile sft \
  --run-root <ABS_RUN_ROOT> \
  --env MODEL=qwen35-9b-judge \
  --env VARIANT=bf16_fsdp \
  --env DATA=<RUN_ROOT>/ce_train.jsonl \
  --env OUT=<ABS judge checkpoint dir> \
  scripts/slurm/sft_variant.sh
```

Three things that will silently do the wrong thing if you deviate:

- **`MODEL` must be set and non-empty.** `${MODEL:-qwen3-8b}` treats empty as unset, so an
  unresolved `--env MODEL=$SOMETHING_UNSET` silently runs a **generator** SFT. Assert the
  resolved `MODEL` in the job's echo before trusting the run.
- **`VARIANT=bf16_fsdp`.** `qlora_r64` would pass `--force_qlora` against a judge config
  that sets `use_qlora: false`; the variant is not cross-checked against the model.
- **`OUT` must be set.** Its default still carries the generator's `prism_full_s42` segment.

`NOPACK=1` is **forced** by the judge aliases and must not be overridden. Under sdpa, TRL's
packing lets one example attend into a neighbour's answer letter — with a one-token target
that is the model reading the answers.

**Checkpoint selection replaces early stopping.** `lora_sft.py` has no `eval_dataset`, no
`eval_strategy` and no `load_best_model_at_end`; adding them would mean editing shared
generator-training code, which is out of scope by decision. Instead: train fixed epochs,
keep all three checkpoints, and pick the best by scoring each on `ce_val.jsonl` through the
single-token eval path. Record all three scores — the selection is then an explicit,
reproducible step rather than an implicit "last checkpoint".

Merge each adapter with `scripts/merge_sft_adapter.py` and validate before evaluating.
Expected wall-clock is minutes to low hours. **If it looks like a multi-hour GRPO job,
something is wrong** — most likely packing, or the target span is the whole prompt.

---

## Task 17: The seven-cell matrix

Requires Task B0. `EVAL_ROOT` must name both arms.

```bash
EVAL_ROOT=/home/lancewicki/projects/turing-rl/results/2026-08-26-single-token-judge-thinking-off \
JUDGE_PROMPT_STYLE=single_token \
THINKING_MODE=off CONFIRM_THINKING_OFF=1 \
JUDGE_4B_CE_MODEL=<merged 4B CE dense model> \
JUDGE_9B_CE_MODEL=<merged 9B CE dense model> \
scripts/cluster_launch.sh --dependency-profile eval \
  --run-root $EVAL_ROOT scripts/launch_judge_eval_matrix.sh
```

`single_token` with `THINKING_MODE=on` is now rejected at submit: the scorer pins
`enable_thinking=False` regardless, so a thinking-on run would attribute every artifact to
the wrong arm.

Same 880-pair parquet, checksum-verified at cell start. Per cell confirm `hard_fail == 0`,
then compute the metrics and flag any degenerate cell (`|a_rate_excess| > 0.2`).

**Sanity check that catches the whole "wrong protocol ran" family:** per-cell timing should
show ~1 completion token. If it shows ~8192, the full-schema path ran under a single-token
label — that is what the timing summary is for.

---

## Task 18: Analysis and results package

**Step 1** — merge the new cells with the reused CSVs via `scripts/merge_judge_comparison.py`.
Beware the duplicate-key hole: `NaN == NaN` but `NaN != "on"`, so a CSV lacking a
`thinking_mode` column can merge the *same* cell twice with different accuracies and no
collision raised. Supply the missing dimension per input, or refuse an input lacking a key
column.

**Step 2** — report, don't adjudicate. There is **no automated switch rule**; you read the
table and the plot and decide. Produce:

- the merged table with the reference cell (`judge-9b-graded-step52`, thinking off,
  **0.7551**) clearly marked;
- per cell `accuracy`, `a_rate`, `expected_a_rate`, `a_rate_excess`, `hard_fail`, the
  `ab_mass` distribution, `brier`, `auc`, and the interval;
- the accuracy plot;
- any degenerate cell flagged regardless of its accuracy.

Ties in the full-schema arm (`rating == 4`, 79 of 880 in the reference cell) are **kept and
counted half-right**, matching the published CSVs. They are not dropped: the single-token
arm never ties, so those are the rows where the protocols differ most.

`order_consistency` will be `None` for every cell and the clustered CI reduces to the naive
one — the 880 rows are 880 unique pairs in one order each, not 440×2. Report that as a
limitation rather than as a computed result.

**Step 3** — pull to `results/2026-08-26-single-token-judge/` with a `README.txt` carrying
**provenance only**: configuration and versions, job IDs and dates, cluster source paths,
artifact filenames and checksums, mechanical validation status, reproduction commands. Per
`CLAUDE.md`, no interpretation, no verdicts, no claims about what the numbers mean.

Record as known provenance gaps: the thinking-ON reference rows come from
`results/2026-08-12-judge-only-rlvr/judge_eval_880.csv`, which is **untracked** (`results/`
is gitignored and packages are force-added selectively), and `MIN_AB_MASS` was set by
argument rather than measurement until Task 15 confirms it.
