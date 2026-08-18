"""Parse one judge rollout into a rating plus independent format components.

veRL cannot constrain rollout decoding (no schema field exists in 0.7.0-0.9.0.dev), so
training rollouts are free-form and frequently imperfect. A rating is therefore
recovered through a fallback ladder, and format quality is scored *separately* rather
than gating the task reward: a malformed-but-parseable verdict still earns task credit,
because it still contains a prediction worth scoring.
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from shared.judge_prompts import TURING_RESPONSE_PROPERTIES
from shared.judge_utils import (
    _coerce_turing_rating,
    _extract_turing_rating,
    _rating_from_turing_score_gap,
)
from shared.prompt_utils import (
    has_hidden_thinking_close,
    prompt_mode_uses_chat_template_thinking,
    split_after_hidden_thinking,
)

TURING_FIELDS: tuple[str, ...] = tuple(TURING_RESPONSE_PROPERTIES)

_JSON_FENCE_RE = re.compile(r"```json\s*(.*?)\s*```", re.DOTALL)


def extract_json_object(text: str | None) -> dict | None:
    """Pull a JSON object out of free-form completion text.

    Mirrors ``reward.py::_extract_json``: tolerates a ```json fence or prose around the
    object, and returns None rather than raising when nothing parses. Duplicated rather
    than imported because ``reward.py`` pulls in aiohttp and veRL at import time, and
    this module must stay importable anywhere.
    """
    if not isinstance(text, str):
        return None
    text = text.strip()
    if not text:
        return None
    match = _JSON_FENCE_RE.search(text)
    if match:
        text = match.group(1)
    else:
        brace_start = text.find("{")
        brace_end = text.rfind("}")
        if brace_start != -1 and brace_end != -1:
            text = text[brace_start : brace_end + 1]
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None

def _scan_string(text: str, i: int, n: int) -> tuple[str | None, int]:
    """``i`` is the opening quote. Returns (contents, index after closing quote), or (None, -1)."""
    j = i + 1
    chars: list[str] = []
    while j < n:
        c = text[j]
        if c == "\\":
            if j + 1 >= n:
                return None, -1
            chars.append(text[j + 1])
            j += 2
            continue
        if c == '"':
            return "".join(chars), j + 1
        chars.append(c)
        j += 1
    return None, -1


def _scan_value(text: str, i: int, n: int) -> int:
    """Index just past a complete JSON value at ``i``, or -1 if truncated or absent."""
    if i >= n:
        return -1
    ch = text[i]
    if ch == '"':
        return _scan_string(text, i, n)[1]
    if ch in "{[":
        depth = 0
        j = i
        while j < n:
            c = text[j]
            if c == '"':
                _, j = _scan_string(text, j, n)
                if j == -1:
                    return -1
                continue
            if c in "{[":
                depth += 1
            elif c in "}]":
                depth -= 1
                if depth == 0:
                    return j + 1
            j += 1
        return -1
    # number / true / false / null: runs until a structural delimiter or whitespace.
    j = i
    while j < n and text[j] not in ',}] \t\r\n':
        j += 1
    return j if j > i else -1


def top_level_json_entries(text: str | None) -> list[tuple[str, str]]:
    """Completed top-level (key, raw value) entries, in emission order.

    Two properties matter, and both are reward-security properties rather than niceties.

    Only depth-1 keys count, so field names occurring inside string values -- notably the
    ``reasoning`` field -- earn nothing. A regex would let the model farm coverage by *talking
    about* the schema.

    Only completed entries count. An earlier version recorded a key on seeing `"name":` and never
    checked that a value followed, so the invalid string `{"immediate_target_a":,...}` -- 37 keys,
    no values, not parseable JSON -- scored coverage 1.0 and a total reward of 0.96, beating an
    honest compact answer at 0.91 for a fraction of the effort. A value must now parse completely
    AND be followed by ',' or '}'.

    Truncation is expected (the 2B saturates the response cap): the scan stops at the cut and
    keeps the entries completed before it, costing at most the one field being written.
    """
    if not isinstance(text, str):
        return []
    start = text.find("{")
    if start == -1:
        return []

    entries: list[tuple[str, str]] = []
    n = len(text)
    i = start + 1
    while i < n:
        while i < n and text[i].isspace():
            i += 1
        if i >= n or text[i] != '"':
            break  # closing brace, truncation, or malformed key position
        key, i = _scan_string(text, i, n)
        if i == -1:
            break
        while i < n and text[i].isspace():
            i += 1
        if i >= n or text[i] != ":":
            break
        i += 1
        while i < n and text[i].isspace():
            i += 1
        value_start = i
        end = _scan_value(text, i, n)
        if end == -1:
            break
        i = end
        while i < n and text[i].isspace():
            i += 1
        if i < n and text[i] in ",}":
            entries.append((key, text[value_start:end]))  # type: ignore[arg-type]
            if text[i] == "}":
                break
            i += 1
            continue
        break
    return entries


def top_level_json_keys(text: str | None) -> list[str]:
    """Keys of the completed top-level entries, in emission order."""
    return [key for key, _ in top_level_json_entries(text)]


def _value_conforms(name: str, raw: str) -> bool:
    """True when this field's emitted value parses and satisfies its declared spec."""
    spec = _FIELD_SPECS.get(name)
    if spec is None:
        return False
    try:
        value = json.loads(raw, parse_constant=_reject_non_finite)
    except (json.JSONDecodeError, ValueError):
        return False
    return _values_are_well_formed({name: value})


def ordered_prefix_coverage(entries: list[tuple[str, str]] | list[str]) -> float:
    """Fraction of TURING_FIELDS emitted, in canonical order, with a schema-conforming value.

    Order is load-bearing: the schema puts the dimension primitives first and ``rating`` last,
    precisely because the rating is *derived* from the primitives. Crediting an unordered set
    would pay for ``{"score_gap", "rating"}`` -- two late fields, asserted rather than derived.

    Values must conform too. Structure-only coverage still paid 0.98 total for a verdict
    asserting dimension scores of 10 (bounded to [0,1]) and a score_gap of 30 (bounded to
    [-3,3]), because the numbers were internally consistent even though the rubric cannot
    produce them. Coverage measures how much of the schema was correctly produced, not how many
    colons were typed.
    """
    matched = 0
    for entry in entries:
        key, raw = entry if isinstance(entry, tuple) else (entry, None)
        if matched >= len(TURING_FIELDS) or key != TURING_FIELDS[matched]:
            break
        if raw is not None and not _value_conforms(key, raw):
            break
        matched += 1
    return matched / len(TURING_FIELDS)


# Constraints come from the schema, not a hand-maintained list: 15 of the 37 fields are strings
# (the per-dimension justifications), and all 22 numeric fields declare minimum/maximum.
_FIELD_SPECS: dict[str, dict] = dict(TURING_RESPONSE_PROPERTIES)


def _values_are_well_formed(parsed: dict) -> bool:
    """Type AND range conformance for every field, against the declared schema.

    Checking types alone was not enough. Dimension scores are bounded to [0,1] and score_gap to
    [-3,3], but a verdict asserting scores of 10 and a gap of 30 is internally consistent, so it
    passed both the type check and the arithmetic check and earned a total reward of 1.0 -- the
    maximum, for numbers the rubric cannot produce.
    """
    for name, value in parsed.items():
        spec = _FIELD_SPECS.get(name)
        if spec is None:
            return False  # not part of the schema at all
        if spec["type"] == "string":
            if not isinstance(value, str):
                return False
            continue
        # bool is an int subclass in Python; True would otherwise satisfy a 0..1 numeric range.
        if isinstance(value, bool):
            return False
        if spec["type"] == "integer":
            if not isinstance(value, int):
                return False
        else:
            if not isinstance(value, (int, float)):
                return False
            if not math.isfinite(value):
                return False
        minimum, maximum = spec.get("minimum"), spec.get("maximum")
        if minimum is not None and value < minimum:
            return False
        if maximum is not None and value > maximum:
            return False
    return True


def _reject_non_finite(constant: str) -> float:
    # json.loads accepts NaN/Infinity/-Infinity by default, none of which are legal JSON. A NaN
    # score_gap silently satisfied every range and arithmetic comparison and scored 1.0.
    raise ValueError(f"non-finite JSON constant: {constant}")


def strict_parse_answer(answer_text: str | None) -> tuple[dict | None, bool]:
    """Parse the answer strictly: whole-string json.loads, no fence stripping, no brace slicing.

    Returns ``(parsed_or_None, exact_ordered_schema)``. The tolerant ``extract_json_object`` is
    wrong here by construction -- slicing first-``{`` to last-``}`` makes partial or prose-wrapped
    output look well-formed, and a metric named "strict" that accepts that is what let the
    compact-JSON shortcut score 0.95.
    """
    if not isinstance(answer_text, str):
        return None, False
    try:
        parsed = json.loads(answer_text.strip(), parse_constant=_reject_non_finite)
    except (json.JSONDecodeError, ValueError):
        return None, False
    if not isinstance(parsed, dict):
        return None, False
    exact = tuple(parsed.keys()) == TURING_FIELDS and _values_are_well_formed(parsed)
    return parsed, exact


_DIMENSION_FIELDS_A = (
    "immediate_target_score_a",
    "human_goal_score_a",
    "communication_style_score_a",
)
_DIMENSION_FIELDS_B = (
    "immediate_target_score_b",
    "human_goal_score_b",
    "communication_style_score_b",
)
_PENALTY_FIELDS_A = (
    "source_copy_penalty_a",
    "wrong_target_or_role_penalty_a",
    "unsupported_adversarial_reframing_penalty_a",
    "assistant_like_penalty_a",
)
_PENALTY_FIELDS_B = (
    "source_copy_penalty_b",
    "wrong_target_or_role_penalty_b",
    "unsupported_adversarial_reframing_penalty_b",
    "assistant_like_penalty_b",
)
_PRIMITIVE_FIELDS = (
    _DIMENSION_FIELDS_A + _DIMENSION_FIELDS_B + _PENALTY_FIELDS_A + _PENALTY_FIELDS_B
)

# The prompt asks for one-decimal scores; 0.05 tolerates rounding without excusing
# a model that simply asserts numbers unrelated to its own dimension scores.
ARITH_TOLERANCE = 0.05


@dataclass(frozen=True)
class JudgeVerdict:
    """One parsed rollout: what it predicted, and how well-formed it was."""

    rating: int | None
    recovery_rung: str
    fmt_json_valid: bool
    fmt_all_fields: bool
    fmt_arith: bool
    fmt_rating_range: bool
    # Scored components. The four booleans above keep their original tolerant meanings so the
    # thinking-OFF ablation runs stay comparable; only these drive format_score.
    fmt_ordered_coverage: float = 0.0
    fmt_exact_schema: bool = False
    fmt_strict_json: bool = False

    @property
    def recovered(self) -> bool:
        return self.rating is not None

    @property
    def format_score(self) -> float:
        """Dense and order-aware, so partial progress toward the schema is paid for.

        The previous flat mean over four booleans made all-37-fields worth 0.25 of a 0.1-weighted
        term -- 0.025 of total reward -- so a correct compact ``{"score_gap","rating"}`` scored
        0.95 against 0.975 for a full verdict with imperfect arithmetic. Both 2B runs found that
        shortcut: fmt_all_fields fell to 0.000 while the aggregate format reward *rose*.
        Coverage now dominates, widening the compact-vs-full gap to ~0.09 without touching the
        0.9/0.1 task/format balance that produced the 9B's 0.752.
        """
        return (
            0.60 * self.fmt_ordered_coverage
            + 0.20 * float(self.fmt_exact_schema)
            + 0.10 * float(self.fmt_strict_json)
            + 0.10 * float(self.fmt_arith)
        )


def _as_float(value: object) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0


def derive_rating(data: dict) -> tuple[int, float]:
    """Recompute score_gap and rating from the primitive dimension + penalty fields.

    Mirrors the arithmetic stated in TURING_PROMPT, and matches reward.py's rule that
    model-emitted arithmetic is never authoritative when the primitives are present.
    """
    base_a = sum(_as_float(data.get(f)) for f in _DIMENSION_FIELDS_A)
    base_b = sum(_as_float(data.get(f)) for f in _DIMENSION_FIELDS_B)
    penalty_a = sum(_as_float(data.get(f)) for f in _PENALTY_FIELDS_A) / 4.0 * 3.0
    penalty_b = sum(_as_float(data.get(f)) for f in _PENALTY_FIELDS_B) / 4.0 * 3.0
    score_a = max(0.0, base_a - penalty_a)
    score_b = max(0.0, base_b - penalty_b)
    score_gap = score_b - score_a
    return _rating_from_turing_score_gap(score_gap), score_gap


def _arithmetic_is_consistent(data: dict, rating: int, score_gap: float) -> bool:
    """True when the model's own stated totals match what its primitives imply."""
    base_a = sum(_as_float(data.get(f)) for f in _DIMENSION_FIELDS_A)
    base_b = sum(_as_float(data.get(f)) for f in _DIMENSION_FIELDS_B)
    penalty_a = sum(_as_float(data.get(f)) for f in _PENALTY_FIELDS_A) / 4.0 * 3.0
    penalty_b = sum(_as_float(data.get(f)) for f in _PENALTY_FIELDS_B) / 4.0 * 3.0
    expected = {
        "base_score_a": base_a,
        "base_score_b": base_b,
        "penalty_a": penalty_a,
        "penalty_b": penalty_b,
        "response_a_score": max(0.0, base_a - penalty_a),
        "response_b_score": max(0.0, base_b - penalty_b),
        "score_gap": score_gap,
    }
    for name, want in expected.items():
        if name not in data:
            return False
        if abs(_as_float(data.get(name)) - want) > ARITH_TOLERANCE:
            return False
    return _coerce_turing_rating(data.get("rating")) == rating


def resolve_answer_text(completion: str, thinking_enabled: bool | None = None) -> str | None:
    """The answer portion of a completion, or None when no answer was produced.

    ``thinking_enabled`` defaults to the propagated PERSONA_ENABLE_THINKING, which the driver
    seeds from ``data.apply_chat_template_kwargs.enable_thinking`` and pushes to every rollout
    worker, so reward scoring sees the same mode generation ran under.

    The mode has to be an input rather than inferred from the text. Inferring it left a large
    hole: under thinking-ON, a response capped before it ever emitted ``</think>`` has no marker,
    so treating "no marker" as "the whole completion is the answer" scored the *reasoning* as an
    answer. A rollout that rambled a full verdict mid-thought and got cut off earned 0.97, while
    one that closed its block and then failed to answer earned 0.00 -- rewarding the model for
    never terminating its reasoning. At the 2B's measured 93.75% clip rate that is the dominant
    path, not an edge case.
    """
    if thinking_enabled is None:
        thinking_enabled = prompt_mode_uses_chat_template_thinking()
    if not thinking_enabled:
        return completion
    if not has_hidden_thinking_close(completion):
        return None  # still reasoning when the cap hit: nothing was answered
    return split_after_hidden_thinking(completion)


def parse_judge_verdict(completion: str | None, *, thinking_enabled: bool | None = None) -> JudgeVerdict:
    """Recover a rating and score format quality from one rollout completion."""
    text = completion if isinstance(completion, str) else ""

    # Format is scored on the ANSWER only. With thinking enabled the completion is
    # `reasoning...</think>\n\n{json}`, and </think> survives skip_special_tokens, so scoring the
    # whole string would fail json.loads on every well-formed answer and would credit field names
    # the model merely mentioned while reasoning.
    resolved = resolve_answer_text(text, thinking_enabled)
    if resolved is None:
        # Thinking was enabled and the block never closed. No answer exists, so there is nothing
        # to score on either axis; judge_reward turns an unrecovered verdict into task 0.0.
        return JudgeVerdict(
            rating=None,
            recovery_rung="unclosed_thinking",
            fmt_json_valid=False,
            fmt_all_fields=False,
            fmt_arith=False,
            fmt_rating_range=False,
        )
    answer_text = resolved
    ordered_coverage = ordered_prefix_coverage(top_level_json_entries(answer_text))
    strict_parsed, exact_schema = strict_parse_answer(answer_text)
    strict_json = strict_parsed is not None

    # The task-reward ladder stays lenient about FORM -- prose around the object, a fenced block,
    # a missing field -- but not about LOCATION: it reads the answer, never the reasoning block.
    # Recovering from the full completion let a rollout that reasoned its way to a full verdict
    # and then emitted no answer at all take the dimensions rung and full task credit. Since
    # judge_reward scores an unrecovered verdict at task 0.0, closing this drops "reason but never
    # answer" from 0.91 to 0.00 and makes answering strictly necessary.
    # When no </think> marker is present, answer_text IS the whole completion, so thinking-OFF
    # rollouts and the retained ablation runs behave exactly as before.
    data = extract_json_object(answer_text)

    if not isinstance(data, dict):
        recovered = _extract_turing_rating(answer_text)
        return JudgeVerdict(
            rating=recovered,
            recovery_rung="rating_text" if recovered is not None else "none",
            fmt_json_valid=False,
            fmt_all_fields=False,
            fmt_arith=False,
            fmt_rating_range=False,
            # Truncated output lands here, and it is exactly the case the dense term exists for:
            # coverage is read from the raw answer, so a cut-off verdict still earns its prefix.
            fmt_ordered_coverage=ordered_coverage,
            fmt_exact_schema=exact_schema,
            fmt_strict_json=strict_json,
        )

    fmt_all_fields = set(data) == set(TURING_FIELDS)
    explicit_rating = _coerce_turing_rating(data.get("rating"))
    fmt_rating_range = explicit_rating is not None
    # Answer-scoped components are identical on every rung below; the rung only decides how the
    # rating was recovered, which is a task-reward concern, not a format one.
    answer_fmt = {
        "fmt_ordered_coverage": ordered_coverage,
        "fmt_exact_schema": exact_schema,
        "fmt_strict_json": strict_json,
    }

    if all(field in data for field in _PRIMITIVE_FIELDS):
        rating, score_gap = derive_rating(data)
        return JudgeVerdict(
            rating=rating,
            recovery_rung="dimensions",
            fmt_json_valid=True,
            fmt_all_fields=fmt_all_fields,
            fmt_arith=_arithmetic_is_consistent(data, rating, score_gap) and _values_are_well_formed(data),
            fmt_rating_range=fmt_rating_range,
            **answer_fmt,
        )

    if "score_gap" in data:
        return JudgeVerdict(
            rating=_rating_from_turing_score_gap(_as_float(data.get("score_gap"))),
            recovery_rung="score_gap",
            fmt_json_valid=True,
            fmt_all_fields=fmt_all_fields,
            fmt_arith=False,
            fmt_rating_range=fmt_rating_range,
            **answer_fmt,
        )

    if explicit_rating is not None:
        return JudgeVerdict(
            rating=explicit_rating,
            recovery_rung="rating_field",
            fmt_json_valid=True,
            fmt_all_fields=fmt_all_fields,
            fmt_arith=False,
            fmt_rating_range=True,
            **answer_fmt,
        )

    recovered = _extract_turing_rating(answer_text)
    return JudgeVerdict(
        rating=recovered,
        recovery_rung="rating_text" if recovered is not None else "none",
        fmt_json_valid=True,
        fmt_all_fields=fmt_all_fields,
        fmt_arith=False,
        fmt_rating_range=False,
        **answer_fmt,
    )
