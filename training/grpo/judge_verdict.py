"""Parse one judge rollout into a rating plus independent format components.

veRL cannot constrain rollout decoding (no schema field exists in 0.7.0-0.9.0.dev), so
training rollouts are free-form and frequently imperfect. A rating is therefore
recovered through a fallback ladder, and format quality is scored *separately* rather
than gating the task reward: a malformed-but-parseable verdict still earns task credit,
because it still contains a prediction worth scoring.
"""

from __future__ import annotations

import json
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
from shared.prompt_utils import split_after_hidden_thinking

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

def top_level_json_keys(text: str | None) -> list[str]:
    """Stream the top-level object's keys, in emission order, tolerating truncation.

    Deliberately not a regex over raw text: field names occur inside string values (notably the
    ``reasoning`` field) and a regex would count those as emitted, which is a reward the model
    could farm by *talking about* the schema. This walks the structure and records only keys at
    depth 1. Truncation is expected -- the 2B saturates the response cap -- so a cut-off string
    or object simply ends the scan and keeps whatever was parsed.
    """
    if not isinstance(text, str):
        return []
    start = text.find("{")
    if start == -1:
        return []

    keys: list[str] = []
    depth = 0
    i = start
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == '"':
            j = i + 1
            chars: list[str] = []
            closed = False
            while j < n:
                c = text[j]
                if c == "\\":
                    # Escapes cannot appear in a canonical field name, so consuming the pair
                    # without unescaping is enough to keep the scanner in sync.
                    j += 2
                    continue
                if c == '"':
                    closed = True
                    break
                chars.append(c)
                j += 1
            if not closed:
                break  # truncated mid-string
            after = j + 1
            while after < n and text[after].isspace():
                after += 1
            if depth == 1 and after < n and text[after] == ":":
                keys.append("".join(chars))
            i = j + 1
            continue
        if ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
            if depth == 0:
                break
        i += 1
    return keys


def ordered_prefix_coverage(keys: list[str]) -> float:
    """Fraction of TURING_FIELDS emitted as an unbroken prefix in canonical order.

    Order is load-bearing: the schema puts the dimension primitives first and ``rating`` last,
    precisely because the rating is *derived* from the primitives. Crediting an unordered set
    would pay a model for emitting ``{"score_gap", "rating"}`` -- two late fields, asserted
    rather than derived -- which is the shortcut this term exists to close.
    """
    matched = 0
    for key in keys:
        if matched < len(TURING_FIELDS) and key == TURING_FIELDS[matched]:
            matched += 1
        else:
            break
    return matched / len(TURING_FIELDS)


# Types come from the schema, not a hand-maintained list: 15 of the 37 fields are strings
# (the per-dimension justifications), and hardcoding "reasoning" as the only one silently failed
# every perfect verdict.
_FIELD_TYPES: dict[str, str] = {
    name: schema["type"] for name, schema in TURING_RESPONSE_PROPERTIES.items()
}


def _values_are_well_typed(parsed: dict) -> bool:
    for name, value in parsed.items():
        declared = _FIELD_TYPES.get(name)
        if declared is None:
            return False  # not part of the schema at all
        if declared == "string":
            if not isinstance(value, str):
                return False
        elif name == "rating":
            if _coerce_turing_rating(value) is None:
                return False
        elif isinstance(value, bool) or not isinstance(value, (int, float)):
            return False
    return True


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
        parsed = json.loads(answer_text.strip())
    except (json.JSONDecodeError, ValueError):
        return None, False
    if not isinstance(parsed, dict):
        return None, False
    exact = tuple(parsed.keys()) == TURING_FIELDS and _values_are_well_typed(parsed)
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


def parse_judge_verdict(completion: str | None) -> JudgeVerdict:
    """Recover a rating and score format quality from one rollout completion."""
    text = completion if isinstance(completion, str) else ""

    # Format is scored on the ANSWER only. With thinking enabled the completion is
    # `reasoning...</think>\n\n{json}`, and </think> survives skip_special_tokens, so scoring the
    # whole string would fail json.loads on every well-formed answer and would credit field names
    # the model merely mentioned while reasoning.
    answer_text = split_after_hidden_thinking(text)
    ordered_coverage = ordered_prefix_coverage(top_level_json_keys(answer_text))
    strict_parsed, exact_schema = strict_parse_answer(answer_text)
    strict_json = strict_parsed is not None

    # The task-reward ladder stays lenient. It prefers the answer, then falls back to the whole
    # completion: extract_json_object slices first-`{` to last-`}`, so a stray brace in the
    # reasoning block would otherwise splice prose into the candidate and lose a recoverable
    # rating that the pre-thinking code would have found.
    data = extract_json_object(answer_text)
    if not isinstance(data, dict):
        data = extract_json_object(text)

    if not isinstance(data, dict):
        recovered = _extract_turing_rating(text)
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
            fmt_arith=_arithmetic_is_consistent(data, rating, score_gap),
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

    recovered = _extract_turing_rating(text)
    return JudgeVerdict(
        rating=recovered,
        recovery_rung="rating_text" if recovered is not None else "none",
        fmt_json_valid=True,
        fmt_all_fields=fmt_all_fields,
        fmt_arith=False,
        fmt_rating_range=False,
        **answer_fmt,
    )
