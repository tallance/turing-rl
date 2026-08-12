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

    @property
    def recovered(self) -> bool:
        return self.rating is not None

    @property
    def format_score(self) -> float:
        components = (
            self.fmt_json_valid,
            self.fmt_all_fields,
            self.fmt_arith,
            self.fmt_rating_range,
        )
        return sum(1.0 for c in components if c) / len(components)


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
        )

    fmt_all_fields = set(data) == set(TURING_FIELDS)
    explicit_rating = _coerce_turing_rating(data.get("rating"))
    fmt_rating_range = explicit_rating is not None

    if all(field in data for field in _PRIMITIVE_FIELDS):
        rating, score_gap = derive_rating(data)
        return JudgeVerdict(
            rating=rating,
            recovery_rung="dimensions",
            fmt_json_valid=True,
            fmt_all_fields=fmt_all_fields,
            fmt_arith=_arithmetic_is_consistent(data, rating, score_gap),
            fmt_rating_range=fmt_rating_range,
        )

    if "score_gap" in data:
        return JudgeVerdict(
            rating=_rating_from_turing_score_gap(_as_float(data.get("score_gap"))),
            recovery_rung="score_gap",
            fmt_json_valid=True,
            fmt_all_fields=fmt_all_fields,
            fmt_arith=False,
            fmt_rating_range=fmt_rating_range,
        )

    if explicit_rating is not None:
        return JudgeVerdict(
            rating=explicit_rating,
            recovery_rung="rating_field",
            fmt_json_valid=True,
            fmt_all_fields=fmt_all_fields,
            fmt_arith=False,
            fmt_rating_range=True,
        )

    recovered = _extract_turing_rating(text)
    return JudgeVerdict(
        rating=recovered,
        recovery_rung="rating_text" if recovered is not None else "none",
        fmt_json_valid=True,
        fmt_all_fields=fmt_all_fields,
        fmt_arith=False,
        fmt_rating_range=False,
    )
