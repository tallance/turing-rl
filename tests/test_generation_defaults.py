"""Sampling-override precedence for heldout generation.

``eval.generate_trained.apply_generation_defaults`` historically forced the domain
decoding defaults (prism -> temperature 0.6) with no way to override them. The test-set
eval needs the Qwen3 model-card sampling used by GRPO validation (0.7 / 0.8 / 20), so
explicit CLI values must win while the no-flag path stays byte-identical.

``tests/test_eval_parity.py`` covers judge-score interpretation only and does NOT guard
this behaviour.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.generate_trained import (  # noqa: E402
    DEFAULT_MODEL_ID,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
    DOMAIN_TEMPERATURES,
    apply_generation_defaults,
)

PRISM_PARQUET = "data/prism/full_s42_history_sft40_grpo60_test10/test.parquet"


def _args(**overrides) -> argparse.Namespace:
    base = dict(
        metric=None,
        gen_num=None,
        max_tokens=None,
        model_id=DEFAULT_MODEL_ID,
        test_parquet=PRISM_PARQUET,
        repetition_penalty=None,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def test_no_flags_keeps_domain_defaults():
    """Unchanged legacy path: prism -> 0.6 / 1.0 / -1."""
    out = apply_generation_defaults(_args())
    assert out.temperature == DOMAIN_TEMPERATURES["prism"] == 0.6
    assert out.top_p == DEFAULT_TOP_P
    assert out.top_k == DEFAULT_TOP_K


def test_explicit_overrides_win():
    """Model-card sampling used by the test-set eval."""
    out = apply_generation_defaults(_args(temperature=0.7, top_p=0.8, top_k=20))
    assert out.temperature == 0.7
    assert out.top_p == 0.8
    assert out.top_k == 20


def test_partial_override_leaves_others_at_domain_defaults():
    out = apply_generation_defaults(_args(temperature=0.7))
    assert out.temperature == 0.7
    assert out.top_p == DEFAULT_TOP_P
    assert out.top_k == DEFAULT_TOP_K


@pytest.mark.parametrize("field,value", [("temperature", 0.0), ("top_p", 0.0), ("top_k", 0)])
def test_explicit_zero_is_respected(field, value):
    """Zero is a real value, not 'unset' — guards a falsy-vs-None regression."""
    out = apply_generation_defaults(_args(**{field: value}))
    assert getattr(out, field) == value


def test_min_p_and_presence_penalty_remain_domain_controlled():
    """Only temp/top_p/top_k are overridable; the rest stay pinned to the domain."""
    out = apply_generation_defaults(_args(temperature=0.7))
    assert out.min_p is None
    assert out.presence_penalty == 0.5
