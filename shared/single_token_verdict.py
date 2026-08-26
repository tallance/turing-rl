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
