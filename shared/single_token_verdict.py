"""Turn a one-token logprobs payload into an A/B verdict and a calibrated p_a.

Kept separate from eval/metrics.py so it can be unit-tested without a server, and so the
training-side and eval-side paths cannot diverge on what "the judge said A" means.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


class HardFail(Exception):
    """The position did not carry a usable A/B verdict. Never silently coin-flip this.

    These failures concentrate on the longest inputs (measured on the 0.8B judge), so
    treating them as chance would bias the result rather than merely thin it.
    """


# Minimum summed A+B probability mass required before a verdict is trusted.
#
# A genuine verdict concentrates several percent even when it is split across the
# "A"/" A"/"a"/"▁A" tokenizer variants, while an incidental A/B token sitting in a
# non-verdict position lands at 1e-4..1e-2. 1% falls in the empty region between the two,
# so the exact value is not near any real decision boundary.
#
# Below such a floor, renormalizing does not merely fail to help -- it MANUFACTURES
# confidence: a 1e-9 A against a 0 B renormalizes to p_a == 1.0, which then enters brier
# as a maximally confident prediction scored 0 or 1 rather than the ~0.25 an honest
# abstention would earn.
#
# Deliberately a module constant plus a keyword-only override, NOT an environment
# variable: env-configured numerics are how two cells in one run end up scored under
# different rules with nothing in the artifacts recording it.
MIN_AB_MASS = 0.01


@dataclass(frozen=True)
class Verdict:
    letter: str    # "A" or "B"
    p_a: float     # P(A) renormalized over A and B only
    ab_mass: float # summed A+B mass before renormalization; the confidence the floor gates
    total: float   # diagnostic: total mass of the supplied top_logprobs (top-k truncated)

    @property
    def off_ab_mass(self) -> float:
        """Upper bound on the mass that is not an A/B verdict.

        Prefer this to a top-k residual (total - ab_mass). A residual is not comparable
        across rows: a flat distribution leaves much of its mass outside the top-k, so an
        identical junk level reads as a SMALLER residual -- quietest exactly when it should
        be loudest. True off-A/B mass is not computable from a truncated top-k at all,
        whereas 1 - ab_mass needs no top-k assumption and errs high, which is the safe
        direction for a gate.
        """
        return 1.0 - self.ab_mass


# SentencePiece renders a leading space as U+2581 (▁); BPE tokenizers (e.g. GPT-2, Qwen)
# use Ġ as the leading-space marker; both appear in the eval matrix (Gemma and Qwen).
# \n and \r are included because a chat template can merge a trailing newline into the
# verdict token (e.g. "\nA"); stripping them must not turn a bare "\n" into a verdict.
_STRIP = " \t\n\r▁Ġ"


def _classify(token: str) -> str | None:
    """Map a raw token to "A", "B", or None. Case- and leading-space-insensitive."""
    cleaned = token.strip(_STRIP).upper()
    return cleaned if cleaned in ("A", "B") else None


def extract_verdict(
    top_logprobs: list[dict],
    *,
    sampled_token: str | None = None,
    min_ab_mass: float = MIN_AB_MASS,
) -> Verdict:
    """Sum probability mass per letter across tokenizer variants, then renormalize.

    ``sampled_token`` is the token the model actually emitted at this position. When
    supplied it is a structural check: if it is not itself an A/B variant then the
    position is not a verdict position at all, which is near-proof and independent of any
    mass threshold. It stays optional so callers holding only top_logprobs still work.
    """
    if sampled_token is not None and _classify(sampled_token) is None:
        raise HardFail(
            f"sampled token {sampled_token!r} is not an A/B verdict; this position is "
            "not a verdict position"
        )

    mass = {"A": 0.0, "B": 0.0}
    total = 0.0
    for entry in top_logprobs:
        p = math.exp(entry["logprob"])
        total += p
        letter = _classify(entry["token"])
        if letter is not None:
            mass[letter] += p

    ab = mass["A"] + mass["B"]
    if ab < min_ab_mass:
        raise HardFail(
            f"A/B mass {ab:.3e} is below the {min_ab_mass:g} floor; "
            f"saw {[e['token'] for e in top_logprobs]!r}"
        )

    p_a = mass["A"] / ab
    # Ties go to B so the mapping stays a total function; p_a == 0.5 exactly is
    # vanishingly rare and is visible in the recorded p_a either way.
    return Verdict(letter="A" if p_a > 0.5 else "B", p_a=p_a, ab_mass=ab, total=total)
