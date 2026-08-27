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
#
# PREMISE, and the one that would break silently: the supplied logprobs are RAW model
# outputs, taken BEFORE temperature/top_k/top_p. The cluster serves vLLM 0.18.0, whose
# ``--logprobs-mode`` defaults to ``raw_logprobs``, and neither serve invocation in
# ``scripts/slurm/judge_sweep_cell.sh`` overrides it. That is what keeps ab_mass and p_a
# comparable across cells even though each cell samples at its own generation_config
# defaults (~temperature 0.7 for Qwen3.5). Serving with ``--logprobs-mode
# processed_logprobs`` would rescale every number gated here, with nothing in the
# artifacts saying so -- if that flag is ever added, re-measure this floor.
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


def is_ab_token(token: str | None) -> bool | None:
    """Is ``token`` itself an A/B verdict variant? ``None`` when no token was reported.

    A RECORDED DIAGNOSTIC, not a gate. The emitted token is a DRAW from the distribution
    -- the cells serve at each model's generation_config defaults (~temperature 0.7 for
    Qwen3.5), no wire override -- so a judge with a healthy 95% A/B mass still samples
    off-A/B on roughly 5% of rows. Failing those would record ~5% abstentions in every
    cell and produce a different abstention set on every re-run, which is why the A/B mass
    floor alone owns abstention and this only rides along in the dump.
    """
    if token is None:
        return None
    return _classify(token) is not None


def extract_verdict(
    top_logprobs: list[dict],
    *,
    min_ab_mass: float = MIN_AB_MASS,
) -> Verdict:
    """Sum probability mass per letter across tokenizer variants, then renormalize.

    The verdict is read from the distribution, not from the sampled token; see
    ``is_ab_token`` for why the sampled token is recorded rather than gated on.
    """
    mass = {"A": 0.0, "B": 0.0}
    total = 0.0
    for entry in top_logprobs:
        try:
            p = math.exp(entry["logprob"])
        except OverflowError:
            p = math.inf
        # A non-finite probability poisons everything downstream QUIETLY: NaN propagates
        # into ab_mass, `nan < min_ab_mass` is False so the floor waves it through, the
        # tie-to-B branch absorbs `nan > 0.5`, and the row lands in the table as a
        # confident B with p_a=nan -- which then makes the cell's brier nan.
        if not math.isfinite(p):
            raise HardFail(
                f"token {entry['token']!r} has a non-finite probability "
                f"(logprob={entry['logprob']!r}); the distribution is unusable"
            )
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
