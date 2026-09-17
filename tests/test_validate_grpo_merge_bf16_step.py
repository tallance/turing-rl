"""Check D's one-bf16-step tolerance.

veRL holds some frozen params in fp32 and writes bf16 on save, so a bit-exact comparison of the
reconstructed backbone against the container cannot hold for those. On judge merge job 23549 all
24 Qwen3.5 Gated-DeltaNet `linear_attn.norm.weight` tensors were exactly one representable step
away and failed the gate, while the trained LoRA itself verified bit-exactly on all 128 targets.

The tolerance has to be tight enough that it cannot excuse a wrong container -- the failure the
gate exists for, which shows differences orders of magnitude larger.
"""

from __future__ import annotations

import math

import torch

ONE_BF16_STEP = 2.0**-7


def _adjacent_bf16(value: float) -> tuple[torch.Tensor, torch.Tensor]:
    """A bf16 value and its immediate neighbour.

    Stepping via torch.nextafter on the float32 view does NOT work: the next float32 rounds
    straight back to the same bf16. One bf16 step at magnitude v is 2**(floor(log2 v) - 7).
    """
    a = torch.tensor([value], dtype=torch.bfloat16)
    exact = a.float().item()
    step = 2.0 ** (math.floor(math.log2(abs(exact))) - 7)
    b = torch.tensor([exact + step], dtype=torch.bfloat16)
    return a, b


def _passes(a: torch.Tensor, b: torch.Tensor) -> bool:
    """The predicate check D applies to a shared tensor."""
    if torch.equal(a, b):
        return True
    return bool(
        a.shape == b.shape
        and torch.isclose(a.float(), b.float(), rtol=ONE_BF16_STEP, atol=0.0).all()
    )


def test_identical_tensors_pass():
    a = torch.tensor([0.9626, 1.167], dtype=torch.bfloat16)
    assert _passes(a, a.clone())


def test_one_bf16_step_is_tolerated():
    """The observed case: adjacent bf16 values around the measured magnitudes."""
    for value in (0.9626, 1.167, 0.5, 0.25):
        a, b = _adjacent_bf16(value)
        assert not torch.equal(a, b), f"{value}: neighbours should not be bit-equal"
        assert _passes(a, b), f"{value}: one step apart must be tolerated"


def test_the_measured_deltas_are_tolerated():
    """Exactly what job 23549 reported: 2^-9 at ~0.96 and 2^-8 at ~1.167."""
    a = torch.tensor([0.9626], dtype=torch.bfloat16)
    b = (a.float() + 2.0**-9).bfloat16()
    assert _passes(a, b)

    a = torch.tensor([1.167], dtype=torch.bfloat16)
    b = (a.float() + 2.0**-8).bfloat16()
    assert _passes(a, b)


def test_two_steps_apart_still_fails():
    """The tolerance is one step, not 'small'."""
    a = torch.tensor([1.0], dtype=torch.bfloat16)
    b = (a.float() + 3.0 * 2.0**-7).bfloat16()
    assert not _passes(a, b)


def test_a_wrong_container_still_fails():
    """The failure this gate exists for: a different backbone differs by ~1e-1, not ~1e-3."""
    a = torch.tensor([0.9626, 1.167, 0.5], dtype=torch.bfloat16)
    b = torch.tensor([0.8100, 1.290, 0.4], dtype=torch.bfloat16)
    assert not _passes(a, b)


def test_a_single_bad_element_fails_the_whole_tensor():
    """`.all()`, not `.any()` -- one genuinely different weight is enough to reject."""
    a = torch.tensor([1.0, 1.0, 1.0], dtype=torch.bfloat16)
    b = torch.tensor([1.0, 1.0, 1.5], dtype=torch.bfloat16)
    assert not _passes(a, b)


def test_zero_is_compared_exactly():
    """atol=0, so a value appearing out of nowhere is never excused as rounding."""
    a = torch.tensor([0.0], dtype=torch.bfloat16)
    b = torch.tensor([2.0**-20], dtype=torch.bfloat16)
    assert not _passes(a, b)
