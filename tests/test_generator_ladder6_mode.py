"""The `ladder6` arm: one pass, six evenly spaced checkpoints, fixed zero-shot judge.

This is the non-alternating control for the judge<->generator loop. It trains the generator
against a FIXED judge for a single pass, and the six checkpoints are the comparison points:

    ckpt 3 = 30 steps x 64 = 1920 samples -- exactly the alternating generator's TOTAL
             (5 rounds x 384 rows), so it answers "same generator data, one judge vs five"
    ckpt 6 = 60 steps x 64 = 3840 samples -- roughly the whole alternating pipeline's data
             (generator 5x384 = 1920, plus judge 5x416 = 2080, so ~4000)

Why 3840 and not the full 4174-row split: the final checkpoint is the one every downstream
eval wants, and it is written ONLY if the last step lands on the save grid. 4174 rows is 65
steps (drop_last), and 65 has no divisor giving 6 checkpoints. The epoch-end hook cannot cover
the gap either -- _resolve_epoch_aligned_save_freq returns None when total_epochs <= 1
(verl_runtime_patch.py:675), so it is inert for a single-pass run. 3840 = 60 x 64 divides by 6
exactly, and it happens to put ckpt 3 on the alternating generator's budget to the sample.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN = (ROOT / "scripts" / "slurm" / "rl_generator_train_9b.sh").read_text()
RUN = (ROOT / "scripts" / "slurm" / "rl_generator_run_9b.sh").read_text()


def _ladder6_arm() -> str:
    """The body of the ladder6 case arm."""
    match = re.search(r"^  ladder6\)(.*?)\)\s*;;", TRAIN, re.DOTALL | re.MULTILINE)
    assert match, "no ladder6 arm in rl_generator_train_9b.sh"
    return match.group(1)


def test_the_mode_is_accepted_by_the_launcher():
    """rl_generator_run_9b.sh validates MODE against a literal list and exits 2 otherwise, so a
    new arm in the trainer alone would be rejected before the trainer ever runs."""
    assert RUN.count("ladder6") >= 2, "ladder6 must be in both the usage line and the validator"


def test_one_epoch():
    assert "trainer.total_epochs=1" in _ladder6_arm()


def test_sixty_steps_of_training_data():
    """3840 = 60 x 64. Not the full 4174: see the module docstring."""
    assert "data.train_max_samples=3840" in _ladder6_arm()


def test_six_checkpoints_with_the_last_one_on_the_grid():
    """60 / 10 = 6, and 60 % 10 == 0 so the final checkpoint is actually written."""
    arm = _ladder6_arm()
    assert "trainer.save_freq=10" in arm
    assert 60 % 10 == 0
    assert 60 // 10 == 6


def test_every_checkpoint_has_a_val_score():
    """test_freq on the same grid as save_freq -- the property the frac10 arm is built around."""
    arm = _ladder6_arm()
    assert "trainer.test_freq=10" in arm
    assert "trainer.val_before_train=True" in arm


def test_no_checkpoint_is_garbage_collected():
    """veRL's default keeps all, but 13634 was submitted with 6 and would have silently deleted
    the earliest saves. With exactly 6 checkpoints, losing any breaks the data-budget ladder."""
    assert "trainer.max_actor_ckpt_to_keep=null" in _ladder6_arm()


def test_validation_matches_the_alternating_generator_rounds():
    """352 = the same 50% val subset frac10ep3 used, so val curves are comparable between this
    control and the alternating rounds rather than being measured on different data."""
    assert "data.val_max_samples=352" in _ladder6_arm()


def test_the_third_checkpoint_is_the_alternating_generators_total():
    """The comparison this arm exists for. 5 alternating rounds x 384 rows = 1920."""
    assert 3 * 10 * 64 == 5 * 384 == 1920


def test_the_other_arms_are_untouched():
    """Additive only: the alternating rounds must keep running byte-identically."""
    assert "data.train_max_samples=384" in TRAIN, "frac10 arm changed"
    assert "trainer.total_epochs=5" in TRAIN, "full5 arm changed"
    assert "  epoch1) OVR+=( trainer.total_epochs=1 ) ;;" in TRAIN, "epoch1 changed"
