"""MERGED_EP3 named two different things; these pin the split and its back-compat.

  merge_grpo_ckpt.sh   the dense model a LoRA is merged ONTO   -> MERGE_CONTAINER
  judge_train_gen.sh   the model fake turns are SAMPLED FROM   -> GEN_MODEL

Only the first equals merged_ep3 for a round-1 generator; from round 2 it is the previous
round's dense model. The old name invited exactly that mistake -- job 19897 folded a round of
LoRA onto the SFT backbone, caught only by gate D with mismatched=128.

MERGED_EP3 stays accepted in both, because other sessions have in-flight commands using it.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MERGE = (ROOT / "scripts" / "slurm" / "merge_grpo_ckpt.sh").read_text()
GEN = (ROOT / "scripts" / "slurm" / "judge_train_gen.sh").read_text()


def _body(text: str) -> str:
    """Script text minus comments, so documentation of the old name is not mistaken for use."""
    return "\n".join(l for l in text.splitlines() if not l.strip().startswith("#"))


# --- the merge container -------------------------------------------------------------------


def test_merge_uses_the_container_name():
    assert "MERGE_CONTAINER" in _body(MERGE)


def test_merge_still_accepts_the_old_name():
    """Other sessions have in-flight commands passing MERGED_EP3; silently ignoring it would
    send the merge to the default container, which is the exact failure being designed out."""
    assert "${MERGED_EP3:-" in _body(MERGE)


def test_merge_passes_the_container_to_both_the_merge_and_the_gate():
    """The gate compares against the same base the merge used. If they could differ, gate D
    would be checking the wrong model and would pass a mis-merged checkpoint."""
    body = _body(MERGE)
    assert body.count('--base "$MERGE_CONTAINER"') == 2


def test_merge_no_longer_reads_the_bare_old_name():
    """Only as a fallback default, never as a live variable."""
    body = _body(MERGE)
    assert '"$MERGED_EP3"' not in body


# --- the generation model ------------------------------------------------------------------


def test_generation_uses_its_own_name():
    assert "GEN_MODEL" in _body(GEN)


def test_generation_still_accepts_the_old_name():
    assert "${MERGED_EP3:-" in _body(GEN)


def test_generation_samples_from_the_gen_model():
    assert '--model_id "$GEN_MODEL"' in _body(GEN)


def test_the_two_scripts_no_longer_share_one_live_variable():
    """The point of the split: the same env var no longer drives two unrelated things."""
    assert '"$MERGED_EP3"' not in _body(GEN)
    assert '"$MERGED_EP3"' not in _body(MERGE)
