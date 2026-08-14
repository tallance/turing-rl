"""Static guards on the judge held-out eval path (merge -> score on the 880 pairs)."""

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAUNCH = os.path.join(ROOT, "scripts", "launch_judge_eval.sh")
MERGE = os.path.join(ROOT, "scripts", "slurm", "merge_grpo_ckpt.sh")
SWEEP = os.path.join(ROOT, "scripts", "slurm", "judge_sweep_cell.sh")


def _text(path: str) -> str:
    with open(path) as handle:
        return handle.read()


def _code(path: str) -> str:
    """Non-comment lines only; the headers name the things they warn about."""
    return "\n".join(l for l in _text(path).splitlines() if not l.strip().startswith("#"))


def test_launcher_is_not_itself_a_job_script():
    text = _text(LAUNCH)
    assert not [l for l in text.splitlines() if l.startswith("#SBATCH")]
    assert not [
        l for l in text.splitlines()
        if "cluster_job_bootstrap.sh" in l and not l.strip().startswith("#")
    ]


def test_launcher_submits_through_the_gateway_with_the_script_boundary():
    code = _code(LAUNCH)
    assert "snapshot_sbatch.sh" in code
    assert "-- scripts/slurm/merge_grpo_ckpt.sh" in code
    assert "-- scripts/slurm/judge_sweep_cell.sh" in code
    for line in code.splitlines():
        assert not line.strip().startswith("sbatch "), "direct sbatch is forbidden"


def test_eval_reuses_the_cell_that_produced_the_baselines():
    """The trained rows are only comparable to the zero-shot rows if the SAME client scored
    both. judge_sweep_cell.sh is what pins the full ordered schema and the 8192-token budget
    (via run_judge_sweep_cell.py), so the eval must not hand-roll its own scoring path."""
    assert "scripts/slurm/judge_sweep_cell.sh" in _code(LAUNCH)
    assert "scripts/run_judge_sweep_cell.py" in _text(SWEEP)


def test_scoring_waits_for_the_merge_gate_to_PASS():
    """validate_grpo_merge exits 5 and LEAVES hf_dense on disk, so 'the directory exists' is
    not evidence the delta is there. Only afterok proves the gate passed; afterany on the
    merge would happily serve an unvalidated -- possibly untrained -- model."""
    code = _code(LAUNCH)
    assert "--dependency=afterok:$merge_jid" in code


def test_arms_do_not_share_one_merge_output_directory():
    """Both arms merge at the same global step; keyed on step alone they would overwrite
    each other and both rows would report whichever merged last."""
    assert "MODEL_TAG=$tag" in _code(LAUNCH)
    assert "${MODEL_TAG:-step${STEP}}" in _code(MERGE)


def test_launcher_rejects_an_unknown_arm():
    assert "arm must be directional or graded" in _text(LAUNCH)


def test_launcher_clears_the_stale_v2_proxy_vars():
    assert "unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY" in _text(LAUNCH)


def test_merge_container_is_overridable_but_defaults_unchanged():
    """Existing 9B callers pass MERGED_EP3 and must keep working; the judge passes CONTAINER."""
    code = _code(MERGE)
    assert "CONTAINER=${CONTAINER:-${MERGED_EP3:-" in code
    assert "merged_ep3" in code, "the 9B default container must survive"


def test_merge_no_longer_references_the_9b_container_variable_in_code():
    """Every consumer must read CONTAINER; a stray $MERGED_EP3 use would silently fold the
    judge delta into the 9B SFT backbone."""
    code = _code(MERGE)
    assert '--base "$CONTAINER"' in code
    assert '--base "$MERGED_EP3"' not in code
    assert '[ -d "$MERGED_EP3" ]' not in code


def test_merge_passes_the_target_count_to_both_the_fold_and_the_gate():
    code = _code(MERGE)
    assert code.count('--expect_targets "$EXPECT_TARGETS"') == 2


def test_shared_atol_reaches_the_gate_and_defaults_to_bit_exact_for_the_9b():
    merge = _code(MERGE)
    assert "SHARED_ATOL=${SHARED_ATOL:-0}" in merge, "the generator path must stay bit-exact"
    assert '--shared_atol "$SHARED_ATOL"' in merge


def test_judge_eval_tolerates_exactly_one_bf16_ulp_and_says_so():
    """0.00390625 = 2**-8. Anything looser would stop the gate catching a wrong container."""
    code = _code(LAUNCH)
    assert "SHARED_ATOL=${SHARED_ATOL:-0.00390625}" in code
    assert "SHARED_ATOL=$SHARED_ATOL" in code, "the merge job must receive it"
