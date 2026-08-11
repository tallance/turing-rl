from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tarfile
from pathlib import Path

import pytest

from scripts.cluster_launch import parse_environment, validate_run_root
from scripts.cluster_workflow import (
    DEPENDENCY_PROFILES,
    clean_worktree,
    redact_argument,
    sha256_file,
    source_manifest,
)
from scripts.publish_cluster_source import remote_publish, verify_extracted_tree
from scripts.record_runtime_manifest import enforced_dependencies
from scripts.retire_cluster_checkout import record_uses_legacy_source


REPO = Path(__file__).resolve().parents[1]


def make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    (repo / "plain.txt").write_text("plain\n")
    executable = repo / "run.sh"
    executable.write_text("#!/bin/bash\ntrue\n")
    executable.chmod(0o755)
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "initial"], cwd=repo, check=True)
    return repo


def test_source_manifest_and_remote_publish_are_content_addressed(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    archive = tmp_path / "source.tar"
    subprocess.run(["git", "archive", "-o", archive, sha], cwd=repo, check=True)
    manifest = source_manifest(repo, sha)
    manifest.update(
        {
            "archive_sha256": sha256_file(archive),
            "published_at_utc": "first",
        }
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    destination = tmp_path / "sources" / sha
    lock = tmp_path / "sources" / ".publish.lock"
    remote_publish(archive, manifest_path, destination, lock)
    assert (destination / "plain.txt").read_text() == "plain\n"
    assert (destination / "run.sh").stat().st_mode & 0o111
    assert not (destination / "plain.txt").stat().st_mode & 0o222

    # First-publication time may differ when identical content is published concurrently.
    manifest["published_at_utc"] = "second"
    manifest_path.write_text(json.dumps(manifest))
    remote_publish(archive, manifest_path, destination, lock)


def test_verifier_rejects_extra_and_missing_files(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    archive = tmp_path / "source.tar"
    subprocess.run(["git", "archive", "-o", archive, sha], cwd=repo, check=True)
    extracted = tmp_path / "extracted"
    extracted.mkdir()
    with tarfile.open(archive) as handle:
        handle.extractall(extracted, filter="data")
    manifest = source_manifest(repo, sha)
    verify_extracted_tree(extracted, manifest)
    (extracted / "extra.txt").write_text("extra")
    with pytest.raises(RuntimeError, match="path mismatch"):
        verify_extracted_tree(extracted, manifest)
    (extracted / "extra.txt").unlink()
    (extracted / "plain.txt").unlink()
    with pytest.raises(RuntimeError, match="path mismatch"):
        verify_extracted_tree(extracted, manifest)


def test_verifier_rejects_extra_empty_directories(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    archive = tmp_path / "source.tar"
    subprocess.run(["git", "archive", "-o", archive, sha], cwd=repo, check=True)
    extracted = tmp_path / "extracted"
    extracted.mkdir()
    with tarfile.open(archive) as handle:
        handle.extractall(extracted, filter="data")
    (extracted / "untracked-empty-directory").mkdir()
    with pytest.raises(RuntimeError, match="extra_dirs"):
        verify_extracted_tree(extracted, source_manifest(repo, sha))


def test_clean_worktree_includes_untracked_files(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    assert clean_worktree(repo) == (True, "")
    (repo / "untracked.txt").write_text("not committed")
    clean, status = clean_worktree(repo)
    assert not clean
    assert "untracked.txt" in status


def test_run_root_policy() -> None:
    state = "/home/lancewicki/projects/turing-rl"
    validate_run_root(state, f"{state}/results/run-a", debug=False, label=None)
    validate_run_root(
        state,
        f"{state}/results/debug/probe-a/run-1",
        debug=True,
        label="probe-a",
    )
    with pytest.raises(ValueError, match="results/debug"):
        validate_run_root(state, f"{state}/results/debug/probe-a", debug=False, label=None)
    with pytest.raises(ValueError, match="probe-a"):
        validate_run_root(
            state,
            f"{state}/results/debug/other",
            debug=True,
            label="probe-a",
        )
    with pytest.raises(ValueError, match="may not contain"):
        validate_run_root(state, f"{state}/results/../outside", debug=False, label=None)
    with pytest.raises(ValueError, match="safe path component"):
        validate_run_root(
            state,
            f"{state}/results/debug/probe-a/run-1",
            debug=True,
            label="nested/probe-a",
        )


def test_legacy_job_detection_does_not_block_snapshot_workdirs() -> None:
    state = Path("/home/lancewicki/projects/turing-rl")
    assert record_uses_legacy_source(
        "JobId=1 Command=/home/lancewicki/projects/turing-rl/scripts/old.sh WorkDir=/tmp",
        state,
    )
    assert record_uses_legacy_source(
        "JobId=2 Command=/bin/bash WorkDir=/home/lancewicki/projects/turing-rl",
        state,
    )
    assert not record_uses_legacy_source(
        "JobId=3 Command=/home/lancewicki/projects/turing-rl-sources/abc/job.sh "
        "WorkDir=/home/lancewicki/projects/turing-rl/results/run/work/job-3",
        state,
    )


def test_environment_rejects_secrets_and_reserved_names() -> None:
    assert parse_environment(["MODE=full5", "N=4", "GEN_KEY=step0"]) == {
        "MODE": "full5",
        "N": "4",
        "GEN_KEY": "step0",
    }
    with pytest.raises(ValueError, match="state-root .env"):
        parse_environment(["WANDB_API_KEY=value"])
    with pytest.raises(ValueError, match="reserved"):
        parse_environment(["TURING_RL_SOURCE_SHA=bad"])


def test_provenance_redacts_generic_secret_assignments() -> None:
    assert redact_argument("--api-key=secret") == "--api-key=<redacted>"
    assert redact_argument("WANDB_API_KEY=secret") == "WANDB_API_KEY=<redacted>"
    assert (
        redact_argument("--export=ALL,API_TOKEN=secret,MODE=full")
        == "--export=ALL,API_TOKEN=<redacted>,MODE=full"
    )


def test_dependency_profiles_scope_enforcement_without_hiding_inventory() -> None:
    inventory = {
        "verl": {"sha": "verl-a"},
        "environments": [
            {"name": "train", "package_list_sha256": "train-a"},
            {"name": "gemma4", "package_list_sha256": "gemma-a"},
            {"name": "sft_qwen35", "package_list_sha256": "sft-a"},
        ],
    }
    selected = enforced_dependencies(inventory, "sft")
    assert selected["profile"] == "sft"
    assert selected["verl"] is None
    assert [item["name"] for item in selected["environments"]] == ["train", "sft_qwen35"]

    unused_changed = json.loads(json.dumps(inventory))
    unused_changed["environments"][1]["package_list_sha256"] = "gemma-b"
    assert enforced_dependencies(unused_changed, "sft") == selected

    used_changed = json.loads(json.dumps(inventory))
    used_changed["environments"][2]["package_list_sha256"] = "sft-b"
    assert enforced_dependencies(used_changed, "sft") != selected
    assert set(DEPENDENCY_PROFILES) == {"all", "data", "eval", "sft", "training"}


def test_runtime_separates_input_and_debug_generated_data(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    scripts = repo / "scripts"
    scripts.mkdir()
    for name in ("cluster_runtime.sh", "cluster_workflow.py", "publish_cluster_source.py"):
        shutil.copy2(REPO / "scripts" / name, scripts / name)
    (repo / "data").mkdir()
    (repo / "data/source-code.txt").write_text("source\n")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "runtime source"], cwd=repo, check=True)
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    archive = tmp_path / "source.tar"
    subprocess.run(["git", "archive", "-o", archive, sha], cwd=repo, check=True)
    manifest = source_manifest(repo, sha)
    manifest.update({"archive_sha256": sha256_file(archive), "published_at_utc": "test"})
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    source = tmp_path / "sources" / sha
    remote_publish(archive, manifest_path, source, tmp_path / "sources/.lock")

    state = tmp_path / "state"
    (state / "data").mkdir(parents=True)
    run = state / "results/debug/profile/run"
    env = os.environ.copy()
    env.update(
        {
            "TURING_RL_CODE_ROOT": str(source),
            "TURING_RL_STATE_ROOT": str(state),
            "TURING_RL_SOURCE_SHA": sha,
            "TURING_RL_RUN_CLASS": "debug",
            "TURING_RL_RUN_ROOT": str(run),
            "TURING_RL_DEPENDENCY_PROFILE": "eval",
        }
    )
    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "$TURING_RL_CODE_ROOT/scripts/cluster_runtime.sh"; '
            'turing_rl_prepare_runtime test; '
            'printf "%s\\n" "$TURING_RL_INPUT_DATA_ROOT" '
            '"$TURING_RL_GENERATED_DATA_ROOT" "$TURING_RL_DATA_ROOT" '
            '"$(readlink "$TURING_RL_WORK_ROOT/data")"',
        ],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    lines = result.stdout.splitlines()
    assert lines == [
        str(state / "data"),
        str(run / "data"),
        str(run / "data"),
        str(source / "data"),
    ]


def test_maintained_launchers_do_not_use_ambiguous_data_root() -> None:
    paths = list((REPO / "scripts/slurm").glob("*.sh"))
    paths.extend((REPO / "scripts").glob("launch*.sh"))
    paths.append(REPO / "scripts/verify_prism_split.sh")
    offenders = [str(path.relative_to(REPO)) for path in paths if "TURING_RL_DATA_ROOT" in path.read_text()]
    assert offenders == []


def test_all_maintained_slurm_scripts_bootstrap_immutable_runtime() -> None:
    scripts = sorted((REPO / "scripts/slurm").glob("*.sh"))
    maintained = [path for path in scripts if "#SBATCH" in path.read_text()]
    assert maintained
    missing = [path.name for path in maintained if "cluster_job_bootstrap.sh" not in path.read_text()]
    assert missing == []


def test_no_shell_launcher_calls_sbatch_directly() -> None:
    direct = re.compile(r"^\s*(?:[A-Za-z_][A-Za-z0-9_]*=\$\()?sbatch(?:\s|$)")
    offenders: list[str] = []
    for path in sorted((REPO / "scripts").rglob("*.sh")):
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if direct.search(line):
                offenders.append(f"{path.relative_to(REPO)}:{number}")
    assert offenders == []


def test_direct_snapshot_gateway_calls_include_script_boundary() -> None:
    offenders: list[str] = []
    for path in sorted((REPO / "scripts").glob("launch*.sh")):
        text = path.read_text()
        for match in re.finditer(r'\("\$SBATCH".*?\)', text, re.DOTALL):
            invocation = match.group(0)
            if '"$@"' not in invocation and not re.search(r"(?:^|\s)--(?:\s|\\)", invocation):
                offenders.append(path.name)
    arm_grid = (REPO / "scripts/slurm/submit_arm_a_grid.sh").read_text()
    if '"$SBATCH" --export=ALL -- scripts/slurm/rl_generator_run.sh' not in arm_grid:
        offenders.append("submit_arm_a_grid.sh")
    assert offenders == []


def test_hydra_entrypoints_use_writable_run_directory() -> None:
    paths = (
        REPO / "scripts/slurm/rl_generator_train.sh",
        REPO / "scripts/slurm/rl_generator_train_9b.sh",
        REPO / "bash_scripts/grpo/train_grpo.sh",
    )
    for path in paths:
        value = path.read_text()
        assert "hydra.run.dir=" in value, path
        assert "hydra.job.chdir=false" in value, path


def test_slurm_launchers_do_not_patch_immutable_source() -> None:
    forbidden = ("sed -i", "YAML_BAK=", "git apply", "git checkout")
    offenders: list[str] = []
    for path in sorted((REPO / "scripts/slurm").glob("*.sh")):
        text = path.read_text()
        for token in forbidden:
            if token in text:
                offenders.append(f"{path.name}: {token}")
    assert offenders == []


def test_snapshot_sbatch_records_submission(tmp_path: Path) -> None:
    code = tmp_path / "sources" / "abc"
    script = code / "scripts/slurm/job.sh"
    script.parent.mkdir(parents=True)
    script.write_text("#!/bin/bash\n")
    run_root = tmp_path / "run"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_sbatch = fake_bin / "sbatch"
    fake_sbatch.write_text(
        '#!/bin/bash\nprintf "%s\\n" "$@" > "$FAKE_SBATCH_ARGS"\necho "12345;cluster"\n'
    )
    fake_sbatch.chmod(0o755)
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "TURING_RL_CODE_ROOT": str(code),
            "TURING_RL_STATE_ROOT": str(tmp_path / "state"),
            "TURING_RL_SOURCE_SHA": "abc",
            "TURING_RL_RUN_CLASS": "retained",
            "TURING_RL_RUN_ROOT": str(run_root),
            "TURING_RL_DEPENDENCY_PROFILE": "eval",
            "FAKE_SBATCH_ARGS": str(tmp_path / "sbatch-args.txt"),
        }
    )
    result = subprocess.run(
        [
            os.fspath(Path(os.sys.executable)),
            os.fspath(REPO / "scripts/snapshot_sbatch.py"),
            "--export=ALL,API_TOKEN=secret,MODE=full",
            "--",
            os.fspath(script),
            "argument-ending-in.sh",
        ],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    assert result.stdout.strip() == "12345"
    record = json.loads(
        (run_root / "provenance/jobs/12345/submission.json").read_text()
    )
    assert record["source_sha"] == "abc"
    assert "API_TOKEN=<redacted>" in record["sbatch_arguments"][0]
    assert any(argument.startswith("--output=") for argument in record["sbatch_arguments"])
    assert any(argument.startswith("--error=") for argument in record["sbatch_arguments"])
    submitted = (tmp_path / "sbatch-args.txt").read_text().splitlines()
    assert str(script.resolve()) in submitted
    assert submitted[-1] == "argument-ending-in.sh"
    assert (run_root / "logs").is_dir()


def test_snapshot_sbatch_rejects_caller_managed_output(tmp_path: Path) -> None:
    code = tmp_path / "sources" / "abc"
    script = code / "job.sh"
    script.parent.mkdir(parents=True)
    script.write_text("#!/bin/bash\n")
    env = os.environ.copy()
    env.update(
        {
            "TURING_RL_CODE_ROOT": str(code),
            "TURING_RL_STATE_ROOT": str(tmp_path / "state"),
            "TURING_RL_SOURCE_SHA": "abc",
            "TURING_RL_RUN_CLASS": "debug",
            "TURING_RL_RUN_ROOT": str(tmp_path / "run"),
            "TURING_RL_DEPENDENCY_PROFILE": "eval",
        }
    )
    result = subprocess.run(
        [
            os.fspath(Path(os.sys.executable)),
            os.fspath(REPO / "scripts/snapshot_sbatch.py"),
            "--output=/tmp/escape.out",
            "--",
            os.fspath(script),
        ],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert result.returncode == 2
    assert "managed below TURING_RL_RUN_ROOT" in result.stderr


def test_snapshot_sbatch_requires_explicit_script_boundary(tmp_path: Path) -> None:
    env = os.environ.copy()
    env.update(
        {
            "TURING_RL_CODE_ROOT": str(tmp_path),
            "TURING_RL_STATE_ROOT": str(tmp_path / "state"),
            "TURING_RL_SOURCE_SHA": "abc",
            "TURING_RL_RUN_CLASS": "debug",
            "TURING_RL_RUN_ROOT": str(tmp_path / "run"),
            "TURING_RL_DEPENDENCY_PROFILE": "eval",
        }
    )
    result = subprocess.run(
        [os.sys.executable, REPO / "scripts/snapshot_sbatch.py", "job.sh"],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert result.returncode == 2
    assert "explicit '--' boundary" in result.stderr
