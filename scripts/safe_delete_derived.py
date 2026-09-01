#!/usr/bin/env python3
"""Delete derived GRPO model artifacts, but only after proving each one is rebuildable.

WHY THIS EXISTS
---------------
A test-eval run leaves ~37 GB of *derived* weights per generator step:

    results/<eval-run>/models/step<N>/
        hf_base/                 18 GB  reconstructed SFT backbone (same content every step)
        hf_base/lora_adapter/   223 MB  the ONLY step-unique content
        hf_dense/                19 GB  container + (alpha/r) * B@A -- what actually gets served

Both big directories are reconstructible from ``merged_ep3`` + the adapter by
``scripts/merge_grpo_adapter.py`` (see ``scripts/slurm/merge_grpo_ckpt.sh``). Eyeballing that
before an ``rm -rf`` is how a model gets lost, so this tool re-derives the proof from the bytes
on disk and refuses to delete anything it could not reconstruct.

THE TRAP THIS TOOL EXISTS TO AVOID
----------------------------------
``lora_adapter/`` lives INSIDE ``hf_base/``. ``rm -rf hf_base`` therefore destroys the one
artifact that makes hf_base *and every sibling hf_dense* recoverable, and it does so silently --
the eval numbers are already written, so nothing downstream complains. This tool relocates the
adapter (and every other non-shard file) to ``hf_base.preserved/`` and verifies the copy
byte-for-byte BEFORE the delete, and rewrites the manifests of other targets in the same batch
to cite the surviving path.

WHAT COUNTS AS PROOF
--------------------
Per target, all of these must hold (Phase 1; nothing is deleted while they run):

  guard     the resolved path is a real directory named hf_base/hf_dense, under an allowed
            root, with a ``results`` component -- a typo cannot reach a checkpoint dir.
  adapter   adapter_config.json parses with r>0/alpha>0, adapter_model.safetensors holds
            exactly 2*--expect-targets paired A/B tensors with ranks consistent with r.
  survives  the adapter is outside the target, or its relocation destination is available.
  base      the merged_ep3 container resolves, has config.json and >=1 weight shard, and
            carries every LoRA target key.
  arith     THE ACTUAL PROOF, on a seeded random sample of target modules read lazily:
              hf_dense:  dense[k] == base[k] + (alpha/r) * B@A  (bit-exact, or within 1 bf16
                         ulp AND closer than the "no delta at all" hypothesis)
              hf_base:   hf_base[k] is bit-identical to base[k] -- it is a copy of merged_ep3,
                         which is why deleting it loses nothing but the adapter.

Phase 2 deletes only targets that passed everything, and drops a
``<target>.deleted.json`` manifest beside each one recording the rebuild command.

DRY RUN IS THE DEFAULT. Pass --delete to actually remove anything.

Usage:
  # prove only -- nothing is touched
  python scripts/safe_delete_derived.py results/<eval-run>/models/step*/hf_dense

  # prove, then delete, keeping the adapters
  python scripts/safe_delete_derived.py --delete \
      results/<eval-run>/models/step*/hf_base results/<eval-run>/models/step*/hf_dense
"""
from __future__ import annotations

import argparse
import errno
import gc
import glob
import json
import os
import random
import shlex
import shutil
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.merge_grpo_adapter import load_adapter  # noqa: E402
from scripts.reuse_test_eval_step import sha256_file  # noqa: E402
from scripts.validate_grpo_merge import Model, normalize_merger_key  # noqa: E402

KIND_DENSE = "hf_dense"
KIND_BASE = "hf_base"
KNOWN_KINDS = (KIND_BASE, KIND_DENSE)

DEFAULT_EXPECT_TARGETS = 128
DEFAULT_SAMPLE = 3
DEFAULT_RESULTS_DIRNAME = "results"
PRESERVE_SUFFIX = ".preserved"
MANIFEST_SUFFIX = ".deleted.json"

# hf_base legitimately lacks the container's mtp.* keys: the actor trained with
# mtp_num_hidden_layers=0. Same allowlist as the merge gate (scripts/validate_grpo_merge.py).
DEFAULT_ALLOW_MISSING = ("mtp.",)

# bf16 keeps 8 log2-mantissa bits, so one ulp is a 2^-8 relative step. The merge stores
# ``(w.float() + delta).to(bf16)``, so recomputing it here should be bit-exact; the ulp bound
# only absorbs a differing float32 matmul reduction order on another machine.
BF16_RELATIVE_ULP = 2.0**-8


class CheckFailure(Exception):
    """A recoverability check said no. The target is skipped, never deleted."""


# --------------------------------------------------------------------------- helpers


def human_bytes(n: int) -> str:
    size = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024 or unit == "TiB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024
    raise AssertionError("unreachable")


def dir_size(path: Path, *, skip: set[Path] | None = None) -> int:
    """Bytes on disk under ``path``, excluding ``skip`` subtrees and never following symlinks."""
    skip = skip or set()
    total = 0
    for root, dirnames, filenames in os.walk(path, followlinks=False):
        root_path = Path(root)
        if root_path in skip:
            dirnames[:] = []
            continue
        dirnames[:] = [d for d in dirnames if (root_path / d) not in skip]
        for name in filenames:
            entry = root_path / name
            if entry.is_symlink():
                continue
            total += entry.stat().st_size
    return total


def expand_targets(patterns: list[str]) -> list[Path]:
    """Expand globs ourselves so a quoted pattern behaves the same as an unquoted one.

    A pattern that matches nothing falls through as a literal path so it surfaces as a normal
    FAIL row (and a nonzero exit) rather than aborting the whole batch.
    """
    out: list[Path] = []
    seen: set[Path] = set()
    for pattern in patterns:
        matches = sorted(glob.glob(pattern)) or [pattern]
        for match in matches:
            # abspath, NOT resolve: resolve() would silently follow a symlinked target and
            # delete whatever it points at, which is exactly what the guard must catch.
            path = Path(os.path.abspath(match))
            if path not in seen:
                seen.add(path)
                out.append(path)
    return out


def file_inventory(path: Path) -> list[dict]:
    """Relative name + size for every file under ``path``, sorted. Sizes only: hashing 19 GB
    of shards to record a manifest would cost more than the rebuild it documents."""
    items = []
    for root, _dirs, filenames in os.walk(path, followlinks=False):
        for name in sorted(filenames):
            entry = Path(root) / name
            if entry.is_symlink():
                continue
            items.append(
                {"name": str(entry.relative_to(path)), "bytes": entry.stat().st_size}
            )
    return sorted(items, key=lambda item: item["name"])


# --------------------------------------------------------------------------- resolution


def check_path_guard(target: Path, allowed_roots: list[Path], results_dirname: str) -> None:
    if target.is_symlink():
        raise CheckFailure(f"target is a symlink: {target}")
    if not target.is_dir():
        raise CheckFailure(f"not a directory: {target}")
    if target.name not in KNOWN_KINDS:
        raise CheckFailure(
            f"unrecognized artifact name {target.name!r}; this tool only deletes {list(KNOWN_KINDS)}"
        )
    # Judge location by the real path, so a symlinked *parent* cannot smuggle a target past
    # the roots. The target itself is already known not to be a link.
    real = target.resolve()
    if results_dirname not in real.parts:
        raise CheckFailure(
            f"refusing: {target} has no {results_dirname!r} path component "
            "(only derived artifacts under a results root are deletable)"
        )
    if not any(real == root or root in real.parents for root in allowed_roots):
        raise CheckFailure(
            f"refusing: {target} is outside the allowed roots "
            f"{[str(r) for r in allowed_roots]} (pass --allowed-root to widen)"
        )


def read_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None


def merge_report_for(target: Path) -> tuple[Path | None, dict | None]:
    """``grpo_merge_report.json`` for this step, whether the target is hf_dense or hf_base."""
    candidates = [target / "grpo_merge_report.json"]
    if target.name == KIND_BASE:
        candidates.append(target.parent / KIND_DENSE / "grpo_merge_report.json")
    for candidate in candidates:
        report = read_json(candidate)
        if report is not None:
            return candidate, report
    return None, None


def sibling_base_dirs(target: Path) -> list[Path]:
    """Where a step's hf_base content can live: in place, or already relocated by this tool."""
    step = target.parent
    return [step / KIND_BASE, step / (KIND_BASE + PRESERVE_SUFFIX)]


def _norm(path: Path) -> Path:
    """Symlink-normalised form, for path COMPARISONS only -- never for writing.

    Both sides of a containment test must be normalised or the test silently inverts.
    ``resolve_adapter`` returns a resolved path while a target is whatever the caller typed,
    and on this cluster ``/home/lancewicki`` is a symlink to ``/storage/home/lancewicki``. So
    "is this adapter inside the directory I am about to delete?" compared
    ``/storage/home/.../hf_base/lora_adapter`` against ``/home/.../hf_base``, answered NO for
    an adapter that was plainly inside, skipped preservation, and deleted the adapter along
    with the shards (judge-r2-4b-graded-thinkon-step52, 2026-09-01). The path guard got this
    right by resolving both sides; these two call sites did not.
    """
    try:
        return path.resolve()
    except OSError:  # a broken symlink still has to compare as itself, not explode
        return path.absolute()


def is_within(child: Path, parent: Path) -> bool:
    """Is ``child`` ``parent`` itself, or underneath it? Symlink-safe on both sides."""
    child_n, parent_n = _norm(child), _norm(parent)
    return child_n == parent_n or parent_n in child_n.parents


def release_mapped_tensors(target) -> None:
    """Drop tensors that are memory-mapped out of the directory we are about to delete.

    ``load_adapter`` returns the A/B pairs as safetensors tensors, which are mmap-backed by
    ``lora_adapter/adapter_model.safetensors``. Holding them keeps that file mapped, and on
    NFS the open mapping makes ``rmdir`` of the enclosing directory fail with EBUSY -- the
    shards unlink fine and then the now-empty ``lora_adapter/`` cannot be removed while this
    process lives. Nothing downstream needs the tensors: the manifest cites r/alpha/sha256,
    which are plain values.
    """
    meta = getattr(target, "adapter_meta", None)
    if isinstance(meta, dict):
        meta.pop("pairs", None)
    gc.collect()


def rmtree_nfs(path: Path, attempts: int = 6) -> None:
    """``shutil.rmtree``, tolerant of NFS close-to-open consistency.

    ``/storage/home`` is FSx over NFSv4. rmtree unlinks a directory's files and then rmdirs
    it, but the server can still report the directory non-empty for a moment after the
    unlinks land -- ENOTEMPTY on a directory that ``ls -A`` shows as empty. It also leaves
    ``.nfsXXXX`` silly-rename placeholders behind for files that were still open, which
    disappear on their own once the handle closes.

    Both resolve by waiting and retrying; the contents are already gone either way. Failing
    here is not dangerous (the adapter is preserved and verified before this runs) but it
    aborted a 23-target batch on its first target twice, so it is worth absorbing.
    """
    delay = 0.5
    for attempt in range(attempts):
        try:
            shutil.rmtree(path)
            return
        except FileNotFoundError:
            return
        except OSError as exc:
            # EBUSY too: an NFS silly-rename placeholder for a file some other process still
            # holds clears on its own once that handle closes.
            if exc.errno not in (errno.ENOTEMPTY, errno.EBUSY) or attempt == attempts - 1:
                raise
            time.sleep(delay)
            delay *= 2


def resolve_adapter(target: Path, explicit: Path | None, report: dict | None) -> Path:
    if explicit is not None:
        return explicit.resolve()
    candidates = []
    if target.name == KIND_BASE:
        candidates.append(target / "lora_adapter")
    else:
        candidates.extend(d / "lora_adapter" for d in sibling_base_dirs(target))
        if report and report.get("adapter"):
            candidates.append(Path(report["adapter"]))
    for candidate in candidates:
        if (candidate / "adapter_model.safetensors").is_file():
            return candidate.resolve()
    raise CheckFailure(
        f"no lora_adapter with adapter_model.safetensors found; tried "
        f"{[str(c) for c in candidates]} (pass --adapter)"
    )


def resolve_base(target: Path, explicit: Path | None, report: dict | None,
                 search_roots: list[Path]) -> Path:
    """Locate the merged_ep3 container the dense model was built from.

    ``grpo_merge_report.json`` records the base as it was seen from inside the job -- a path
    under the run's staged ``work/launcher-*/`` copy, which is routinely gone by the time
    anyone reclaims disk. So a recorded-but-missing base is re-anchored by its
    ``checkpoints/...`` suffix under the known roots rather than treated as fatal.
    """
    if explicit is not None:
        return explicit.resolve()
    recorded = report.get("base") if report else None
    if not recorded:
        raise CheckFailure(
            f"no grpo_merge_report.json recording the base container for {target} (pass --base)"
        )
    recorded_path = Path(recorded)
    if recorded_path.is_dir():
        return recorded_path.resolve()
    parts = recorded_path.parts
    if "checkpoints" in parts:
        suffix = Path(*parts[parts.index("checkpoints") :])
        for root in search_roots:
            candidate = root / suffix
            if candidate.is_dir():
                return candidate.resolve()
    raise CheckFailure(
        f"base container recorded as {recorded} does not exist and was not found under "
        f"{[str(r) for r in search_roots]} (pass --base)"
    )


# --------------------------------------------------------------------------- checks


def check_adapter_complete(adapter_dir: Path, expect_targets: int) -> dict:
    """Reuses merge_grpo_adapter.load_adapter so the pairing/scaling logic cannot drift from
    the merge it certifies -- an adapter this loader rejects is one the rebuild would reject."""
    config = read_json(adapter_dir / "adapter_config.json")
    if config is None:
        raise CheckFailure(f"adapter_config.json missing or unparseable in {adapter_dir}")
    try:
        rank, alpha = int(config["r"]), int(config["lora_alpha"])
    except (KeyError, TypeError, ValueError) as exc:
        raise CheckFailure(f"adapter_config.json has no usable r/lora_alpha: {exc}") from exc
    if rank <= 0 or alpha <= 0:
        raise CheckFailure(f"implausible LoRA config r={rank} alpha={alpha} in {adapter_dir}")

    try:
        pairs, scaling = load_adapter(adapter_dir)
    except (ValueError, OSError, KeyError) as exc:
        raise CheckFailure(f"adapter unreadable: {exc}") from exc
    if len(pairs) != expect_targets:
        raise CheckFailure(
            f"adapter has {len(pairs)} LoRA targets, expected {expect_targets} "
            "(wrong adapter, or a truncated file)"
        )
    # Orientation guard: A is (r, in) and B is (out, r). A swapped pair would still merge and
    # would still be "rebuildable", just into a different model.
    for key, (lora_a, lora_b) in pairs.items():
        if lora_a.shape[0] != rank or lora_b.shape[1] != rank:
            raise CheckFailure(
                f"{key} has lora_A{tuple(lora_a.shape)} lora_B{tuple(lora_b.shape)}, "
                f"inconsistent with r={rank}"
            )
    return {"pairs": pairs, "scaling": scaling, "r": rank, "alpha": alpha}


def open_model(path: Path, label: str, normalize=None) -> Model:
    try:
        return Model(path, normalize=normalize)
    except SystemExit as exc:  # Model exits the process on an empty/unreadable dir
        raise CheckFailure(f"{label}: {exc}") from exc


def check_base_loadable(base_dir: Path) -> Model:
    if not (base_dir / "config.json").is_file():
        raise CheckFailure(f"base container {base_dir} has no config.json")
    if not sorted(base_dir.glob("*.safetensors")):
        raise CheckFailure(f"base container {base_dir} has no safetensors shards")
    return open_model(base_dir, f"base container {base_dir}")


def sample_keys(keys: list[str], count: int, seed: int) -> list[str]:
    ordered = sorted(keys)
    if count >= len(ordered):
        return ordered
    return sorted(random.Random(seed).sample(ordered, count))


def verify_dense_arithmetic(target: Path, base: Model, adapter: dict, *, count: int,
                            seed: int, ulps: float) -> list[dict]:
    """dense == base + (alpha/r) * B@A on a sample. This is the whole proof for hf_dense."""
    dense = open_model(target, f"target {target}")
    pairs, scaling = adapter["pairs"], adapter["scaling"]

    missing_in_base = sorted(set(pairs) - base.keys())
    if missing_in_base:
        raise CheckFailure(
            f"{len(missing_in_base)} adapter targets absent from the base container, e.g. "
            f"{missing_in_base[:3]} -- wrong base, so the rebuild would not reproduce this model"
        )
    missing_in_dense = sorted(set(pairs) - dense.keys())
    if missing_in_dense:
        raise CheckFailure(
            f"{len(missing_in_dense)} adapter targets absent from {target}, e.g. {missing_in_dense[:3]}"
        )

    results = []
    for key in sample_keys(list(pairs), count, seed):
        w_base = base.get(key)
        w_dense = dense.get(key)
        lora_a, lora_b = pairs[key]
        # float32 accumulate then cast back -- byte-for-byte what merge_grpo_adapter does.
        delta = (lora_b.float() @ lora_a.float()) * scaling
        if delta.shape != w_base.shape:
            raise CheckFailure(
                f"{key}: delta{tuple(delta.shape)} does not match base{tuple(w_base.shape)}"
            )
        expected = (w_base.float() + delta).to(w_base.dtype)
        bit_exact = bool(torch.equal(w_dense, expected))
        err = float((w_dense.float() - expected.float()).abs().max())
        realized = float((w_dense.float() - w_base.float()).abs().max())
        tol = ulps * BF16_RELATIVE_ULP * float(expected.float().abs().max())
        # Tolerating up to an ulp is not enough on its own: on these adapters the delta itself
        # is sub-ulp for some tensors, so an unmerged copy of the base would slip through. Also
        # require the residual to be smaller than the change the delta actually produced, i.e.
        # "base + B@A" explains this tensor better than "base alone" does.
        ok = bit_exact or (err <= tol and err < realized)
        if realized == 0.0:
            raise CheckFailure(f"{key} in {target} is unchanged from the base -- no GRPO delta")
        if not ok:
            raise CheckFailure(
                f"{key}: dense != base + {scaling}*B@A (max abs err {err:.4g}, tol {tol:.4g}, "
                f"observed delta {realized:.4g}) -- the rebuild recipe does NOT reproduce this file"
            )
        results.append({
            "module": key,
            "shape": list(w_base.shape),
            "dtype": str(w_base.dtype),
            "scaling": scaling,
            "bit_exact": bit_exact,
            "max_abs_err": err,
            "tolerance": tol,
            "observed_delta_absmax": realized,
        })
    return results


def verify_base_identity(target: Path, base: Model, *, count: int, seed: int,
                         allow_missing: tuple[str, ...]) -> list[dict]:
    """hf_base is a copy of merged_ep3, so bit-identity on a sample IS its recoverability."""
    raw = open_model(target, f"target {target}")
    # verl.model_merger saves through AutoModelForVision2Seq and nests the keys one level too
    # deep. Pick the mapping empirically -- renaming already-correct keys would corrupt the
    # comparison, so keep whichever aligns better with the container (same rule as the gate).
    hf_base = raw
    renamed = open_model(target, f"target {target}", normalize=normalize_merger_key)
    if len(renamed.keys() & base.keys()) > len(raw.keys() & base.keys()):
        hf_base = renamed

    shared = sorted(hf_base.keys() & base.keys())
    if not shared:
        raise CheckFailure(f"{target} shares no tensor keys with the base container")
    bad_missing = [k for k in sorted(base.keys() - hf_base.keys()) if not k.startswith(allow_missing)]
    bad_extra = [k for k in sorted(hf_base.keys() - base.keys()) if not k.startswith(allow_missing)]
    if bad_missing:
        raise CheckFailure(
            f"{len(bad_missing)} container keys absent from {target}, e.g. {bad_missing[:3]} "
            "-- this is not a faithful copy of the base"
        )
    if bad_extra:
        raise CheckFailure(f"{len(bad_extra)} unexpected keys in {target}, e.g. {bad_extra[:3]}")

    results = []
    for key in sample_keys(shared, count, seed):
        w_base = base.get(key)
        w_target = hf_base.get(key)
        if not torch.equal(w_base, w_target):
            err = float((w_target.float() - w_base.float()).abs().max())
            raise CheckFailure(
                f"{key} differs from the base container (max abs err {err:.4g}) -- "
                f"{target} is not reproducible from it"
            )
        results.append({
            "module": key,
            "shape": list(w_base.shape),
            "dtype": str(w_base.dtype),
            "bit_identical_to_base": True,
        })
    return results


# --------------------------------------------------------------------------- preservation


def preservable_entries(target: Path) -> list[Path]:
    """Everything under hf_base that is not a weight shard: lora_adapter/, merge_provenance.json,
    config, tokenizer. Kilobytes against 18 GB, and none of it is re-derivable for free."""
    return sorted(
        entry
        for entry in target.iterdir()
        if not (entry.is_file() and entry.name.endswith(".safetensors"))
    )


def _tree_digest(root: Path) -> dict[str, str]:
    digest = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and not path.is_symlink():
            digest[str(path.relative_to(root))] = sha256_file(path)
    return digest


def _entry_digest(entry: Path) -> dict[str, str]:
    if entry.is_dir():
        return {f"{entry.name}/{rel}": value for rel, value in _tree_digest(entry).items()}
    return {entry.name: sha256_file(entry)}


def check_preservation_available(target: Path, destination: Path) -> None:
    """Phase 1: confirm the relocation can happen. Nothing is written here."""
    entries = preservable_entries(target)
    if not entries:
        raise CheckFailure(f"{target} has nothing to preserve -- is it really an hf_base dir?")
    if destination.exists():
        if not destination.is_dir():
            raise CheckFailure(f"preservation destination {destination} exists and is not a dir")
        expected: dict[str, str] = {}
        for entry in entries:
            expected.update(_entry_digest(entry))
        actual = _tree_digest(destination)
        if actual != expected:
            raise CheckFailure(
                f"preservation destination {destination} already exists with different content; "
                "refusing to overwrite it (move it aside or pass --preserve-dir)"
            )


def relocate_preserved(target: Path, destination: Path) -> dict:
    """Copy the non-shard content out of hf_base and prove the copy byte-for-byte.

    This runs BEFORE the rmtree, and a mismatch aborts the delete: the adapter inside hf_base
    is the only step-unique artifact in the whole step directory.
    """
    entries = preservable_entries(target)
    expected: dict[str, str] = {}
    for entry in entries:
        expected.update(_entry_digest(entry))

    if not destination.exists():
        # This tool gets reached for precisely because the quota is full, and on this cluster a
        # refused write surfaces as a bare OSError. Turn it into a skip: a half-copied adapter
        # followed by a successful rmtree is the one outcome that must never happen.
        try:
            destination.mkdir(parents=True)
            for entry in entries:
                if entry.is_dir():
                    shutil.copytree(entry, destination / entry.name, symlinks=True)
                else:
                    shutil.copy2(entry, destination / entry.name)
        except OSError as exc:
            raise CheckFailure(
                f"could not write the preserved copy to {destination}: {exc} -- NOT deleting "
                f"{target}. Free space elsewhere first, or pass --preserve-dir."
            ) from exc

    actual = _tree_digest(destination)
    if actual != expected:
        only_src = sorted(set(expected) - set(actual))
        only_dst = sorted(set(actual) - set(expected))
        corrupt = sorted(k for k in set(expected) & set(actual) if expected[k] != actual[k])
        raise CheckFailure(
            f"preserved copy at {destination} does not match {target}: "
            f"missing={only_src[:3]} extra={only_dst[:3]} corrupt={corrupt[:3]} -- NOT deleting"
        )
    return {
        "destination": str(destination),
        "files": len(actual),
        "bytes": dir_size(destination),
        "sha256": actual,
    }


# --------------------------------------------------------------------------- per-target flow


class Target:
    def __init__(self, path: Path):
        self.path = path
        self.kind = path.name if path.name in KNOWN_KINDS else "?"
        self.failure: str | None = None
        self.checks: dict[str, str] = {}
        self.adapter: Path | None = None
        self.base: Path | None = None
        self.samples: list[dict] = []
        self.adapter_meta: dict = {}
        self.size_bytes = 0
        self.preserve_dst: Path | None = None
        self.preserved: dict | None = None
        # Set when the files went but the (now empty) directories could not be rmdir'd yet.
        self.residue: str | None = None
        self.deleted = False
        self.provenance: dict | None = None
        self.reclaimed = 0

    @property
    def ok(self) -> bool:
        return self.failure is None

    def fail(self, name: str, message: str) -> None:
        self.failure = f"{name}: {message}"
        self.checks[name] = "FAIL"


def inspect_target(target: Target, args, allowed_roots: list[Path], search_roots: list[Path]) -> None:
    """Phase 1 for one target. Reads only; never deletes, never writes."""
    try:
        check_path_guard(target.path, allowed_roots, args.results_dirname)
        target.checks["guard"] = "ok"
    except CheckFailure as exc:
        target.fail("guard", str(exc))
        return
    # Sized up front so the summary reports what a *failed* target is still costing.
    target.size_bytes = dir_size(target.path)

    _, report = merge_report_for(target.path)
    prov_dirs = [target.path] if target.kind == KIND_BASE else sibling_base_dirs(target.path)
    for prov_dir in prov_dirs:
        target.provenance = read_json(prov_dir / "merge_provenance.json")
        if target.provenance is not None:
            break

    try:
        target.adapter = resolve_adapter(target.path, args.adapter, report)
        target.adapter_meta = check_adapter_complete(target.adapter, args.expect_targets)
        # merge_grpo_ckpt.sh fingerprints the adapter at merge time precisely because a stale
        # hf_base (wrong step, wrong run tag) is internally self-consistent and passes every
        # arithmetic check while being the wrong model. Re-check it whenever it is recorded.
        recorded_sha = (target.provenance or {}).get("adapter_sha256")
        adapter_sha = sha256_file(target.adapter / "adapter_model.safetensors")
        if recorded_sha and recorded_sha != adapter_sha:
            raise CheckFailure(
                f"adapter sha256 {adapter_sha[:16]} does not match the {recorded_sha[:16]} "
                "recorded in merge_provenance.json -- these artifacts are mismatched"
            )
        target.adapter_meta["sha256"] = adapter_sha
        target.checks["adapter"] = (
            f"ok r={target.adapter_meta['r']} alpha={target.adapter_meta['alpha']} "
            f"targets={len(target.adapter_meta['pairs'])} sha={adapter_sha[:12]}"
            + ("" if recorded_sha else " (no recorded fingerprint to cross-check)")
        )
    except CheckFailure as exc:
        target.fail("adapter", str(exc))
        return

    # The adapter must outlive the deletion. Inside hf_base it does not, unless relocated.
    nested = is_within(target.adapter, target.path)
    try:
        if nested:
            target.preserve_dst = (
                Path(args.preserve_dir).resolve()
                if args.preserve_dir
                else target.path.with_name(target.path.name + PRESERVE_SUFFIX)
            )
            check_preservation_available(target.path, target.preserve_dst)
            target.checks["survives"] = f"relocate -> {target.preserve_dst.name}/"
        else:
            target.checks["survives"] = "adapter is outside the target"
    except CheckFailure as exc:
        target.fail("survives", str(exc))
        return

    try:
        target.base = resolve_base(target.path, args.base, report, search_roots)
        base_model = check_base_loadable(target.base)
        target.checks["base"] = f"ok {target.base}"
    except CheckFailure as exc:
        target.fail("base", str(exc))
        return

    try:
        if target.kind == KIND_DENSE:
            target.samples = verify_dense_arithmetic(
                target.path, base_model, target.adapter_meta,
                count=args.sample, seed=args.seed, ulps=args.ulps,
            )
            exact = sum(1 for s in target.samples if s["bit_exact"])
            target.checks["arith"] = f"ok {len(target.samples)} sampled ({exact} bit-exact)"
        else:
            target.samples = verify_base_identity(
                target.path, base_model, count=args.sample, seed=args.seed,
                allow_missing=tuple(args.allow_missing_prefix),
            )
            target.checks["arith"] = f"ok {len(target.samples)} sampled bit-identical to base"
    except CheckFailure as exc:
        target.fail("arith", str(exc))
        return


def rebuild_commands(target: Target, adapter_path: Path, expect_targets: int) -> list[str]:
    """The literal commands that put this artifact back. Written into the manifest, because a
    'recoverable' deletion nobody knows how to reverse is just a deletion."""
    base = str(target.base)
    if target.kind == KIND_DENSE:
        return [
            shlex.join([
                "python", "scripts/merge_grpo_adapter.py",
                "--base", base, "--adapter", str(adapter_path),
                "--out", str(target.path), "--expect_targets", str(expect_targets),
            ]),
            shlex.join([
                "python", "scripts/validate_grpo_merge.py",
                "--base", base, "--dense", str(target.path), "--adapter", str(adapter_path),
            ]),
        ]
    actor = (target.provenance or {}).get("actor_dir")
    commands = []
    if actor:
        commands.append(
            shlex.join([
                "python", "-m", "verl.model_merger", "merge", "--backend", "fsdp",
                "--local_dir", actor, "--target_dir", str(target.path),
            ])
            + ("" if Path(actor).is_dir() else "   # NOTE: actor_dir no longer exists")
        )
    commands.append(
        f"# hf_base weights were verified bit-identical to {base}; if the actor is gone, "
        f"use {base} directly as the backbone (it is what hf_dense is built from anyway)."
    )
    return commands


def write_manifest(target: Target, adapter_path: Path, args, deleted: bool,
                   inventory: list[dict]) -> Path:
    manifest = {
        "tool": "scripts/safe_delete_derived.py",
        "written_utc": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "deleted": deleted,
        "target": str(target.path),
        "kind": target.kind,
        "bytes_on_disk": target.size_bytes,
        "bytes_reclaimed": target.reclaimed,
        "files": inventory,
        "adapter": {
            "path": str(adapter_path),
            "relocated_from": str(target.adapter) if adapter_path != target.adapter else None,
            "r": target.adapter_meta.get("r"),
            "lora_alpha": target.adapter_meta.get("alpha"),
            "scaling": target.adapter_meta.get("scaling"),
            "n_targets": len(target.adapter_meta.get("pairs", {})),
            "sha256": sha256_file(adapter_path / "adapter_model.safetensors"),
        },
        "base": {
            "path": str(target.base),
            "config_sha256": sha256_file(target.base / "config.json"),
            "shards": [
                {"name": p.name, "bytes": p.stat().st_size}
                for p in sorted(target.base.glob("*.safetensors"))
            ],
        },
        "verification": {
            "sample_size": args.sample,
            "seed": args.seed,
            "ulps": args.ulps,
            "expect_targets": args.expect_targets,
            "checks": target.checks,
            "sampled_modules": target.samples,
        },
        "preserved": target.preserved,
        "merge_provenance": target.provenance,
        "rebuild": rebuild_commands(target, adapter_path, args.expect_targets),
    }
    path = target.path.with_name(target.path.name + MANIFEST_SUFFIX)
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return path


# --------------------------------------------------------------------------- main


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("targets", nargs="+", help="hf_dense/hf_base dirs or globs over them")
    ap.add_argument("--delete", action="store_true",
                    help="actually delete. Without this the tool only proves recoverability.")
    ap.add_argument("--base", type=Path, default=None,
                    help="merged_ep3 container (default: from grpo_merge_report.json)")
    ap.add_argument("--adapter", type=Path, default=None,
                    help="lora_adapter dir (default: <step>/hf_base/lora_adapter)")
    ap.add_argument("--sample", type=int, default=DEFAULT_SAMPLE,
                    help=f"target modules to spot-check per artifact (default {DEFAULT_SAMPLE})")
    ap.add_argument("--seed", type=int, default=0, help="sampling seed, recorded in the manifest")
    ap.add_argument("--ulps", type=float, default=1.0,
                    help="bf16 ulps of slack allowed when the recomputation is not bit-exact")
    ap.add_argument("--expect-targets", type=int, default=DEFAULT_EXPECT_TARGETS,
                    help="LoRA target count (Qwen3.5-9B: 32*mlp3 + 8*attn4 = 128)")
    ap.add_argument("--allowed-root", type=Path, action="append", default=None,
                    help="root a target must live under (repeatable; default: cwd)")
    ap.add_argument("--search-root", type=Path, action="append", default=None,
                    help="root to re-anchor a recorded-but-missing base under (default: cwd)")
    ap.add_argument("--results-dirname", default=DEFAULT_RESULTS_DIRNAME,
                    help="path component a target must contain (default: results)")
    ap.add_argument("--preserve-dir", default=None,
                    help="where to relocate hf_base's non-shard content "
                         f"(default: <target>{PRESERVE_SUFFIX})")
    ap.add_argument("--allow-missing-prefix", nargs="*", default=list(DEFAULT_ALLOW_MISSING),
                    help="key prefixes hf_base may legitimately lack (default: mtp.)")
    ap.add_argument("--json", type=Path, default=None, help="write the full run report here")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.sample < 1:
        raise SystemExit("FAIL: --sample must be >= 1; the arithmetic proof is the point")

    cwd = Path.cwd().resolve()
    allowed_roots = [p.resolve() for p in (args.allowed_root or [cwd])]
    search_roots = [p.resolve() for p in (args.search_root or [cwd])]
    targets = [Target(p) for p in expand_targets(args.targets)]
    if args.preserve_dir and len(targets) > 1:
        raise SystemExit(
            f"FAIL: --preserve-dir applies to a single target but {len(targets)} were given; "
            "several hf_base dirs would collide in one destination"
        )
    print(f"=== safe_delete_derived: {len(targets)} target(s), "
          f"mode={'DELETE' if args.delete else 'DRY RUN'} ===")

    for target in targets:
        inspect_target(target, args, allowed_roots, search_roots)
        status = "PASS" if target.ok else "FAIL"
        print(f"\n[{status}] {target.path}")
        for name, detail in target.checks.items():
            print(f"    {name:9s} {detail}")
        if target.failure:
            print(f"    -> SKIPPED, not deletable: {target.failure}")

    passed = [t for t in targets if t.ok]

    def final_adapter(target: Target) -> Path:
        """An adapter under a doomed hf_base moves, so every manifest in this batch must cite
        where the adapter actually ends up -- not where it happened to be read from."""
        for other in targets:
            if other.preserve_dst is None:
                continue
            # Deliberately NOT gated on other.ok. A target whose REMOVAL failed after its
            # relocation succeeded is marked failed, but its adapter has still moved and its
            # original copy is already unlinked -- so a sibling that cites the old path names
            # a file that no longer exists, and hashing it for the manifest raises
            # FileNotFoundError mid-batch. What matters is the verified relocation, not
            # whether the rmtree that followed it happened to finish.
            if args.delete and other.preserved is None:
                continue
            doomed = other.path
            if is_within(target.adapter, doomed):
                return other.preserve_dst / _norm(target.adapter).relative_to(_norm(doomed))
        return target.adapter

    if args.delete and passed:
        print("\n--- phase 2: preserve, then delete ---")
        for target in passed:
            if target.preserve_dst is not None:
                try:
                    target.preserved = relocate_preserved(target.path, target.preserve_dst)
                    print(f"    preserved {target.preserved['files']} files "
                          f"({human_bytes(target.preserved['bytes'])}) -> {target.preserve_dst}")
                except CheckFailure as exc:
                    target.fail("preserve", str(exc))
                    print(f"    ABORT {target.path}: {exc}")

        for target in [t for t in passed if t.ok]:
            inventory = file_inventory(target.path)
            preserved_bytes = target.preserved["bytes"] if target.preserved else 0
            target.reclaimed = target.size_bytes - preserved_bytes
            # One target failing must not abandon the rest of the batch half-done: the
            # preserved copies are already written and verified, so the remaining targets are
            # still safe to process. Manifest writing is inside the guard too -- it hashes the
            # adapter, and an unreadable adapter is a reason to skip this target, not to
            # strand every later one.
            try:
                manifest_path = write_manifest(target, final_adapter(target), args, True, inventory)
                release_mapped_tensors(target)
                rmtree_nfs(target.path)
            except OSError as exc:
                # The bytes are what we came for. If every FILE is gone and only empty
                # directories are left, the reclaim succeeded and the residue is cosmetic --
                # an NFS handle somewhere keeps rmdir returning EBUSY until this process
                # exits, after which a plain rmdir clears it. Calling that a FAIL both
                # under-reports the space freed and makes a nonzero exit meaningless to a
                # caller. Report it as deleted, and name the residue.
                remaining = [p for p in target.path.rglob("*") if p.is_file()]
                if remaining:
                    target.fail("delete", f"{type(exc).__name__}: {exc}")
                    print(f"    ABORT {target.path}: removal failed, "
                          f"{len(remaining)} file(s) still present: {exc}")
                    continue
                target.deleted = True
                target.residue = str(target.path)
                print(f"    deleted {target.path} (+{human_bytes(target.reclaimed)}), "
                      f"manifest {manifest_path.name}")
                print(f"    NOTE empty directories remain at {target.path} ({exc.strerror}); "
                      "rmdir them once this process has exited")
                continue
            target.deleted = True
            print(f"    deleted {target.path} (+{human_bytes(target.reclaimed)}), "
                  f"manifest {manifest_path.name}")

    failed = [t for t in targets if not t.ok]
    reclaimable = sum(
        t.size_bytes - (t.preserved["bytes"] if t.preserved else 0)
        for t in targets if t.ok
    )
    print("\n=== summary ===")
    print(f"{'RESULT':7s} {'KIND':9s} {'SIZE':>10s}  TARGET")
    for target in targets:
        verdict = "DELETED" if target.deleted else ("PASS" if target.ok else "FAIL")
        print(f"{verdict:7s} {target.kind:9s} {human_bytes(target.size_bytes):>10s}  {target.path}")
    verb = "reclaimed" if args.delete else "reclaimable (dry run, nothing deleted)"
    print(f"{human_bytes(reclaimable)} {verb}; {len(failed)} target(s) failed")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({
            "delete": args.delete,
            "bytes_reclaimable": reclaimable,
            "targets": [
                {
                    "path": str(t.path),
                    "kind": t.kind,
                    "ok": t.ok,
                    "deleted": t.deleted,
                    "bytes": t.size_bytes,
                    "failure": t.failure,
                    "checks": t.checks,
                    "adapter": str(final_adapter(t)) if t.adapter else None,
                    "base": str(t.base) if t.base else None,
                    "sampled_modules": t.samples,
                }
                for t in targets
            ],
        }, indent=2) + "\n")

    if failed:
        print(f"\nFAILED: {len(failed)} target(s) were NOT proven recoverable and were not deleted:")
        for target in failed:
            print(f"  - {target.path}: {target.failure}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
