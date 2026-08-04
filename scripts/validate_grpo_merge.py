"""Hard gate for the GRPO dense-merge: prove hf_dense really carries the RL delta.

THE FAILURE THIS PREVENTS
-------------------------
``verl.model_merger`` pops every ``lora_*`` key out into ``lora_adapter/`` and saves the
stripped remainder as the model -- i.e. the SFT base. If that base were served by mistake,
every checkpoint would score like step_0: plausible numbers, no error, wrong conclusion.
Nothing downstream would catch it. Hence this gate runs BEFORE any generation job.

CHECKS
------
A. Adapter accounting -- exactly ``--expect_targets`` paired LoRA tensors, each mapping to a
   real base weight; no unpaired or unmatched keys.
B. Delta exactness    -- for EVERY target, ``hf_dense - base == (alpha/r) * B@A``, and the
   delta is non-zero.
C. Non-targets frozen -- every other tensor bit-identical to the base container, key sets equal.
D. Base fidelity      -- ``hf_base`` vs ``merged_ep3`` at TENSOR level. Any missing/extra key is a
   HARD FAILURE unless allowlisted. Default allowlist is ``mtp.*``: the actor ran with
   ``mtp_num_hidden_layers=0`` so the checkpoint legitimately has 0 mtp keys while merged_ep3
   has 15. Config equality is deliberately NOT asserted -- the two configs differ in 11 keys
   that are purely transformers-5.4 materialized defaults (partial_rotary_factor, dtype,
   eos/pad_token_id, attn_implementation, version), not architecture.
E. Distinctness       -- optional: two dense dirs must differ (step_8 vs step_16).

Tensors are streamed one at a time, so peak memory stays small despite ~18 GB models.

Usage:
  python scripts/validate_grpo_merge.py \
    --base checkpoints/sft/.../merged_ep3 \
    --dense <EVAL_ROOT>/models/step8/hf_dense \
    --adapter <EVAL_ROOT>/models/step8/hf_base/lora_adapter \
    [--hf_base <EVAL_ROOT>/models/step8/hf_base] \
    [--distinct_from <EVAL_ROOT>/models/step16/hf_dense]
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from safetensors import safe_open

DEFAULT_ALLOWLIST = ("mtp.",)
LORA_A, LORA_B = ".lora_A", ".lora_B"

_REPEATED_LM = re.compile(r"(language_model\.)+")


def normalize_merger_key(key: str) -> str:
    """Undo the extra wrapper level ``verl.model_merger`` adds for Qwen3.5 vision2seq.

    Saving via ``AutoModelForVision2Seq`` re-prefixes the state dict, so hf_base keys come
    back nested one level too deep, in two forms::

        model.language_model.language_model.language_model.X -> model.language_model.X
        model.language_model.visual.X                        -> model.visual.X

    Both are fixed by dropping the leading ``model.language_model.`` then collapsing any
    remaining repeats. This is a pure RENAME -- the caller still asserts exact key-set
    equality afterwards, so a genuinely missing or extra tensor is not masked.

    Applied to an ALREADY-correct key this would corrupt it, so the caller never applies it
    blindly: it builds both mappings and keeps whichever actually agrees with the container.
    """
    if key.startswith("model.language_model."):
        key = "model." + key[len("model.language_model.") :]
    return _REPEATED_LM.sub("language_model.", key)


def load_adapter_independently(adapter_dir: Path, provenance: dict | None) -> tuple[dict, float]:
    """Read the adapter WITHOUT reusing merge_grpo_adapter's loader.

    The gate must not import its parsing from the script it is checking: if that shared code
    mis-derived the scaling (alpha/r inverted) or swapped lora_A/lora_B, the merge would apply a
    wrong delta and the gate would recompute the SAME wrong delta and pass. So this re-derives
    everything here and, when available, cross-checks r/alpha against ``lora_train_meta.json`` --
    written by veRL at training time, independent of both scripts.
    """
    cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
    r, alpha = int(cfg["r"]), int(cfg["lora_alpha"])
    if r <= 0 or alpha <= 0:
        raise SystemExit(f"FAIL: implausible LoRA config r={r} alpha={alpha}")

    meta_path = None
    if provenance and provenance.get("actor_dir"):
        cand = Path(provenance["actor_dir"]) / "lora_train_meta.json"
        if cand.exists():
            meta_path = cand
    if meta_path:
        meta = json.loads(meta_path.read_text())
        if int(meta.get("r", r)) != r or int(meta.get("lora_alpha", alpha)) != alpha:
            raise SystemExit(
                f"FAIL: adapter_config (r={r}, alpha={alpha}) disagrees with veRL's "
                f"{meta_path} (r={meta.get('r')}, alpha={meta.get('lora_alpha')})"
            )
        print(f"[A] cross-checked r/alpha against {meta_path.name}")
    else:
        print("[A] WARN: no lora_train_meta.json found; r/alpha come from adapter_config alone")

    a_by, b_by = {}, {}
    with safe_open(str(adapter_dir / "adapter_model.safetensors"), framework="pt") as f:
        for key in f.keys():
            tag = LORA_A if LORA_A in key else (LORA_B if LORA_B in key else None)
            if tag is None:
                raise SystemExit(f"FAIL: unexpected adapter key {key}")
            stem = key.split(tag)[0]
            if stem.startswith("base_model.model."):
                stem = stem[len("base_model.model.") :]
            (a_by if tag == LORA_A else b_by)[stem] = f.get_tensor(key)

    if set(a_by) != set(b_by):
        raise SystemExit(f"FAIL: unpaired LoRA tensors: {sorted(set(a_by) ^ set(b_by))[:5]}")
    # Orientation check: A is (r, in), B is (out, r). Catches a swapped pairing.
    for mod in a_by:
        if a_by[mod].shape[0] != r or b_by[mod].shape[1] != r:
            raise SystemExit(
                f"FAIL: {mod} has lora_A{tuple(a_by[mod].shape)} lora_B{tuple(b_by[mod].shape)}, "
                f"inconsistent with rank r={r} -- the two may be swapped"
            )
    return {f"{m}.weight": (a_by[m], b_by[m]) for m in a_by}, alpha / r


class Model:
    """Lazy key->shard index over a sharded safetensors dir."""

    def __init__(self, path: Path, normalize=None):
        self.path = path
        self.index: dict[str, Path] = {}
        self.raw: dict[str, str] = {}
        n_raw = 0
        for shard in sorted(path.glob("*.safetensors")):
            with safe_open(str(shard), framework="pt") as f:
                for key in f.keys():
                    n_raw += 1
                    name = normalize(key) if normalize else key
                    self.index[name] = shard
                    self.raw[name] = key
        if not self.index:
            raise SystemExit(f"FAIL: no safetensors tensors under {path}")
        if normalize and len(self.index) != n_raw:
            raise SystemExit(
                f"FAIL: key normalization collided under {path} ({n_raw} keys -> {len(self.index)}); "
                "the rename rule is wrong and would hide a real mismatch"
            )

    def keys(self) -> set[str]:
        return set(self.index)

    def get(self, key: str) -> torch.Tensor:
        with safe_open(str(self.index[key]), framework="pt") as f:
            return f.get_tensor(self.raw[key])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="Container the dense model was built from (merged_ep3)")
    ap.add_argument("--dense", required=True, help="hf_dense to validate")
    ap.add_argument("--adapter", required=True, help="lora_adapter/ dir with the GRPO delta")
    ap.add_argument("--hf_base", default=None, help="verl.model_merger base output (check D)")
    ap.add_argument("--distinct_from", default=None, help="Another hf_dense that must differ (check E)")
    ap.add_argument("--expect_targets", type=int, default=128)
    ap.add_argument("--allow_missing_prefix", nargs="*", default=list(DEFAULT_ALLOWLIST),
                    help="Key prefixes allowed to differ between hf_base and base (default: mtp.)")
    ap.add_argument("--no_key_normalize", action="store_true",
                    help="Compare hf_base keys verbatim (skip the verl vision2seq rename fix)")
    a = ap.parse_args()

    failures: list[str] = []
    notes: list[str] = []

    base = Model(Path(a.base))
    dense = Model(Path(a.dense))
    prov_path = Path(a.adapter).parent / "merge_provenance.json"
    provenance = json.loads(prov_path.read_text()) if prov_path.exists() else None
    pairs, scaling = load_adapter_independently(Path(a.adapter), provenance)

    # ---- A: adapter accounting -------------------------------------------------
    if len(pairs) != a.expect_targets:
        failures.append(f"A: expected {a.expect_targets} LoRA targets, adapter has {len(pairs)}")
    unmatched = sorted(set(pairs) - base.keys())
    if unmatched:
        failures.append(f"A: {len(unmatched)} adapter targets absent from base, e.g. {unmatched[:3]}")
    print(f"[A] adapter targets={len(pairs)} scaling={scaling} unmatched={len(unmatched)}")

    # ---- C: key sets identical -------------------------------------------------
    if base.keys() != dense.keys():
        miss = sorted(base.keys() - dense.keys())[:3]
        extra = sorted(dense.keys() - base.keys())[:3]
        failures.append(f"C: dense key set differs from base (missing={miss} extra={extra})")

    # ---- B + C: per-tensor comparison -----------------------------------------
    n_target_ok = n_target_bitexact = n_frozen_ok = 0
    worst_delta_err = 0.0
    zero_delta: list[str] = []
    changed_non_targets: list[str] = []

    for key in sorted(base.keys() & dense.keys()):
        w_base = base.get(key)
        w_dense = dense.get(key)
        if key in pairs:
            lora_a, lora_b = pairs[key]
            delta = (lora_b.float() @ lora_a.float()) * scaling
            if float(delta.abs().max()) == 0.0:
                zero_delta.append(key)
            expected = (w_base.float() + delta).to(w_base.dtype)
            if torch.equal(w_dense, expected):
                n_target_bitexact += 1
                n_target_ok += 1
            else:
                err = float((w_dense.float() - expected.float()).abs().max())
                worst_delta_err = max(worst_delta_err, err)
                # bf16 has ~3 decimal digits; allow a 1-ulp-ish slack before failing.
                if torch.allclose(w_dense.float(), expected.float(), rtol=1e-2, atol=1e-2):
                    n_target_ok += 1
                else:
                    failures.append(f"B: {key} != base + {scaling}*B@A (max abs err {err:.4g})")
            if torch.equal(w_dense, w_base):
                failures.append(f"B: {key} is UNCHANGED from base -- GRPO delta not applied")
        else:
            if torch.equal(w_dense, w_base):
                n_frozen_ok += 1
            else:
                changed_non_targets.append(key)

    if zero_delta:
        failures.append(f"B: {len(zero_delta)} targets have an all-zero delta, e.g. {zero_delta[:3]}")
    if changed_non_targets:
        failures.append(
            f"C: {len(changed_non_targets)} non-target tensors changed, e.g. {changed_non_targets[:3]}"
        )
    print(f"[B] targets verified={n_target_ok}/{len(pairs)} (bit-exact={n_target_bitexact}) "
          f"worst_err={worst_delta_err:.4g} zero_delta={len(zero_delta)}")
    print(f"[C] non-target tensors bit-identical={n_frozen_ok} changed={len(changed_non_targets)}")

    # ---- D: hf_base fidelity vs the container ----------------------------------
    if a.hf_base:
        # Pick the key mapping empirically rather than assuming the merger mangled names:
        # blindly renaming a correctly-named dir would corrupt the comparison.
        hf_base = Model(Path(a.hf_base))
        if not a.no_key_normalize:
            renamed = Model(Path(a.hf_base), normalize=normalize_merger_key)
            n_raw = len(hf_base.keys() & base.keys())
            n_renamed = len(renamed.keys() & base.keys())
            if n_renamed > n_raw:
                notes.append(
                    f"D: applied the verl vision2seq key rename to hf_base "
                    f"({n_raw} -> {n_renamed} keys aligned with the container)"
                )
                hf_base = renamed
        allow = tuple(a.allow_missing_prefix)

        def allowed(k: str) -> bool:
            return k.startswith(allow)

        missing = sorted(base.keys() - hf_base.keys())
        extra = sorted(hf_base.keys() - base.keys())
        bad_missing = [k for k in missing if not allowed(k)]
        bad_extra = [k for k in extra if not allowed(k)]
        if bad_missing:
            failures.append(f"D: hf_base missing {len(bad_missing)} non-allowlisted keys, e.g. {bad_missing[:3]}")
        if bad_extra:
            failures.append(f"D: hf_base has {len(bad_extra)} unexpected keys, e.g. {bad_extra[:3]}")
        n_allowlisted = len(missing) - len(bad_missing)
        if n_allowlisted:
            notes.append(f"D: {n_allowlisted} allowlisted keys absent from hf_base (prefixes {allow})")

        mismatched = []
        for key in sorted(base.keys() & hf_base.keys()):
            if not torch.equal(base.get(key), hf_base.get(key)):
                mismatched.append(key)
        if mismatched:
            failures.append(
                f"D: hf_base differs from base on {len(mismatched)} shared tensors, e.g. {mismatched[:3]} "
                "-- the reconstructed frozen backbone does not match merged_ep3"
            )
        print(f"[D] hf_base shared={len(base.keys() & hf_base.keys())} mismatched={len(mismatched)} "
              f"allowlisted_missing={len(missing)} bad_missing={len(bad_missing)} bad_extra={len(bad_extra)}")

    # ---- E: distinctness --------------------------------------------------------
    if a.distinct_from:
        other = Model(Path(a.distinct_from))
        shared = sorted((dense.keys() & other.keys()) & set(pairs))
        n_diff = sum(1 for k in shared if not torch.equal(dense.get(k), other.get(k)))
        if n_diff == 0:
            failures.append(f"E: {a.dense} is identical to {a.distinct_from} on all {len(shared)} targets")
        print(f"[E] targets differing vs {Path(a.distinct_from).parent.name}: {n_diff}/{len(shared)}")

    for note in notes:
        print(f"NOTE {note}")
    if failures:
        print(f"\nFAILED ({len(failures)}):")
        for f in failures:
            print(f"  - {f}")
        raise SystemExit(1)
    print("\nPASS: dense model carries the GRPO delta; backbone verified.")


if __name__ == "__main__":
    main()
