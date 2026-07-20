"""Stage-1 probe for Qwen3.5-9B SFT (see plan: SFT the Qwen3.5-9B generator).

Loads Qwen/Qwen3.5-9B in the transformers-5.x SFT env and prints the facts that decide
the lora_sft.py changes, so we never guess at class/module names for a multi-hour run:

  1. AutoConfig  -> architectures, model_type, nested text config summary.
  2. Working loader class (AutoModelForCausalLM, else ImageTextToText/ConditionalGeneration).
  3. Decoder layer class name (for FSDP transformer_layer_cls_to_wrap).
  4. Linear module names, split text-tower vs vision-tower, + leaf suffixes (LoRA targets).
  5. One PRISM SFT row through build_chat_template_sft_features (chat template + mask path).
  6. A 1-step forward/backward under the chosen attn impl (does DeltaNet run w/o extra kernels?).

Run on the cluster with 1 GPU in the new env, e.g.:
  ssh ... "cd repo && .../turing-rl-sft-qwen35/bin/python training/sft/probe_qwen35.py \
           --data_path data/sft/prism_full_s42_sft_cot.jsonl"
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from training.sft.lora_sft import MODEL_MAP, build_chat_template_sft_features  # noqa: E402

SECTION = "\n" + "=" * 78 + "\n"


def _hr(title: str) -> None:
    print(f"{SECTION}## {title}{SECTION}", flush=True)


def probe_config(model_id: str):
    from transformers import AutoConfig

    _hr("1. AutoConfig")
    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    print("model_type       :", getattr(cfg, "model_type", None))
    print("architectures    :", getattr(cfg, "architectures", None))
    text_cfg = getattr(cfg, "text_config", None)
    if text_cfg is not None:
        print("has text_config  : True")
        for k in ("model_type", "num_hidden_layers", "hidden_size", "vocab_size",
                  "layer_types", "full_attention_interval"):
            if hasattr(text_cfg, k):
                v = getattr(text_cfg, k)
                if k == "layer_types" and isinstance(v, (list, tuple)) and len(v) > 12:
                    v = f"{list(v[:6])} ... ({len(v)} total)"
                print(f"  text_config.{k:24}: {v}")
    else:
        print("has text_config  : False (top-level is the text config)")
    return cfg


def probe_loader(model_id: str, attn: str):
    """Return (model, loader_name). Tries CausalLM first, then multimodal wrappers."""
    import torch

    _hr("2. Loader class")
    candidates = ["AutoModelForCausalLM", "AutoModelForImageTextToText",
                  "AutoModelForConditionalGeneration"]
    import transformers
    last_err = None
    for name in candidates:
        klass = getattr(transformers, name, None)
        if klass is None:
            print(f"  {name}: not present in transformers")
            continue
        try:
            model = klass.from_pretrained(
                model_id, dtype=torch.bfloat16, trust_remote_code=True,
                attn_implementation=attn, low_cpu_mem_usage=True, device_map="cuda:0",
            )
            print(f"  WORKS -> {name}")
            return model, name
        except Exception as e:  # noqa: BLE001
            last_err = e
            print(f"  {name}: FAILED -> {type(e).__name__}: {str(e)[:200]}")
    raise RuntimeError(f"No loader worked; last error: {last_err}")


def probe_modules(model):
    import torch

    _hr("3. Module class names (decoder layer for FSDP)")
    class_names = sorted({type(m).__name__ for m in model.modules()})
    decoderish = [c for c in class_names if "DecoderLayer" in c or "Block" in c]
    print("candidate decoder/block classes:", decoderish)
    print("all distinct module classes ({}):".format(len(class_names)))
    for c in class_names:
        print("   ", c)

    _hr("4. Linear modules (LoRA targets) — text vs vision")
    lin_names = [n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)]
    # Heuristic split by common multimodal prefixes.
    vision_markers = ("visual", "vision", "image", "patch", "vit", "connector", "merger",
                      "mm_projector")
    text_names, vis_names = [], []
    for n in lin_names:
        (vis_names if any(mk in n.lower() for mk in vision_markers) else text_names).append(n)

    def leaf_suffixes(names):
        return sorted({n.rsplit(".", 1)[-1] for n in names})

    print(f"total Linear modules: {len(lin_names)}  (text={len(text_names)}, vision={len(vis_names)})")
    print("\nTEXT leaf suffixes  :", leaf_suffixes(text_names))
    print("VISION leaf suffixes:", leaf_suffixes(vis_names))
    overlap = set(leaf_suffixes(text_names)) & set(leaf_suffixes(vis_names))
    print("OVERLAP (text&vision):", sorted(overlap))
    print("\n-- sample TEXT linear full-names (first 40) --")
    for n in text_names[:40]:
        print("   ", n)
    if vis_names:
        print("\n-- sample VISION linear full-names (first 15) --")
        for n in vis_names[:15]:
            print("   ", n)
    return leaf_suffixes(text_names)


def probe_chat_template(model_id: str, data_path: str):
    from transformers import AutoTokenizer

    _hr("5. Chat template + completion mask (one PRISM row)")
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    with open(data_path) as f:
        row = json.loads(f.readline())
    feats = build_chat_template_sft_features(tok, row["messages"])
    n = len(feats["input_ids"])
    n_target = sum(feats["completion_mask"])
    print(f"input_ids len={n}  target(masked-in) tokens={n_target}")
    print("first 30 input_ids:", feats["input_ids"][:30])
    return tok, feats


def probe_fwd_bwd(model, tok, feats, attn: str):
    import torch

    _hr(f"6. 1-step forward/backward (attn={attn})")
    ids = feats["input_ids"][:512]
    mask = feats["completion_mask"][:512]
    input_ids = torch.tensor([ids], device="cuda:0")
    labels = torch.tensor([[t if m else -100 for t, m in zip(ids, mask)]], device="cuda:0")
    model.train()
    model.gradient_checkpointing_enable()
    out = model(input_ids=input_ids, labels=labels)
    print("loss:", float(out.loss))
    out.loss.backward()
    grad_params = sum(1 for p in model.parameters() if p.grad is not None)
    print(f"backward OK; params with grad: {grad_params}")
    print("max mem allocated (GB):", round(torch.cuda.max_memory_allocated() / 1e9, 2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen35-9b", choices=list(MODEL_MAP.keys()))
    ap.add_argument("--data_path", default="data/sft/prism_full_s42_sft_cot.jsonl")
    ap.add_argument("--attn", default="sdpa", help="sdpa|flash_attention_2|eager")
    args = ap.parse_args()

    model_id = MODEL_MAP[args.model]
    import transformers
    print("transformers:", transformers.__version__)
    print("model_id    :", model_id)

    probe_config(model_id)
    try:
        model, loader = probe_loader(model_id, args.attn)
    except Exception:
        traceback.print_exc()
        raise
    probe_modules(model)
    tok, feats = probe_chat_template(model_id, args.data_path)
    try:
        probe_fwd_bwd(model, tok, feats, args.attn)
    except Exception as e:  # noqa: BLE001
        _hr("6. forward/backward FAILED")
        print(f"{type(e).__name__}: {e}")
        print("If this is a missing-kernel error, install flash-attn / fla / causal-conv1d "
              "in the SFT env and re-run, or try --attn eager.")
        traceback.print_exc()

    _hr("PROBE SUMMARY")
    print(f"loader class : {loader}")
    print("Use the decoder class from section 3 for --fsdp_transformer_layer_cls,")
    print("and the TEXT leaf suffixes from section 4 as LoRA target_modules (excluding vision).")


if __name__ == "__main__":
    main()
