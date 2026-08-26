"""Find where Qwen3.5-9B forward first goes non-finite (nan-hunt via forward hooks).

The Stage-1 probe showed loss=nan on a 1-step bf16/sdpa forward even with fla installed.
This isolates the first module whose OUTPUT is non-finite, and compares dtypes/attn impls,
so we fix the real cause instead of guessing.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

MODEL_ID = "Qwen/Qwen3.5-9B"


def tensor_nonfinite(x):
    if isinstance(x, torch.Tensor) and x.is_floating_point():
        return not torch.isfinite(x).all().item()
    return False


def any_nonfinite(obj):
    if isinstance(obj, torch.Tensor):
        return tensor_nonfinite(obj)
    if isinstance(obj, (list, tuple)):
        return any(any_nonfinite(o) for o in obj)
    return False


def run(dtype, attn, seqlen, use_random):
    print("\n" + "=" * 78)
    print(f"RUN dtype={dtype} attn={attn} seqlen={seqlen} random_input={use_random}")
    print("=" * 78, flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=dtype, trust_remote_code=True,
        attn_implementation=attn, low_cpu_mem_usage=True, device_map="cuda:0",
    )
    model.eval()

    if use_random:
        vocab = model.config.get_text_config().vocab_size
        ids = torch.randint(0, vocab, (1, seqlen), device="cuda:0")
    else:
        with open("data/sft/prism_full_s42_sft_cot.jsonl") as f:
            msgs = json.loads(f.readline())["messages"]
        text = tok.apply_chat_template(msgs, tokenize=False, enable_thinking=False)
        ids = tok(text, return_tensors="pt").input_ids[:, :seqlen].to("cuda:0")

    # First-nonfinite hook.
    first = {}
    def mk(name):
        def hook(mod, inp, out):
            if not first and any_nonfinite(out) and not any_nonfinite(inp):
                first["name"] = name
                first["type"] = type(mod).__name__
        return hook
    handles = [m.register_forward_hook(mk(n)) for n, m in model.named_modules()]

    # Weight sanity.
    bad_w = [n for n, p in model.named_parameters() if not torch.isfinite(p).all()]
    print("non-finite weights:", bad_w[:5], "..." if len(bad_w) > 5 else "", f"({len(bad_w)})")

    with torch.no_grad():
        out = model(input_ids=ids)
    for h in handles:
        h.remove()
    logits = out.logits
    print("logits finite:", torch.isfinite(logits).all().item(),
          "| logits dtype:", logits.dtype)
    if first:
        print(f">>> FIRST non-finite module: {first['name']}  ({first['type']})")
    else:
        print(">>> no module flagged (input already bad, or finite throughout)")
    del model
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqlen", type=int, default=256)
    args = ap.parse_args()
    import transformers
    print("transformers", transformers.__version__, "torch", torch.__version__)
    # 1) the failing config: bf16 + sdpa, real data
    run(torch.bfloat16, "sdpa", args.seqlen, use_random=False)
    # 2) bf16 + eager
    run(torch.bfloat16, "eager", args.seqlen, use_random=False)
    # 3) fp32 + sdpa (does precision fix it? 9B fp32 ~36GB weights — short seq, no_grad)
    try:
        run(torch.float32, "sdpa", min(args.seqlen, 64), use_random=False)
    except Exception as e:  # noqa: BLE001
        print("fp32 run failed:", type(e).__name__, str(e)[:160])
    # 4) bf16 + sdpa, random tokens (rule out data)
    run(torch.bfloat16, "sdpa", args.seqlen, use_random=True)


if __name__ == "__main__":
    main()
