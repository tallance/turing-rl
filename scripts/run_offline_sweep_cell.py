"""Offline batched vLLM sweep cell for Qwen3-8B thinking-off. Reuses the reward-dump
schema helper so the GUI viewer renders these identically to server cells."""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO_ROOT))
import pandas as pd
from shared.judge_prompts import TURING_PROMPT, TURING_RESPONSE_SCHEMA
from shared.judge_utils import build_source_copy_warning, format_source_copy_watchlist
from training.grpo.reward import _build_reward_dump_row   # DRY: shared viewer contract

MODEL_ID = "Qwen/Qwen3-8B"
# frozen sampling = Qwen generation_config defaults (Task 1); matches served cells
SAMPLING_OFF = dict(temperature=0.6, top_p=0.95, top_k=20, min_p=0.0, max_tokens=8192)

def build_prompt(row: dict, generated_is_b: bool) -> str:
    resp_a, resp_b = (row["human"], row["generated"]) if generated_is_b else (row["generated"], row["human"])
    wa = build_source_copy_warning(resp_a, thread_context=row["context"])
    wb = build_source_copy_warning(resp_b, thread_context=row["context"])
    return TURING_PROMPT.format(persona=row.get("persona", ""), user_history=row["user_history"],
        context=row["context"], response_a=resp_a, response_b=resp_b,
        source_copy_watchlist=format_source_copy_watchlist([wa, wb], item_label="Response",
            labels=["Response A", "Response B"]))

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)   # .../raw/sweep/qwen3-8b/off_offline
    ap.add_argument("--tensor_parallel_size", type=int, default=8)
    args = ap.parse_args()
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import GuidedDecodingParams
    pairs = pd.read_parquet(args.pairs).to_dict(orient="records")
    (args.out_dir / "reward").mkdir(parents=True, exist_ok=True)
    llm = LLM(model=MODEL_ID, tensor_parallel_size=args.tensor_parallel_size,
              gpu_memory_utilization=0.85, max_model_len=32768, dtype="bfloat16")
    tok = llm.get_tokenizer()
    guided = GuidedDecodingParams(json=TURING_RESPONSE_SCHEMA)
    sp = SamplingParams(guided_decoding=guided, **SAMPLING_OFF)
    prompts, meta = [], []
    for row in pairs:
        for gib in (True, False):
            chat = tok.apply_chat_template([{"role": "user", "content": build_prompt(row, gib)}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False)
            prompts.append(chat); meta.append((row, gib))
    t0 = time.time(); outs = llm.generate(prompts, sp); dt = time.time() - t0
    path = args.out_dir / "reward" / f"reward-offline-{os.getpid()}.jsonl"
    with path.open("w") as fh:
        for i, ((row, gib), o) in enumerate(zip(meta, outs)):
            first = o.outputs[0]
            fh.write(json.dumps(_build_reward_dump_row(
                generated_is_b=gib, human_side=("A" if gib else "B"),
                randomized_order=("gt_first" if not gib else "gen_first"),
                rating_gt_first=None, rating_gen_first=None,
                response=row["generated"], ground_truth=row["human"], context=row["context"],
                user_history=row["user_history"], judge_response={}, judge_prompt="",
                judge_raw_content=first.text, judge_reasoning="", judge_latency_ms=None,
                judge_finish_reason=first.finish_reason, judge_model=MODEL_ID, judge_usage={},
                final_reward=0.0, turing_judge_score_raw=0.0, turing_judge_score_clipped=0.0,
                source_copy_penalty=0.0, assistant_like_penalty=0.0,
                wrong_target_or_role_penalty=0.0, unsupported_adversarial_reframing_penalty=0.0,
                call_id=i, user_id=row["user_id"], post_id=row["post_id"],
                target_idx=row["target_idx"], persona="", ts=time.time(), worker_pid=os.getpid()),
                default=str) + "\n")
    (args.out_dir / "run_metadata.json").write_text(json.dumps({"model": MODEL_ID,
        "thinking_mode": "off", "backend": "offline", "tensor_parallel_size": args.tensor_parallel_size,
        "sampling": SAMPLING_OFF, "n_pairs": len(pairs), "n_calls": len(prompts),
        "wall_seconds": dt, "req_per_s": len(prompts) / dt if dt else 0.0,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID")}, indent=2))
    print(f"[offline] {len(prompts)} calls in {dt:.1f}s -> {path}", flush=True)

if __name__ == "__main__":
    main()
