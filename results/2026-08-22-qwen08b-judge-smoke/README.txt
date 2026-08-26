Qwen3.5-0.8B as a GRPO training judge -- acceptance smoke, REJECTED
===================================================================
Provenance only. No interpretation of the numbers is recorded here.

Purpose: decide whether to spend ~25 h of 16-GPU time on a third judge arm using
Qwen3.5-0.8B, alongside the completed Qwen3.5-9B and gemma-4-12B arms over the same fixed
384-row slice. The gate answered no in 10 minutes.

Job 18901, 2026-08-22, 1 node / 8 GPUs, CANCELLED by the submitter after the verdict was
in (gate 4 is throughput tuning for a judge that will not be used, and it was holding 8
GPUs). Gates 1-3 completed; gate 4 is partial.


1. RESULT
---------
Gate 1 identity  PASS   /v1/models advertises Qwen/Qwen3.5-0.8B
Gate 2 training  FAIL   usable 0.175, hard fail 0.825      <- the go/no-go
Gate 3 schema    FAIL   usable 0.200 / 0.193 over two passes

                          usable   hard fail   finish_reason=length
  training, json_object    0.175     0.825            0.825
  eval,     json_schema    0.200     0.800            0.800
  eval,     json_schema    0.193     0.807            0.807
  Qwen3.5-9B, same prompts 1.000     0.000            0.000

The 9B baseline is not a fresh measurement: job 15371 already recorded a verdict for every
one of those 352 val prompts, so it is read out of that run's reward dump.

The two schema-mode passes agree to within 0.007, so the failure is not sampling noise.


2. WHY IT FAILED
----------------
Not malformed JSON. 165 of 200 training-mode responses ended with finish_reason="length":
the model does not stop thinking and is truncated mid-verdict at the 8192-token cap.
finish_reasons = {"stop": 35, "length": 165}.

This is why enabling the ordered 37-field schema -- the mitigation that exists for exactly
this class of problem, and the one that rescued gemma-4-31B from a faulty rating-only
schema -- changed nothing here. A schema constrains the shape of a response, not the
length of the reasoning preceding it.

Throughput fails in the opposite direction from intuition: 0.636 req/s at p50 199 s,
because nearly every request runs to the cap. A 0.8B judge is SLOWER than the 9B, not
cheaper, so the "small judge means a faster run" assumption does not hold.

Recorded in docs/judge-response-schema.md so this is not retried from first principles.


2b. FOLLOW-UP: THINKING DISABLED (job 18913, COMPLETED 0:0, 11:28)
------------------------------------------------------------------
Re-ran the same battery with PERSONA_JUDGE_ENABLE_THINKING=0 to test the one remaining
lever. Full transcript in nothink_18913_full.log.

  thinking  mode           usable          hard fail       throughput
  on        json_object    0.175           0.825           0.64 req/s, p50 199 s
  off       json_object    0.840           0.160           2.42 req/s, p50 32 s
  off       json_schema    0.886 / 0.855   0.114 / 0.145

  gate 4 concurrency sweep, thinking off:
    conc=8   usable 0.891  hard fail 0.109  1.105 req/s  p50 22.14 s
    conc=16  usable 0.875  hard fail 0.125  1.646 req/s  p50  9.27 s
    conc=32  usable 0.875  hard fail 0.125

Gate 2 detail with thinking off: clean 168, recovered 0, failed 0, error 32,
finish_reasons {"stop": 168, "length": 32}. Every remaining failure is still the 8192 cap.

The band is stable across concurrency, so ~11-16% is a floor rather than a load artefact,
against the Qwen3.5-9B's 0.000 on the same prompts. Throughput improves 3.8x.


3. NOTE ON GATE 3
-----------------
Gate 3's reference dump path names gemma's sweep output regardless of which model is under
test, so it runs for any judge rather than skipping. For Qwen3.5-0.8B it is therefore NOT
the equivalence test its name suggests -- the reference ratings came from a different
model, and the summary labels that comparison "CROSS-JUDGE (not a gate)". What it does
measure here, and the reason it is kept, is the served model's parse rate under the
ordered schema, which is precisely the mitigation gate 2's failure would send you to.


4. FILES HERE
-------------
gate1_identity.txt            /v1/models check and engine readiness
gate2_training_summary.json   the go/no-go: parse outcomes under {"type":"json_object"}
gate2_qwen_baseline.json      Qwen3.5-9B on the same prompts, read from job 15371's dump
gate2_training.log            full gate 2 transcript including the baseline block
gate3_eval_summary.json       two passes under the ordered 37-field schema
gate4_concurrency.log         partial -- job cancelled during the sweep
nothink_18913_full.log        complete driver log of the thinking-off re-run (all 4 gates)


5. REPRODUCTION
---------------
  scripts/cluster_launch.sh --dependency-profile training \
    --run-root /home/lancewicki/projects/turing-rl/results/runs/judge-smoke-qwen08b \
    --env SMOKE_MODEL=Qwen/Qwen3.5-0.8B --env SMOKE_PARSER=qwen3 --env SMOKE_N=200 \
    scripts/submit_snapshot_job.sh --export=ALL -- \
    scripts/slurm/gemma4_judge_training_smoke.sh

The battery defaults to gemma; SMOKE_MODEL/SMOKE_PARSER point it at any judge the serve
script can bring up. Add --env SMOKE_THINKING=0 for the thinking-off re-run (job 18913,
run root .../judge-smoke-qwen08b-nothink).

PREREQUISITE, and the one thing preflight caught: judge_serve_9b_replicas.sh exports
HF_HUB_OFFLINE=1, so the model must already be in the cache or the server cannot load it.
Qwen3.5-0.8B was not, and was fetched first (1.2 G, snapshot
2fc06364715b967f1860aea9cf38778875588b17):

  HF_HOME=/home/lancewicki/data/hf_cache HF_HUB_CACHE=/home/lancewicki/data/hf_cache \
  python -c "from huggingface_hub import snapshot_download; \
             snapshot_download('Qwen/Qwen3.5-0.8B')"

Note the cache is flat at that root, not under hub/.


6. CLUSTER SOURCE PATHS
-----------------------
Run root  /home/lancewicki/projects/turing-rl/results/runs/judge-smoke-qwen08b
Gates     $RUN_ROOT/work/job-18901-8756c7170696/results/judge-smoke/Qwen-Qwen3.5-0.8B/18901
Logs      $RUN_ROOT/logs/slurm-gemma_judge_smoke-18901.{out,err}
Baseline  results/grpo/rl-generator/9b_frac10_10ep_kl1e4_lr1e4_temp1/reward_dump
Source    commit 8756c71


7. HOW THIS DIRECTORY WAS BUILT
-------------------------------
Assembled 2026-08-22, pulled over the SSH tunnel (ssh -p 2223 lancewicki@localhost). Every
file except README.txt is machine-generated cluster output copied verbatim.
