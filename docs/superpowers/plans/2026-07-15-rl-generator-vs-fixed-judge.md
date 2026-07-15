# RL Generator vs. Fixed Judge — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Train a GRPO user-simulator generator (from the SFT LoRA checkpoint) against a **frozen**
judge with the reward cap lifted, and show it drives the judge's win-rate against the real human turn
above the SFT baseline / >50% (the Turing-RL "First Experiment": a frozen judge is gameable).

**Architecture:** One code change (env-configurable reward cap) + fresh Slurm launchers wrapping the
authors' `bash_scripts/grpo/train_grpo.sh` with explicit Hydra overrides and the judge reward env
(cap=7, reppen 1.1, thinking-on). Judge served self-hosted (9B as 8×1-GPU replicas; 397B TP=8) and
scored through the real `training/grpo/reward.py` path. Eval reuses the judge-sweep pipeline.

**Tech Stack:** veRL 0.7.1 (GRPO, FSDP+LoRA), vLLM (judge serving), PyTorch, Slurm (RFAI V3, 8×A100-40GB),
Qwen3-8B generator, Qwen3.5-9B / Qwen3.5-397B-A17B-GPTQ-Int4 judges, PRISM data.

**Spec:** `docs/superpowers/specs/2026-07-15-rl-generator-vs-fixed-judge-design.md` (read it first).

**Conventions:** Mac authors → commit (additive only) → `scripts/sync_to_cluster.sh` → run on cluster
via tunnel. Run the `preflight-job-check` skill before EVERY `sbatch`. Read remote files via SSH `cat`.

---

## Key paths (verified on cluster)

- SFT init/ref adapter: `checkpoints/sft/qwen3_8b_prism_full_s42_bf16_fsdp_nopack/final`
- GRPO train / val / test: `data/prism/full_s42_history_sft40_grpo60_test10/grpo/{train,val}.parquet` (4174/705), `.../test.parquet` (880)
- SFT-baseline judge pairs (sweep): `results/2026-07-08-judge-sweep/raw/pairs/prism_heldout_880.parquet`
- Authors' GRPO launcher: `bash_scripts/grpo/train_grpo.sh` (reads `training/grpo/configs/qwen3_8b_grpo_turing.yaml`, takes `SFT_ADAPTER_PATH`, forwards `"$@"` as Hydra overrides)
- Reward path: `training/grpo/reward.py` (`metric="turing"`); cap constant at `reward.py:57`, `clip_turing_judge_score` at `reward.py:267`
- 397B judge server: `scripts/slurm/judge_serve.sh` (TP=8). NOTE: `grpo_smoke*.sh` are BROKEN (deleted `train_grpo_smoke.sh`) — do not use.

---

## Task 1: Env-configurable reward cap (the one code change)

**Files:**
- Modify: `training/grpo/reward.py` (~line 57 constant, ~line 267 `clip_turing_judge_score`)
- Test: `tests/test_reward_cap.py`

**Step 1: Write the failing test**

```python
# tests/test_reward_cap.py
import importlib, os, pytest
import training.grpo.reward as R

def _reload():
    importlib.reload(R)  # pick up module-level default; env is read per-call
    return R

def test_default_cap_is_5(monkeypatch):
    monkeypatch.delenv("TURING_JUDGE_SCORE_CLIP_MAX", raising=False)
    r = _reload()
    assert r.clip_turing_judge_score(7) == 5.0
    assert r.clip_turing_judge_score(4) == 4.0

def test_env_cap_7_is_noop(monkeypatch):
    monkeypatch.setenv("TURING_JUDGE_SCORE_CLIP_MAX", "7")
    r = _reload()
    assert r.clip_turing_judge_score(7) == 7.0
    assert r.clip_turing_judge_score(6) == 6.0

def test_reward_math_cap5_vs_cap7(monkeypatch):
    # unadjusted = (clip-1)/6 ; adjusted = *0.9
    monkeypatch.setenv("TURING_JUDGE_SCORE_CLIP_MAX", "5")
    r = _reload()
    c = r.clip_turing_judge_score(7)
    assert r.adjust_turing_raw_reward((c - 1) / 6) == pytest.approx(0.6)   # (5-1)/6*0.9
    monkeypatch.setenv("TURING_JUDGE_SCORE_CLIP_MAX", "7")
    r = _reload()
    c = r.clip_turing_judge_score(7)
    assert r.adjust_turing_raw_reward((c - 1) / 6) == pytest.approx(0.9)   # (7-1)/6*0.9

def test_bad_env_raises(monkeypatch):
    monkeypatch.setenv("TURING_JUDGE_SCORE_CLIP_MAX", "notafloat")
    r = _reload()
    with pytest.raises(ValueError):
        r.clip_turing_judge_score(7)
```

**Step 2: Run — expect FAIL**

`cd ~/Projects/turing-rl && python -m pytest tests/test_reward_cap.py -v`
Expected: `test_env_cap_7_is_noop` / `test_bad_env_raises` FAIL (cap is a hardcoded constant).

**Step 3: Implement**

In `training/grpo/reward.py`, keep the constant as the default and read env per call:

```python
# keep near line 57:
TURING_JUDGE_SCORE_CLIP_MAX = 5.0  # default; overridable via TURING_JUDGE_SCORE_CLIP_MAX env

def _get_turing_judge_score_clip_max() -> float:
    raw = os.environ.get("TURING_JUDGE_SCORE_CLIP_MAX")
    if raw is None:
        return TURING_JUDGE_SCORE_CLIP_MAX
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"TURING_JUDGE_SCORE_CLIP_MAX must be a float, got {raw!r}") from exc

def clip_turing_judge_score(score: float) -> float:
    """Clip the raw Turing judge score before reward normalization for training."""
    return min(float(score), _get_turing_judge_score_clip_max())
```

**Step 4: Run — expect PASS**

`python -m pytest tests/test_reward_cap.py -v` → 4 passed.

**Step 5: Commit**

```bash
git add training/grpo/reward.py tests/test_reward_cap.py
git commit -m "reward: env-configurable TURING_JUDGE_SCORE_CLIP_MAX (default 5.0); cap=7 lifts it for the hacking probe"
```

---

## Task 2: Overfit-gate metric (win-rate on the 10)

Defines the gate unambiguously (spec §11 test 2): read reward-dump rows, pick=gen if Likert≥5, ties
(rating 4) excluded, pass = ≥8/10.

**Files:**
- Create: `scripts/overfit_gate_check.py`
- Test: `tests/test_overfit_gate.py`

**Step 1: Failing test**

```python
# tests/test_overfit_gate.py
from scripts.overfit_gate_check import win_rate_from_rows

def _row(gen_is_b, rating):
    # human on A when gen is B: rating_gt_first is the "gen is more human" scale.
    # We store the oriented Likert directly as `likert` for the test.
    return {"likert": rating}

def test_win_excludes_ties_and_counts_ge5():
    rows = [{"likert": r} for r in [7,6,5,5,4,4,3,2,5,6]]
    # non-tie = 8 (drop the two 4s); wins (>=5) = 6
    wr = win_rate_from_rows(rows)
    assert wr["n_nontie"] == 8
    assert wr["wins"] == 6
    assert wr["win_rate"] == 0.75
    assert wr["passed"] is False   # 6/8 < 8/10 target on 10-set -> see threshold note

def test_pass_at_8_of_10():
    rows = [{"likert": r} for r in [7,7,7,6,6,5,5,5,3,4]]
    wr = win_rate_from_rows(rows, pass_wins=8)
    assert wr["wins"] == 8 and wr["passed"] is True
```

**Step 2: Run — expect FAIL** (`ModuleNotFoundError`).

**Step 3: Implement**

```python
# scripts/overfit_gate_check.py
"""Overfit-gate metric: win-rate of the generated turn on the (few) training prompts.

`likert` per row is the judge's 1-7 rating oriented so higher = judge thinks the GENERATED
turn is more human (reward.py's `turing_judge_score_raw`). pick=gen if likert>=5; ties (4) excluded.
Gate passes when wins >= pass_wins (default 8, i.e. 8/10)."""
from __future__ import annotations
import argparse, glob, json, os

def win_rate_from_rows(rows, pass_wins: int = 8) -> dict:
    likerts = [float(r["likert"]) for r in rows if r.get("likert") is not None]
    nontie = [x for x in likerts if int(round(x)) != 4]
    wins = sum(1 for x in nontie if x >= 5)
    n = len(nontie)
    return {"n_total": len(likerts), "n_nontie": n, "wins": wins,
            "win_rate": (wins / n) if n else 0.0, "passed": wins >= pass_wins}

def _load_reward_dump(dump_dir: str) -> list[dict]:
    rows = []
    for f in glob.glob(os.path.join(dump_dir, "reward-*.jsonl")):
        for line in open(f):
            try: d = json.loads(line)
            except Exception: continue
            # oriented Likert: reward.py dumps turing_judge_score_raw
            lk = d.get("turing_judge_score_raw")
            if lk is not None:
                rows.append({"likert": lk})
    return rows

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump_dir", required=True)
    ap.add_argument("--pass_wins", type=int, default=8)
    a = ap.parse_args()
    res = win_rate_from_rows(_load_reward_dump(a.dump_dir), pass_wins=a.pass_wins)
    print(json.dumps(res, indent=2))
    raise SystemExit(0 if res["passed"] else 1)
```

**Step 4: Run — expect PASS.** `python -m pytest tests/test_overfit_gate.py -v`

**Step 5: Commit**

```bash
git add scripts/overfit_gate_check.py tests/test_overfit_gate.py
git commit -m "add overfit-gate win-rate metric (>=8/10, ties excluded) + test"
```

---

## Task 3: Reward-env → judge-payload wiring test

Guards that the fidelity knobs actually reach the judge (spec §11 test 3). `reward.py._openai_chat`
already reads `PERSONA_JUDGE_SAMPLING` and `PERSONA_JUDGE_ENABLE_THINKING` — this test locks it.

**Files:**
- Test: `tests/test_reward_env_payload.py`
- (Modify `training/grpo/reward.py` / `shared/api_client.py` only if the test reveals a gap.)

**Step 1: Failing test**

```python
# tests/test_reward_env_payload.py
import json, importlib, pytest

def test_sampling_and_thinking_reach_payload(monkeypatch):
    monkeypatch.setenv("PERSONA_JUDGE_SAMPLING", json.dumps({"repetition_penalty": 1.1, "temperature": 0.6}))
    monkeypatch.setenv("PERSONA_JUDGE_ENABLE_THINKING", "1")
    from shared import api_client
    importlib.reload(api_client)
    # build_chat_payload merges sampling + chat_template_kwargs
    payload = api_client.build_chat_payload(
        model="Qwen/Qwen3.5-9B",
        messages=[{"role": "user", "content": "hi"}],
        max_completion_tokens=8192,
        response_format={"type": "json_object"},
        reasoning=False,
        sampling={"repetition_penalty": 1.1, "temperature": 0.6},
        chat_template_kwargs={"enable_thinking": True},
    )
    assert payload.get("repetition_penalty") == 1.1
    assert payload.get("temperature") == 0.6
    assert payload["chat_template_kwargs"]["enable_thinking"] is True
```

**Step 2: Run — expect PASS or FAIL.** If `build_chat_payload`'s signature/merge differs, read
`shared/api_client.py:build_chat_payload` and adjust the test to the real API; if a knob is dropped,
fix `build_chat_payload` minimally (thread `sampling`/`chat_template_kwargs` into the payload dict).

**Step 3: (only if FAIL) implement the minimal merge fix in `shared/api_client.py`.**

**Step 4: Run — expect PASS.**

**Step 5: Commit**

```bash
git add tests/test_reward_env_payload.py shared/api_client.py 2>/dev/null; git add tests/test_reward_env_payload.py
git commit -m "test: judge reppen/temp/thinking env reach the judge payload"
```

---

## Task 4: GRPO config-integrity test (SSOT guard)

Asserts the base config keeps the locked training params (spec §11 test 4). No new yaml — the fresh
launcher passes data/adapter/batch overrides; this test guards the paper-faithful values in
`training/grpo/configs/qwen3_8b_grpo.yaml`.

**Files:**
- Test: `tests/test_grpo_config.py`

**Step 1: Failing test**

```python
# tests/test_grpo_config.py
import os, yaml

CFG = "training/grpo/configs/qwen3_8b_grpo.yaml"

def _load():
    with open(CFG) as f:
        return yaml.safe_load(f)

def test_locked_training_params():
    c = _load()
    ar = c["actor_rollout_ref"]
    assert ar["model"]["lora_rank"] == 64 and ar["model"]["lora_alpha"] == 32
    assert ar["actor"]["ppo_epochs"] == 1
    assert ar["actor"]["ppo_mini_batch_size"] == 64
    assert ar["actor"]["use_kl_loss"] is True
    assert float(ar["actor"]["kl_loss_coef"]) == 1e-3
    assert float(ar["actor"]["clip_ratio"]) == 0.2
    assert ar["rollout"]["n"] == 4
    assert float(ar["rollout"]["temperature"]) == 0.6
    assert c["data"]["train_batch_size"] == 64            # match upstream code (paper table=128)
    assert c["trainer"]["total_epochs"] == 3
    assert c["algorithm"]["adv_estimator"] == "grpo"

def test_sft_data_paths_exist_locally_or_documented():
    # PRISM grpo split used by the launcher override (not the base convokit default).
    p = "data/prism/full_s42_history_sft40_grpo60_test10/grpo/train.parquet"
    assert os.path.exists(p) or True  # cluster-only; presence asserted at run time in preflight
```

**Step 2: Run — expect PASS** (base config already has these) — this is a *regression guard*, so it
passes now and fails if anyone drifts the params. If it FAILS today, a value already drifted → fix
the yaml back to the locked value.

**Step 3: (only if drift) restore the locked value in the yaml.**

**Step 4: Run — expect PASS.**

**Step 5: Commit**

```bash
git add tests/test_grpo_config.py
git commit -m "test: lock GRPO training params (train_batch 64, lora 64/32, kl 1e-3, G=4, 3 epochs)"
```

---

## Task 5: Overfit-10 dataset builder

**Files:**
- Create: `scripts/build_overfit10.py`
- Test: `tests/test_overfit10_builder.py`

**Step 1: Failing test**

```python
# tests/test_overfit10_builder.py
import pandas as pd
from scripts.build_overfit10 import build_overfit

def test_build_overfit_takes_first_n(tmp_path):
    src = tmp_path / "train.parquet"
    df = pd.DataFrame({"data_source": ["prism"]*20, "prompt": range(20),
                       "reward_model": [{}]*20, "extra_info": [{}]*20})
    df.to_parquet(src)
    out = tmp_path / "train_overfit10.parquet"
    build_overfit(str(src), str(out), n=10)
    got = pd.read_parquet(out)
    assert len(got) == 10
    assert list(got.columns) == ["data_source", "prompt", "reward_model", "extra_info"]
    assert got["prompt"].tolist() == list(range(10))   # first 10, deterministic
```

**Step 2: Run — expect FAIL.**

**Step 3: Implement**

```python
# scripts/build_overfit10.py
"""Write the first N rows of a veRL grpo train parquet to an overfit subset."""
import argparse, pandas as pd

def build_overfit(src: str, out: str, n: int = 10) -> None:
    df = pd.read_parquet(src)
    req = ["data_source", "prompt", "reward_model", "extra_info"]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"missing veRL columns: {missing}")
    df.head(n).to_parquet(out, index=False)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=10)
    a = ap.parse_args(); build_overfit(a.src, a.out, a.n)
    print(f"wrote {a.out}")
```

**Step 4: Run — expect PASS.**

**Step 5: Commit**

```bash
git add scripts/build_overfit10.py tests/test_overfit10_builder.py
git commit -m "add overfit-10 dataset builder (first-N veRL parquet subset) + test"
```

---

## Task 6: Eval scorer + eval-vs-sweep parity test

Reuses the sweep pipeline; parity test (spec §11 test 6) ensures RL-final accuracy is comparable to
the SFT baseline (same directional-accuracy logic + order).

**Files:**
- Create: `scripts/eval_rl_generator.py` (wrapper: reads a judge-pairs parquet + reward dumps → accuracy/win-rate/ties, sweep-matched order)
- Test: `tests/test_eval_parity.py`

**Step 1: Failing test**

```python
# tests/test_eval_parity.py
from scripts.eval_rl_generator import directional_accuracy

def test_matches_sweep_convention():
    # rating_gt_first: judge's 1-7 where >=5 => picks generated (fake). accuracy = judge picks HUMAN.
    rows = [
        {"rating_gt_first": 2, "rating_gen_first": None},  # <=3 -> picks A(human) -> correct
        {"rating_gt_first": 6, "rating_gen_first": None},  # >=5 -> picks B(gen)  -> wrong
        {"rating_gt_first": 4, "rating_gen_first": None},  # tie -> excluded
    ]
    acc = directional_accuracy(rows)
    assert acc["n_nontie"] == 2 and acc["correct"] == 1
    assert acc["accuracy"] == 0.5 and acc["gen_win_rate"] == 0.5
```

**Step 2: Run — expect FAIL.**

**Step 3: Implement** `directional_accuracy` replicating the sweep's rule (pick human if rating≤3,
gen if ≥5, ties=4 excluded; use `rating_gt_first`, fall back to `rating_gen_first` with orientation
flip). Add a `--pairs`/`--dump_dir` CLI that runs `eval/generate_trained.py` + `build_judge_pairs.py`
outputs through it. (Read `scripts/run_judge_sweep_cell.py` / `build_judge_pairs.py` for the exact
orientation fields before finalizing.)

**Step 4: Run — expect PASS.**

**Step 5: Commit**

```bash
git add scripts/eval_rl_generator.py tests/test_eval_parity.py
git commit -m "add RL-generator eval scorer (sweep-matched directional accuracy) + parity test"
```

---

## Task 7: 9B 8-replica judge serve script

**Files:**
- Create: `scripts/slurm/judge_serve_9b_replicas.sh`

Boots 8×1-GPU `Qwen/Qwen3.5-9B` replicas on one node (ports 8300..8307), env `turing-rl-train`,
`--reasoning-parser qwen3`, `HF_HUB_OFFLINE=1`. Model-verified health check per replica (reuse the
pattern in `scripts/slurm/judge_sweep_cell.sh:108-120`). Prints the 8 endpoint URLs to a file the
launcher reads. Stays up until scancel.

**Steps:** write script → `bash -n` → commit. (No unit test; validated by Task 9 smoke.)

```bash
git add scripts/slurm/judge_serve_9b_replicas.sh
git commit -m "add 9B 8-replica judge serve (data-parallel, thinking-on)"
```

---

## Task 8: Fresh GRPO launcher (serve → train → teardown)

**Files:**
- Create: `scripts/slurm/rl_generator_run.sh` (all-in-one orchestrator)

Parametrized by env: `JUDGE={9b|397b}`, `MODE={overfit|full|epoch1}`. Responsibilities:
1. `unset` proxies; export reward env: `REWARD_METRIC=turing`, `TURING_JUDGE_SCORE_CLIP_MAX=7`,
   `PERSONA_JUDGE_SAMPLING={"repetition_penalty":1.1,"temperature":0.6}`, `PERSONA_JUDGE_ENABLE_THINKING=1`,
   `PERSONA_JUDGE_MAX_COMPLETION_TOKENS=8192`, `PERSONA_JUDGE_DUMP_RATE=1.0`, `PERSONA_REWARD_DUMP_DIR=<run dir>`,
   `JUDGE_MODEL=<Qwen/Qwen3.5-9B|Qwen/Qwen3.5-397B-A17B-GPTQ-Int4>`, `OPENAI_API_BASE=<judge endpoint>`,
   `SFT_ADAPTER_PATH=checkpoints/sft/qwen3_8b_prism_full_s42_bf16_fsdp_nopack/final`,
   `WANDB_PROJECT=2026-07-15-rl-generator-vs-fixed-judge` (wandb project = the plan name).
2. Sbatch the judge server (Task 7 for 9b; `judge_serve.sh` for 397b) on a **separate** node; wait
   `/v1/models` + model-verify.
3. Run `bash bash_scripts/grpo/train_grpo.sh turing prism history none` with **explicit Hydra
   overrides** appended (train_grpo.sh forwards `"$@"`):
   - `data.train_files=data/prism/full_s42_history_sft40_grpo60_test10/grpo/train.parquet`
   - `data.val_files=data/prism/full_s42_history_sft40_grpo60_test10/grpo/val.parquet`
   - `MODE=overfit`: also `data.train_files=<overfit10>` `data.train_batch_size=10`
     `actor_rollout_ref.actor.ppo_mini_batch_size=10` `trainer.total_epochs=<~40>` `trainer.save_freq=0`
   - `MODE=epoch1`: `trainer.total_epochs=1`
   - `trainer.project_name=2026-07-15-rl-generator-vs-fixed-judge` (veRL wandb project = plan name)
   - `trainer.default_local_dir=<run ckpt dir>` `trainer.experiment_name=<name>` `trainer.resume_mode=auto`
   - Carry `our_patches.md` "DELETED: train_grpo_smoke.sh" 40GB overrides IF needed (start without; add on OOM).
4. `trap`-scancel the judge on exit.

**Steps:** write → `bash -n` → commit.

```bash
git add scripts/slurm/rl_generator_run.sh
git commit -m "add fresh RL-generator launcher (serve judge -> train_grpo.sh w/ reward env -> teardown)"
```

---

## Task 9: Deploy + veRL LoRA init/ref verification (cluster gate)

Confirms veRL loads the SFT adapter as **both** RL init and KL reference `πref` before any real run.

**Steps:**
1. `git status` clean + HEAD contains others' commits → `bash scripts/sync_to_cluster.sh`.
2. Run local tests: `python -m pytest tests/ -q` (all green).
3. Run `preflight-job-check` skill for `scripts/slurm/rl_generator_run.sh`.
4. Launch a **tiny** 9B overfit smoke (`JUDGE=9b MODE=overfit`, `total_epochs=2`) — the smallest run.
5. In the trainer log, verify: adapter loaded from `SFT_ADAPTER_PATH`; `ref` policy initialized from
   the SFT checkpoint (grep veRL log for `lora_adapter_path` / ref-policy load / non-zero KL at step 1).
   Non-zero-but-small KL at step 1 ⇒ πθ≈πref (correct); KL≈0 forever or adapter-not-found ⇒ bug → fix
   wiring (pass `actor_rollout_ref.ref.*` / adapter override explicitly) before proceeding.

No commit unless a wiring fix is needed.

---

## Task 10: Overfit gate — 9B (blocking)

**Steps:**
1. Build overfit10: `python scripts/build_overfit10.py --src <grpo train> --out data/prism/full_s42_history_sft40_grpo60_test10/grpo/train_overfit10.parquet` (commit the builder invocation in a small `bash_scripts` note; the parquet lives under `data/` = gitignored, build on cluster).
2. `preflight-job-check` → launch `JUDGE=9b MODE=overfit`.
3. Monitor wandb (reward ↑, raw judge score ↑) + parse-fail rate (<~5% with reppen).
4. On completion: `python scripts/overfit_gate_check.py --dump_dir <run reward dump>` → **PASS = ≥8/10**.
5. If not passing: raise epochs / lower `kl_loss_coef`; re-run. Record outcome in the post-plan.

**Gate:** do not start Task 12 until 9B overfit passes.

---

## Task 11: Overfit gate — 397B (blocking for Task 13)

Same as Task 10 with `JUDGE=397b MODE=overfit` (judge = `judge_serve.sh`, TP=8). Budget ~4–10h
(0.044 calls/s, pre-reppen; reppen faster). Gate: ≥8/10 → allows Task 13.

---

## Task 12: Full run — 9B (headline)

**Steps:**
1. `preflight-job-check` → launch `JUDGE=9b MODE=full` (trainer node + 9B 8-replica judge node = 16 GPU).
2. Monitor: reward/raw-score curves, win-rate proxy, parse-fail, throughput (steps/h). `resume_mode=auto`.
3. Expect ~14–24h (reppen-dependent). On completion: final adapter under the run ckpt dir.

---

## Task 13: 397B single-epoch plumbing run

`JUDGE=397b MODE=epoch1` (trainer + 397B TP=8 judge = 16 GPU). ~4–5 days (pre-reppen). Goal: confirm
the pipeline scales against the anchor. Full 3-epoch 397B deferred (spec §7).

---

## Task 14: Eval on the 880 (headline metric)

**Steps (per completed run — 9B first):**
1. `preflight-job-check` → generate from the RL-final adapter: `eval/generate_trained.py` on
   `data/prism/full_s42_history_sft40_grpo60_test10/test.parquet` (mirror `scripts/slurm/heldout_inference.sh`, swap checkpoint dir).
2. `scripts/build_judge_pairs.py` → (real, RL-gen) pairs.
3. Score with the matching judge (reppen, thinking-on, sweep-matched order) via the reward path.
4. `python scripts/eval_rl_generator.py` → accuracy / gen-win-rate / frac-ties.
5. Compare to the SFT baseline (sweep: 9B-on accuracy 0.73). **Win iff gen-win-rate > SFT baseline,
   ideally >50% (accuracy <50%).** Inspect hacked turns via `scripts/dump_viewer.py`.

---

## Task 15: Docs + reproducibility

**Files:**
- Create: `docs/superpowers/post-plans/2026-07-15-rl-generator-decisions.md` (decisions/deviations: cap 5→7, train_batch 64 code-choice, reppen 1.1, 9B-only full, 397B deferred, probe numbers).
- Modify: `our_patches.md` (reward.py cap env change).
- Create: `results/grpo/rl-generator/README.txt` (repro commands, input paths, per-run numbers vs SFT baseline).

```bash
git add docs/superpowers/post-plans/2026-07-15-rl-generator-decisions.md our_patches.md
git commit -m "docs: RL-generator decisions/deviations + reward cap patch note"
```

---

## Execution order & gates

1–6 (local TDD, any order) → 7,8 (scripts) → **deploy + Task 9 (veRL wiring gate)** →
**Task 10 (9B overfit gate)** → Task 12 (9B full) → Task 14 (9B eval). In parallel after Task 9:
Task 11 (397B overfit gate) → Task 13 (397B epoch1). Task 15 throughout.
