"""Sweep-cell client: run one (judge, thinking-mode) cell through the REAL reward path.

One *process per endpoint*. The pair-set is sharded across endpoints
(``pairs[endpoint_index::num_endpoints]``), so each process owns a disjoint slice
and sets ``OPENAI_API_BASE`` exactly once (to its own shard endpoint) before any
scoring happens. This avoids the v1 race where a single process mutated
``os.environ["OPENAI_API_BASE"]`` per async task.

Scoring is delegated to the production reward path
(``training.grpo.reward.score_turing_with_info`` -> ``_score_pairwise_likert_with_info``),
so the judge prompt, both-orderings randomization, JSON-schema response format, and
the reward-layer + HTTP dump wiring (Task 6) are exercised exactly as in training.
Reward rows land in ``<cell>/<mode>/reward/`` and raw judge HTTP dumps in
``<cell>/<mode>/http/`` automatically via the env this client locks.

Env is applied to ``os.environ`` *before* importing ``training.grpo.reward`` so its
env reads pick up the locked values. The reward import lives inside ``async_main`` so
this module imports cleanly on a machine with no live judge server.

Usage:
  python scripts/run_judge_sweep_cell.py \\
      --pairs raw/pairs/prism_heldout_880.parquet \\
      --endpoints http://node-a:8123/v1,http://node-b:8123/v1 \\
      --model Qwen/Qwen3-8B \\
      --thinking_mode off \\
      --out_dir raw/sweep \\
      --endpoint_index 0 --num_endpoints 2 \\
      --concurrency_per_endpoint 16
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def cell_env(
    *,
    model_id: str,
    mode: str,
    out_dir: str,
    sampling: dict | None = None,
) -> dict[str, str]:
    """Return the locked judge env for one sweep cell.

    ``mode`` is ``"on"``/``"off"`` for judge chain-of-thought (``enable_thinking``).
    ``out_dir`` is the per-(cell, mode) directory; HTTP dumps go to ``out_dir/http``
    and reward-layer dumps to ``out_dir/reward``.

    ``sampling`` is accepted for interface compatibility but intentionally IGNORED:
    Task 1 froze the policy to "no wire override; vLLM uses each model's
    generation_config.json defaults", so ``PERSONA_JUDGE_SAMPLING`` is NOT emitted.
    """
    del sampling  # deliberately unused (see docstring / Task-1 decision)
    return {
        "PERSONA_JUDGE_JSON_SCHEMA": "1",
        "PERSONA_JUDGE_DUMP_RATE": "1.0",
        "PERSONA_JUDGE_ENABLE_THINKING": "1" if mode == "on" else "0",
        "PERSONA_DISABLE_OPENROUTER_EXTRAS": "1",
        "JUDGE_MODEL": model_id,
        "PERSONA_JUDGE_MAX_COMPLETION_TOKENS": "8192",
        "PERSONA_JUDGE_DUMP_DIR": os.path.join(out_dir, "http"),
        "PERSONA_REWARD_DUMP_DIR": os.path.join(out_dir, "reward"),
    }


def shard_indices(items: list, endpoint_index: int, num_endpoints: int) -> list:
    """Round-robin shard: this process's slice of ``items``."""
    if num_endpoints < 1:
        raise ValueError(f"num_endpoints must be >= 1, got {num_endpoints}")
    if not 0 <= endpoint_index < num_endpoints:
        raise ValueError(
            f"endpoint_index {endpoint_index} out of range for num_endpoints {num_endpoints}"
        )
    return items[endpoint_index::num_endpoints]


def cell_output_dirs(base: str, cell_name: str, mode: str) -> dict[str, str]:
    """Return {"reward": .../{cell}/{mode}/reward, "http": .../{cell}/{mode}/http}."""
    mode_dir = os.path.join(base, cell_name, mode)
    return {
        "reward": os.path.join(mode_dir, "reward"),
        "http": os.path.join(mode_dir, "http"),
    }


def model_cell_name(model_id: str) -> str:
    """Slug for a model id, e.g. ``Qwen/Qwen3-8B`` -> ``qwen3-8b``."""
    return model_id.split("/")[-1].strip().lower()


def _final_metadata(
    base: dict,
    *,
    n_pairs: int,
    wall_seconds: float,
    ended_ts: float,
    ok: int,
    err: int,
) -> dict:
    """Merge post-scoring throughput fields onto the pre-scoring metadata base.

    Emits the keys ``scripts/calibration_report.py`` consumes (``n_pairs`` and
    ``wall_seconds``) so the throughput/>4h gate sees real numbers instead of
    ``n=0, wall=0`` (which extrapolated to ``inf``).
    """
    return {
        **base,
        "n_pairs": n_pairs,
        "wall_seconds": wall_seconds,
        "ended_ts": ended_ts,
        "ok": ok,
        "err": err,
    }


def _raise_on_scoring_errors(err: int) -> None:
    """Make incomplete shards fail after all assigned pairs are attempted."""
    if err:
        raise RuntimeError(f"judge shard completed with {err} scoring error(s)")


def _parse_endpoints(raw: str) -> list[str]:
    endpoints = [e.strip().rstrip("/") for e in raw.split(",") if e.strip()]
    if not endpoints:
        raise ValueError("--endpoints must contain at least one endpoint")
    return endpoints


def _api_key_for_endpoint(endpoint: str, resolve_remote_key: Callable[[], str]) -> str:
    """Use a dummy key for loopback vLLM; keep strict lookup for remote APIs."""
    hostname = urlparse(endpoint).hostname
    if hostname in {"localhost", "127.0.0.1", "::1"}:
        return os.environ.get("OPENAI_API_KEY") or "EMPTY"
    return resolve_remote_key()


async def async_main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", required=True, help="Pair-set parquet (from build_judge_pairs.py)")
    parser.add_argument("--endpoints", required=True,
                        help="Comma-separated judge endpoints (each like http://host:8123/v1)")
    parser.add_argument("--model", required=True, help="Judge model id, e.g. Qwen/Qwen3-8B")
    parser.add_argument("--thinking_mode", required=True, choices=["on", "off"])
    parser.add_argument("--out_dir", required=True, help="Sweep base dir (raw/sweep)")
    parser.add_argument("--cell_name", default=None,
                        help="Cell name (defaults to a slug of --model)")
    parser.add_argument("--concurrency_per_endpoint", type=int, default=16)
    parser.add_argument("--endpoint_index", type=int, default=0)
    parser.add_argument("--num_endpoints", type=int, default=1)
    parser.add_argument("--max_pairs", type=int, default=None,
                        help="Cap total pairs (applied before sharding) for calibration")
    args = parser.parse_args()

    endpoints = _parse_endpoints(args.endpoints)
    if args.endpoint_index >= len(endpoints):
        raise SystemExit(
            f"endpoint_index {args.endpoint_index} >= number of endpoints {len(endpoints)}"
        )
    # Guard against silent pair loss: shards are pairs[i::num_endpoints], so if
    # num_endpoints > len(endpoints) the shards with index >= len(endpoints) never
    # run and their pairs vanish from the cell. Require an exact match.
    if args.num_endpoints != len(endpoints):
        raise SystemExit(
            f"num_endpoints ({args.num_endpoints}) must equal the number of "
            f"endpoints ({len(endpoints)}) so every shard is executed"
        )
    my_endpoint = endpoints[args.endpoint_index]
    cell_name = args.cell_name or model_cell_name(args.model)

    dirs = cell_output_dirs(args.out_dir, cell_name, args.thinking_mode)
    mode_dir = os.path.dirname(dirs["reward"])  # base/cell/mode
    os.makedirs(dirs["reward"], exist_ok=True)
    os.makedirs(dirs["http"], exist_ok=True)

    # Lock env BEFORE importing the reward module and set this shard's endpoint once.
    env = cell_env(model_id=args.model, mode=args.thinking_mode, out_dir=mode_dir)
    os.environ.update(env)
    os.environ["OPENAI_API_BASE"] = my_endpoint

    # Imports that read env / need aiohttp live here so the module imports cleanly
    # on a machine without a live judge (the unit test only touches the pure helpers).
    import aiohttp
    import pandas as pd

    from shared.api_client import resolve_judge_api_key
    from training.grpo.reward import score_turing_with_info

    df = pd.read_parquet(args.pairs)
    all_pairs = df.to_dict("records")
    if args.max_pairs is not None:
        all_pairs = all_pairs[: args.max_pairs]
    my_pairs = shard_indices(all_pairs, args.endpoint_index, args.num_endpoints)

    print(
        f"[sweep-cell] model={args.model} mode={args.thinking_mode} cell={cell_name} "
        f"endpoint_index={args.endpoint_index}/{args.num_endpoints} endpoint={my_endpoint} "
        f"pairs_total={len(all_pairs)} pairs_this_shard={len(my_pairs)} "
        f"concurrency={args.concurrency_per_endpoint}",
        flush=True,
    )

    api_key = _api_key_for_endpoint(my_endpoint, resolve_judge_api_key)
    started = time.time()

    # Rank-0 writes run metadata (before scoring so it exists even if the run is killed).
    if args.endpoint_index == 0:
        metadata = {
            "model": args.model,
            "thinking_mode": args.thinking_mode,
            "cell_name": cell_name,
            "endpoints": endpoints,
            "num_endpoints": args.num_endpoints,
            "concurrency_per_endpoint": args.concurrency_per_endpoint,
            "sampling": os.environ.get("PERSONA_JUDGE_SAMPLING")
            or "generation_config_defaults (no wire override)",
            "json_schema": os.environ.get("PERSONA_JUDGE_JSON_SCHEMA"),
            "enable_thinking": os.environ.get("PERSONA_JUDGE_ENABLE_THINKING"),
            "max_completion_tokens": os.environ.get("PERSONA_JUDGE_MAX_COMPLETION_TOKENS"),
            "disable_openrouter_extras": os.environ.get("PERSONA_DISABLE_OPENROUTER_EXTRAS"),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "pair_source": os.path.abspath(args.pairs),
            "n_pairs_total": len(all_pairs),
            "max_pairs": args.max_pairs,
            "started_ts": started,
            "started_iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(started)),
        }
        with open(os.path.join(mode_dir, "run_metadata.json"), "w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2)

    semaphore = asyncio.Semaphore(max(1, args.concurrency_per_endpoint))
    counters = {"ok": 0, "err": 0}

    async def _score_one(session: aiohttp.ClientSession, pair: dict) -> None:
        async with semaphore:
            try:
                await score_turing_with_info(
                    session,
                    api_key,
                    str(pair.get("generated", "") or ""),
                    str(pair.get("human", "") or ""),
                    str(pair.get("user_history", "") or ""),
                    str(pair.get("context", "") or ""),
                    user_id=pair.get("user_id", ""),
                    post_id=pair.get("post_id", ""),
                    target_idx=pair.get("target_idx", ""),
                )
                counters["ok"] += 1
            except Exception as exc:  # noqa: BLE001 - one bad pair must not kill the shard
                counters["err"] += 1
                print(
                    f"[sweep-cell] scoring error pair_id={pair.get('pair_id')!r}: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )

    connector = aiohttp.TCPConnector(limit=max(1, args.concurrency_per_endpoint) * 2)
    timeout = aiohttp.ClientTimeout(total=float(os.environ.get("PERSONA_OPENAI_TIMEOUT_SECONDS", "400")))
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        await asyncio.gather(*(_score_one(session, pair) for pair in my_pairs))

    elapsed = time.time() - started
    print(
        f"[sweep-cell] done shard endpoint_index={args.endpoint_index}: "
        f"ok={counters['ok']} err={counters['err']} elapsed_s={elapsed:.1f}",
        flush=True,
    )

    # Rank-0 re-loads the pre-scoring metadata and adds the throughput fields the
    # calibration report reads (n_pairs / wall_seconds). Written after gather so
    # the numbers reflect the run that actually completed.
    if args.endpoint_index == 0:
        meta_path = os.path.join(mode_dir, "run_metadata.json")
        with open(meta_path, encoding="utf-8") as fh:
            base_metadata = json.load(fh)
        ended = time.time()
        final_metadata = _final_metadata(
            base_metadata,
            n_pairs=len(my_pairs),
            wall_seconds=elapsed,
            ended_ts=ended,
            ok=counters["ok"],
            err=counters["err"],
        )
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(final_metadata, fh, indent=2)

    _raise_on_scoring_errors(counters["err"])


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
