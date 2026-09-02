"""Per-example conversation trajectory dump for RL-generator overfit runs.

For one run's reward-*.jsonl dumps: group rows by (user_id, post_id, target_idx),
order each example's rows by ts and chunk into epochs of `group_size` (GRPO G)
rollouts. Emit a text file where, for each of the 10 overfit examples (chained),
we print the conversation CONTEXT and the GROUND-TRUTH human turn once, then the
GENERATED turn(s) for every epoch (all G rollouts, each with its judge Likert),
so the evolution over training is readable top-to-bottom.

Usage:
  python scripts/dump_conversation_trajectory.py \
    --dump_dir results/grpo/rl-generator/<tag>/reward_dump \
    --out results/grpo/rl-generator/<tag>/conversation_trajectory.txt [--group_size 4]
"""
from __future__ import annotations
import argparse, glob, json, os


def _load(dump_dir: str) -> list[dict]:
    rows = []
    for f in glob.glob(os.path.join(dump_dir, "reward-*.jsonl")):
        for line in open(f):
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def _key(r: dict) -> tuple:
    return (r.get("user_id"), r.get("post_id"), r.get("target_idx"))


def _likert(r: dict):
    x = r.get("turing_judge_score_raw")
    if x is None:
        return None
    try:
        return int(round(float(x)))
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--group_size", type=int, default=4)
    # train rows are rollout.n per epoch (G=4); val rows are val_kwargs.n=1 per pass,
    # so a val trajectory needs --split val --group_size 1 or the passes get merged.
    ap.add_argument("--split", choices=["train", "val"], default=None,
                    help="keep only this split (default: all rows)")
    a = ap.parse_args()
    G = a.group_size

    rows = _load(a.dump_dir)
    if a.split:
        rows = [r for r in rows if r.get("split") == a.split]
        if not rows:
            raise SystemExit(f"no rows with split={a.split!r} in {a.dump_dir}")
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault(_key(r), []).append(r)
    for k in groups:
        groups[k].sort(key=lambda r: r.get("ts") or 0)

    lines: list[str] = []
    lines.append(f"# Conversation trajectory — {a.dump_dir}")
    lines.append(f"# {len(groups)} examples; G={G} rollouts/epoch; win = Likert>=5, tie = 4, "
                 f"parse-fail = 0/None. Generated turn shown per rollout per epoch.")
    lines.append("# Score column: p_human (single-token judge; renormalised P(generated turn "
                 "is the human), 0-1) where logged, else Likert (full-schema judge, 1-7). "
                 "For single-token rows the [tag] is just p_human thresholded at 0.5.")
    lines.append("")

    for si, k in enumerate(sorted(groups, key=lambda t: tuple(str(x) for x in t)), 1):
        v = groups[k]
        head = v[0]
        n_epochs = (len(v) + G - 1) // G
        lines.append("=" * 100)
        lines.append(f"SAMPLE {si}/{len(groups)}  key={k[0]}/{k[1]}/t{k[2]}  "
                     f"({len(v)} rollouts, {n_epochs} epochs)")
        lines.append("=" * 100)
        lines.append("[CONTEXT — conversation so far]")
        lines.append(str(head.get("context", "")).rstrip())
        lines.append("")
        lines.append("[GROUND TRUTH — real human turn]")
        lines.append(str(head.get("ground_truth", "")).strip())
        lines.append("")
        lines.append("[GENERATED TURN BY EPOCH]  (epoch 0 = initial SFT policy, pre-update)")
        for e in range(n_epochs):
            chunk = v[e * G:(e + 1) * G]
            lines.append(f"  --- epoch {e} ---")
            for ri, r in enumerate(chunk, 1):
                lk = _likert(r)
                tag = "win" if (lk is not None and lk >= 5) else ("tie" if lk == 4 else
                      ("parsefail" if (lk == 0 or lk is None) else "human-wins"))
                gen = str(r.get("response", "")).replace("\n", " ").strip()
                # Single-token rows carry p_human, the actual reward and a continuous
                # [0,1]. Their "Likert" is only the 1/7 endpoint stand-in for the A/B
                # letter, so it cannot separate a coin-flip verdict from a certain one.
                # Prefer p_human where present; fall back to Likert for full-schema rows.
                ph = r.get("p_human")
                score = (f"p_human={float(ph):.3f}" if isinstance(ph, (int, float))
                         else f"Likert={lk}")
                hf = "  HARDFAIL" if r.get("hard_fail") else ""
                lines.append(f"    rollout {ri}  {score} [{tag}]{hf}  {gen}")
        lines.append("")

    with open(a.out, "w") as fh:
        fh.write("\n".join(lines))
    print(f"wrote {a.out}  ({len(groups)} examples)")


if __name__ == "__main__":
    main()
