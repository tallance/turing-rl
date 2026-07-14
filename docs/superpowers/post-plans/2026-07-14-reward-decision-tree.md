# Reward computation — decision tree (`training/grpo/reward.py`, metric="turing")

How the GRPO reward for one generated user-turn is computed. Source of truth:
`training/grpo/reward.py` (`score_turing_with_info` / `_score_pairwise_likert_with_info`).
Note: in the **judge sweep** the final reward is NOT used (`final_reward=None` in dumps);
only the derived `rating` is recorded/analyzed. This tree documents the full training-time path.

```
compute reward for one generated response (metric="turing")
│
├─ CoT has no meaningful thinking?  (thinking_hard_zero)
│     └─ YES → reward = 0.0   ── STOP (hard zero)
│
├─ parsed response empty?
│     └─ YES → total = max(0, format_score − length_penalty)   ── STOP
│
└─ call judge  (score_turing_with_info)
      │
      ├─ pick A/B order: generated_is_b = SHA256(user_id|post_id|target_idx|response) % 2 == 0
      │        (deterministic, ~50/50, fixed per pair, identical across cells)
      │
      ├─ ONE judge call → parse JSON → data
      │
      ├─ derive rating:
      │     ├─ has_score_fields?  (any of the 6 dimension scores present in JSON)
      │     │     ├─ YES → RECOMPUTE (model's own "rating" ignored):
      │     │     │        base_{a,b}   = immediate_target + human_goal + communication_style      (0–3)
      │     │     │        penalty_{a,b}= mean(source_copy, assistant_like, wrong_target_or_role,
      │     │     │                            unsupported_adversarial_reframing) × 3               (0–3)
      │     │     │        resp_{a,b}   = max(0, base − penalty)
      │     │     │        score_gap    = resp_b − resp_a
      │     │     │        rating       = _rating_from_turing_score_gap(gap)
      │     │     │                       (≤−2→1, ≤−1→2, ≤−0.25→3, <0.25→4, <1→5, <2→6, else 7)
      │     │     └─ NO  → rating = model's explicit "rating"
      │     └─ judge_parse_error? → likert_score = 0
      │
      ├─ order-normalize to "generated looks human" (higher = generated wins):
      │     ├─ gt_first  (human=A, gen=B):  likert_score = rating
      │     └─ gen_first (gen=A, human=B):  likert_score = 8 − rating
      │
      └─ return likert_score
│
├─ clip:      clipped   = min(likert_score, 5.0)              (TURING_JUDGE_SCORE_CLIP_MAX)
├─ normalize: unadjusted= (clipped − 1) / 6                   → [0, 0.667]
├─ scale:     adjusted  = unadjusted × 0.9                    (TURING_RAW_REWARD_SCALE)
├─ reward     = adjusted
│
└─ total_score = max(0, reward + format_score − length_penalty)   ── FINAL reward
        format_score   = generator CoT/response format bonuses (prefix, nonempty reasoning,
                         schema, clean tail); 0 if thinking hard-zero
        length_penalty = bounded penalty for response length vs ground-truth (deadband + cap)
```

## Key points
- The judge's own `rating` is **usually discarded** — the rating is re-derived from the
  **6 dimension scores + 8 penalties**, so those rubric fields (not the number) drive the reward.
  The explicit `rating` is used only when the model emits no dimension scores.
- Judge score is **capped at 5** before normalizing → ratings 5/6/7 all map to the same max
  (`(5−1)/6 = 0.667`), then ×0.9 ≈ **0.6 max** contribution from the judge.
- Generator-side `format_score` (+) and `length_penalty` (−) are added on top; hard-zeros
  (no thinking / empty response) short-circuit to ~0.
- The `0.9` scale, the `×3` penalty weighting, and the clip-at-5 are **upstream extras beyond
  the paper's plain `(min{s,5}−1)/6`** (inherited from commit 6aaecfb; see paper audit).
