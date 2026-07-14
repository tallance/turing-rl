# Post-Plans — decisions, discoveries & deviations during execution

This folder is the **running record of how reality diverged from the plan/spec** while
executing the judge-sweep work (and beyond). The plan (`docs/superpowers/plans/`) and spec
(`docs/superpowers/specs/`) are written *before* the work; this folder captures what we
**learned, decided, and changed** *during* it — the things a future reader (or a fresh agent
after compaction) needs to understand why the code/config looks the way it does.

## What goes here
- **Deviations** from the original plan/spec or upstream code (with the reason and whether it's
  a documented, intentional divergence).
- **Decisions** made at forks in the road (options considered, what we picked, why).
- **Discoveries** — bugs/gotchas/facts found during execution (e.g. the packing contamination,
  the single-GPU launch OOM) and how they changed course.

## How to use it
- Append dated entries to the current log (newest at top within each file). One file per
  work-phase/day is fine; keep entries short and concrete.
- Each entry: **Context → Decision/Finding → Rationale → Impact** (+ links to plan task #,
  commit SHAs, job ids where relevant).
- This is the human-facing *narrative*; the code-patch registry lives in `our_patches.md`
  (Task 22). When a deviation touches upstream code, record it in **both** — here for the
  "why", there for the "what changed".

## Index

Post-plans for the judge-sweep plan (`plans/2026-07-08-judge-sweep-implementation.md`) live
in the [`2026-07-08-judge-sweep/`](2026-07-08-judge-sweep/) subfolder:
- [2026-07-10 — judge-sweep execution decisions & deviations](2026-07-08-judge-sweep/2026-07-10-decisions-and-deviations.md)
- [2026-07-13 — paper vs. code methodology audit](2026-07-08-judge-sweep/2026-07-13-paper-vs-code-methodology-audit.md) — 8-point table (arXiv 2606.19336); all points match the paper or are unspecified, no conflicts.
- [2026-07-14 — reward computation decision tree](2026-07-08-judge-sweep/2026-07-14-reward-decision-tree.md)
- [2026-07-14 — thinking-ON CoT parse-failure diagnostic + 397B speed follow-up](2026-07-08-judge-sweep/2026-07-14-cot-failure-diagnostic.md)
