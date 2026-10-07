# record_merge_v1 — records-only merge of `local-record-2026-10-06`

**Status:** `CLOSED` (2026-10-07) — **MERGED**. Registered 2026-10-08 (plan date), run before any other node.
Kind: housekeeping.

**Outcome.** The 41 nodes that existed only on `local-record-2026-10-06`, with their attachment folders and index
rows, are on `reference-physics` at `e448a8d5` (299 files, all additions). Each carries the banner
`Produced on local-record-2026-10-06 at afea04bd; code not merged; old physics.` The 24 nodes that differ between
the branches were left as on `reference-physics`. All three checks pass: hygiene 15 → 14 failures (the AGENTS.md
entry-point check now resolves; the other 14 predate the merge); `src/`, `tests/` and `experiments/` unchanged;
every node the programme cites resolves.

## Goal
Every lineage cited by the paper resolves to a node file on the working branch, with its attachments, and no
simulator code changes from the other branch enter silently.

## Scope (fixed now)
- Bring over: the 41 lineage nodes that exist only on `local-record-2026-10-06`, their attachment folders,
  and their `LINEAGES.md` index rows.
- Do **not** bring over: any change under `src/`, tests, configs, or `.gitignore` code-related entries.
  The 8 conflicting code files (`infrastructure.py`, `autoscaler.py`, `simulation.py`, `model.py`,
  `executesimulation.py`, GNN scheduler, peer-greedy scheduler, determinism test) stay as on the working branch.
- Where a merged node describes results produced by code that differs from the working branch, add a one-line
  banner: `Produced on local-record-2026-10-06 at <hash>; code not merged; old physics.`

## Checks
1. Hygiene test passes (every index row has a node; no new failures).
2. A diff of `src/` before vs after the merge is empty.
3. Every node cited in [`reference_physics_programme.md`](reference_physics_programme.md) and its dependants resolves.

## Separate job (not this node)
A code merge of the other branch requires a replay of every registered gate's default path and is its own
lineage if ever needed.

## Record

### 2026-10-07 — merged
Merged by a separate session on branch `rp/record-merge` (`e448a8d5`), reviewed and fast-forwarded into
`reference-physics` by the coordinator: diff is additions only, under `docs/lineages/` and `LINEAGES.md`; banner present
on all 41 nodes. Tests were run with the main checkout's interpreter (the worktree's pipenv env is empty).
