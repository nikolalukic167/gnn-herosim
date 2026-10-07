# record_merge_v1 — records-only merge of `local-record-2026-10-06`

**Status:** `REGISTERED` (no runs). Kind: housekeeping. Can run in parallel with `kpa_scaleout_v1`,
but not while any node is writing to `LINEAGES.md`.

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
