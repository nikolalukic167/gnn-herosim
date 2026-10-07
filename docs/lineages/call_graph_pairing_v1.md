# call_graph_pairing_v1 — trace-derived pairing with synthetic exchange semantics

**Status:** `REGISTERED` (no runs). Depends on: `r1_attribution_v1` (read). Plan: [`reference_physics_programme.md`](reference_physics_programme.md).
Created 2026-10-08 from W1 of the withdrawn draft `workload_redesign_v1` (never committed).

## Labelling (must appear wherever this workload is described)
**Trace-derived pairing, synthetic exchange semantics.** The Alibaba trace records caller→callee calls. This node uses
the reconstructed call graph only to decide *which* sibling tasks of a request exchange data and *how many* partners
each has. That siblings exchange data at all, and how much, remains a modelling assumption (payloads from WF1).
It is not a call-edge or DAG workload.

## Why not a true DAG workload here
Parent→child data flow changes the decision structure (tasks no longer arrive as a burst; the live stack batches only
tasks whose parents finished and has no DAG block; see `docs/hard-stops.md`, route_b audit 2026-09-06). That needs its
own lineage with learned arms designed for it, outside this programme.

## Data
- Casper (docc-lab/casper) run on the 2021 dataset. No licence: ship only our extraction script; cite Huye et al.,
  ICPE'24, and the Dataverse release. Measure data volume before planning.
- Pairing rule (fixed now): two tasks in the same request exchange data if they share a caller in the reconstructed
  graph and are both leaf-or-internal calls within 2 levels of each other; partner count per task capped at 2
  (as today) so only the *identity* of partners changes, not their number.
- Reconstruction quality reported: share of requests whose call graph is connected after Casper, and how many groups
  change size versus the current fan-out library.

## Design
- Physics R1, workload WF1 except pairing, rungs from `load_recalibration_v1`.
- Arms: classical arms; learned arms from `r1_attribution_v1` evaluated **zero-shot** and labelled so. Retraining on
  this workload is a dated amendment, allowed only if the stopping rule has not triggered.

## Primary family
CD vs each other arm per rung (Holm).

## Predictions
1. CD remains first.
2. Exchange share changes by < 20 % relative to random pairing (partner count is unchanged).
3. Rankings among classical arms unchanged.

## Outcomes
Rankings that survive are reported as robust to partner identity; any flip is a sensitivity finding.
