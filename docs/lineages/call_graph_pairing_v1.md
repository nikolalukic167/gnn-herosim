# call_graph_pairing_v1 — trace-derived pairing with synthetic exchange semantics

**Status:** `REGISTERED` (no runs). Depends on: `r1_attribution_v1` (read). Plan: [`reference_physics_programme.md`](reference_physics_programme.md).
Created 2026-10-08 from W1 of the withdrawn draft `workload_redesign_v1` (never committed).


**Pre-run amendment (2026-10-07, decided by the coordinator).** The registered rule cannot select partners: our peer
groups already *are* fan-outs, calls of one trace sharing an rpcid parent (`scripts_cosim/grounded_workload_v1_extract.py`),
so every pair in a group "shares a caller within 2 levels". Measured on the first 6 M rows of `MSCallGraph_0.csv`
(128,665 traces, one fan-out drawn per trace): fan-out size mean 3.78, median 3; **33 %** have ≥ 4 siblings, the only
groups where a 2-partner cap leaves a choice; 77 % of siblings are leaves; 42 % have a sibling calling the same
service; 13 % share a downstream service with a sibling.
- **Source:** the same raw `cluster-trace-microservices-v2021` shard rows that built the fan-out library (same trace
  ids), so group membership and sizes are unchanged and partner identity is the only change. **Casper is not used**:
  its repairs change group membership (connected traces 58.32 % → 83.82 %, Huye et al., ICPE'24), and the Dataverse
  copy (doi 10.7910/DVN/RXIC9Z, 22.6 GB, CC0) is shuffled, with trace-id correspondence unchecked. A Casper-rebuilt
  library would be its own node, labelled as a group-membership change.
- **Rule:** within a fan-out, rank each task's candidate partners by (1) same callee service (`dm`), (2) a shared
  downstream service in the two calls' subtrees, (3) dispatch-time proximity; keep the top 2. Report the share of
  tasks whose partner set differs from today's random draw, by the tier that decided it.
- **Expected size:** prediction 2 is near-guaranteed by construction; this node can only show a small effect.

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
