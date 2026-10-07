# Reference-physics programme (plan, 2026-10-07; revised 2026-10-08)

**Status:** `REGISTERED` (programme plan). No result in any node below exists yet.
Purpose: rebuild every comparison on a physics that has been scrutinised and frozen, on a workload that has been
fixed, then write results. Nothing from the old physics enters a results section except as a labelled
sensitivity finding.

## Revision 2026-10-08 (before any run of nodes 2–9)
Following the feasibility review:
- W2–W4 (payloads, access-link classes, task types) moved ahead of retraining as `workload_fix_v1`, and load
  recalibration now runs on the fixed workload. Learned arms are trained once, on the final workload.
- W1 split out as `call_graph_pairing_v1` (trace-derived pairing, synthetic exchange semantics). A true DAG
  workload is a future lineage, not part of this programme.
- `physics_audit_v1` gains I11 (co-simulation reproduces live under R1), I12 (policy time scale fixed across
  rungs) and a pre-stated fallback for I5.
- `r1_attribution_v1` declares its candidate-pruning rule and ablation before labels are generated.
- `replica_placement_v1` replaces the live one-step lookahead with offline snapshot labels plus an analytic rule.

## Order (each node depends on the one above unless marked)

| # | Node | Kind | Gate to proceed |
|---|------|------|-----------------|
| 1 | `kpa_scaleout_v1` | physics factor (third factor of `transfer_physics_v1`) | read complete, predictions scored |
| 2 | `physics_audit_v1` | invariant checks + freeze of reference physics **R1** | all invariants pass (or I5 fallback as stated) |
| 3 | `workload_fix_v1` | W2 payloads, W3 access links, W4 task types on R1; freeze workload **WF1** | each stage read; WF1 committed |
| 4 | `load_recalibration_v1` | load rungs on R1 + WF1 by measured queue share | rungs fixed on held-out cells |
| 5 | `r1_attribution_v1` | train learned arms on R1 + WF1; attribution battery | read complete |
| 6 | `record_merge_v1` | housekeeping, no runs (between node reads) | hygiene test passes |
| 7 | `call_graph_pairing_v1` | trace-derived pairing on R1 + WF1; sensitivity | read complete |
| 8 | `replica_placement_v1` | hypothesis lineage (conditional on 1–5) | precondition, then exploratory |
| 9 | `access_link_contention_v1` | hypothesis lineage (conditional on 1–5) | precondition, then exploratory |

One physics or workload change per node. Nodes 8 and 9 never run together.
Minimum paper: nodes 1–5 suffice for the attribution and physics-sensitivity paper. Nodes 7–9 are extensions.

## Shared rules (apply to every node unless the node overrides them explicitly)

**Registration.** Node file and index row are committed before any condition run. Predictions, bars and the
analysis script path are in the node. Changes after registration are dated amendments.

**Replay check.** Any code change ships with a replay of the previous default path; the node is not read until
the replay is identical field-for-field (known hung runs excepted and named).

**Protocol deviations.** Any departure from a registered procedure (e.g. reruns cancelled before the registered
timeout) is recorded in the node as a deviation, with a sensitivity check showing whether conclusions depend on it.

**Provenance labels.** Every reported number carries: transfer model, replica-release flag, scale-out regime,
origin model, payload model, link classes, load-rung definition, policy time scale. Old-physics numbers are
labelled `old physics (store-and-forward, held replicas, reachability scale-out)`.

**Statistics.** Unit of analysis = topology (seeds averaged within topology first; 19 topologies). Paired
Wilcoxon signed-rank, two-sided, Holm across the node's registered primary family. Report median paired %
difference, wins out of 19, and Holm-adjusted p. "Win" = median ≤ −5 % and Holm p < 0.05.
"Direction only" = Holm p < 0.05 but |median| < 5 %.

**Failures.** A run that fails after one rerun at 3× timeout is a failure of its own arm; only that arm's paired
tests drop the cell; nothing is imputed. Sensitivity re-read with any topology that has a failure removed for
every arm.

**Metrics always reported.** Mean and median task latency; P95 and P99; queue time in seconds and queue share;
exchange share; cold starts per task; replicas per type over time (median, p90, max); per-replica busy fraction
(median, max); decision time per task (wall clock, same CPU, single thread, not in simulated time).

**Baselines always present.** CD (exact-cost coordinate descent; cite Bi & Zhang TWC'18, DROO TMC'20),
locality-first, one-pass greedy, self-predict, Knative (labelled with its scale-out regime). Knative is context
only, never the headline comparator.

**Learned arms.** Any learned arm evaluated on a physics or workload it was not trained on is labelled
`zero-shot` and excluded from claims.

## Stopping rule (registered 2026-10-07, scope updated 2026-10-08)
If, after nodes 1–5, no learned arm reaches a **win** against CD in any registered regime, the programme stops
searching for a learned-performance result and writes the attribution and physics-sensitivity paper. Nodes 7–9
may still run, each on its own registered bars; a positive result there is reported as its own finding, not
retrofitted.
