# replica_placement_v1 — where new replicas go, under eager scale-out (hypothesis)

**Status:** `REGISTERED` as hypothesis (no runs). Conditional on: nodes 1–5 read; closed as NO-LEVER if
`physics_audit_v1` records I5 as FAIL-WITH-CAUSE. Runs alone (never with `access_link_contention_v1`).
Plan: [`reference_physics_programme.md`](reference_physics_programme.md).
Revision 2026-10-08 (before any run): the live one-step lookahead is replaced by offline snapshot labels plus an
analytic rule (SimPy has no state fork; a live lookahead is not deployable).

## Hypothesis
Under KPA scale-out on R1 + WF1, choosing **which node receives a new replica** (trading cold-start cost against future
co-location with partners and load spread) is a joint decision that per-task scoring and CD over task placement do not
capture.

## Precondition (checked first, no policy comparison)
From `kpa_scaleout_v1` and `physics_audit_v1` logs on R1 + WF1: scale-up events per minute and cold-start share of
latency. Proceed only if cold starts are ≥ 10 % of mean latency at the moderate rung; otherwise close as NO-LEVER
without training anything.

## Decision added
At each scale-up event, the policy chooses the node for the new replica among nodes with memory headroom. Task
placement is held fixed at CD for every arm, so only the replica-placement rule varies.

## Arms (replica-placement rules)
| Arm | Rule |
|---|---|
| RP-default | current: first eligible node with headroom |
| RP-least-loaded | node with lowest in-flight concurrency |
| RP-partner | node hosting the most partners of queued tasks of that type |
| RP-analytic | minimise (cold start on node) + Σ over queued tasks of that type of expected exchange cost given partners' current locations + expected queue wait from in-flight concurrency |
| RP-learned | GNN scorer over (pending tasks, partners, nodes), trained on offline labels |
| RP-learned-twin | same, message passing off |

**Offline labels.** At captured scale-up events (co-simulation snapshots, valid under I11), simulate each candidate
node for one stable window and label the best. Labels are offline only; no arm consults the simulator at decision time.

## Primary family (Holm)
RP-learned vs RP-analytic and vs best of {RP-least-loaded, RP-partner}, at moderate and heavy rungs.

## Predictions
1. Precondition passes at the heavy rung only (or heavy is unreachable and the node closes).
2. RP-partner beats RP-default by ≥ 5 %.
3. RP-analytic is within 3 % of the offline-label oracle; RP-learned does not beat RP-analytic.
4. RP-learned vs twin: tie.

## Outcomes
Any win is reported as its own finding on this node's bars; it does not reopen task-placement claims. A
"near-oracle quality at lower decision time" claim must be made against RP-analytic, the deployable rule.
