# access_link_contention_v1 — peer exchange through shared slow access links (hypothesis)

**Status:** `REGISTERED` as hypothesis (no runs). Conditional on: nodes 1–5 read, using the W3 link classes from
`workload_fix_v1` (WF1). Runs alone (never with `replica_placement_v1`). Plan: [`reference_physics_programme.md`](reference_physics_programme.md).


**Pre-run amendment (2026-10-07).** The W3 classes sit on every node's access links, servers included (see
`workload_fix_v1`), so the precondition measures link wait on the server links that carry peer exchange. With classes
on client links only, exchange would never cross a slow link and the precondition could not pass.

## Hypothesis
When peer exchanges take capacity on shared links (instead of being a computed delay), placements that each
look cheap in isolation can collide on one slow access link. A plan that accounts for these interactions
beats per-task exact-cost rules. This is the mechanism in SONIC's fan-out bandwidth bottleneck and in
collaborative-edge-computing work on contended flows.

## Relation to prior record
`link_contention_v1` (1.5 and 0.5 MB/s links) and `dag_fabric_contention_v1` (≤ 100 MB/s) were **offline**
label studies: contention was null or optimal plans routed around it. Neither tested a live scheduler, and
neither used heterogeneous access-link classes. This node tests the live seat; a null result here is expected
and is a valid outcome.

## Physics change (one change)
Route peer exchange through the existing link capacity resources (`network_fabric.py` `pipe()`), as task
ingress already does, under the pipelined transfer model. New flag (proposed): `HEROSIM_EXCHANGE_PIPES=1`.
Replay of R1 with the flag off must be identical.

## Precondition (checked first)
With W3 link classes on R1 + WF1, measure link wait share of exchange time under CD. Proceed only if link wait is
≥ 15 % of exchange time at the moderate rung.

## Arms
CD (with its cost model **unchanged**, so it does not see contention), CD-contention (cost model includes
current link occupancy), locality-first, GNN-eng and Twin-eng retrained on labels from this physics
(brute force on small batches), MLP-same.

## Primary family (Holm)
Best learned vs CD-contention, at moderate and heavy rungs.

## Predictions
1. Precondition passes only at the heavy rung with Wi-Fi/cellular clients.
2. CD-contention beats CD by ≥ 5 % where the precondition passes.
3. No learned arm beats CD-contention.
4. GNN vs twin: GNN ahead by < 5 % (direction only), the most favourable case for message passing in this programme.

## Outcomes
A learned win over CD-contention would be the first registered learned-performance result; it must then survive
a second topology draw before being reported as robust.
