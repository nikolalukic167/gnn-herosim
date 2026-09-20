# rollout_imitation_v1 — learn to beat the greedy by one step of policy improvement on the simulator

**Status:** `REGISTERED` (2026-09-20) — design and bars signed; **not built** until
`joint_burst_v1` reads, because it shares that lineage's capture and its answer decides whether a
better label is the remaining lever. Its label engine is a horizon return, which this record has
found to be deterministic chaos once (`objective_pivot_v1` Phase 2), so the rank-stability control
below is a blocking bar, not a diagnostic.

**Parents:** [`peer_greedy_live_v1`](peer_greedy_live_v1.md) (the rule to improve on),
[`joint_burst_v1`](joint_burst_v1.md) (the served-distribution corpus), `objective_pivot_v1` (the
chaos control).

**Question.** The immediate rule is a one-step greedy on the physics: it prices a candidate by
the platform's drain and the exchange to partners already placed, and ignores the rendezvous of
queued tasks and the arrivals about to land. One step of policy improvement — for each decision,
the candidate whose *realised* group-local cost is lowest when the simulator is rolled forward
with the rule — is at least as good as the rule by construction. Can a graph model learn that
improved policy and serve it per arrival, with no rollout at serving time, and beat the rule?

**Design.**
- **States.** The rule's own live runs on the 24 training topologies (unburst, the study's
  arrival structure), one snapshot per decision at a stride; ≥ 4,000 labelled decisions.
- **Label.** For decision (task i, candidates C): for each c ∈ C re-run the trace from t = 0
  with the rule and task i forced to c (a forced-placement hook in `PeerGreedyNetworkScheduler`,
  the same mechanism `src/policy/determined` uses), truncated N arrivals after i; cost =
  elapsed of i + Δelapsed of i's partners + Δelapsed of the tasks queued behind i on that
  platform within the horizon. Label = argmin. Horizons N ∈ {20, 50, 100}.
- **Chaos control (blocking, Phase A).** Over ≥ 300 decisions, the label's candidate ranking at
  N = 20 vs 50 vs 100: median Spearman ≥ 0.80 and the argmin agreeing on ≥ 80 % → the label is
  a property of the decision; otherwise the label is downstream chaos and the lineage closes
  `LABEL-IS-CHAOS` with the measurement, nothing trained (as `objective_pivot_v1` P3 did).
- **Phase A.2 (blocking).** The label must differ from the rule's own choice on ≥ 10 % of
  decisions with a median realised gain ≥ 0.3 s where it differs; otherwise the greedy is
  one-step optimal at this rung and the lineage closes `GREEDY-IS-ONE-STEP-OPTIMAL`.
- **Model and serving.** `gnnedge0` on 1-task batches with the prefix block carrying the placed
  partners (the serving path's `peers_outside_batch` handling), per-arrival serving, no wait;
  MP-OFF twin. **Gate:** vs the immediate rule (primary, checkpoint unit, rule broadcast), vs
  Knative, vs the twin, on `unsaturated_edge_v1`'s 16 environments per rung.

**Bars.** R0 rank-stability (above); R1 label-vs-rule gain (above); R2 model vs rule
`MODEL-BEATS-RULE` / `RULE-FASTER` / `NOT-SEPARATED`; R3 vs reactive; R4 vs twin; the chain's
5 % / p < 0.05 / n = 16.

**Registered expectation (signed 2026-09-20).** R0 passes 50 % (the group-local cost is far
less exposed to downstream chaos than a full horizon return, and it is still a rollout). R1
passes 60 %. R2 MODEL-BEATS-RULE 30 %.

**Cost.** Labelling is the expensive part: ~4 forced runs per decision at ~20 s each; 4,000
decisions ≈ 90 CPU-hours, ~2 h at 48-wide. Then the `joint_burst_v1` training and gate recipe.

## Record (newest first)

- 2026-09-20 — Registered; build deferred until `joint_burst_v1` reads.
