# rollout_imitation_v1 — learn to beat the greedy by one step of policy improvement on the simulator

**Status:** `ACTIVE` (build started 2026-09-21) — triggered by [`joint_burst_v2`](joint_burst_v2.md)
closing `GNN-BEATS-GREEDY / CD-STILL-AHEAD`: the coordinate-descent greedy is now the ceiling, and
this lineage's one-step policy improvement is the pre-signed lever to reach it. Building the label
engine (forced-placement rollout over the real engine) and running the two blocking phases below
FIRST — Phase A (rank-stability) and A.2 (label-vs-rule gain) — before spending the ~90 CPU-h
labelling. Its label engine is a horizon return, which this record has found to be deterministic
chaos once (`objective_pivot_v1` Phase 2), so the rank-stability control below is a blocking bar,
not a diagnostic.

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

- 2026-09-21 — **Corpus build.** Minted 96 more training topologies (cc40s9225..9320, clone of the
  base cell with a new `network.topology.seed`; `rollout_imitation_v1_mint_cells.py`) → 120 total,
  because exploitable decisions are sparse (~a handful of usable per topology). Enriched the label
  engine to capture per-candidate score-term FEATURES ([drain, cold, exec, latency, exchange], the
  rule's own terms) alongside the rollout label, so each decision is a self-contained (features,
  label) example. Labelling all 120 topologies × 2 non-saturating rates (f700, f1000) via a
  throttled local submitter. Next: offline train-screen (`rollout_imitation_v1_train_screen.py`) —
  can a learned scorer on the rule's features learn the rollout label and beat the rule on realised
  cost across held-out topologies — which ORDERS the live gate (rule 6), then the live gate itself.
- 2026-09-21 — **R1 (Phase A.2) PASSES: `LABEL-DIFFERS-FROM-RULE`.** On the 440 pooled decisions,
  the rollout label differs from the rule's OWN choice on **305/440 = 69.3 %** (bar 10 %), median
  realised gain **451.6 s** where it differs (bar 0.3 s). The greedy is NOT one-step optimal: its
  one-pass drain estimate mis-ranks the two candidates on two-thirds of multi-candidate decisions.
  Note the gain is a horizon-GROUP-SUMMED cost (~100 tasks), so the magnitude is large by
  construction and the 0.3 s bar is trivially cleared; the differ FRACTION is the substantive read.
  Both blocking phases now pass → build the corpus, train `gnnedge0` (+ mpoff twin) on the rollout
  label, gate vs rule (primary), reactive, twin, and CD greedy (disclosed).
- 2026-09-21 — **R0 PASSES, powered: `LABEL-IS-A-PROPERTY-OF-THE-DECISION`.** Forced-rollout label
  engine built (`scripts_cosim/rollout_imitation_v1_label.py`, `datalab/rollout_imitation_v1_phaseA.sbatch`,
  `..._phaseA_read.py`) and run on the 24 training topologies at two non-saturating unburst rates
  (f700 + f1000). **440 pooled decisions, median Spearman(20,100) = 1.0, argmin(20==100) agreement
  0.886** (both bars cleared) — the group-local rollout label is NOT the horizon chaos that closed
  `objective_pivot_v1` P3. Build notes, disclosed: (a) the study's arrival rate saturates a 6-server
  training cell over the horizon (rule places onto an unreachable node, fails loud), so the label
  runs use non-saturating rates f700/f1000 — a rate choice for a well-defined cost, not the serving
  rate; (b) exploitable decisions are SPARSE and CONCENTRATED — most (task, state) pairs give the
  rule a single reachable initialised replica (no choice); multi-candidate decisions (exactly 2
  candidates) appear only after replicas warm (~pos 248) and only on a subset of topologies (9 of
  24 gave none); (c) a forced candidate can drive the truncated trace into the saturated regime —
  that decision is skipped, disclosed. Next: A.2 (R1, label-vs-rule gain, blocking).
- 2026-09-21 — Build started. `joint_burst_v2` closed with CD greedy as the ceiling, so the
  one-step-improvement label is the live lever. Order: build the forced-placement rollout label
  engine → Phase A (R0 rank-stability, ≥300 decisions, N∈{20,50,100}) → Phase A.2 (R1
  label-vs-rule gain). Both are blocking; only on a pass do the 4,000-decision corpus, training and
  gate run. Adding CD greedy as a disclosed gate arm (v2's honest bar) alongside the pre-signed
  R2-vs-rule primary.
- 2026-09-20 — Registered; build deferred until `joint_burst_v1` reads.
