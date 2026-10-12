# r1_attribution_v1 — train on R1 + WF1 and run the attribution battery

**Status:** `CLOSED` (2026-10-10) — **CD-FASTER**. Registered 2026-10-08; every bar below was signed before its data.

**Outcome.** On R1.1 + WF1 at 40c × 6s, no learned arm beats CD, which matches prediction 1. The stopping rule applies.
- **Primary family.** The best learned arm, gnn_eng_physmp (declared on validation), against CD on 19 test topologies:
  +3.42 / +5.90 / +5.14 % (light / moderate / heavy). It is faster in only 1 / 1 / 2 of 19 topologies, Holm p .0001 /
  <.0001 / .018, so CD is Holm-confirmed faster at every rung.
- **Message passing does not separate from its twin.** gnn_eng vs twin_eng: −0.19 / −0.07 / +1.00 %, not separated.
  gnn_eng vs set_transformer: also not separated.
- **Physics columns inside message passing help a little.** physMP vs gnn_eng: −0.04 / −0.72 / −1.91 %, Holm-confirmed
  at moderate and heavy.
- **Against the same-input MLP, the GNN wins at light and loses at heavy:** −0.74 % at light, +1.68 % at heavy.
- **Seeded search wins, but MP is not the reason.**
  - CD←GNN beats CD: −0.38 / −1.87 / −1.66 %, all Holm-confirmed.
  - CD←GNN ties CD←Twin, so this is a learned-seed win, not a message-passing win.
  - CD←random is slower than CD: +1.35 / +4.28 / +3.91 %.
- **Other arms:**
  - Random placement (fixed, 454 of 456 cells): +32.8 / +119.6 / +154.9 %.
  - Knative (the queue-keyed arm, blind under replica release; gate-tools 2026-10-11): +26 % at light and collapses at
    moderate and heavy. The concurrency-key arm does not collapse (`accel_nosplit_v1`).
  - CD decides 10× faster than any learned arm (prediction 7).
- **gnn_raw addendum (S1, does not change the verdict).** Message passing helps the raw-plan arm: gnn_raw vs twin_raw
  −4.75 / −4.62 / −7.99 %, Holm-confirmed at every rung (19/17/18 of 19). Prediction 2 holds in direction, below its
  8–24 % range at light and moderate. Both raw arms are far slower than CD: gnn_raw +11.3 / +20.6 / +19.9 %.
- **Successor:** `scale_160_v1` tests the same question at 160c × 24s.
- **Read:** `r1_attribution_v1/gate_read_primary_2026-10-10.json`, from S4's `rp/r1a-gate` a6713375. 5,444 summaries;
  28 failed (rate- or projection-killed, excluded and counted).

Depends on: `physics_audit_v1` (R1, including I11 pass),
`workload_fix_v1` (WF1), `load_recalibration_v1` (rungs). Plan: [`reference_physics_programme.md`](reference_physics_programme.md).
Revision 2026-10-08 (before any run): trains on WF1 instead of today's workload; candidate pruning declared.

**Pipeline, corpus and gate record (2026-10-08 → 2026-10-10), moved to [`r1_attribution_v1/pipeline_and_gate_record.md`](r1_attribution_v1/pipeline_and_gate_record.md)** (2026-10-12). It holds the I11 preconditions, B2 dry runs and fixes, the ranking tests, replay variants D/E, the production corpus at E (pin e647446f), training, selections, every arm's gate read and the gnn_raw addendum. The verdict and its numbers are in the head above.

**Pipeline readiness and pre-run amendments (2026-10-08, coordinator, before any corpus exists; S6's check at
`rp/wf1-corpus-check` `7c924b8a`, smoke data only on calibration topologies 9601 and 9607).**
- **Corpus route: live capture → `make_warm_corpus` → `executecosimulation` → `prepare_graphs_cache`** (the
  `small_batch_v1_capture` pattern). It carries WF1 with the inputs: single origin, wf1_v1 payloads, directional access
  links, four types, the R1.1 physics env, and the 300 s timeout in the stats. `generate_gnn_datasets_fast.py` grids aren't used:
  they have no live snapshot seed, no arrival process, no WF1 payloads or access classes, and draw origins per task.
- **Label: the plain brute-force optimum (default label) for every arm.** The `rtt_drift` shaping reads
  `drainable_objective_v1/drain_table.json`, measured on 2026-09-14 physics, with no entries for rf/cnn. Re-measuring it would
  add a second calibrated object, and the node's label was always the brute-force optimum.
- **F1, a feature-direction bug under W3, fixed before training.** The partial-state exchange columns price the
  candidate → peer transfer (`prepare_graphs_cache` node_exchange via `route_hops_and_bottleneck(a, b)`, read in
  `reduced_features.py`, and the same in `prefix_serving.py`). The simulator charges the task the pull peer → candidate and
  the partner the reverse. On 9601 a cellular candidate reads 4 MB/s in the feature against 75 MB/s pulled (18.75×). Fix:
  carry both directions under a new partial-state contract version; train/serve parity is required before any training.
- **F3, four task types visible to every learned arm.** Corrected by S6 the same day: under partial_state v3/v4 the
  task's own type is recoverable from the krank block (`KRANK_TYPES=4`), and graph arms also get `task_type_onehot4`.
  What is blind: the legacy 3-dim task block (rf/cnn rows all zero) and the platform replica flags (has_dnn1/has_dnn2
  only). Both are widened, with columns appended so no index moves.
- **The new contract, `partial_state_v5`** (S6's design, accepted): v4's 25 columns, with columns 7–8 = the pull peer →
  candidate in seconds (log1p) and column 23 (committed service) on the pull direction; plus 2 appended columns, the reverse
  candidate → peer committed exchange and peer mass (width 27). Old contracts stay byte-identical.
- **`partial_state_v5` built and checked (S6, `rp/r1a-features` `95e6cea6`).**
  - v3/v4 are byte-identical on 400 real graphs and on whole smoke caches, and the test failure set is unchanged.
  - 11 tests pass, among them the real 9601 W3 pair: pull 75 MB/s, push 4 MB/s.
  - The loader refuses a contract mismatch in both directions.
  - Train/serve parity holds on 6 WF1 smoke datasets. The only differences are in `peer_norm` on single-node candidate sets, and v5 doesn't use it.
  - Not yet exercised: training. Missing: v5 arm configs and gate kinds, and an MLP-same layout. **MLP-same and Set-transformer have no implementation in the tree; both must be built before the gate.**
- **Physics env in the cache (decided).** `node_exchange` depends on `HEROSIM_TRANSFER_MODEL` (hops counted once under pipelined),
  and the cache recorded neither. A first v5 cache built without the R1 env had exchange 3–4× off, caught only by parity.
  From now on the cache writes TRANSFER_MODEL, REPLICA_RELEASE and SCALEOUT into its metadata, and the trainer and serving refuse a
  mismatch.
- **Single-candidate-node batches are filtered at corpus build** (there's nothing to decide, and `PartialStateContext` raises on them),
  and they're counted in the B2 dry run.
- **B3 cause: KPA's first pass in a co-sim episode.** The KPA process runs its scale-down pass before its first wait, so at
  t0 of every episode it removes idle replicas that are candidates in the forced plans. That made 3,542 of 5,600 plans fail on ds_00002;
  under legacy scale-out it's 5,600/5,600. A live decision is made before the next tick, so this is a restart-order artifact.
  **Decision:** a co-sim episode starts KPA at the live tick phase carried by the snapshot (or, if the snapshot lacks it, at
  one reconcile interval), never with a pass at t0, under the same autoscaler settings I11 validated. A plan that still
  fails discards its dataset, and the discard rate is reported. Before any label is built, I11 is re-checked on 20 states
  with the change, and must show no change. Pruning out candidates and silently discarding were rejected: the first biases the
  plan space, the second the snapshots.
- **Amended the same day: the corpus co-sim must run the autoscaler I11 validated.** S6 found three differences between
  the recipe and `i11_replay --params live`:
  - KPA target 100 (the legacy constant) against 0.7;
  - tick interval 1e12 (stretched for cost) against 1 s × time scale;
  - no `HEROSIM_SNAPSHOT_FIDELITY` at capture, so there's no KPA window history and no tick phase.

  A label made under other autoscaler settings is a different physics, and isn't covered by I11. **Decision:** captures set
  `HEROSIM_SNAPSHOT_FIDELITY=1`, and co-sim episodes use `live_run_params()` (target 0.7, 1 s ticks, keep-alive, start
  phase from `next_wake`). That replaces the t0 rule above. Its cost is measured on the smoke sets and the B2 dry run before the
  capture. If 5,000 batches don't fit in about 2 days of datalab, the episode is cut once the batch's last task completes,
  and that cut is itself checked against live, I11-style (20 states, p95 ≤ 1 %), before it's used. The stretched, frozen-autoscaler
  variant is rejected.
- **Cost measured: no cut needed.** S6 recaptured the 6 smoke sets with fidelity and replayed the full candidate product
  under `live_run_params()`: 21,989 plans in 3.5 min on 28 workers, 0 failures, ds_00002 5,600/5,600. Each plan takes 0.07–0.24 s
  (p95 ≤ 0.5 s), over a horizon of 77–214 s. That projects to about 1,070 core-hours for 5,000 batches (about 17 h on 62 cores). Measured on
  the ×11.61 snapshots only.
- **Dataset structure and label (decided).** With fidelity, the snapshot carries the already-queued tasks (28–114 per
  snapshot against 4–6 batch tasks). They are **state, not decisions**: their placements stay as captured. The decision is
  the batch's placement, and **the label is I11's quantity, Σ over the batch's tasks of (done − scheduled)**. Arrival to
  decision is the same for every plan, so this ranks plans exactly as end-to-end latency does, and it is the quantity I11
  validated (median 0.000 %, p95 0.126 %). The corpus replay must call the same code path as `i11_replay --params live`.
  If it does, S5's I11 run covers it and no separate 20-state check is needed.
- **Fidelity corpus wired (S6, `rp/r1a-features` `23bce622`).**
  - **Same code path as I11:** `src/placement/fidelity_replay.py` calls the same `execute_simulation` with
    `live_run_params()`, mirroring `i11_replay.py:245–360`, and refuses mismatched params. An independent harness
    re-simulated 135 labels with max |diff| 1.8e-13.
  - **Identity on 3 live cells:** unchanged; only `total_rtt_plus_inference` differs, because it adds wall-clock time.
  - **Smoke:** 6/6 sets complete (ds_00002 5,600/5,600), and the cache builds under v5 with the R1 env.
  - **Single-candidate-node filter:** rejects 0/3 snapshots on 9601 and 6/16 on 9607. The old route's capture sim had
    also shrunk the candidate slate, so the earlier 9607 degeneracy was partly that artefact.
  - **Still owed before training: a fidelity-aware train/serve parity check.** Replay the batch decision in the live
    scheduler with the queued state, and compare its features with the cache's, attribute by attribute.
  - `HEROSIM_SNAPSHOT_FIDELITY=1` is required at capture and at corpus build.
- **Fidelity-aware parity (S6, `38186d45`, jobs 843602/843603): passes on 6/6 smoke sets.** Two are identical. Four differ only on
  platforms holding an in-flight task the replay resumes: those tasks aren't exposed as `platform.current_task` inside the replay,
  so a builder there reads zero remaining work (≤ 1.8e-3 normalised, backlog ≤ 1.9e-2 s). The cache now carries the snapshot's own
  in-flight terms, which is what real live serving sees. Accepted: the replay is the blind side, not the cache.
  Fixed in the same pass:
  - replica flags for types outside the batch had been 0 in training and 1 in serving, which also affects the old so1load corpora;
  - batches holding several disconnected peer groups are rejected and counted (`--require-connected-batch`), because the
    serving scheduler would split them.
- **Candidate slate: the declared pruning, applied identically at corpus build and at serving (decided).**
  `make_warm_corpus` drew a balanced random slate (≤ 20,000 plans), and live serving offers every reachable replica. Neither
  matches the node's declared rule. Both now use it: each task's top 5 candidates by exact standalone cost (CD's cost model),
  with sub-batches of ≤ 4 tasks when a batch exceeds 100,000 plans. The node's pruning ablation (labels on 300 small batches;
  the best arm served with and without pruning on 6 test topologies) stays the check on this rule. Cost is re-measured in the
  B2 dry run.
- **Pruning built (S6, `f6a0c1cd`, `src/placement/declared_slate.py`, one definition for both sides).**
  - **The rule:** standalone cost = drain + cold + exec + latency, the CD/peer-greedy base score
    (`peer_greedy_network/scheduler.py:283`), tie-broken by (cost, node, platform).
  - **Serving:** `GNN_SERVE_CANDIDATE_SLATE=declared_pruning_v1`, recorded in the sidecar and refused on mismatch both ways.
  - **Parity:** passes 6/6 in this mode and with a forced sub-batching cap; labels re-simulate to 1e-13.
  - **Disclosed approximations:**
    - `cold` follows the snapshot's initialized flag rather than the live warm-by-previous-task test;
    - in a sub-batch, siblings outside the chunk are absent from the replay, because the capture records no `chosen` placement.

    If more than 10 % of production batches are sub-batched, the sibling rule is revisited before training.
  - Also counted from now on: `no_choice` rejections (fewer than 2 candidates for most tasks, or a plan space of 1).
- **Topologies (decided).** All share the 9473–9568 generator template (40 clients, 6 servers).
  - **Train:** 9201–9296 and 9101–9124 (120).
  - **Held-out validation:** 9297–9320 (24).
  - **Never used:** the 19 test topologies and the calibration set (9601, 9602, 9607, 9608). The rungs were fitted there, so
    training on them would be in-sample for the load definition.
  - The 77 unselected pool ids aren't used.
  - Before capture, every training and validation id passes the live-topology feasibility check
    (`workload_fix_v1_reachability_live.py`). Infeasible cells are dropped and listed.
  - **Result (S6, `96a09da4`):** 41 of the 144 are infeasible in all four windows, each one because a sending client can't
    reach a dnn2-capable server. That's 34 train and 7 held-out ids; the list is in datalab
    `workload_fix_v1/corpus_prod_prov/reach_live_144_infeasible.json`. **Decision: they're dropped and the lists aren't
    extended.** That leaves train 86 and held-out 17, which is enough for about 5,000 batches. Inputs for the 103 are built
    (W3 diff 206/206 ok), and they're rebuilt at the recalibrated rungs once those exist.
- **F2:** the link-graph feature reads `bandwidth_mbps` = min(out, in). It's fixed (out and in as two features) only if a
  registered arm reads the link graph; otherwise that's recorded as a limit.
- **B2, a degeneracy check before the 5,000-batch capture.** In 6/6 smoke datasets the optimum put the whole group on
  one node with zero exchange. A 100-dataset dry run, from the recalibrated rungs once they exist, reports the share of
  optima with zero exchange or one node and the distribution of plan counts. **If more than 80 % of optima are
  single-node with zero exchange, the node pauses before training,** because the label would then teach one move that
  a hand rule already makes. The dry run also diagnoses the incomplete sweep (B3: 2,058 of 5,600 rows) and checks queue
  seeding under R1.1.

## Questions
- Q1 (performance): does any learned arm, trained on R1 + WF1, beat CD there?
- Q2 (attribution): where learned arms differ, is it message passing, relational features, or set context?
- Q3 (hybrid): does a learned seed improve CD, and is any gain from message passing?

## Labels (fixed now)
Brute-force optima from co-simulation under R1 + WF1 (valid only after I11 passes), same capture protocol and batch
sizes as the original corpus (batches of ≤ 10 tasks). One corpus, shared by every learned arm; size matched to the
original (~5,000 batches) and recorded. Train/validation split by topology; the 19 test topologies are excluded.

**Candidate pruning (declared before labels exist).** Eager scale-out can raise candidates per task beyond what brute
force can enumerate. If a captured batch has more than 5 candidates for any task:
- keep each task's top 5 candidates ranked by that task's exact standalone cost (cost model shared by CD and the
  simulator), and enumerate the pruned plan space;
- if a batch still exceeds 100,000 plans, split it into sub-batches of ≤ 4 tasks.
Record the fraction of pruned batches and of sub-batched batches.

**Pruning ablation (part of this node).** On 300 captured batches small enough to enumerate unpruned, report the
optimality gap of pruned vs unpruned labels; and at serving time, evaluate the best learned arm with and without the
pruning rule on 6 test topologies. If the label gap exceeds 2 % (median), the node is paused and the rule amended
before training.

## Arms
| Arm | What it isolates |
|---|---|
| CD, locality-first, one-pass greedy, self-predict | classical reference (Knative as context) |
| GNN-eng | engineered relational features + message passing (retrained `so1load` recipe) |
| Twin-eng | as GNN-eng, message passing off |
| MLP-same | same inputs, no graph |
| GNN-raw / Twin-raw | no hand-built relational features, MP on / off |
| GNN-eng-physMP | GNN-eng with physics columns live inside message passing (`NEAR_RTT_MP_BIPARTITE_EDGE_ATTR_ZERO=0`) |
| Set-transformer | same inputs, all-pairs attention over the batch, no graph edges |
| CD←GNN, CD←Twin, CD←random | CD started from each plan (random start is the control for the "worse basin" reading) |

**Fair-comparison budget (after Errica et al., ICLR'20):** every learned arm gets the same hyperparameter search
budget (6-configuration grid, same epoch cap, same early stopping, 3 training seeds). The configuration per arm is
chosen on validation topologies only, and the "best learned arm" for the primary family is declared from validation
before any test read.

## Cells
19 test topologies × 3 recalibrated rungs × seeds as in `transfer_physics_v1`.

## Primary family (Holm, 3 tests)
Best learned arm vs CD at each rung.

## Secondary families (Holm within each)
- S1 attribution: GNN-eng vs Twin-eng; GNN-raw vs Twin-raw; GNN-eng vs MLP-same; GNN-eng vs Set-transformer;
  GNN-eng-physMP vs GNN-eng (5 contrasts × 3 rungs).
- S2 hybrid: CD←GNN vs CD; CD←GNN vs CD←Twin; CD←random vs CD (3 × 3).

## Predictions
1. Q1: no learned arm wins against CD at any rung.
2. S1: GNN-raw beats Twin-raw (old 8–24 % pattern, smaller on R1); GNN-eng ties Twin-eng.
3. S1: GNN-eng beats MLP-same at moderate and heavy rungs, by less than on old physics.
4. S1: Set-transformer ties GNN-eng.
5. S1: physics-live MP within ±3 % of GNN-eng.
6. S2: CD←GNN does not beat CD; CD←random ≤ CD←GNN.
7. Decision time: CD ≥ 10× faster per task than any learned arm.

## Outcomes
- Q1 win anywhere → registered result; checked again in `call_graph_pairing_v1`.
- Q1 no win → stopping rule applies; S1 and S2 form the paper's attribution section.
- If prediction 2 fails, "message passing substitutes for feature engineering" is old-physics only and labelled so.

## Cost
Corpus capture and co-simulation under R1 + WF1 (largest item); about 126 trainings (7 trained models × 3 seeds ×
6 configs); evaluation ≈ 15 arms (5 classical incl. Knative, 7 learned, 3 seeded-CD) × 19 × 3 rungs × seeds.
