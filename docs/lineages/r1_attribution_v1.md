# r1_attribution_v1 — train on R1 + WF1 and run the attribution battery

**Status:** `REGISTERED` (no runs). Depends on: `physics_audit_v1` (R1, including I11 pass),
`workload_fix_v1` (WF1), `load_recalibration_v1` (rungs). Plan: [`reference_physics_programme.md`](reference_physics_programme.md).
Revision 2026-10-08 (before any run): trains on WF1 instead of today's workload; candidate pruning declared.

**Precondition met (2026-10-08, before any run): I11 holds on R1.1 + WF1.** S5, job 843489, `rp/i11-wf1` `ac1915b1`
(audit drivers only on `97269192`). The replay builds the live topology, with W3's directional links and access classes, and
replays per-task types (W4). Cells: calibration 9601/9602/9607/9608 × ×0.2666 / ×11.61, CD, window g0, 6,000 arrivals each;
12 states from t ≥ 360 s in the first 3,000 arrivals plus 20 beyond them. **255 of 256 states: median 0.000 %, p95 0.126 %,
max 0.656 %, none above 5 %**; beyond 3,000 arrivals the p95 is 0.105 %. One miss is on the truth side (the isolated live run
scheduled the tasks at another instant) and stays in the denominator. Limits: one window; at ×11.61 the states reach only
t ≤ 1,059 s. Re-check 20 states at the recalibrated heavy rung, late in the trace, before labels are built there. Per-state rows:
datalab `simulation_data/workload_fix_v1/i11_wf1/cells/*/replay.jsonl`.

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
