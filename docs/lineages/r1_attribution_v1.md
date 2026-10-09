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

**Late-state re-check at the recalibrated heavy rung: PASS (2026-10-09; S5, jobs 843776 and 843791, `rp/recal`
`293fd8c4`).**
- **Rung:** ×11.6139 (`load_recalibration_v1`), using the finals' cells, window g0, full 50,000-arrival captures.
- **Last third, 20 states (5 per calibration topology, t ≥ 6,270 s):** median 0.000 %, p95 0.033 %, max 0.578 %. The max
  is an absolute miss of 0.001 s. Nothing is above 5 %, and every state replayed.
- **Backlog states:**
  - None exist in the last third: the largest replica lock backlog there is under 100 s, and the run-wide maxima of
    75–518 s all fall in the first 1,300 s.
  - A separate set of 24 states with backlog > 100 s (9607 and 9608) scores max 0.075 %. It includes a batch that waited
    1,935 s live, reproduced within 0.029 %.
- **Data:** `simulation_data/load_recalibration_v1/i11_heavy{,_bl}/`.
- **Unverified, open:** against the *continuing* live run, as opposed to the isolated truth, two early 9608 states
  differ more: 4.603 s live against 4.110 s replayed, and 237.5 s against 230.7 s. Most likely this is the batch-label horizon
  (arrivals after the snapshot aren't in the label) rather than a replay error, but the cause has not been checked.

**B2 dry run, heavy portion submitted (2026-10-09, S6).** Rung ×11.61; 12 topologies (8 train, 4 held-out); inputs job 843775
(W3 diff passes; access classes wired 0.40, wifi 0.39, cellular 0.21); capture job 843881 and build job 843882 (afterok, LIMIT=4),
with fidelity capture, `--require-connected-batch` and declared pruning. Moderate (×5.94) is held until its finals finish.

**Pre-production gate (2026-10-09, coordinator).** The ~17 h production capture launches only after two things: the B2 dry-run
read, and an independent read-only bug audit of the pipeline by S7 (label, physics flags, train/serve parity, determinism,
silent drops, cost).

**Pipeline audit (2026-10-09; S7, read-only at `rp/r1a-features` `96a09da4`; one test, job 843910).**
- **Result:** no blocker for the capture, but seven fixes are required first.
- **Confirmed by reading:**
  - A truncated sweep aborts the build, then slips back in on resume, because the resume skip and the cache loader
    check only that files exist, not `sweep_complete`.
  - A partial capture passes as complete, there is no per-run timeout, and an `afterok` build would never start.
  - LIMIT=90 gives about 111k datasets, not 5,000; snapshots are not topped up; shard drops are not counted.
  - `COSIM_PLACEMENT_TIMEOUT_S` never fires.
- **Confirmed by run:** the cache is not bit-reproducible. Winner ties are picked in worker order, which permutes 24 of
  229 platform rows; labels are unaffected.
- **Suspected:** t < 360 s states and the light and moderate rungs had no I11 check.
- **Passed:**
  - the label is I11's quantity, with queued tasks as state;
  - the physics flags and KPA phase match live;
  - v5 features and the declared pruning use one function on both the train and serve sides;
  - placements, best and workload rebuild identically;
  - cache-metadata refusal works.
- **Cost:** recomputed with pruning on 12,614 B2 heavy snapshots, about 450 core-hours, about 7 h on 62 cores for
  5,000 datasets. That excludes capture time and the heavy tail.
- **Decision:**
  - S6 fixes items 1–7: discard and count truncations; require `sweep_complete`; a capture sentinel, a per-run timeout
    and `afterany` plus a count check; target 5,000 batches with top-up; deterministic tie-breaks and platform order;
    `--min-time 360`; a real placement timeout; plus hygiene.
  - S7 reviews the diff and reruns the A/B test, which must come out bit-identical.
  - S5 runs I11 at light and moderate.
  - Production waits for all three.

**I11 at light and moderate: PASS (2026-10-09; S5, job 843926, `293fd8c4`, window g0, 4 calibration topologies,
50,000 arrivals each; `simulation_data/load_recalibration_v1/i11_lm/`).**
- **Light ×1.2584:** uniform, 20 states: median 0.000 %, p95 0.081 %, max 1.77 % (an absolute miss of 0.002 s). Last
  third, 20 states: max 0.40 %. No backlog > 100 s states exist.
- **Moderate ×5.9402:** uniform, 20 states: max 0.032 %. Last third, 20 states: max 0.44 %. Backlog, 10 states (123–194 s,
  first ~500 s only): max 0.010 %.
- **Open:**
  - One reported figure is inconsistent (881 s against 866 s, quoted as 0.006 %); I asked S5.
  - The continuing-vs-isolated gap recurs, up to 5 % (14.09 against 13.38 s). S5 is testing whether it's the label
    horizon, meaning post-snapshot arrivals.

**Fixes committed (2026-10-09; S6, `rp/r1a-features` `c1fb0185`; 29/29 tests).**
- **Sweeps:** a truncated or failed sweep is discarded and counted. One `sweep_status.py` (which requires
  `sweep_complete` and rows = `num_placements`) gates resume, cache and check.
- **Capture:** a `.done` sentinel, a per-run timeout, and a sentinel check before the build.
- **Volume:** 5 batches per cell (about 6,180), with sub-batches counting toward their parent, top-up, and fail-loud on
  a shortfall.
- **Determinism:** plan-lexicographic tie-breaks and sorted platform rows.
- **Build window and timeout:** `--min-time 360`; a stall watchdog for the placement timeout.
- **End check:** volume, discards ≤ 2 % and fidelity.
- **New finding:** the 300-snapshot cap at stride 31 stopped at task 37–41k of 50k in the B2 heavy captures, so the last
  fifth of each trace went unsampled. The cap is now 600 and fails loud when reached. The B2 heavy dry run (843881)
  predates the fix, so its read is disclosed as covering about the first 80 % of each trace.
- **Pending:** determinism A/B (843932), a 2-topology × 3-rung smoke (843934–843937), and S7's review. Production
  waits for all of them.

**Label horizon (2026-10-09; S5, job 843931; `load_recalibration_v1/gap.json`).** The continuing-vs-isolated gap is the
arrivals after the snapshot, exactly.
- On the 5 largest-gap states (2–22 %), admitting arrivals with time ≤ t + w closes the gap to below 0.001 % once the
  window covers the arrivals that overlap the batch (1–20 s, 2–152 arrivals). At w = 0 it reproduces the isolated
  value.
- No other residual shows at the printed digits. That is inferred from the exact match, not read directly.
- The isolated label is therefore a clean lower bound that omits the cost later arrivals add.
- **Decision, thresholds signed before data.** S5 builds a ranking test that replays the top isolated plans with
  post-snapshot arrivals admitted, placed by the capture policy.
  - Capture can proceed.
  - The **label build** waits if the isolated argmin's median regret under that evaluation is > 1 %, or argmin
    agreement is < 80 %.

**B2 heavy capture: starved-client spin (2026-10-09 10:45; S6, job 843881).**
- **Hung cells:** 8 of 48 heavy cells (17 %) spin on "No compatible hardware available for dnn2" from sim time ≈ 990 s,
  under the capture policy `peer_greedy_network_batch`: 9103 g0, 9105 g2/g3, 9202 g3, 9203 g1, 9298 g2, 9300 g3,
  9302 g3. Their snapshot counts creep (36–231).
- **The other 40 cells** hit the 300 cap at task ~38–41k of 50k.
- **Decision:**
  - The B2 build moves to `afterany` and the 7 slow tasks are cancelled.
  - Spin cells are excluded from the B2 read, named, and counted.
- **Production rule, signed now:**
  - A hung cell is excluded and counted.
  - If more than 5 % of a rung's cells hang, production pauses for diagnosis. That is already triggered for heavy.
  - S5 diagnoses the spin: release/scale-out, a reservation defect, W3 reachability, or legitimate overload.

**Fix review (2026-10-09; S7, jobs 843939 and 844128).**
- **Accepted at c1fb0185:** findings 1–4, 6 and 7.
- **Finding 5:** `--min-time 360` is accepted. Its I11 part is closed by S5's job 843926.
- **A/B:** cache tensors, `placements.jsonl` rows, `best.json` and `workload.json` are all identical. `optimal_result.json`
  differs only in wall-clock `schedulingTime` and end-of-run list order, neither of which the cache reads.
- **Snapshot cap:** 600 covers heavy, which projects 385–402 snapshots. Light and moderate are not projected.
- **Two new defects, to fix before production:**
  - resume doesn't count skipped datasets toward `--limit-batches` (confirmed: a second pass built 2 more);
  - cap-hit cells are used silently.
- **Fixed in `31d8620b`** (resume pass 845772: a resubmitted cell builds 0; determinism file passed, 4/4 graphs,
  0 differences). Cap-hit cells now fail the volume check.
- **`2a7c85c9` adds a selection rule, `unplaced_partner`.** It rejects a batch whose out-of-batch peer partner was not
  placed at capture, because the replay refuses it (`simulation.py:799`). That was the cause of B2 9101 ds_00400's
  empty "skipped" sweep.
- **Threshold, signed now:** if more than 10 % of offered batches at any rung are rejected as `unplaced_partner`,
  production pauses and the rule is revisited. Serving still sees those batches.

**B2 dry run, heavy read (2026-10-09; S6, build 843882 at `96a09da4`, PER_CELL=4, 300-snapshot cap;
`corpus_b2_heavy/optima_read.json`).** Neither line trips, so there is no pause.
- **Scope:** 38 of 48 cells. The 8 hung spin cells are excluded, and so are 9101 g2/g3, which never ran after task 0
  died on the unplaced-partner refusal.
- **Datasets:** 151 complete sweeps from 143 batches, 0 discarded.
- **Rejections:** 31 of 182 offered (17 %): single_candidate_node 6.0 %, no_choice 10.4 %, disconnected 0.
- **unplaced_partner,** computed offline with the same predicate (`c5b9a8c6`): 0.46 % of 11,103 offered, against the
  10 % line.
- **Pruning:** 51.7 % of batches pruned, **4.2 % sub-batched** (against 10 %).
- **Optima:** **single-node with zero exchange 66.2 %** (against 80 %). Optima span 2, 3 and 4 nodes in 42, 5 and 4
  datasets.
- **Plan counts:** median 75, p90 27,648, max 100,000.
- **Thin:** 11 topologies, heavy only, the first 75–80 % of each trace. Light and moderate come from the smoke.

**Final fix check (S7, job 846492, `2a7c85c9`): all three accepted, no new defect.**
- **Resume:** the second pass builds 0.
- **Cap-hit:** a cap-hit cell fails the volume check (exit 1).
- **`unplaced_partner`:** agrees exactly with the replay guard on 12,073 B2 heavy snapshots; 1.4 % of snapshots
  (S6's 0.46 % uses offered batches as the denominator).
- **Pipeline code is cleared for production** at `2a7c85c9`. The capture still waits for the spin diagnosis (the heavy
  hung-cell rate is 17 %, against the 5 % line).
- **The spin is not specific to the capture policy (2026-10-09 11:00).**
  - CD 9607 g1 at ×5 (S4's job 843900, old-ladder context) busy-loops at 99.7 % CPU and 3.5 GB RSS for 43 min,
    against 44–54 s for its siblings, with an empty log. The same cell finishes at ×5.94.
  - CD 9608 g1 at ×0.2666 hit the 2,700 s wall limit in both passes; likely the same failure, unconfirmed.
  - S6's smoke has 9298 heavy g1 spinning, with a request_timeout signature.
  - S5 diagnoses the spin on the CD calibration cells first.

**Smoke (2026-10-09; S6, jobs 843935–843937, `2a7c85c9` lineage; 9101 and 9298 × 3 rungs × g0/g1).**
- **Sentinels:** 12/12 ok, no cap hit, coverage to task 49.5–49.9k. 9298 heavy g1 finished on its own after 1 h 34 min,
  so the hung-sentinel path is still unexercised; S6 is forcing it.
- **Build:** 12 batches, 14 datasets, 0 discards, 0 shortfall.
- **Rejections:** 5 of 19 (no_choice 21 %).
- **Sub-batched:** 8.3 %.
- **Single-node, zero-exchange optima:** 11 of 14 (78.6 %, n tiny); the per-rung split is requested.
- **Fidelity check:** 0 problems.
- **unplaced_partner:** 0.42–0.59 % per rung.
- **Per rung, single-node zero-exchange:** light 4/4, moderate 4/4, heavy 3/6. no_choice: light 1/5, moderate 3/8,
  heavy 0/6. n is 4–6, so these are not readings.
- **Hung path proven** (jobs 849836–849838, CAPTURE_TIMEOUT_S=300): the sentinel reads hung, the build runs afterany and
  skips the cell, and the volume check lists it.
- **Decision (2026-10-09, before any light or moderate B2 data):**
  - The B2 dry run extends to light and moderate on the same 12 topologies and the same lines, at the merged R1.1-T-fix
    code, with heavy re-run there too.
  - **If a rung's single-node zero-exchange share exceeds 80 %, production pauses for that rung** and the coordinator
    decides, recorded before production, whether it stays in the corpus.

**Pipeline at the R1.1-T fix (2026-10-09; S6, `rp/r1a-features` `d28d1bb0` = merge `a5806044` + a snapshot change).**
- **Tests:** 44 pass. Determinism A/B (job 850069): 0 differences, and a resubmit builds 0.
- **Snapshot change.** `snapshot_fidelity` records a planned-then-deferred partner as unplaced (`peer_node_name` uses
  `peer_is_placed`), so the replay matches the live rendezvous state.
- **S7 accepts it.** It can't mark a placed partner unplaced. `unplaced_partner` rejections rise by exactly the batches
  with such a partner; the size is measured per rung in the extended B2, against the 10 % line.
- **Extended B2** is ready: 12 topologies × 3 rungs, PER_CELL=4, target 576. It waits for S5's I11 spot check at
  `d28d1bb0`.

**Arms built (2026-10-09; S7, `rp/r1a-arms` `b454377f`, smoke job 850090).**
- **New arms:** MLP-same (flat per-edge vector, no graph) and Set-transformer (SAB over tasks, ISAB over candidates,
  mean attention). Both read edge columns through the shared `gather_scorable_edges`, whose refactor is bit-identical on 4
  configs.
- **Configs:** 7 arms × a 6-point grid (lr {5e-4, 1e-3, 2e-3} × width {64, 128}) plus smoke variants.
- **Sidecar:** carries `arm_kind`, and serving refuses mismatches.
- **Determinism:** smoke-trained twice per arm, bit-identical for all 7. 27 new tests pass.
- **Parameter counts differ** by arm design (8k–249k at width 64). Disclose with the read.
- **Open:**
  - merge the R1.1-T fix;
  - set `HEROSIM_INFLIGHT_CAPTURE` for the `ra_*` kinds from cache/serve feature parity, not by guess;
  - one servesmoke per arm.

**I11 at the fix: PASS (2026-10-09; S5, jobs 850058 at `c8eb73c2` and 850086 at `d28d1bb0`, numbers identical).**
- Moderate 9602 g0 (CD, the cell the fix moved most), 20 states: median 0.004 %, max 0.19 % (an absolute miss of 0.002 s).
- No state hit the unplaced-partner refusal. How many sampled states had a planned-then-deferred partner open was not
  counted.
- **Extended B2 submitted** at `d28d1bb0`. The ranking test runs on its datasets.

**Arms at the fix (2026-10-09; S7, `rp/r1a-arms` `4e294122`; merge `50d64ae6` of `d28d1bb0`).**
- **Suite:** the same 26 pre-existing failures; 1,793 pass.
- **`HEROSIM_INFLIGHT_CAPTURE` stays unset (legacy) for `ra_*`.** Capture clears every `HEROSIM_*` variable, and capture
  and serve compute backlog with the same function.
- **Parity job 850130:** 13 of 14 datasets clean. Its two modes can't be told apart in replay, so legacy rests on the code
  argument.
- **Servesmoke job 850196:** 7 of 7 arms finish, with queue share 0.04–0.08 and provenance per arm.
- **Open:**
  - record the capture mode in the cache metadata and refuse a mismatch;
  - a 0.55 s `backlog_s` diff in ds_01800 (feature bug or replay artifact?);
  - ds_02201 splits a 4-task batch into 2 in replay (real batching divergence?);
  - the suite rerun at head.
- **Resolved at `0699a0e0`** (suite: 1,798 pass, the same 26 pre-existing failures).
  - **Capture mode:** `physics_env.inflight_capture` is now read off the datasets; mixed-mode caches are refused, and so
    is a serving or training mismatch.
  - **The 0.55 s backlog diff** (ds_02000, not ds_01800) is a replay-harness blind spot. A replayed ghost is not
    `platform.current_task`, so the replay side reads 0. The cache and serving use the same capture function, and a new
    test pins that.
  - **ds_02201 is a harness artifact.** Live serving collects the whole peer group and cuts it with the same
    `declared_slate.sub_batches`. Parity job 850834: 14 of 14 clean.
  - **Accepted, and disclosed:** later sub-batches don't see earlier sub-batch placements, on either the train or the
    serve side, which is consistent.
  - **Verdict: the arms are gate-ready**, pending only the production cache.

**Extended B2 read (2026-10-09; S6, capture 850131, build 850132, check 853176, `d28d1bb0`; `corpus_b2_ext/by_rung.json`).**
No line trips.

| Rung | Datasets | Single-node zero-exchange | Sub-batched | no_choice (of offered) | unplaced_partner | Median plans |
|---|---|---|---|---|---|---|
| light | 197 | 51.8 % | 2.1 % | 29.5 % | 0 % | 8 |
| moderate | 198 | 79.3 % | 2.1 % | 10.9 % | 0 % | 25 |
| heavy | 209 | 79.4 % | 7.4 % | 6.2 % | 0.4 % | 32 |

- **Hung:** 1 of 144 cells (0.7 %), 9103 heavy g0, stopped by user decision and not rerun. The volume target became 572.
- **Discards:** 0. The fidelity check found 0 problems.
- **Moderate and heavy sit 0.6 points under the 80 % line.** The corpus will be mostly single-node optima at those rungs;
  quote that with any model-class result.
- **Decision:** the production capture launches now. The label build waits for the ranking test, per the signed rule.
- **Production layout (2026-10-09).**
  - Held-out = the registered 9297–9320 ∩ the 103 feasible topologies. That gives 17: 9298, 9300–9302, 9304–9309,
    9312–9315, 9317–9319. The other 86 are train.
  - Production ROOT goes on `/share/nikola.lukic`, because /home has about 56 GB free against a capture of about 47 GB
    plus the build.

**Ranking test, gate 3 FAILS (2026-10-09; S5, job 853231, `rp/forced-batch-d28` `bdea4592`, `corpus_b2_ext`).**
- **Ran:** only 34 of 223 plan evaluations (6 of 40 datasets). 30 match their label exactly, and 4 (ds_03200) miss by
  12 % (0.381 against 0.335 s). No ranking was read, per the signed rule.
- **Refused:**
  - 154 plans place a task on a (node, platform) that is not an existing replica at that instant. The co-sim creates
    replicas on demand, and the forced live hook refuses them.
  - 30 are batch mismatches: the dataset's batch is a sub-batch of the live decision's batch.
  - 5 are timing mismatches.
- **Open question, before any option is chosen:** is the co-sim plan space the same as live serving's candidate set,
  with or without on-demand replica creation? S7 is checking the code. The label build stays held, and the production
  capture continues, since it doesn't depend on the label.
- **Answered (S7, code and data read, `0699a0e0`): train and serve candidate sets are identical.**
  - Both use existing reachable replicas, the top 5 under declared pruning.
  - The co-sim slate is the snapshot's own `candidates`, and 16,784 sampled rows have 0 placements outside them.
  - Serving draws its edges from `system_state.replicas` with the same top 5 and sub-batch cut. CD uses the same set.
  - The 154 "non-replica" plans are most likely a harness task-order mapping error: dataset task order differs from
    snapshot batch order in 425 of 604 datasets, and the mis-mapping reproduces a 70.5 % rate.
  - S5 fixes the mapping and reruns gate 3. ds_03200's 12 % miss is rechecked under the corrected mapping first.

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
