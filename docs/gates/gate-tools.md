# GATE TOOLS — correctness fixes

> **Status:** `REFERENCE` &nbsp;·&nbsp; **Index:** [LINEAGES.md](../../LINEAGES.md)
>
> A gate that lies is worse than no gate. This table is deliberately **not** filed under
> any lineage: someone re-running a gate in six months needs to find out what changed about
> the tool without reading a lineage's story to get there.

Facts about the *gates themselves*, kept out of the lineage narratives on purpose. A gate
that lies is worse than no gate, and someone re-running one of these in six months needs to
find out what changed about the tool without reading a lineage's story to get there.

## Breaking changes, dated (compiled 2026-10-12)

One row per change after which earlier numbers no longer mean what they said. Each date and effect was read at the home named in the last column.

| Date | Change | What it invalidates | Home |
|---|---|---|---|
| 2026-08-17 | Cache labels came from `optimal_result.json`, not the `placements.jsonl` minimum: 24/200 `shallow_v1` labels wrong, mean regret 15.78 % | Any ablation trained on those caches (the 2026-08-17 `shallow_v1` ablation) | `docs/lessons.md` "CACHE LABELS ≠ SWEEP OPTIMA" |
| 2026-08-18 | Workload draw was unseeded (global `random.randint`, no seed) | Reproducing any corpus or live number from before it by its seed | `docs/lessons.md` "THE WORKLOAD DRAW WAS UNSEEDED" |
| 2026-08-20 | Checkpoints carried no infrastructure provenance: the sealed holdout served 40–50-node cells to 20 × 20-trained models unnoticed; the feature-layout guess was closed 2026-08-23 | Live numbers read as in-distribution before the `corpus` sidecar block; checkpoints served under a guessed layout | [table](gate-tools-table-2026-08-19-to-09-14.md), rows 2026-08-20 and 2026-08-23 |
| 2026-08-21 | Every datalab gate ran in an undeclared third environment (`pipenv run` past `micromamba activate`) | Nothing: library versions measured at exactly 0.0 logit change, 0/256 argmax flips; it invalidates venue-gap explanations | [table](gate-tools-table-2026-08-19-to-09-14.md) row 2026-08-21; `PARITY.md` |
| 2026-09-16 | Size-free served representation (`partial_state_v3`: rank one-hot → two scalars) | Serving a pre-v3 checkpoint at another cluster size; v3 ties v2 at 6 servers | [partial_state_v3](../lineages/partial_state_v3.md) |
| 2026-09-18 | Bipartite penalty aggregates by `sum` over a candidate set that grows 3.55 → 47.92 per task (6 → 80 servers); use `mean` | `sum`-aggregated bipartite arms at 80 servers as an architecture verdict | [bipartite_aggr_v1](../lineages/bipartite_aggr_v1.md) |
| 2026-09-20 | Paired statistic: pair on the environment before the median | Every client-rung "vs reactive" number before it (the −20.8 % headline was an unpaired ratio of medians) | [unsaturated_edge_v1](../lineages/unsaturated_edge_v1.md) |
| 2026-09-20 | A hand rule with the model's information is the bar (peer-greedy rule −12.96 / −16.01 % vs reactive) | "Beats Knative" as evidence for a learned arm | [peer_greedy_live_v1](../lineages/peer_greedy_live_v1.md) |
| 2026-09-20 | Serving cap blocked the label's co-location move (read 09-20, audit-confirmed 09-21) | Capped burst-seat learned-arm losses (`joint_burst_v1` +16.8 % vs the greedy) | [joint_burst_v2](../lineages/joint_burst_v2.md) |
| 2026-09-25 | Uncapped-rung `krank` serving defect fixed at 721d44f (zero-demand nodes ranked first; 56/60 batches) | Uncapped burst-seat learned-arm numbers before 721d44f; bounded: the fix made `gnnedge0` +1.4 % slower | [load_repr_v1](../lineages/load_repr_v1.md) |
| 2026-10-07 | Single request origin per group, not scattered | Every learned-vs-CD win before it (grounded, peak-load, small-batch): single-origin, no learned arm beats CD | [small_batch_so_v1](../lineages/small_batch_so_v1.md), [client_local_v1](../lineages/client_local_v1.md) |
| 2026-10-08 | Reference physics R1 → R1.1 (pipelined, release, KPA, time scale 1.0) and workload WF1 frozen | Every pre-R1 number is another physics; R1 co-sim labels before the I11 fix 033811d6 | [physics_audit_v1](../lineages/physics_audit_v1.md), [workload_fix_v1](../lineages/workload_fix_v1.md) |
| 2026-10-09 | Load rungs fixed by CD effective queue share (light ×1.26, moderate ×5.94, heavy ×11.61) | Rung labels from before it | [load_recalibration_v1](../lineages/load_recalibration_v1.md) |
| 2026-10-10 | Multiplier convention: double scaling at 160 × 24 | "CD saturated at m0" (retracted) | [scale_160_v1](../lineages/scale_160_v1.md) |
| 2026-10-10 | Hand bar moves above CD: `cd_expand` (α-expansion) and `cd_exactS` (exact S over the slate), −7 to −11 % vs CD | "Beats CD" as "beats the best hand search" | [accel_replica_v1 post-close diagnostics](../lineages/accel_replica_v1/post_close_diagnostics.md) |
| 2026-10-12 | Declared top-5 slate charges cold 0 on initialized platforms; `cd_exactS` is not slate-restricted | GNN-vs-hand-search contrasts as scorer verdicts: matched-slate, the GNN self-search wins −4.85 / −1.94 % | [slate_bar_v1](../lineages/slate_bar_v1.md); the section below |
| 2026-10-12 | Hand bar moves to `cd_exactS_ib0` (S without in-batch stacking): −4.73 / −2.27 % vs `cd_exactS`, 12/12 | `cd_exactS` as the bar | [`ib0_confirm_v1`](../lineages/ib0_confirm_v1.md) |

### Numbers measured before a change that affects them (audit 2026-10-12)

Scope: the AGENTS.md standing block, the `throughline.md` head, the first 25 `LINEAGES.md` rows and the heads of the nodes they link. "Slate" is the 2026-10-12 declared-slate cold defect, "ib0" the 2026-10-12 bar move, "R1" the 2026-10-08 physics and workload freeze. Every row marked NO now carries a caveat at its quote site.
- Not affected by these changes: hand-vs-hand numbers (`cd_exactS` and `cd_exactS_ib0` vs CD, Knative +48 / +68 % vs CD, `high_rho_s0_v1`, `sparse_obs_s0_v1` on `ib0`), because CD's cost is not the declared slate.
- The accel rungs (×11.6139 / ×27.6227) were calibrated in `accel_replica_v1` on 2026-10-10, after the multiplier retraction. So no accel-seat number predates the rung or multiplier changes; `r1_attribution_v1` ran on `load_recalibration_v1`'s rungs.

| Number | Quoted in | Measured (date, against) | Affected by | Re-measured? |
|---|---|---|---|---|
| "The bar is `cd_exactS` … no learned arm beats it" | AGENTS lead | 2026-10-10/11, vs CD | ib0 | Superseded by [ib0_confirm_v1](../lineages/ib0_confirm_v1.md); caveat added |
| `gnn_eng_nosplit` −6.9 / −7.7 % vs CD | AGENTS, throughline, [accel_nosplit_v1](../lineages/accel_nosplit_v1.md) | 2026-10-11, gnn-eng g4 on `declared_pruning_v1` | Slate | NO (`slate_fix_v1` running) |
| `gnn_selfsearch` −8.5 / −10.1 % vs CD; heavy −3.1 / −3.8 %; ties MP-OFF twin | AGENTS, throughline, [selfsearch_confirm_v1](../lineages/selfsearch_confirm_v1.md) | 2026-10-11/12, same corpus and slate | Slate (twin contrast: both sides) | vs CD NO; vs `cd_exactS` matched-slate by [slate_bar_v1](../lineages/slate_bar_v1.md) |
| gnn_eng −5.45 % / physmp −6.05 % vs CD; CD←GNN −5.03 / −3.40 % | [accel_replica_v1](../lineages/accel_replica_v1.md) | 2026-10-10 | Slate | NO |
| G8 self-search −9.7 / −1.5 % vs CD, +0.5 / +5.5 % vs `cd_exactS` | AGENTS (closed levers), [big_groups_s0_v1](../lineages/big_groups_s0_v1.md) | 2026-10-11 | Slate; ib0 | NO |
| self-search −1.4 / +1.2 % vs CD; "hand search stays the bar" | [link_bound_s0_v1](../lineages/link_bound_s0_v1.md) row | 2026-10-11 | Slate; ib0 | NO |
| GNN 4–12 % ahead of CD per block; deficit to `cd_exactS` +2.7 pp / 24 h | AGENTS, [long_horizon_s0_v1](../lineages/long_horizon_s0_v1.md) | 2026-10-12, vs `cd_exactS` | Slate; ib0 | NO |
| over-cap self-search +0.2 / +2.5 % vs `cd_exactS` | AGENTS, [gnn_overcap_v1](../lineages/gnn_overcap_v1.md) | 2026-10-12 | Slate (in head); ib0 | NO |
| top-k prune −1.4 / −2.1 % vs `cd_exactS`'s fallback; GNN order ties cost order | AGENTS, [prune_g8_s0_v1](../lineages/prune_g8_s0_v1.md) | 2026-10-12 | ib0 (prune not tried on `ib0`); slate (GNN order only) | NO |
| label oracle −0.9 / −4.2 % vs `cd_exactS`; hand ib0 −4.6 / −3.2 % | [plan_rerank_v1](../lineages/plan_rerank_v1.md) row | 2026-10-11/12, 6 topologies | ib0 | YES: [ib0_confirm_v1](../lineages/ib0_confirm_v1.md) (ib0 −4.73 / −2.27 % on 12 fresh topologies; oracle vs ib0 −2.82 / −1.96 %) |
| matched-slate self-search −4.85 / −1.94 % vs restricted `cd_exactS` | AGENTS, throughline, rows | 2026-10-12 | None for its claim (scoped to `cd_exactS`; both sides on the same slate) | n/a |
| GNN seed −0.99 / −1.22 % vs hand seed; +0.73 / +0.41 % vs `ib0` | throughline, [ib0_confirm_v1](../lineages/ib0_confirm_v1.md) | 2026-10-12 | Slate (seed checkpoint) | NO |
| MLP V ties `cd_exactS` | [cost_to_go_v1](../lineages/cost_to_go_v1.md) row | 2026-10-11 | Slate (training tables); ib0 | NO (caveat already present) |
| argmin-S optimal 79 % (regret 6.1 %), median 3 tasks | AGENTS ("little room"), [big_groups_s0_v1](../lineages/big_groups_s0_v1.md) | 2026-10-11, 413 held-out accel tables | Slate: tables enumerate the declared top 5 | NO |
| raw MP vs twin −4.75 / −4.62 / −7.99 %; physmp +3.4 / +5.9 / +5.1 % vs CD | throughline, AGENTS | 2026-10-09/10, 40 × 6, declared pruning | Slate | NO |
| +5.14 pp offline; −15.94 % live | [peer_affinity_v1](../lineages/peer_affinity_v1.md) row | 2026-09, pre-R1 | R1 | NO; caveat added |
| −11.9 % vs one-pass; CD +12.5 % | [joint_burst_v2](../lineages/joint_burst_v2.md) row | 2026-09-20, burst seat | R1; `krank` fix 721d44f | krank bounded (+1.4 %, [load_repr_v1](../lineages/load_repr_v1.md)); R1 NO; caveat added |
| Phase 2 TIE p = 0.25 | [route_b_v1](../lineages/route_b_v1.md) row | 2026-08/09, co-sim | R1 | NO; caveat added |
| CD 5.6 / 5.0 / 5.6 % above the batch optimum | [live_headroom_v1](../lineages/live_headroom_v1.md) row | 2026-10-01, scattered origins | R1; origin (caveat present) | NO; physics caveat added |
| D1: H-MYOPIA | [drainable_debug_v1](../lineages/drainable_debug_v1.md) row | 2026-09 | R1 | NO; caveat added |
| re-gate 5/5 | [siv1_full_corpus](../lineages/siv1_full_corpus.md) row | 2026-08 | R1 | NO; caveat added |
| mixed_dispatch / radical_physics pilots | rows | 2026-09, old physics (heads say so) | R1 | NO; row tag added |
| L2D: MP ties at the authors' lr, +1.5–5 pp converged | [literature_reeval_v1](../lineages/literature_reeval_v1.md) row | 2026-09, upstream L2D env | None (not HeROsim) | n/a |

## 2026-10-13 — Fidelity capture booked priced exchanges into the access-class tally (fixed 6d1764bd, 2026-10-13)

`snapshot_fidelity._expected_input_seconds` priced each popped task's input stage with `Platform._peer_exchange_time`, which also calls `fabric.record_exchange`. So every capture booked transfers that never happened into the access-class tally. The fix adds `_peer_exchange_time(task, record=False)` for the capture; the physics call keeps `record=True`. Found by `uncut_groups_s0_v1` identity (m).
- **What is affected:** every run that captured fidelity snapshots (`cd_oracle` arms, warm-snapshot corpora, `link_bound_s0_v1` capture cells) carries inflated `peerExchangeByAccessClass.*.{seconds,transfers}` only.
- **What is not:** `averageElapsedTime`, `steady_state` and every other key are bit-equal, per the hook-on/off read (16490 g0 heavy `cd_exactS`: 8 keys differ, all in that tally). Trajectories are unaffected.
- **Scope (verified by the expert):** a grep of `docs/` and `LINEAGES.md` finds no node quoting a `peerExchangeByAccessClass` number from a capture run. The five hits (workload_fix_v1, r1_attribution_v1, link_bound_s0_v1, gnn_overcap_v1, prune_g8_s0_v1) describe the W3 access classes, and link_bound's "links never bind" metric is not that tally.
- No quoted number is affected and no audit of past result files is needed. Any future quote of that tally from a pre-6d1764bd capture run must be re-derived.

## 2026-10-13 — A cell `batch_size` outside the GNN range was ignored with a print

`gnn/orchestrator.py:72–84` silently ignored a cell batch_size outside [1, 16] and served GNN_BATCH_SIZE (4). This hit only offline snapshot replays of > 16-task batches: fidelity_cd_plans 855859/855861 (40/104 each), plan_rerank S_alpha 859727 (1/3,103) and cost_to_go train_h5 (1/2,173). No live gate. Fixed by uncut_groups_s0_v1 ruling C (raise).
- **Scan:** a grep of every surviving `.log`/`.out`/`.txt` under datalab `simulation_data`, `/share/nikola.lukic` and `~/gnn-herosim/logs` (job 864525, 2026-10-13); deleted or rotated logs are not covered. No argmax-mode ([2, 4]) hit.
- The 67.2 % / 55.3 % hit rates (2–8-task datasets) are inside the range and unaffected.
- fidelity_cd_plans' > 16-task tables were never recorded and are not to be quoted; if that diagnostic is ever written up, those 40/104 replays are rerun first.

## 2026-10-12 — `run_one` rebound the cell name: per-cell `.tasks.npy` unusable since 6ad51eaf

`fresh_topo_burst_v1_gate.py` `run_one` checks the served alpha env with `for name in ("HEROSIM_PG_DRAIN_ALPHA", "HEROSIM_PG_BATCH_ALPHA")` (plan_rerank, 6ad51eaf). `name` is also the cell name under which `write_task_times` writes `<name>.tasks.npy`, so every cell reaching that loop wrote its per-task times to `HEROSIM_PG_BATCH_ALPHA.tasks.npy`, one file overwritten by every cell (concurrent cells also raced on its `.partial`, which killed the first warm_surge check cells). The sparse_obs check loop (`for name in OBS_ENV`) did the same. Summaries, `steady_state` keys and failed-cell records use their own path variables and are unaffected; only the per-task npy is lost. Every gate run between 6ad51eaf and the fix has no usable `.tasks.npy` (slate_fix_v2 at 6a5c922c, sparse_obs `gate_s2`, ib0_confirm). Fixed at e3d5e069 (loop variable renamed); `tests/test_gate_name_not_shadowed.py` fails if `run_one` rebinds `name` in a `for`. A reader that needs per-task times from a gate older than the fix must rerun it.

## 2026-10-13 — A median paired rule is blind to a 25–45 % collapse tail: every surge gate reports a collapse count per arm

`warm_surge_s0_v1` Phase 0b: the registered primary (median paired % of post-switch mean, ib0 vs exactS) read "not moved" while `cd_exactS_ib0` ran away on 3/12 topologies at heavy ×2 and 5/12 at heavy ×3 (post-switch mean 5–55 s against 0.5–1.0 s, end − last arrival up to +1,094 s), where `cd_exactS` collapsed on 1/12 and `cd` on 0. Medians against the collapsing arm (cd −32 %, SS −39 % at heavy ×3) are collapse artefacts, never a win. **From 2026-10-13 every surge gate reports, beside the median, a collapse count per arm and condition: COLLAPSE = post-switch mean > 3× the same cell's `cd` mean; TAIL FLAG (separate count) = end − last arrival > 120 s.** Thresholds, from the 0b data (576 cells): the largest non-collapsed ratio is 2.4× and the smallest collapsed 8.1×, so 3× sits in the gap with margin both ways. The tail alone does not separate: non-collapsed cells reach 100 s, and one learned cell with a normal mean (1.5 s, 0.9× cd) has a 142 s tail, so the tail is a separate count, not part of the collapse. Reader: `scripts_cosim/warm_surge_extra_read.py collapse`. A condition with a collapse count above 0 is also read against every robust hand arm, not only the bar.

## 2026-10-12 — Audit after three breaking changes

An independent audit (relayed by the coordinator) of the three 2026-10-12 breaking changes: the `cd_exactS_ib0` bar, the declared-slate cold defect, and the WF1 / multiplier rows. It found no cited number wrong, and four MAJOR reproducibility or contract items, which S6 is fixing on `rp/slate-fix`.

**Verified correct.**
- **The `ib0` headline**, recomputed from the 624 summaries: −4.734 / −2.267 %, 12/12 ([ib0_confirm_v1](../lineages/ib0_confirm_v1.md)).
- **The multiplier is applied once.** Realised arrival rates are 5.347 /s = 0.46039 × 11.6139 (moderate) and 12.717 /s = 0.46039 × 27.6227 (heavy).
- **Labels are the `placements.jsonl` minimum**, 42 of 42 sampled.
- **`cd_exactS` is bit-equal** across 79ea002c, d192081f and 4d90119c on 16490 g0, at both rungs.
- **Batch α multiplies only committed in-batch service** (`scheduler.py:501–502` and `:1044–1045` on `rp/plan-rerank`).
- **Venue parity:** logits max|Δ| 0, 0/256 argmax flips.

**MAJOR findings and their fixes.**
- **M1 — the primary comparison pairs two commits on divergent branches.** No branch has both, and `reference-physics` has neither. Fix: integration branch `rp/slate-fix-int`; the read records both commits.
- **M2 — `experiments/accel_replica_v1_split.json` is untracked on every branch**, though every accel and `slate_fix` sidecar names it (sha f5bc4f51…). Fix: track it and add a test.
- **M3 — warmth physics is not enforced for any GNN arm.**
  - Sidecars and caches carry no `warmth_physics`.
  - `resolve_warmth_physics` defaults to `platform_reuse_v1` (`src/placement/warmth.py:123–138`). Live runs get `node_disk_v2` only from the gate's environment: `HEROSIM_WARMTH_PHYSICS="node_disk_v2"` in `scripts_cosim/fresh_topo_burst_v1_gate.py`, at :535 on `reference-physics` and :769 on `rp/slate-fix`.
  - Fix: a physics environment block plus a serving whitelist. The R1.1 definition now names the warmth physics (`physics_audit_v1`, `workload_fix_v1`, AGENTS seat line).
- **M4 — the uncapped slate doubles the sub-batched share:** 20.2 % vs 10.3 % of tasks on 19420 heavy. Its counters go in the `slate_fix_v1` read.

## 2026-10-12 — The declared slate charges no cold start on an `initialized` platform; the simulation does

**Where.** `declared_slate.standalone_cost` (`src/placement/declared_slate.py:46` at the measured commits on `rp/slate-bar` / `rp/plan-rerank`; :48 on `rp/slate-fix`, after two RULE lines) sets cold = 0 whenever the candidate payload's
`initialized` flag is true. That flag is `platform.initialized.triggered` (`src/placement/live_audit.py:197`), a platform-level
event.

**What the simulation does.** The live simulation applies the table cold start unless the platform's **previous task was the
same function type** (`sandbox_is_warm`, `src/placement/infrastructure.py:1616`, applied at `:1623–1627`).
- CD's `_pg_standalone` charges `incoming_cold_start_time` (`src/placement/scheduling_cost.py:71`), which follows the
  simulation here, so the defect is in the slate, not in CD.

**Replayed proof** (`slate_bar_v1_recall.py cold-probe`, job 863493 at ddc3b5cd). Five batch tasks were forced onto couples the
two costs price differently.
- Every target platform was `initialized`, with no previous or current task.
- The simulation cold-started all five, at exactly the table value.
- Example: `cc40s16223_heavy_g2` snapshot 287, a cnn task forced to platform 887 on node 176 (xavier).
  - Declared cost 0.60 s; CD cost 38.60 s.
  - Simulated `coldStarted = true`, `coldStartTime = 38.0` s.
- The other four: 0.98, 0.98, 10.62 and 0.096 s, each equal to the table value.

**Consequences.**
- **Every artifact on `declared_pruning_v1` ranks with this cost:** the accel corpus, `candidate_slate` in the checkpoint
  sidecars, the GNN's served slate, and `cd_declared`.
  - On the accel snapshots the declared top 5 differs from CD's own top 5 on 26 % of tasks.
  - It loses about 10 pp of batch recall of the hand rules' picks at G=8 (`slate_bar_v1`, recall audit).
- **v1 stays as it is.** The existing artifacts keep it, since it is what the GNN saw, and the matched-information live bar uses it.
- **A future capped build uses `declared_pruning_v2`** (`slate_bar_v1`): the simulation's cold charge plus the partner composition.

**A second, smaller deviation, in CD's cost.** `incoming_cold_start_time` returns 0 whenever the platform has a `current_task`
(`scheduling_cost.py:73`). Under `HEROSIM_REPLICA_RELEASE=1`, though, `previous_task` is set to the in-flight task right after its
cold start (`infrastructure.py:1651`). An incoming task of a different type therefore cold-starts on a busy platform.
- Not measured here.
- It is a hand-rule finding for whichever node next changes CD's cost.

## 2026-10-11 — Under replica release, the Knative arm's least-connected key is blind

`knative_network/scheduler.py:211–213` ranks replicas by `len(queue.items)`. Under `HEROSIM_REPLICA_RELEASE=1` (R1),
tasks leave `queue` at once and wait on `compute_lock`, so the key reads 0 on 80–84 % of placements. The tie then goes
to the lowest (node, platform) id 99.8 % of the time. The arm herds onto a few low-id slow platforms and collapses on
lock wait while GPUs sit idle.

With release=0 the same arm is healthy on 3 of 4 cells (`accel_nosplit_v1`, 2026-10-11). Any reactive or Knative
number under R1, including r1_attribution_v1's collapse, reflects this blind key, not Knative's concurrency-based
balancing. Quote it with that caveat, or use a key that counts queue + lock waiters + running.

## 2026-10-11 — `test_trainer_determinism.py` skips 8 of 18 checks without the smoke caches

The pre-training gate in AGENTS.md (`tests/test_trainer_determinism.py`) passes while skipping 8 of its 18 tests when
`simulation_data/graphs_cache_route_b_smoke_s_dag*` is absent, as it is in a fresh worktree (measured 2026-10-11: 10
passed, 8 skipped). A green run therefore does not cover the DAG partial-state trainer paths. Run it with `-rs` and read
the skips. Treat a skipped trainer as unverified, and rebuild the smoke cache before gating a run on those paths.

## 2026-10-10 — Fidelity replays froze the autoscaler and ran the monitor off phase

Fidelity replays under any non-determined scheduler (PG, CD, GNN) froze the replayed KPA. Its first tick waits on
`_kpa_start_gate`, which only `DeterminedScheduler` released. Fix for the GNN-family path: 8e19fcab.

Separately, any replay longer than one tick restarted the orchestrator's 1-s `monitor_process` at t0. That put it off the
live run's absolute phase, and later off its within-instant order: live runs the KPA tick first, then the monitor. The
tick's `env.step()` then re-phased scale actions by up to a tick, which moved cold starts and co-location. Fixed tool-side
in `rollout_replay` (daf74ffa, a96b8f27); the generic src fix is pending.

Shielded: everything computed at the decision, plus sweep-row labels, so every reported number is intact
(`accel_replica_v1`, 2026-10-10 23:00). Exposed: latency replayed after the decision. A rare residual remains: a KPA
action one tick late, in 6 of 300 state-horizon pairs (`cost_to_go_v1`).

## 2026-10-08 — Queue share misses pre-execution starvation; read p95 and run end too

Under WF1 at ×11.61, Knative's queue share was 0.001 while its mean latency was 602 s: p50 1 s, p95 5,366 s, and runs ending at
about 2× the last arrival (`workload_fix_v1`). The wait sits before execution and outside the queue, exchange and
rendezvous stages the summaries measure. The 300 s request timeout doesn't cover it, because the tasks are unplaced. A
stability or admission guard that reads queue share alone passes a collapsed arm. Read p95 per-task latency and the run's
end time against its last arrival as well.

**Second blind spot (2026-10-09, `load_recalibration_v1`).** Under `HEROSIM_REPLICA_RELEASE=1`, `started` is stamped
before `_serve_task` requests `compute_lock`. So the FIFO backlog on a saturated replica lands in **compute time**, not
queue time. For CD at ×24.71 (9601 g0), 40.7 of 42.0 s per task was lock wait, and queue share read 0.010 instead of
about 0.97. Fix: use effective queue share = (queue + compute_start − io_end) / elapsed. Never read
`averageComputeTime` as execution under R1.

**Correction (2026-10-09).** On the recalibration cells, Knative's collapse is lock wait, not unplaced tasks: placement
wait is 0.006 s and lock wait is 826 of 837 s at ×11.61 (`load_recalibration_v1`). The "unplaced" explanation above is
not supported there. Whether the `workload_fix_v1` W4 cells collapse by the same mechanism is unverified.

## 2026-10-08 — The autoscaler loops call `env.step()`: any extra scheduled event is a physics change

`Autoscaler.autoscaler_process` and `_kpa_autoscaler_process` (`src/placement/autoscaler.py`) end every iteration
with `self.env.step()`, which processes the **next queued event, whatever it is**, and moves the clock to it. So
what each loop iteration consumes depends on everything else in the event queue: adding a SimPy process that
reads and writes nothing can still change a run, and autoscaler ticks are not `last_tick + interval`.
Measured (`physics_audit_v1`, 2026-10-08): `scripts_cosim/physics_audit/env_step_repro.py` changes job completion
times with one extra `Timeout` in a standalone model; in the simulator (2,000 arrivals) 0, 1 or 100 idle timeouts
leave total RTT at 1288.517 s, 1,000 give 1311.277 s (+1.8 %), and a 0.1 s occupancy sampler gave 1077.7 s
(−16 %). **Instrumentation must schedule nothing** (`HEROSIM_AUDIT_TRACE` does not), and a new process in a run
needs a replay-identity check like any physics change. A snapshot replay must read the next tick from the
Timeout the autoscaler is suspended on, not compute it.

## 2026-10-07 — `bandwidth_mbps` is MB/s, and transfers are store-and-forward

`network.backbone.bandwidth_mbps` (and `core_bandwidth_mbps`, `network.bandwidth`, `--link-bandwidth-mbps`,
`HEROSIM_LINK_BANDWIDTH_MBPS`) is read as **megabytes per second**: every charge divides bytes by
`value × 1024²` (`src/placement/scheduling_cost.py`, `transfer_time`; `Platform._payload_transfer_time`).
The grounded cells' `1000` is therefore about 8.4 Gbps per link, not 1 Gbps. The record described it as
"1 Gbps" until 2026-10-07; those lines now say MB/s. The key keeps its name because every config and corpus
uses it. **Write any new link rate in MB/s**: a 100 Mbps Wi-Fi uplink is `12.5`, not `100`.

A multi-hop transfer is charged **store-and-forward** by default: hops × size ÷ slowest link, plus the route
latency. On `small_batch_confirm_v1`'s 19 topologies a server-to-server route is 2–8 hops, mean 4.96, so an
exchange costs about 5× a pipelined transfer. `HEROSIM_TRANSFER_MODEL=pipelined` (size ÷ bottleneck) is opt-in
from 2026-10-07 (`transfer_physics_v1`). Quote every exchange number with its transfer model.

## 2026-09-23 — Assert peer physics in mini co-sim screens

The 32-task pair/triple/block move screens in
[the fixed-replica MP ablation](../lineages/gnn_seeded_cd_mp_ablation_v1.md)
passed workloads containing `peer_exchange` to a direct co-sim evaluator but did
not set `HEROSIM_PEER_EXCHANGE=1`. The evaluator silently assigned zero peer
exchange time, making both screens' apparent headroom and hand-control gaps
invalid for the intended environment. A two-plan probe found a moved pair cost
of 15.363 with peer physics off versus 149.378 with it on; the baseline cost
was 17.219 in both cases.

For every direct evaluator or gate, assert the effective physics environment
before comparing plans, then run a discriminating placement with a nonzero
`averagePeerExchangeTime` (or equivalent peer metric) when peer traffic is
expected. A matching baseline cost alone does not verify the physics contract:
it can remain unchanged even when peer physics is disabled.

## 2026-09-23 — Do not require peer transfer in every selected live arm

The initial [proposal frontier gate](../lineages/proposal_frontier_v1.md#2026-09-23--initial-gate-invalid-at-peer-accounting-audit) rejected topology 10305 because its learned-selector arms measured zero peer-transfer time while the hand arm measured positive time. The audit treated positive transfer in *every* arm as proof that peer physics was enabled. A selected plan may co-locate communicating tasks and legitimately incur zero transfer, so that assertion confounds a policy outcome with the physics switch and invalidated the incomplete gate before quality metrics were read.

For live policy comparisons, check the effective peer-physics setting and frozen peer-enabled workload, then require each arm's peer accounting to be finite and nonnegative. Verify the switch separately with a discriminating placement that must transfer. Keep the positive peer-metric assertion for that probe, where transfer is expected by construction.

## 2026-09-23 — Bound the warm-corpus selector by its effective limit

`scripts_cosim/make_warm_corpus.py::choose_candidates` accepts `max_combos` but never
reads it; candidate selection checks `target_combos` alone. For a 32-task batch,
the default `min_choice_fraction=0.5` requires at least 16 tasks to retain two
choices, hence at least 2^16 = 65,536 plans, above the default
`target_combos=20,000`. All 16 captured full-batch snapshots were rejected in
the dry run. Increasing `max_combos` alone cannot change that result.

This is a corpus-apparatus limit, not evidence that the 32-task environment
lacks headroom. The exploratory move screen and its scope are in the
[owning diagnostic](../lineages/gnn_seeded_cd_mp_ablation_v1.md#2026-09-22--fixed-seat-debugging-and-move-screens).
The historical generator source remains unchanged; any successor must set a
feasible `target_combos` or change the candidate-selection contract explicitly
and account for the larger search cost.

## 2026-09-22 — Acquire conditional resource locks only when optional work runs

The first DAG rematerialization gate included recomputed-parent locks whenever
an output's recompute bit was set, even if that output was already cached.
This changed execution order without performing recompute work and made a
nonzero plan improvement appear in rows with zero rematerialization events.
The first gate's objective comparisons are invalid for qualification. A
conditional action must have no effect when its trigger is absent; check this
with a no-op ablation in addition to actual-simulator parity, since two
implementations can agree on the same mistaken contract.

The corrected replay and live adapter compute effective locks from cache
state at dispatch and retain that lock for the active task. A regression
checks that setting a recompute bit for an always-cached output leaves the
objective unchanged. New calibration and fresh seeds are fixed in the
[owning record](../lineages/dag_remat_s0_v1.md).

## 2026-09-22 — Translate schedule semantics between search decoders

Event-dispatch priorities and serial reservation priorities are different search
representations. Reusing the same numbers may turn a good feasible incumbent into
a poor initialization even when both decoders cover it. The irregular-DAG
diagnostic found this effect; its measured sizes belong to the linked record.
The original numerical comparisons remain valid, but this weakens a claim that
the reservation family received a strong search challenge.

The successor constructs chronological ranks (ties by the original rank), checks
that reservation initialization does not worsen its non-delay incumbent, and
translates before cross-decoder proposals. Both materialization and translation
count against the serving budget. Preserve the old fingerprints and use new
qualification seeds. See the [owning record](../lineages/mixed_dispatch_v1.md#2026-09-22--dag-result-and-schedule-translation-successor).

## 2026-09-22 — Read DAG job latency from terminal completion, not summed task RTT

`stats.total_rtt` sums per-task elapsed times. For a zero-overhead chain those
durations telescope to job completion; parallel branches do not. The initial
irregular-DAG smoke check therefore disagreed on the aggregate even though every
operation's start and completion matched independent replay. The historical
field is not redefined and no frozen source is changed.

The successor `local-record-2026-10-06:src/placement/radical/dag_reservation_live.py` (not merged) reports
`job_completion_sum` from the unique terminal sink of each zero-release-time
application and retains the original aggregate as `total_task_rtt`. Gates use
the former for the declared summed job completion objective. Its diamond
regression has job completion 9 and summed task elapsed 12, so substituting the
old aggregate fails even when all task timings are correct. Before extending
this adapter to nonzero job release times, subtract each job's release time.
Validate sink coverage, the actual served DAG and every operation's completion.

## 2026-09-22 — Filter timing eligibility before choosing a live control

The frozen `local-record-2026-10-06:scripts_cosim/mixed_dispatch_block_screen.py` (not merged) selected its live hand
control by maximum objective improvement without first excluding arms with a
wall-time repeat above budget. Its separate timing gate correctly failed, but
the selected arm could not be called a qualified equal-wall-time control.

Keep the original source and artifacts for provenance. Run
`scripts_cosim/complete_dispatch_block_budget_gate.py --out <screen-directory>`
to filter on **every measured repeat**, choose the strongest remaining fixed
arm, and execute/audit the missing live replays. The correction freezes its own
protocol and hashes and preserves the original screen report. The regression in
`local-record-2026-10-06:tests/test_dispatch_block_moves.py` (not merged) rejects an over-budget winner and raises
when no eligible control exists. New gates must apply this filter before
selection. The experimental outcome belongs to
[mixed_dispatch_v1](../lineages/mixed_dispatch_v1.md#2026-09-22--capacity-scaled-block-search-result).

The [DAG memory screen](../lineages/dag_memory_lifetime_s0_v1.md) exposed the
other timing failure: allocating the whole budget to search omitted baseline
construction and evaluation of the hand initializers. Its first calibration
stopped before fresh inputs; the corrected gate charges the complete path and
leaves a reserve for the last candidate evaluation. Timing eligibility must
use that complete path before choosing a control.

## 2026-09-21 — Rule availability observations and validation source identity

The peer rule used `platform_queue_drain_seconds()` as availability, although that
legacy snapshot field excludes the task already removed from the queue. An idle
platform and one executing ten more seconds could both score zero drain. Changing
the field itself would double-count work in snapshot replay, which separately adds
current-task and communication residuals.

`local-record-2026-10-06:src/placement/availability.py` (not merged) introduces a separate observation and estimate.
`Platform.platform_process` records phases from dequeue through ingress, cold
start, input I/O, peer rendezvous, execution and output I/O, plus virtual warmup.
Existing SimPy events and their durations are preserved. Timed work decreases with
simulation time; resource and peer waits retain an unresolved flag until released.
The estimate separates queued work, current known service and remaining virtual
backlog. Future I/O not yet priced by the runtime is listed in `unpriced_stages`;
known work is not an exact predicted release time. No future placements or queues
are consulted.

Rule serving opts in with `HEROSIM_PG_DRAIN_CONTRACT=availability_v2`. The default
`backlog_only_v1` preserves historical behavior and snapshot semantics. Under v2,
the rule prefers candidates without unresolved waits, then minimizes its score;
if all candidates are blocked it still places a task, since placement can release
a peer. This ordering is a new rule contract, not a measured policy improvement.
Both learned rollout seats reject v2 rather than silently changing checkpoint
inputs. Results carry the contract in `schedulerCounters.pg_drain_contract`;
the live CLI also records it in `run_provenance.pg_drain_contract` and decision
captures carry `drain_contract`. The closed lookahead comparison/probe CLIs reject
v2 before their environment reset; their hand controls remain on the old contract.

Validation preflight now checks source-file SHA-256 and physical infrastructure
identity before launching workers, and workers recheck the bytes they consume.
The `physical_infrastructure_v2` fingerprint is SHA-256 of compact, sorted JSON of
`network`, `nodes`, `link_topology`, `compute_slots_per_node`,
`ingress_bandwidth_mbps`, and `warmth_physics` from `config.infrastructure`.
Queue draws and forced placements do not turn one infrastructure into independent
replicates. The launch manifest adds these explicit fingerprints and source hashes;
the older opaque hash entries are retained for historical reference. Existing
result artifacts and their embedded pre-run configurations are unchanged.

Regression coverage: `local-record-2026-10-06:tests/test_availability.py` (not merged), snapshot tests and source
preflight tests. The retained 16-task immediate-rule case reproduces its task
timings, placements and total RTT exactly under the legacy contract; v2 completes
a small rule-only smoke with its contract recorded. This maintenance neither
reopens `peer_lookahead_v1` nor justifies model training.

## 2026-08-19 → 2026-09-14 — the correction tables (summary; full rows in [`gate-tools-table-2026-08-19-to-09-14.md`](gate-tools-table-2026-08-19-to-09-14.md))

53 dated rows, moved verbatim on 2026-10-12. Each is fixed in code unless marked. The rules still in force:

- **Topology and seeds.** `--seed` overrides the config's topology seed (2026-08-20): a "multi-seed" live gate is a multi-topology gate. Set-ordered tie-breaks in the autoscalers are not pinned by `PYTHONHASHSEED`; keys must be total (2026-08-21).
- **Provenance.** A live gate can measure an uncommitted code diff; read `run_provenance` in the result JSON, never an sbatch banner (2026-08-21). `pipenv run` under sbatch escapes `micromamba activate gnn`; use `${HEROSIM_PY:-pipenv run python3}` (2026-08-21). The shared training wrapper's `PROJECT_ROOT` default escaped pinned worktrees (2026-08-30).
- **Datalab.** `micromamba shell hook --shell bash`, never `--bash` (2026-08-20). A task that wrote a result and then FAILED is treated as complete by skip-existing guards; a prefix read does not validate a document (2026-08-29).
- **Checkpoint contracts.** A checkpoint was served under a guessed feature layout (2026-08-23), and an MP-OFF checkpoint was scored with MP on, offline (2026-09-03) and in the live loader (2026-09-06): architecture flags come from the `.contract.json`, never from weight shapes.
- **Co-sim corpora.** A corpus generated without `HEROSIM_COSIM_KEEP_ALIVE` reports SUCCESS and is silently wrong (2026-08-27). A truncated sweep counted as SUCCESS (2026-08-27; fixed 2026-09-03). Skip reasons are mislabelled under `replica_overlap`, and a zero-replica task type is not classified (2026-08-27/28). Run `scripts_cosim/audit_cache_live_divergence.py` before trusting a collection you have not gated.
- **Serving.** `GNN_BATCH_SIZE` is overridden by the cell config's `scheduler.batch_size` (2026-09-03). The legacy_v0 queue column a live GNN ranks on is out of its trained range (2026-09-12). Import-time `basicConfig` in `prepare_graphs_cache` disables later logging and fills disks (2026-09-13). A starved client spins at 100 % CPU in both the GNN and the reactive batch scheduler (2026-09-14).
- **Gate design.** Gate statistics that drift with sweep size cannot gate (2026-08-19); a straddling CI is not a FAIL, and sign-disagreement checks are per seed (2026-08-19); a decoder tie-break or a censored denominator can read as physics (2026-08-27); a registered bar can be uncomputable on its own corpus (2026-08-27); a control bar that reads the wrong key passes silently (2026-09-14).

## 2026-09-14 — a landed lineage's corpus directory was extended IN PLACE, so "the corpus" is now the split, not the directory

Found by `drainable_objective_v1` when it tried to rebuild a cache over T1b's corpus.

`gnn_datasets_peer_affinity_v1_c3_x200_train2` held **346** datasets when
`peer_affinity_v1` T1b was built on it. A later generation job (the `drainable_debug_v1`
H2 corpus ladder, since abandoned) appended to the same directory with `--resume`, and it
now holds **1,500**. Appending is correct and non-destructive — `ds_00000`–`ds_00345` are
untouched — but it means **`--base-dirs <that directory>` no longer reconstructs the corpus
any landed result was measured on.** A cache built that way would have been 1,670 datasets
against T1b's 516, and a contrast against T1b would then be a contrast about **corpus size**,
which `docs/lessons.md` L73 records as the largest measured lever in this program.

**The rule:** a corpus is defined by its **split artifact**, which names its parents
explicitly, not by the contents of a directory. To rebuild one, derive a parent list from
the split and pass it to `prepare_graphs_cache --oversample-manifest` (every weight 1 makes
it a pure filter; `prepare_graphs_cache.py:1981` skips any dataset absent from the
manifest). Then **assert the resulting `num_datasets`** — `drainable_objective_v1`'s cache
job fails loud at anything other than 516, because the failure this guards against is
silent and produces a perfectly well-formed cache of the wrong thing.

Second, smaller: **`gnn_datasets_peer_affinity_v1_c3_x200_train` (the 136-dataset half of
T1b) is not on the cluster at all** — only `train2` was ever synced. A rebuild there needs
it rsynced first. `models/` and corpora are gitignored, so "the repo is in sync" says
nothing about whether a corpus is.

### Addendum: the cluster's copy of a corpus can be missing the SSC the cache build needs

Same day, chasing the same rebuild. After both missing corpora were rsynced, the cluster
build still died:

```
Missing system_state_captured_unique.json for ds_00000;
  run scripts_cosim/refresh_optimal_full_stats.py --repair
```

`system_state_captured_unique.json` is written by `refresh_optimal_full_stats --rewrite-ssc`,
which is step 1 of the cache recipe — and **T1b's cache was built locally and rsynced**
(its own sbatch header says so: *"Cache and split are built LOCALLY and rsynced (never race
on them here)"*), so the SSC rewrite only ever ran on the local copy. The cluster's
`train2` has the sweeps but not the snapshots.

Two things follow:

* **`--oversample-manifest` filters at graph-build time, not at load time**
  (`prepare_graphs_cache.py:1981` versus `:718`). Every dataset under `--base-dirs` is
  loaded first, so a parent filter does **not** save you from a broken dataset outside the
  filter. Point `--base-dirs` at directories that hold only what you want, or repair the
  whole directory.
* **Rebuild where the corpus is complete.** The local tree held exactly T1b's 516 with SSC
  on all of them, so the V = 1 cache was built there and rsynced, which is also how T1b
  itself was built. Check `ls ds_*/system_state_captured_unique.json | wc -l` against the
  dataset count before choosing where to build.

## 2026-09-15 — `best_val_regret_topk` in a W&B summary holds the **masked_topo** value

`train_near_rtt.py` selects its checkpoint on `val/regret_masked_topo` (the decoder the live path
actually runs) but writes the selected value into the summary key **`best_val_regret_topk`**. The
name is wrong and has been wrong on every run that logged it. Verified on `drainable_objective_v1`
job 766897 seed 2: the summary reads `best_val_regret_topk = 128.606` and
`final/val/regret_masked_topo = 128.606`, while the run's own `val/regret_topk` minimum over the
300 epochs is 51.79 — a 2.5× difference, in the direction that makes a run look worse than it is.

Read one of these instead, never the key's name:

* the trainer's own stdout line, `Best val regret_<metric>: <value>` — it names the metric;
* `final/val/regret_masked_topo` in the summary, which is the selected checkpoint re-evaluated;
* `scripts_cosim/read_training_curves.py`, whose SELECTION block infers the selector by matching
  the recorded best against each curve's minimum and prints which metric it landed on.

**Not renamed on purpose.** The key is in every historical run's summary; renaming it would split
the series and make old and new runs incomparable in the W&B UI, which is the one place the key is
read. Fix the reader, not the key.

## 2026-09-15 — a "did the policy do what its name says" bar must be sized against the CONTROL's own counter

`drainable_objective_v1`'s C0 required every learned arm to show ≤ 20 % incomplete
peer-group batches. All 31 arms failed it at 34.8–35.5 %. Measuring the **control** with the
same counter settles what that meant: T1b at the same cell sits at **37.0 %**, so the bar
fails the thing the treatment is compared against, and the treatment arms are slightly
*better* than the control on the very statistic that flagged them. The bar could not
separate treatment from control at any value the arms could reach.

That bar was written from what "peer-group batching" ought to look like, not from what the
control actually does. The sibling bar in the same registration, C1, was sized against a
measured control number (T1b's 35.5 % above-shallowest, halved to 18 %) and did its job:
it separated the arms and failed them in an interpretable direction.

**Rule:** before signing a control bar that asserts a policy behaved as named, compute that
counter on the control arm and put the number in the registration next to the bar. A bar the
control cannot clear is not a control.

**And do not move it afterwards.** C0 stands as signed, with the control's value recorded
beside it in the node; the verdict is carried by the bar that does separate. Re-tuning a bar
after seeing the arms fail it is how a registration stops meaning anything.

## 2026-09-15 — a "clears in ≥ N families" bar must check the SIGN, not just |ρ|

`offline_live_transfer_v1` registered two bars of the same shape a few paragraphs apart and
only one of them checks direction.

* **R1's within-arm bar** requires |ρ| ≥ 0.50 **in the same arm and the same sign** in ≥ 2
  families. When x800p3/`gnn` read −0.553 and x800p2/`gnn` read +0.150, it correctly refused
  to count them together.
* **R2's family sub-bar** requires |ρ| ≥ 0.40 in ≥ 2 families and says nothing about sign. So
  `depth` scored "2 of 3 families clearing" on **−0.490 and +0.710** — two opposite
  relationships counted as agreement.

It changed no verdict (that candidate failed the other two sub-bars and nothing was promoted),
and the bar was left as signed rather than retuned mid-lineage. But a candidate that cleared
the ρ and p bars could have been promoted on evidence that points both ways.

**Rule:** any bar of the form "the effect reproduces in ≥ N groups" must require the effect to
have the **same sign** in those groups. Two strong correlations in opposite directions are
evidence *against* reproducibility, and an |ρ| threshold scores them as evidence for it.

Cheap check when writing one: ask what the bar does with −0.9 and +0.9. If it passes, it is
measuring magnitude, not reproducibility.

## 2026-09-15 — an ssh-polling wait loop reads a connection failure as job completion

The obvious way to wait for a SLURM array is to poll `squeue` over ssh until it reports
nothing:

```bash
until ! ssh datalab 'squeue -j $JOB -h | grep -q .'; do sleep 180; done   # WRONG
```

`ssh` returns non-zero when the **connection** fails, not only when the job is gone, so the
negation makes a login-node outage look exactly like a finished job. The datalab login node
refuses connections in bursts (seen twice on 2026-09-15, clearing after ~45 s), so this is
not hypothetical: a watcher on `serving_stability_v1`'s S3 gate announced the array complete
while **90 of 96** arms had finished and 7 were still running. A read fired on that signal
would have scored an incomplete gate and, worse, scored it as though the missing arms had
**hung** — which is precisely what that lineage's blocking deadlock bar counts.

Require the poll to **succeed** and report zero, so a failed connection retries instead of
terminating the wait:

```bash
until out=$(ssh -o ConnectTimeout=20 datalab 'squeue -j $JOB -h | wc -l' 2>/dev/null) \
      && [ "$out" = "0" ]; do sleep 90; done
```

**General rule:** when a wait loop's exit condition is "the remote says no", make sure the
loop can tell "the remote said no" from "the remote said nothing". Any check whose failure
mode is indistinguishable from its success mode is not a check.

## 2026-09-15 — name the PAIRED test when the arms are paired, before the data decides for you

`serving_stability_v1`'s S3-d compares the same 16 seeds with and without a serving knob —
paired by construction — and the bar I signed named **Mann-Whitney**, which is unpaired. On
the cell where the effect was small and perfectly consistent the two tests disagree completely:

| cell | seeds better | Mann-Whitney (registered) | Wilcoxon (matched) |
|---|---|---|---|
| s9001 `gnn` | **16/16** | **p = 0.127** | **p = 0.000031** |

A 16/16 sweep reading p = 0.13 is the tell: an unpaired test throws away the pairing, so a
small, utterly consistent shift sits inside the between-seed spread and vanishes. Under the
matched test that lineage's S3-d verdict flips from GUARDRAIL-DOES-NOT-HELP to GUARDRAIL-HELPS
and its outcome from STABILITY-NOT-THE-LEVER to HELPS-NOT-ENOUGH.

**The verdict was left as signed and the sensitivity recorded beside it** — swapping a test
after seeing which way it moves the answer is how a registration stops meaning anything. But
the fix is to not be in that position: **when the arms share seeds, register Wilcoxon
signed-rank (or a paired t) and report the unpaired test as the conservative secondary**, not
the other way round. `drainable_objective_v1` carried the same mis-specification three weeks
earlier and got away with it only because both tests agreed there.

Cheap check when signing a bar: ask whether swapping which arm a given seed belongs to would
change the statistic. If it would not, the test is ignoring the pairing you paid for.

| 2026-09-15 | `queue_range_v1_read.py` — Q0 (bar design, disclosed and NOT moved) | **A "does it get worse late?" bar that names specific deciles assumes a drift.** Q0 asked whether the served queue column leaves its trained range more in deciles **9–10** than in 1–2. The mechanism is there on 2 of 3 cells and it is an **excursion**, not a drift: `cell_s7901` peaks at deciles **4–5** (out-of-range share 0.36) and recovers to 0.03 by decile 9; `cell_s9002` peaks at decile **7** (0.83). Read at deciles 9–10 the bar fires on one cell and the verdict is `COLUMN-IN-CONTRACT` — the opposite of what the per-decile curve shows. The bar was mis-**shaped**, not mis-tuned, and moving it after seeing the curve would have been tuning on the data. | Verdict stands as signed; the curve is in the node and the reading is "fires in shape on 2/3, not on the bar". **Rule: a positional bar must name the statistic (`any decile above X`, `max over deciles`) rather than a position, unless the registration also argues why the effect must be monotone.** `docs/lessons.md` already separates excursions from drifts (`offline_live_transfer_v1` R3) — that distinction belongs in the bar, not only in the interpretation. |

| 2026-09-15 | `queue_range_v1_read.py` — Q4 (bar design, disclosed and NOT moved) | **A "does X extend Y?" bar sized against a control value measured on ONE cell when the family has three.** Q4 counted deciles in which the arm's queue is below reactive's, bar ≥ 4, against a control value of **2** taken from `serving_stability_v1` S1. S1 reported that deciles 1–2 are ahead; it never asked how many deciles are ahead. Measured here, the **control itself** reads 2 / **10** / **4** on s7901 / s9001 / s9002 — it clears the bar on 2 of 3 cells before any intervention, and all three arms (`plain`, `pinned`, `inrange`) return **identical** counts. Q4 therefore fires `ADVANTAGE-EXTENDS` while the intervention changed nothing. | Verdict recorded as a **control artifact** and the counts kept, because the control reading is the interesting one (on s9001 the arm leads reactive on queue in **10/10** deciles and still loses the cell — see the per-term decomposition). Same error shape as `drainable_objective_v1`'s C0, whose bar failed its own control. **Rule, now twice: measure the control's own value of the bar statistic, on every cell in the family, before signing the threshold.** |

| 2026-09-16 | `scheduler_residence_v1_r3_read.py` — the confound control (disclosed, NOT moved) | **A ratio control is degenerate when the primary is null.** R3 registered: if the structure predicts *reactive's own* queue at least `R3_CONTROL_RATIO = 0.75` as strongly as it predicts the learned arm's excess, with matching sign, the primary reads `CONFOUNDED-ENVIRONMENT` whatever its ρ. Correct in shape — `drainable_objective_v1`'s C0 and `peer_affinity_v1`'s H5 both died of a control that moved with the treatment. It then fired on a primary ρ of **+0.041**, where the threshold it computes is 0.031, so any control ρ above noise trips it: the gate printed `CONFOUNDED-ENVIRONMENT` for a plain **null** (p = 0.885, wrong sign, bar \|ρ\| ≥ 0.60). | Verdict stands as signed; the node records both the printed verdict and the substantive reading (`LOPSIDEDNESS-DOES-NOT-PREDICT`), and the read's own output prints the primary ρ and p beside it so the degeneracy is visible. **Rule: a control expressed as a fraction of the primary's magnitude needs an absolute floor on that magnitude, below which it reports NOT-APPLICABLE — a third branch, not a retuned ratio.** A null and a confound are different findings and the instrument must be able to say which; printing the one that means "a real effect was obscured" for "there was no effect" inverts the conclusion for anyone reading the verdict line. |

| 2026-09-16 | live gates that mint cells from topology seeds (measurement, not a code bug) | **5 of 20 independent topology draws are unservable by EVERY policy.** `scheduler_residence_v1` R3 minted 20 cells differing from a working base in exactly one field (`network.topology.seed`); 5 hung with the documented starved-client signature — 100 % CPU, no log growth, frozen on `Gateway: Processing event` — and `knative_network` hung on them too, so it is a property of the draw, not of a scheduler. Batch 1 lost 4 of 12 and spent ~18 CPU-hours at a 30-minute `--time` waiting for spins that were never going to finish. | **Budget 25 % attrition when sizing any gate whose cells come from fresh seeds**, and set `--time` to ~3× a servable arm (here 15 min against ~5 min) so a hang is identified in a third of the wall clock. Also: **whether the attrition biases the sample must be measured, not assumed** — R3 compared the hung cells' statistics against the readable ones and found the primary's hung range *inside* the readable range (no selection) but the second statistic's low end truncated (selection). Both are in the node; neither was corrected for. |

| 2026-09-16 | `cluster_scale_v1_s0b.sbatch` — arm naming collision (measurement defect, fixed) | **An arm name that does not carry every axis the gate varies makes the idempotence guard a collision.** Job 769390 wrote ARM as `${CELL}__${KIND}` (cell + policy), but the gate also varies the workload rung. All three rungs resolved to the same summary path, so the f300 rung found the f1000 rung's file and exited in 1 s per task with "exists, not re-running". Nothing failed loud — the guard did exactly what it was designed to do against the wrong key, and the f300 rung produced no data. | Fixed: ARM is now `${CELL}__${KIND}__${RUNG}`. The transferable rule — every axis a gate varies must appear in the arm name — is filed once, in [`docs/lessons.md`](../lessons.md). The tool-level cause: an idempotence guard tests "did this arm already run", so when two distinct arms share a name the second silently inherits the first's file. This is the same class of silent re-use defect as `drainable_serving_config_v1`'s window override, where the cell config silently overrode `GNN_BATCH_TIMEOUT`. |

| 2026-09-16 | `partial_state_v3_cache.sbatch` / any cache over the T1b corpus (venue trap, documented) | **The T1b/`drainable_objective_v1` caches were built on the LOCAL checkout, not on datalab, and cannot be rebuilt there.** Job 769544 died in 8:44 on `Missing system_state_captured_unique.json for ds_00000` in `gnn_datasets_peer_affinity_v1_c3_x200_train2`: datalab's copy is the 1500-dataset in-place extension and carries **0** SSC files, while the local copy is the original 346 parents with all 346. `sacct` shows no datalab job at the v2 cache's build time (2026-09-15 08:07 UTC) — it was built locally and rsynced, and every earlier `dobj-cache` job on datalab (766312/766313/766316) had failed the same way without anyone reading the failure. | Build caches over this corpus **locally** (same command, same env), run P0 locally, rsync cache + P0 artefact with md5. **Rule: a cache's `metadata.json` names its base dirs but not the host; when a cache exists and no job built it, look for the local build before assuming the cluster did it.** Repairing datalab's `train2` (`refresh_optimal_full_stats.py --repair` on the 346 parents) is a separate, unregistered chore. |

| 2026-09-16 | `partial_state_v3_read.py` — P3-a as first coded (disclosed; tool corrected, verdict recorded as printed) | **"Every arm must complete" was coded as "any missing arm is a representation failure."** P3-a's registered text names the failure classes it is about — a `krank_node_order` raise or a contract refusal at 12+ servers, the reason the lineage exists — but `read_p3` printed `STILL-PINNED` for *any* learned arm short of its expected count. Job 769872 lost 2 of 144 arms (cs12s9005 gnn s4, cs24s9005 gnn s5) to **OUT_OF_MEMORY at the 48 GB cap after ~17.5 min**, both past the last arrival of their trace (sim t = 54,392 s and 28,244 s against traces of ~54,300 s and ~27,150 s) — a resource kill at the end of a run, the same failure class as a hang, and nothing to do with the representation, which served **all 140 other scaled arms** without a raise. The read printed `STILL-PINNED` by arithmetic. | Printed verdict recorded as printed; the substantive P3-a reading (no representation failure on any of 142 arms) recorded beside it; the two arms re-run at 120 GB (job 770351) and the read re-issued on the completed set. **Rule (the R3 ratio-control rule again, one level up): a bar that names a failure CLASS must be read by cause, not by count — a read that cannot tell a hang from a raise needs a third branch (`INCOMPLETE-OTHER`, disclosed, rung read on the completed arms) rather than firing the class it was written for.** `read_p3` now takes per-arm causes; unclassified losses are disclosed and never counted as pinned. |
| 2026-09-17 | `peer_only_v1` Phase B training — the corpus, not the gate (two blocking defects, both fixed) | **A split artifact minted for one corpus, and a cache candidate the sweep never priced.** (a) Job 782116 died in 9 s on every arm: `assert_split_artifact_covers` requires the artifact to enumerate **exactly** the cache's parents, and the T1b artifact knows 516 while the cache holds 1,657. (b) Job 782166 died at ~2.5 min on every arm: `refresh_partial_state_edge_attr ... candidate absent from the partial-state context (missing key (22, 118))`. The demand table is built from the **placement sweep's rows**; the graph's candidate set lists **every replica of the task's type**. In 3 of 1,141 new `train2` datasets a replica exists that no sweep row ever places on — sweeps complete, `sweep_complete` true, dataset structurally valid — so the trainer has no cost for a candidate it is asked to score. **0 of the 516 T1b parents** has it, which is why 3.5 months of training never saw it. | Both are now pre-training guards rather than post-mortems. The split artifact is **minted from the cache in the same job** (`peer_only_v1_split.py`, test and val carried verbatim so the corpus contrast pairs like with like), and `peer_only_v1_candidate_check.py` walks every graph's `task_logit_to_placement` against its `partial_state_ctx["demand"]` and **exits 1 with the offender list** — it read `offenders=0` on the 1,654 rebuild, so the class is proven absent, not assumed. **Rule: a corpus is validated by the trainer's own contracts before a GPU is booked, not by the generator's status codes.** `SUCCESS` and a complete sweep do not imply every candidate is priced. |
| 2026-09-17 | any SLURM gate array wider than the account's `MaxSubmit` | **An array counts every task against the submit limit, so a 96-task gate is rejected outright.** `sbatch --array=36-131%12` returned `AssocMaxSubmitJobLimit` / `Job violates accounting/QOS policy` against `MaxSubmit = 50`. The `%12` throttle limits **concurrency**, not submission, and does not help. It surfaced at the worst moment — inside a chain, after a 5-hour training array had already drained — and the chain reported `PHASE B GATE NOT SUBMITTED` with an empty job id. | Phase B was submitted as two 48-task halves, chained. **Check before writing any chain: `sacctmgr -n show assoc user=$USER format=MaxSubmit` and split the array below it.** A chain that submits a later stage must also **verify the returned job id is numeric** before waiting on it — the wait loop polled `squeue -j ''`, got zero rows, and would have declared the stage complete had the chain not tested the id. |
| 2026-09-17 | any live gate that crosses checkpoints with cells (`peer_only_v1` A2/B2; disclosed, bar NOT moved) | **`n` pairs is not `n` independent draws, and the read prints only the pair count.** B2 crossed `CKSEEDS = (1, 2, 4, 5)` with 4 topology cells and reported **16/16, p = 0.0004** — as registered (`A2_MIN_PAIRS = 12` pairs by (cell, checkpoint seed)), so no bar was violated. But the headline is a claim about an **architecture**, whose independent unit is the checkpoint, and there were **four**. A two-sided sign test on 4 units cannot go below **p = 0.125**. The same read quoted "−41.4 %, 16/16 vs reactive" from **one** reactive run per cell, re-used four times. The sister lineage's "13/16 seeds" (`peer_affinity_v1`) *was* 16 training seeds, so the two numbers look identical in a node and are not. | Amendment 1 registered `B3`: the same contrast collapsed to one value per checkpoint (median over its cells, `collapse_to_seed`, which fails loud on a ragged row) over all 16 trained seeds, with the consequence signed before the data. **Rule: the registration must name the independent unit for the claim being made, and the read must print the pair-level and unit-level statistics side by side.** Crossing K checkpoints with C cells buys precision on the cell average; it does not buy degrees of freedom for a statement about the model. Corollary for sizing: train 16 seeds and then serve 4 of them is the wrong split of a fixed budget when the claim is architectural — the gate is ~2 min per arm here against ~40 min to train one. **Measured price, same day:** serving the other 12 checkpoints (96 arms, ~40 min) took the headline from **−15.68 %** to **−5.75 %** — the 4-checkpoint read overstated the effect by more than **2×**, 2 of 16 checkpoints changed sign, and every descriptive figure moved with it (the twin reads −36.81 % against reactive, not −28.00 %). Amendment 2 then did the same for the opposing clause and it moved +10.40 % → **+11.17 %**, i.e. a thin `n` is not biased toward the exciting answer — it is simply *noisy*, and which way it errs is not knowable in advance. This is [`docs/lessons.md`](../lessons.md) → "inheriting a statistic does not inherit its power" priced in wall-clock. |

### A reader that prints an absent count as `None` (2026-09-18)

`bipartite_edge_v1`'s gate reader printed **`None/16`** for the ahead-count of every bar,
beside medians that were entirely correct — including a `−20.37 %` headline. The cause is
mundane: `paired_tie` returns the count as **`v3_ahead`** (its first caller was
`partial_state_v3`) and the formatter read `ahead`.

It matters more than a cosmetic slip because the ahead-count is what separates a median carried
by all 16 checkpoints from one carried by nine. The corrected read turned D2 at R3 from
"clears its bar" into "clears its bar at **11/16 with p = 0.0437**", which is most of the reason
the lineage closed on `CONV-IS-THE-LEVER` rather than on the attributes.

**The rule: a missing statistic must refuse, never render.** `_fmt` now raises rather than
printing `None`, so a headline cannot be produced without its count. Any `dict.get` feeding a
printed number wants the same treatment — `None` in a report reads as a reported value.

### A gate task table that outgrew `MaxArraySize` (2026-09-18)

`peer_only_v1_gate.sbatch`'s task table has been appended to by five lineages and reached
1,108 entries. This cluster's `MaxArraySize` is **1001**, so `sbatch --array=964-1011` is
rejected outright with *"Invalid job array specification"* — not when the task runs, but at
submission, and only for the blocks whose indices cross 1000. `bipartite_aggr_v1`'s first
block (916–963) submitted and completed normally; the second failed instantly.

The tempting fix is a second gate script. **Do not** — two gates that share cells, workloads,
physics and a summary schema drift apart in exactly one constant, and then a paired comparison
is quietly a cross-gate one. That risk is what kept `bipartite_edge_v1` and `bipartite_aggr_v1`
inside this file in the first place.

The fix is an offset, so one task table keeps being the single source of truth:

```bash
sbatch --array=0-47 --export=ALL,PO_TASK_OFFSET=964 scripts_cosim/datalab/peer_only_v1_gate.sbatch
```

`T = SLURM_ARRAY_TASK_ID + ${PO_TASK_OFFSET:-0}`, defaulting to 0 so every earlier submission
spelling is unchanged, and the task logs the mapping when the offset is non-zero.

**The rule: check `scontrol show config | grep MaxArraySize` before appending a block that
would cross it**, and treat a growing task table as a thing with a ceiling rather than a list.

## "1670" is a label, not a count — the corpus is 1,654 datasets (2026-09-19)

Every arm tag, split filename and cache directory in the `peer_only_v1` family says **1670**:
`peer_only_v1_1670_split.json`, `graphs_cache_peer_only_v1_1670_psv3`, the arm labels
`1670_mpoff` / `be1670_gnnedge0` / `ba1670_gnnedgesum`. The corpus behind all of them is
**1,654 datasets** (split 1524 / 96 / 34), which is what the record itself has always quoted —
CLAUDE.md and `peer_only_v1`'s own head both say "the 1,654-dataset corpus".

`pointwise_baseline_v1` asserted `num_datasets == 1670` in its training sbatch, on the
strength of the name, and **failed 9 of its own selection tasks**. That is the guard working:
the alternative was training a corpus-matched arm against a number nobody had checked.

**The rule:** take a corpus size from the cache's `metadata.json` and the split artifact's own
totals, never from a filename or an arm tag. If a guard asserts a size, derive the expected
value the same way — and when a name and a measurement disagree, the measurement is the corpus
(`docs/lessons.md` → "A corpus is its SPLIT, not its directory").

The names are **not** being renamed: 1,670-tagged checkpoints and result files are on disk
across five lineages, and a rename would break every arm label the readers match on. The tag
is a name; this entry is the translation.

## 2026-09-20 — `collapse_to_seed` compared the arm's median cell with reactive's median cell: an unpaired ratio of medians

`scripts_cosim/peer_only_v1_read.py::collapse_to_seed` produced one value per checkpoint as the
**median over cells of the arm's raw elapsed**, and the chain then divided that by the same
collapse of reactive (`paired_tie`, relative). The two medians come from different cells. On the
B7 client rungs reactive's elapsed spanned 18–64 s across four cells, so the arm's median cell and
reactive's median cell were never the same cluster, and the ratio said **−20.3 %** for an arm
that loses three cells of four (+4.6 / +75.8 / +9.7 / −36.8 %). Every "beats reactive" number
read on `cc40*` / `cc80*` cells by `peer_only_v1`, `bipartite_edge_v1`, `bipartite_aggr_v1`,
`best_arm_v1` and `corpus_matched_v1` used it; paired per cell every one flips sign
(`docs/lineages/unsaturated_edge_v1.md`). Arm-vs-arm contrasts on the same cells are unaffected
only where both arms' median cell happened to coincide — treat them as directions, not magnitudes.

**The fix is the statistic `unsaturated_scale_v2` introduced**: relative % against the reference
**on the same environment first**, then the median over environments
(`unsaturated_scale_v2_read.checkpoint_stats`); arm-vs-arm pairs raw elapsed on environment AND
checkpoint (`pair_checkpoint_stats`). A reader that collapses cells before it pairs is wrong
whenever the cells differ in load — which is exactly when a screen would have excluded some of
them. Compounding defect: two of the four cells (9002, 9005) were saturated on w0 (share 0.87,
0.80) and supplied the only wins; the server ladder had an admissibility screen, the client
ladder never did.

## 2026-09-22 — Research guidance after the AGENTS.md migration

The old hygiene test demanded that retained `.claude/CLAUDE.md` acquire the date
of every new lineage, conflicting with the Codex migration's rule that guidance
routes to result records and the retained Claude source is not edited as part of
this workflow. The check now preserves the Claude migration baseline, verifies
AGENTS.md's research-entry-point paths, and continues to enforce node-head
freshness and index/node status agreement. No result-freshness check on the
owning lineage nodes was removed.

## 2026-09-22 — Determinism smoke isolation in the managed workspace

The A1 end-to-end determinism tests claimed to use a temporary model directory but
actually ran from the repository and selected the newest production checkpoint.
Here that failed on the inherited unwritable model directory; it could also race
with another training job. The smoke now invokes the absolute runner from its own
temporary working directory and reads only that directory's checkpoints. Its W&B
artifact cache is isolated there too, avoiding the read-only home cache. The tiny
12-graph smoke uses zero DataLoader workers: the previous forked-worker attempt
hung after earlier in-process Torch tests. Subprocesses have a 120-second timeout
so recurrence fails loudly. Same-seed equality and different-seed divergence checks
remain unchanged; production trainer defaults are unchanged.

## 2026-09-23 — Sampled-slate lookup censored decoded validation plans

The 32-task sampled-slate trainer selected epoch-zero checkpoints because its
validation reader assigned a worst-regret fallback when a decoded plan was not
one of 16 pre-scored slate rows. Final-epoch models often proposed such plans;
the lookup measured slate membership, not their simulator RTT. Exact replay on
the assigned validation topology changed all three architecture selections to
final checkpoints ([g32_exact_val_selection_v1](../lineages/g32_exact_val_selection_v1.md)).

For a decoder that can leave a finite label slate, select checkpoints using
exact simulator replay of the decoded plans under the gate's physics. Verify
physics parity by replaying at least one cached slate plan and matching its RTT
exactly before interpreting new plans. Keep the validation topology separate
from the held-out test and fresh live gate.

## 2026-09-24 — AGENTS.md is canonical; `.claude/CLAUDE.md` is a thin import

The 2026-09-22 split (retained `.claude/CLAUDE.md` as the continuously-updated
narrative, AGENTS.md as routing-only) avoided narrative duplication by keeping
Claude Code's default read path — CLAUDE.md, when present, is read instead of
AGENTS.md, per Claude Code's own AGENTS.md-interop rules — the sole source. It
still left every *other* section (rules, commands, architecture, feature
contracts) duplicated verbatim across both files by hand, with no test over
that half and a Codex skill already citing "AGENTS.md rule 6" against a file
that did not carry the rules.

Reversed instead: AGENTS.md is now the single canonical file, and
`.claude/CLAUDE.md` is a two-line `@../AGENTS.md` import plus a short
Claude-only appendix (hook/settings pointers), enforced by
`test_claude_md_is_a_thin_import`. Claude Code resolves the import at read
time, so both tools now read one file with no manual mirroring anywhere.
`test_retained_claude_guidance_has_migration_baseline` and
`test_agents_guidance_routes_to_existing_lineage_records` are replaced by
`test_agents_md_has_standing_answer_stamp`, `test_agents_research_entry_points_resolve`,
and `test_agents_md_stays_within_budget` (30,000 B — the merged file's real size
plus headroom, not the old 22,000 B CLAUDE.md-only budget).

## 2026-09-24 — "Reactive" in the rollout_imitation_v1 gate was Knative-ECT

`scripts_cosim/datalab/rollout_imitation_v1_gate.sbatch` runs `knative_network_ect` and its reader
labels that arm "reactive" (bar R3). `peer_greedy_live_v1` and `joint_burst_v2` gate against plain
`knative_network` and report ECT separately; ECT is the weaker arm (it loses +12.8 % to
shortest-queue, `batch_window_edge_v1`). So `rollout_imitation_v1`'s "beats reactive −7.8 % / −28.6 %"
is a margin over ECT, not over the programme's reactive baseline. Found by the sanity line of
`scripts_cosim/selfpredict_bar_v1_read.py`: on that gate's summaries the rule beats "reactive"
−28.7 % / −32.0 %, against −12.96 % / −16.01 % over `knative_network` in `peer_greedy_live_v1`.
Before quoting any "vs reactive", name the arm.

## 2026-09-24 — Scale-down could strand a platform's worker forever (fixed in `src/placement/autoscaler.py`)

Scale-down reset a removed platform with `platform.initialized = env.event()` unconditionally. When
the old event had never fired, `platform_process` could already be parked on it (`yield
self.initialized`); scale-up then fired the NEW event and the worker never woke. Every task placed on
that platform afterwards queued forever, the run never finished, and the per-second monitor grew
memory until SLURM's OOM killer ended it. Found in `selfpredict_burst_v1` (cc40s9101 w0,
`peer_greedy_selfpredict_network`, 48 GB OOM after 17 min): 301 `dnn2` tasks behind node1/203, whose
worker waited on an untriggered event that was no longer `platform.initialized`. Any policy can reach
it; a run that completed never hit it, since hitting it is a permanent hang. Fix: replace the event
only if it has fired. The same cell then completes (6.29 s/task). Neutrality check: `selfpredict_burst_v1`
reran every arm at the fix commit and compared each completed cell against the pre-fix run.

## 2026-09-25 — A rate ladder scaled the cfg's time constant and missed the hardcoded ones

`cd_gap_v1` B stretched every arrival timestamp ×2 and, following the hard-stop rule ("scale every policy
time constant"), scaled the one constant a cell config carries, `scheduler.batch_timeout`. It did not
scale `KEEP_ALIVE` (30 s) or `RECONCILE_INTERVAL` (1 s). Both are hardcoded in
`src/placement/constants.py` and passed by `src/executesimulation.py`, so no cfg or env var exposed
them. Replicas therefore idled out twice as often relative to traffic. The tell was a sign: at the
lighter rate every batched arm's queue fell, while self-predict's and reactive's *rose*. With both
constants ×2, self-predict got 15–22 % faster on the cells checked. **Fix:** `HEROSIM_POLICY_TIME_SCALE`
multiplies both; unset is bit-identical (venue witness reproduced). `fresh_topo_burst_v1_gate.py` sets it
from the cell's recorded `cd_gap_v1_rate_scale.factor` and fails a run whose provenance lacks it.
**The general rule:** before a rate ladder, list the policy time constants from the code, not from the
config. After it, check that each arm's queue moves in the direction the load moved.

## 2026-09-25 — The uncapped burst seat served a node rank the model was never trained on

`joint_burst_v2` moved the served decoder to the uncapped rung (`alpha = inf`), and nobody re-ran the
train/serve parity check (`scripts_cosim/peer_affinity_live_serve_check.py`) on that seat until
`load_repr_v1`.

- **The defect.** `prefix_serving.attach_live_prefix_block` built caps as `inf × peak demand for m > 0`.
  That dropped every node whose candidates all have zero memory demand (`pynqFpga` `dnn1` = 0.0), and
  `krank_node_order` then ranked those nodes first at cap 0.0. The cache's uncapped rung is `{}`, which
  puts every node at 0.0.
- **The size.** On 60 held-out jb2 batches, `jb2` `gnnedge0` s1 was served a different `krank` on 56
  and decoded a different plan on 8. Every burst-seat learned-arm gate before 721d44f carries it.
- **Fix:** 721d44f serves `{}` at `alpha = inf`. The decoder reads `caps.get(n, inf)`, so only the rank
  changes. Parity after the fix: 58–60/60, with the leftovers same-node sibling-platform ties at equal
  RTT.
- **Measured cost:** `gnnedge0` fixed vs as served is +1.4 % (0/12). The defect did not inflate any
  learned headline by more than that, and it did not favour the rule arms.
- **The general rule:** run the parity check whenever a decode rung, a contract or a serving flag
  changes. A seat change is a new serving path, even when the checkpoint is not new.

## 2026-10-05 — `total_rtt_plus_inference` times the batch schedulers only, CD included

Decision time is wall clock (`default_timer`) around the batch decision in `GNNScheduler._process_task_batch_prefix`
(`src/policy/gnn/scheduler.py`), split per task into `gnn_decision_time` and summed by the orchestrator. Every policy
that inherits that batch path is timed: the GNN and MLP arms **and** CD, `cdextr`, the one-pass greedy and
locality-first (`PeerGreedyNetworkBatchScheduler` subclasses `GNNScheduler`). The per-arrival schedulers (Knative,
self-predict, drain, random, Decima's rule) never set it and read 0.

- **Measured on the 19 grounded topologies** (server-only gates, 62 simulations per node, single-threaded): per
  50,000-task run CD takes 3.7 / 5.1 / 11.2 s at ×2 / ×3 / ×5, `sb1load` 92 / 93 / 102 s (8–24× CD, paired, 19/19),
  `lf1gnn` 172–191 s, `lf1mlp` 66–80 s. Both are ≤ 0.02 % of total latency. Ratios are fair; absolute values are
  inflated by the shared node.
- **Rule:** a "latency + inference" contrast against a per-arrival rule charges only one side. Say which arms are timed.

## 2026-10-05 — Raw results on the home quota: three gates at once fill it

`fresh_topo_burst_v1_gate.py` writes each simulation's ~250 MB raw result next to its summary and deletes it once
summarised, so a gate at 62 parallel holds ~15 GB at any moment. Three such gates plus training reached the 250 G home
quota at 19:09 on 2026-10-05: runs ended with rc 120 (Python's stdout flush failing at exit, no traceback) and training
tasks died mid-epoch with nothing in their logs.

- **Fix:** `HEROSIM_RAW_DIR` (default: the gate dir, so older sbatches are unchanged) moves the raw file to node-local
  `/tmp` (936 G free on CPU-amd nodes); `client_local_v1_*.sbatch` and the newer gate sbatches set it.
- **Rule:** rc 120, or a run whose log stops mid-simulation with no error, is the quota until shown otherwise; check
  `du -sh ~` before rerunning, and remove those records — they are not outcomes.

## 2026-10-12 — `read_training_curves.py` read rising accuracies as falling and train/ce against the wrong floor

The wandb curve audit (343 run dirs) found two defects in the reader. Neither changes a checkpoint, selection or live number, but both are in curve reads quoted since the plan-exact metrics landed.
- **Peak direction.** The SELECTION block took `max` only for names ending in `acc`. So `val/mt_plan_exact`, `val/mt_task_acc_choice` and `val/task_acc_choice` had their *minimum* reported as their peak, e.g. "peaks at epoch 0" when they peak at epochs 57–99. Now a name containing acc, exact, choice or agreement is higher-is-better unless it is a regret, ce or loss (`higher_is_better`). Read the prefix-free ones with [lessons.md "Prefix-free accuracy drifts down as the selector improves"](../lessons.md): their max is not the run's decision.
- **train/ce floor.** train/ce was compared with `baseline/val_chance_ce`, the val split's one-term plan NLL. Under `NEAR_RTT_FULL_CONTEXT_CE_WEIGHT=1.0`, train/ce adds a second plan NLL, so 25 grounded runs read "train/ce NEVER BEATS CHANCE" while val/ce beat it. Only val/ce is now read against the floor, and the output says why. The weight is in the checkpoint contract, not the W&B config, so the floor can't be scaled from the run dir.
- Tests: `tests/test_read_training_curves_direction.py`, on two trimmed saved runs in `tests/fixtures/wandb_curves/` (history and summary records only).
