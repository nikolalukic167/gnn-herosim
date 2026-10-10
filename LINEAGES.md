# LINEAGES.md — the index of what is known

**Read this before starting work.** It is the one entry point to this repo's research
record: every lineage, every gate-tool correction, every transferable rule, and every
closed question. `simulation_data/REGISTRY.json` does the same for datasets.

This file is an **index only**. It carries a status and a one-line outcome per lineage;
the full record lives in the node file each row links to. Nothing here is a summary you
may cite — cite the node.

## The graph

The routing table — which file owns which kind of fact — lives in
[`AGENTS.md`](AGENTS.md) under **Where knowledge lives**, and is not
repeated here. It was duplicated in four places and had already drifted: this file claimed
`docs/lessons.md` held 350 rules when it held 67.

In short: **this file is an index**, one status and one line per lineage. The record lives
in [`docs/lineages/`](docs/lineages/). Nothing here is a summary you may cite — cite the
node. `tests/test_record_hygiene.py` enforces it.

## Statuses

| Status | Meaning |
|---|---|
| `ACTIVE` | Current work. Change this code. |
| `REGISTERED` | Pre-registered, not yet run. The registration is signed off and dated; the result is not in. |
| `PARKED` | Paused by a dated decision, not answered. Every measured result in the node stands; resuming requires a signed amendment in the node, not just picking it back up. |
| `CLOSED` | The question was answered. Result stands, no further work — re-opening needs a new question, not a re-run. |
| `SUPERSEDED` | Replaced by a later lineage. Result still stands; don't build on it. |
| `FAILED` / `FALSIFIED` | The gate failed, or the hypothesis was disproven. **Do not revive without new evidence.** |
| `SYNTHESIS` | Not an experiment — a cross-lineage reading of several. |
| `PAPER` | Frozen because the paper cites it. Change only with a paper edit. |

## Open — the current program

| Lineage | Status | Outcome |
|---|---|---|
| [**accel_nosplit_v1**](docs/lineages/accel_nosplit_v1.md) | `ACTIVE` | Accel GNN served with split 9–10-task groups decoded whole vs CD, 24 fresh topologies, both rungs, Holm over 4; cd_pull descriptive. |
| [**mixed_dispatch_v1**](docs/lineages/mixed_dispatch_v1.md) | `ACTIVE` | Reservations win 11/16 observed comparisons, but the trained DAG successor loses to hand search; dag_resume_s0_v1 also closes its bounded continuation recipe with no relative gain. |
| [**radical_physics_v1**](docs/lineages/radical_physics_v1.md) | `ACTIVE` | One-shot and pair-selector pilots are negative; execution-order successor mixed_dispatch_v1 passes stronger-control headroom screens, without a GNN result. |
| [**transfer_physics_v1**](docs/lineages/transfer_physics_v1.md) | `ACTIVE` | 2×2 read: CD is fastest in every condition and topology; held replicas made queues dominate (released: queue share 0.05–0.08); Knative's gap shrinks; zero-shot GNN falls further behind. Scale-out factor registered as kpa_scaleout_v1. |
| [**reference_physics_programme**](docs/lineages/reference_physics_programme.md) | `REGISTERED` | Programme plan (9 nodes): KPA scale-out → audit and freeze R1 → fix workload WF1 → recalibrate load → train once and attribute; stopping rule after nodes 1–5. |
| [**call_graph_pairing_v1**](docs/lineages/call_graph_pairing_v1.md) | `REGISTERED` | Trace-derived pairing, synthetic exchange semantics: Casper call graphs decide which siblings exchange (2 partners kept); learned arms zero-shot. |
| [**access_link_contention_v1**](docs/lineages/access_link_contention_v1.md) | `REGISTERED` | Hypothesis: peer exchange through shared access-link pipes; precondition link wait ≥ 15 % of exchange; best learned vs contention-aware CD. |
| [**live_headroom_v1**](docs/lineages/live_headroom_v1.md) | `REGISTERED` | Offline single-batch headroom of CD on the live states it produces at ×2/×3/×5 (exhaustive sweep, no synthetic backlog): is the environment closed for any learned model? |
| [**joint_burst_v2**](docs/lineages/joint_burst_v2.md) | `CLOSED` | GNN-BEATS-GREEDY / CD-STILL-AHEAD: uncapped, gnnedge0 beats the 1-pass greedy in its own seat (K1 −11.9%, 13/13), reactive (−34.7%) and random (~−48%) — v1's +16.8% loss was the SERVING CAP (K6 uncap-alone −8.8%; K5 corpus-neutral), not the model. Loses to CD greedy (K2 +12.5%); ties the MLP twin (K4). Next: rollout_imitation_v1. |
| [**drainable_debug_v1**](docs/lineages/drainable_debug_v1.md) | `REGISTERED` | Why do both learned arms lose to reactive Knative at ρ ≈ 0.16? Bars signed before any datum exists; parents are `drainable_regime_v1` and `drainable_serving_config_v1`. |
| [**peer_affinity_v1**](docs/lineages/peer_affinity_v1.md) | `ACTIVE` | Message passing beats its own MP-OFF twin **+5.14 pp** offline (p = 0.001, 13/16 seeds) at 482 datasets — **offline only**. The edge reverses live, is contingent on one platform type the corpus never contained, and reads **−15.94 %** at a defensible load. Never quote a live number without its load factor. |
| [**literature_reeval_v1**](docs/lineages/literature_reeval_v1.md) | `ACTIVE` | **L2D closed out** across 6×6 / 10×10 / 15×15: message passing ties at the authors' learning rate and wins ~1.5–5 pp once converged, so the authors' recipe under-trains. FDD/MWKR beats the learned policy at every size. |
| [**route_b_env_pivot_v1**](docs/lineages/route_b_env_pivot_v1.md) | `PARKED` | **PARKED.** The screen could not measure S0 on its overlap rungs, and even a pass would have fed an objective that option 1 had already closed. Resuming needs a signed amendment in the node. |
| [**route_b_v1**](docs/lineages/route_b_v1.md) | `ACTIVE` | **Stage 1 PASS** — contention plus coupling does produce joint structure. Phase 2's retrain confirms a **TIE** (p = 0.25) once the censored checkpoint selector is fixed: message passing is redundant with prefix conditioning, not harmful, and corpus size is not the lever. |
| [**siv1_full_corpus**](docs/lineages/siv1_full_corpus.md) | `ACTIVE` | First real live-gate FAILED, then **SUPERSEDED the same day** — it measured an uncommitted code diff, not the model. The synced-code re-gate (job 709163) wins **5/5 on `workload-150-100` and `workload-175-100`**, 2W/1T/2L on `workload-125-225`. Corrected-cache retrain (job 709234) is ungated. |
| [**trainer_determinism_v1**](docs/lineages/trainer_determinism_v1.md) | `ACTIVE` | The seed fix reached **1 of 4 trainers**. Three defect classes now — newest: `prepare_graphs_cache` seeded 42 at module import and clobbered every GNN draw's seed. `tests/test_trainer_determinism.py` covers every trainer; run it before training anything you intend to gate. |

## Closed — answered, do not re-run

| Lineage | Status | Outcome |
|---|---|---|
| [**accel_moderate_confirm_v1**](docs/lineages/accel_moderate_confirm_v1.md) | `CLOSED` | DIRECTION-ONLY: 24 fresh topologies, moderate; gnn_eng −4.4 %, physmp −3.8 % vs CD (Holm .09); CD←GNN −3.8 % 24/24. Lead real in direction, ~−4 % magnitude. |
| [**accel_replica_v1**](docs/lineages/accel_replica_v1.md) | `CLOSED` | NOT-SEPARATED: replicas on fastest platform; GNN −5.5/−6.1 % vs CD at moderate (Holm .07/.08), +2.7 % at heavy; CD←GNN −5.0/−3.4 % (12/12, learned seed). No win. |
| [**scale_160_v1**](docs/lineages/scale_160_v1.md) | `CLOSED` | CD-FASTER at 160c×24s: gnn_eng +1.2/+4.0 %, physmp +1.1/+3.7 % vs CD (heavy Holm-confirmed, moderate not separated); CD←GNN −3.5/−3.3 % (learned seed, no twin). |
| [**r1_attribution_v1**](docs/lineages/r1_attribution_v1.md) | `CLOSED` | CD-FASTER: best learned arm (physMP) +3.4/+5.9/+5.1 % vs CD, Holm-confirmed at all 3 rungs; MP ties its twin; CD←GNN −0.4/−1.9/−1.7 % but ties CD←Twin (learned seed, not MP). Successor scale_160_v1. |
| [**load_recalibration_v1**](docs/lineages/load_recalibration_v1.md) | `CLOSED` | RUNGS-FIXED on R1.1 + WF1 by CD effective queue share (queue + compute-lock wait): light ×1.26, moderate ×5.94, heavy ×11.61; all CD guards pass; Knative collapses at moderate and heavy. |
| [**workload_fix_v1**](docs/lineages/workload_fix_v1.md) | `CLOSED` | WF1-FROZEN: W2 payloads + W3 access links + W4 four types on R1.1; CD first at both rungs (self-predict's W2 lead gone); reactive saturates at heavy. |
| [**physics_audit_v1**](docs/lineages/physics_audit_v1.md) | `CLOSED` | R1-FROZEN: pipelined + release + KPA at time scale 1.0; all invariants pass but I5 (reachability-driven replicas); co-sim reproduces live (216 states, p95 0.15 %); labels from t ≥ 360 s. R1.1 (leak fix + 300 s timeout) passes pass 3. |
| [**replica_placement_v1**](docs/lineages/replica_placement_v1.md) | `CLOSED` | NO-LEVER by the registered rule: under R1 only 22 % of CD's replica creations are load-caused; never run. |
| [**kpa_scaleout_v1**](docs/lineages/kpa_scaleout_v1.md) | `CLOSED` | SELFPREDICT-BEATS-CD-UNDER-RELEASE: under KPA scale-out with released replicas self-predict beats CD −6.5 to −9.5 % (19/19, Holm); CD first elsewhere; KPA, not the autoscaler swap, causes it; enters candidate R1. |
| [**record_merge_v1**](docs/lineages/record_merge_v1.md) | `CLOSED` | MERGED: 41 old-physics nodes from local-record-2026-10-06 brought over records-only, with banners; no code changed. |
| [**residency_placement_v1**](docs/lineages/residency_placement_v1.md) | `CLOSED` | Four attempts, 18 checkpoints and 128 audited container runs: no established MP benefit; both learned arms lose to exact on the mixed probe. Fixed eviction; two physical blocks only. |
| [**shuffle_placement_s0_v1**](docs/lineages/shuffle_placement_s0_v1.md) | `CLOSED` | 48 real-TCP streams audit; adaptive search has small exploratory gains, fluid timing validation fails on 2/4 configurations. No GNN training qualification. |
| [**online_reservation_s0_v1**](docs/lineages/online_reservation_s0_v1.md) | `CLOSED` | Hidden-arrival two-decision search gains 0% median over a matched hand rollout portfolio in both fresh phases; 96 materialized HeROsim runs audit, and no GNN is trained. |
| [**l2d_rollout_headroom_v1**](docs/lineages/l2d_rollout_headroom_v1.md) | `CLOSED` | Full future-action rollout misses the two-size 5% training bar, even after guarded four-rule continuation improves 6×6 by 6.38% but 10×10 by only 4.31% on fresh cases. |
| [**l2d_learning_efficiency_v1**](docs/lineages/l2d_learning_efficiency_v1.md) | `CLOSED` | Size-specific GNN fails the both-controls, both-sizes data-efficiency rule at 3× (1/4 contrasts) and 1.5× (3/4); typed-MLP-only midpoint gains do not establish a message-passing win. |
| [**l2d_size_transfer_v1**](docs/lineages/l2d_size_transfer_v1.md) | `CLOSED` | Zero-shot 6×6 L2D GNN transfer fails all four registered bars against MP-OFF and typed MLP at fresh 10×10 and 15×15 sizes. |
| [**proposal_frontier_v1**](docs/lineages/proposal_frontier_v1.md) | `CLOSED` | Corrected eight-topology live gate: full GNN gains 0.251% over time-feasible new MP-OFF (5/8), missing both signed bars despite a 5.048% gain over hand multistart; 24 runs audit. |
| [**dag_resume_s0_v1**](docs/lineages/dag_resume_s0_v1.md) | `CLOSED` | Resume contract audits, but one-decision and beam retry both gain 0% median over matched hand on 32 fresh workflows; 128 live runs pass and no model is trained. |
| [**dag_block_proposer_128_s0_v1**](docs/lineages/dag_block_proposer_128_s0_v1.md) | `CLOSED` | Widened 128-operation block proposals gain only 0.44% median over a strong start; 49 fresh live replays audit, so this move-pool line stops before GNN training. |
| [**g32_coordinated_valley_s0_v1**](docs/lineages/g32_coordinated_valley_s0_v1.md) | `CLOSED` | Material two-move valleys yield only 1.03% median bounded gain; equal-budget route-aware hand search captures 91–100% of broader pair headroom, with fresh live physics audited. |
| [**g32_core_prefetch_s0_v1**](docs/lineages/g32_core_prefetch_s0_v1.md) | `CLOSED` | Core-only contention and 13.72% median bounded-oracle gain survive fresh live audit, but route-aware hand controls capture 72–100% of the gain, closing this local-move training target. |
| [**g32_peer_prefetch_s0_v1**](docs/lineages/g32_peer_prefetch_s0_v1.md) | `CLOSED` | Prefetch pipes cause large RTT penalties, but access links dominate: core wait clears the signed median bar on only 1/4 training topologies; twelve fresh live runs audit. |
| [**g32_peer_fabric_s0_v1**](docs/lineages/g32_peer_fabric_s0_v1.md) | `CLOSED` | Actual core peer-link waiting is under 1.1% of RTT at every training-topology median despite route overlap; fresh paired live gate confirms the platform-start physics fails its manipulation bar. |
| [**g32_fourtype_move_s0_v1**](docs/lineages/g32_fourtype_move_s0_v1.md) | `CLOSED` | Four-type move pool clears oracle headroom, but leave-one-topology-out hand-feature controls capture most gain; residual GNN-training gate fails after audited live manipulation. |
| [**g32_exact_val_selection_v1**](docs/lineages/g32_exact_val_selection_v1.md) | `CLOSED` | Exact-validation-selected final weights still miss all bipartite bars on eight fresh live topologies; full, peer-only and MP-OFF each beat hand. |
| [**g32_sampled_slate_v1**](docs/lineages/g32_sampled_slate_v1.md) | `CLOSED` | Matched 32-task live gate: full GNN loses to peer-only by 1.69% direct and 1.87% in the equal-size portfolio at the median; no bipartite-specific win. |
| [**gnn_proposal_selector_v1**](docs/lineages/gnn_proposal_selector_v1.md) | `CLOSED` | Corrected eight-topology live gate: adding the GNN proposal improves the hand+MP-OFF selector 3.93% median (8/8 positive); no isolated message-passing claim. |
| [**gnn_seeded_cd_v1**](docs/lineages/gnn_seeded_cd_v1.md) | `CLOSED` | Under dynamic autoscaling, matched GNN-seeded CD splits 2–2 with hand CD on four burst topologies, median +1.69% elapsed with far more inference; no win in that seat. |
| [**gnn_seeded_cd_fixed_v1**](docs/lineages/gnn_seeded_cd_fixed_v1.md) | `CLOSED` | Fixed-replica fresh live gate: GNN-seeded CD beats matched hand CD on 7/8 topologies, median 2.73% lower elapsed (p = 0.035); its MP-OFF twin later wins 8/8. |
| [**gnn_seeded_cd_mp_ablation_v1**](docs/lineages/gnn_seeded_cd_mp_ablation_v1.md) | `CLOSED` | In the fixed-replica seat, the matched learned MP-OFF seed beats the GNN seed on 8/8 live topologies (median 1.68%) and also beats hand search on 8/8. |
| [**dag_remat_s0_v1**](docs/lineages/dag_remat_s0_v1.md) | `CLOSED` | Corrected rematerialization gains only 5.90% reference headroom over timed hand; actual recomputation runs on 2/16 fresh plans; 49 live runs audit. |
| [**dag_memory_value_learning_v1**](docs/lineages/dag_memory_value_learning_v1.md) | `CLOSED` | Six trained checkpoints and 144 live runs: GNN has no supported edge over MP-OFF or hand-feature MLP and trails 250 ms hand search by 2.75% median. |
| [**dag_memory_lifetime_s0_v1**](docs/lineages/dag_memory_lifetime_s0_v1.md) | `CLOSED` | Output-cache pressure binds in 16/16 fresh DAGs, but feasible-reference gain over timed hand is 6.32% below the 8% bar; 49 live runs audited. |
| [**dag_block_stepwise_s0_v1**](docs/lineages/dag_block_stepwise_s0_v1.md) | `CLOSED` | Corrected 384-operation block screen: 3.05% feasible-reference gain vs timed hand, 0% median block gain; 52 live runs audited. |
| [**dag_reservation_learning_v1**](docs/lineages/dag_reservation_learning_v1.md) | `CLOSED` | Four trained models, 96 live runs: GNN ties MP-OFF, loses 1.81% to hand search; all 64 raw learned plans lose the guard. Full search and event audits pass. |
| [**mixed_pair_learning_v1**](docs/lineages/mixed_pair_learning_v1.md) | `CLOSED` | Nine models and 384 live runs: zero median GNN gain over both learned controls and hand rule; full serving exceeds 5 ms. |
| [**mixed_dispatch_learning_v1**](docs/lineages/mixed_dispatch_learning_v1.md) | `CLOSED` | Nine models and 384 live runs: GNN is 19.03% slower than the hand rule and adds no gain over matched learned controls. |
| [**mixed_dispatch_state_v1**](docs/lineages/mixed_dispatch_state_v1.md) | `CLOSED` | Ready-set imitation repairs unstable labels but GNN ties learned controls, loses 9.74% to the hand rule, and misses 5 ms. |
| [**mixed_dispatch_value_v1**](docs/lineages/mixed_dispatch_value_v1.md) | `CLOSED` | Outcome values give GNN a small uncertain learned-control edge, but it loses 8.98% to the hand rule and misses 5 ms. |
| [**mixed_physics_learning_v1**](docs/lineages/mixed_physics_learning_v1.md) | `CLOSED` | Integrated setup/locks/power has headroom, but nine models and 768 live runs leave GNN at zero median gain over learned controls and 4.97% behind hand search. |
| [**workflow_basin_v1**](docs/lineages/workflow_basin_v1.md) | `CLOSED` | Perfect twelve-start selection gains only 2.58% on qualification and 1.60% in 128 fresh live runs; insufficient headroom for GNN training. |
| [**workflow_move_value_v1**](docs/lineages/workflow_move_value_v1.md) | `CLOSED` | Exact scoring of all 80 moves costs 0.20 ms; compiled descent beats the GNN hybrid on 32 fresh workflows and 224 live runs. No move-value training justified. |
| [**workflow_proposal_v1**](docs/lineages/workflow_proposal_v1.md) | `CLOSED` | Guarded GNN beats Knative but ties MP-off and loses to timed hand search on 32 fresh workflows; 384 live runs reject the 2 ms proposal recipe. |
| [**workflow_amortized_v1**](docs/lineages/workflow_amortized_v1.md) | `CLOSED` | Real planning headroom, negative nine-model pilot: GNN ties MP-off and loses to hand MLP and timed hand search across 256 workflows and 3,001 live replays. |
| [**peer_lookahead_v1**](docs/lineages/peer_lookahead_v1.md) | `CLOSED` | DO-NOT-ADVANCE: first-task two-hop gains 0.021% vs immediate and 2.823% vs peer mass on 16 fresh live infrastructures; fails the 5% benefit bar. No learned-MP claim. |
| [**graph_decision_witness_v1**](docs/lineages/graph_decision_witness_v1.md) | `CLOSED` | Distant information flips the best first placement, but hand min-sum and min-cut match the exhaustive live optimum in all 12 constructed cells; no training justified. |
| [**graph_three_host_screen_v1**](docs/lineages/graph_three_host_screen_v1.md) | `CLOSED` | Nine-task, three-host graphs remain solved by hand min-sum and sub-millisecond exhaustive search; two full live sweeps confirm the objective. No training justified at this size. |
| [**graph_size_screen_v1**](docs/lineages/graph_size_screen_v1.md) | `CLOSED` | At 12/24/48 tasks, CD without MP matches all 12 certified optima in 1–14 ms; selected plans pass live replay. Size alone preserves easy co-location. |
| [**graph_unary_screen_v1**](docs/lineages/graph_unary_screen_v1.md) | `CLOSED` | Heterogeneous execution creates a real live trade-off, but graph-cut search leaves under 0.8% RTT regret in eight cases; no material learned-model headroom established. |
| [**small_batch_so_v1**](docs/lineages/small_batch_so_v1.md) | `CLOSED` | **`NO-WIN`.** Retrained on single-origin groups the engineered GNN still loses or ties CD (+13 / +7 / −4 %); the raw-plan GNN beats its MLP at ×2/×3 but CD is faster (+13–19 %). |
| [**hetero_conv_v1**](docs/lineages/hetero_conv_v1.md) | `CLOSED` | **`NO-GAIN`.** Per-relation bipartite weights: beats the MP-OFF twin −12 to −24 %, plain `lf1gnn` only at ×5 (−7 %); CD faster at ×2/×3 (+15 / +24 %). |
| [**client_local_v1**](docs/lineages/client_local_v1.md) | `CLOSED` | **`CLIENT-EXECUTION-NO-GAIN / ORIGIN-MODEL-MATTERS`.** Clients barely host replicas (0.1–2 % local). One origin per group: rules 26–82 % faster and no learned arm beats CD. |
| [**scale_sweep_v1**](docs/lineages/scale_sweep_v1.md) | `CLOSED` | **`SCALE-NOT-THE-LEVER`.** At ~2× / ~4× tasks per batch and on 12-server cells the GNN-vs-twin margin stays −2 to −4 % (10-task GNN loses to its twin); search-rule wins hold. |
| [**rule_baselines_v1**](docs/lineages/rule_baselines_v1.md) | `CLOSED` | **`BEATS-NAIVE-RULES / CO-LOCATION-RULES-FASTER`.** `lf1gnn` beats random, least-loaded and Decima's weighted fair at every rung (−12 to −85 %); locality-first, self-predict and one-pass greedy are faster (up to +78 %). |
| [**local_features_v1**](docs/lineages/local_features_v1.md) | `CLOSED` | **`NO-WIN`.** Per-candidate-only features, 19 unseen topologies: the GNN beats the same-input MLP −11 / −24 / −20 % (Holm, every seed) but CD is faster at every rung (+14 / +34 / +6 %). The gap is queue. |
| [**small_batch_confirm_v1**](docs/lineages/small_batch_confirm_v1.md) | `CLOSED` | **`SEARCH-WIN-NOT-MP`.** 19 unseen topologies, 8 seeds: the small-batch GNN beats CD −10 / −35 / −17 % and CD+ext at every rung (Holm), but ties its MP-OFF twin (within 1 %). Learned-scorer win, not MP. |
| [**small_batch_v1**](docs/lineages/small_batch_v1.md) | `CLOSED` | **`BATCH-SIZE-FIX-HELPS`.** Retraining on live-sized (~4-task) batches fixes the GNN's ×5 loss to its twin and adds 5–12 % at ×3 / ×5; the dev-topology twin margin did not replicate. |
| [**raw_plan_v2**](docs/lineages/raw_plan_v2.md) | `CLOSED` | **`NO-WIN (CD-FASTER)`.** Raw-plan GNN with a load-sum channel (`rawS`) beats the MLP and its twin at ×2 / ×3 on unseen topologies but loses to CD +20 to +33 % and CD+ext; its twin is faster at ×5. |
| [**raw_plan_v1**](docs/lineages/raw_plan_v1.md) | `CLOSED` | **`MP-BEATS-TWIN / CD-FASTER`.** Raw plan, no engineered context: the GNN beats its MP-OFF twin −14 / −24 / −12 % (12/12 each) but both lose to CD (+24 to +53 %); the gap is queue. |
| [**peak_controls_v1**](docs/lineages/peak_controls_v1.md) | `CLOSED` | **`LEARNED-SCORER-WIN / NOT-MP`.** The MP-OFF twin beats CD −11 / −19 / −20 % and CD+externality at every rung; it ties the GNN at ×2 / ×3 and beats it at ×5 (+13 %). The externality closes ~half of the GNN–CD gap. |
| [**peak_load_v1**](docs/lineages/peak_load_v1.md) | `CLOSED` | **`PEAK-LOAD-WIN`.** At published peak/surge loads the self-refined GNN beats CD: w0 ×1.5 −24.6 %; grounded ×1.5 −5.3 %, ×2 −12.5 %, ×3 −20.4 %, ×5 −9.2 % (all CONFIRMED); beats one-pass, Decima, Knative, random; ties self-predict. ×2–×5 rungs post hoc; ×3/×5 overload. |
| [**grounded_workload_v1**](docs/lineages/grounded_workload_v1.md) | `CLOSED` | **`NOT-SEPARATED`.** On Alibaba-grounded windows (≈4-task groups, ms sibling spacing, ×1) the learned arm ties CD (+0.4 %, p=0.57) and self-predict (−0.9 %); beats Knative −16 %, random −40 %, one-pass −7 %, Decima −20 % (12/12). |
| [**x11_confirm_v1**](docs/lineages/x11_confirm_v1.md) | `CLOSED` | **`NOT-CONFIRMED`.** The ×1.1 w0 win over CD on unseen seeds 2–4: −11.5 % (p=0.092, 9/12); 4 seeds −11.8 % (p=0.064). Magnitude replicates, significance doesn't: 9434 collapses on every seed. |
| [**backlog_corpus_v1**](docs/lineages/backlog_corpus_v1.md) | `CLOSED` | Synthetic-backlog corpus does not move the live queue (L1 +0.83 % NOT-SEPARATED, L2 +6.6 % vs CD); the gap is in-batch stacking; self-refine on the same weights ties CD (+1.5 %, p = 0.15). |
| [**fullctx_refine_v1**](docs/lineages/fullctx_refine_v1.md) | `CLOSED` | Training on full-batch context makes the self-refined GNN tie CD from the fast side: F1 `CLOSES-GAP` −1.76 % (p=0.76, 6/11); F2 −2.23 % vs `bc1load_selfref` (10/11, direction only); −13.9 % vs self-predict (12/12). |
| [**exchange_seconds_v1**](docs/lineages/exchange_seconds_v1.md) | `CLOSED` | Exchange in log1p seconds on top of `fc1load`: X1 `CLOSES-GAP` −0.94 % vs CD (p=0.92, 5/10); X2 `NOT-SEPARATED` −1.34 % vs `fc1load_selfref` (p=0.054, 10/11) but 9461 got worse (+3.9 %); −14.6 % vs self-predict (11/11). |
| [**seeded_cd_xs1_v1**](docs/lineages/seeded_cd_xs1_v1.md) | `CLOSED` | CD refine seeded by `xs1load` beats CD −3.74 % (p=0.002, 10/10, direction only; 9461 −0.9 %); MP-OFF seed −3.07 %, so ~4/5 is any learned seed; −17.8 % vs self-predict. |
| [**burst_ladder_v1**](docs/lineages/burst_ladder_v1.md) | `CLOSED` | w0 ×1: learned −11.9 % vs CD over 4 draws (9/12, p=0.15; 9434 collapses, 9461's old −61 % was a single-draw CD collapse). ×1.5 (inadmissible): −28.1 % vs CD over 4 draws (11/12, p=0.009) — survives rule scatter. |
| [**capacity_sweep_v1**](docs/lineages/capacity_sweep_v1.md) | `CLOSED` | C1 `NO-CAPACITY-GAIN` (knee higher 5, lower 1, tied 5 of 11; ratio 1.05); C2 `PLACEMENT-LEAD` direction only (−8.4 % vs CD without replica churn, 10/12); ×1.1: −13.7 % vs CD on 1 seed, not confirmed on 4 (−11.8 %, p=0.064; `x11_confirm_v1`). |
| [**decima_rule_v1**](docs/lineages/decima_rule_v1.md) | `CLOSED` | Decima's tuned weighted-fair rule (α=+1) is +46 % vs self-predict (0/12) and +66 % vs CD: locality-blind, not a bar here. Learned arm beats it −41 % (11/11). |
| [**load_repr_v1**](docs/lineages/load_repr_v1.md) | `CLOSED` | LOAD-HELPS / NARROWS: CD's load terms in seconds (`partial_state_v4`) make the burst-seat GNN −5.3 % vs its twin (11/11) and −8.5 % vs self-predict (10/11); CD still +6.7 % ahead (was +12.4 %). Found and fixed an uncapped-rung serving rank defect (cost < 1.5 %). |
| [**cd_gap_v1**](docs/lineages/cd_gap_v1.md) | `CLOSED` | SCORE-EXPLAINS / CD-FASTER: the burst-seat GNN trails CD because its score cannot load-balance; the label (CD-imitator +13.6 %), decode order and load (+13.8 % at half rate) don't explain it. GNN-seeded CD beats CD −3.8 % (10/10, direction only). |
| [**fresh_topo_burst_v1**](docs/lineages/fresh_topo_burst_v1.md) | `CLOSED` | MP-EDGE-GENERALISES (direction only): on 11 fresh topologies `gnnedge0` beats its MP-OFF twin −4.2 % (11/11, p = 0.001), but ties self-predict (−1.2 %, p = 0.70); MP-OFF loses to it (+2.8 %, 2/11); CD ahead of all. |
| [**selfpredict_burst_v1**](docs/lineages/selfpredict_burst_v1.md) | `CLOSED` | **`GNN-BEATS-SELFPREDICT`, burst-seat bar CD.** Uncapped gnnedge0 beats the self-predict rule −7.25 % (13/13 ckpt; 9/16 env, thin) in the burst seat — not an MP win (ties its MP-OFF twin); the CD greedy stays ahead of both (+12.5 / +21.3 %). |
| [**hidden_exec_s0_v1**](docs/lineages/hidden_exec_s0_v1.md) | `CLOSED` | **`NO-HEADROOM`.** faas-sim-style hidden execution time (node speed × co-execution × noise): a rule that knows it ties or loses to the table-reading self-predict rule (C40 +0.40 %, C80 −0.05 %); exec is 0.4 % of latency here. Nothing to learn. |
| [**selfpredict_bar_v1**](docs/lineages/selfpredict_bar_v1.md) | `CLOSED` | **`BAR=SELFPREDICT`.** The rule + a price for unarrived partners at the node the rule would give them beats Knative −19.4/−21.4 % (16/16), the old rule −6.2/−8.5 % and the CD greedy −7.8/−8.9 % (C40 not separated): the programme's best policy and the bar for any learned arm. |
| [**lookahead_mp_v1**](docs/lineages/lookahead_mp_v1.md) | `CLOSED` | **`HAND-COORDINATION-RECOVERS`.** Pricing unarrived partners is real live headroom (oracle −6.6/−8.2 % vs the rule), but a hand self-predict rule gets 93–95 % of it (−6.2/−8.5 %): coordination, not a message-passing lever. P1 never started; that rule is the new bar. |
| [**rollout_imitation_v1**](docs/lineages/rollout_imitation_v1.md) | `CLOSED` | **`RULE-FASTER-LIVE`.** One step of policy improvement: the rollout label is horizon-stable and a pointwise scorer on the rule's own terms beats the rule −12.7% OFFLINE (held-out), but served live is +14–28% SLOWER (R2 FAILS, 0/16 C40). Beats Knative-ECT/random, loses to CD greedy — another offline/live reversal; the rule stays the unbeaten bar. |
| [**joint_burst_v1**](docs/lineages/joint_burst_v1.md) | `CLOSED` | **`NO-GNN-WIN`.** Trained on the served distribution (bursts, loaded states, group-optimum labels): the burst-trained `gnnedge0` beats reactive Knative −12.47 % (15/15) — first learned win in the programme — and beats its own cold twin −7.82 % (J5 HELPS, 15/15), but still loses to the batched greedy +16.83 % (0/15). Necessary, not sufficient; triggers `rollout_imitation_v1`. |
| [**backbone_sparsity_v1**](docs/lineages/backbone_sparsity_v1.md) | `CLOSED` | **`TOO-FEW-UNSATURATED-ENVIRONMENTS` on both variants.** A 250 MB/s backbone saturates reactive on every cell (a load lever, like payload ×3); p = 0.4 hangs reactive on 39/48 cells (a reachability lever). No study; distance cannot be read at this rate. |
| [**burst_groups_v1**](docs/lineages/burst_groups_v1.md) | `CLOSED` | **`WAIT-COLLAPSED` · `gnnedge0` NOT-SEPARATED from Knative (−2.2 %) · `RULE-FASTER-THAN-GRAPH-ARM` +25 %.** Bursts remove the 7 s wait and Knative's rendezvous; the rule reads −26 % (8/8), the learned arm ties. Screen TOO-FEW (22/48 reactive hangs); read on 8 environments under two signed amendments, disclosed. |
| [**payload_scale_v1**](docs/lineages/payload_scale_v1.md) | `CLOSED` | **`CO-LOCATION-PAYS-FROM-x1` (rule) · at no measured scale (`gnnedge0`).** At 20 MB the rule ties Knative and every batching arm pays its 7 s wait (+15–16 %); at 600 MB and 2 GB reactive saturates on every cell, a load lever at this rate. |
| [**peer_greedy_live_v1**](docs/lineages/peer_greedy_live_v1.md) | `CLOSED` | **`HAND-RULE-BEATS-REACTIVE-WITHOUT-WAITING` · `RULE-FASTER-THAN-GRAPH-ARM`.** Queue drain in seconds + exchange to partners already placed, no wait: −12.96 % (C40) / −16.01 % (C80) vs Knative, 16/16 each, and −13.2 % vs `gnnedge0` in its own seat. The bar for every learned arm is now the rule. |
| [**batch_window_edge_v1**](docs/lineages/batch_window_edge_v1.md) | `CLOSED` | **`WINDOW-NOT-THE-LEVER` · `BATCHING-RELOCATES-WAITING`.** 2/4/8/16 s windows on the screen topology: the wait falls 7.15 → 1.66 s and the total vs Knative stays +7.7 to +8.8 %; the arm's shorter queue and rendezvous are the scheduler wait relocated. Genuine gain: −1.05 s of exchange per task from co-location. |
| [**unsaturated_edge_v1**](docs/lineages/unsaturated_edge_v1.md) | `CLOSED` | **`HEADLINE-WAS-AN-UNPAIRED-STATISTIC`.** The −20.8 % came from a ratio of medians over 4 cells, 2 saturated; paired per cell `gnnedge0` is +7 %. On 16 environments every learned arm loses to Knative (+6.7 to +14.4 %, 0/16) while `gnnedge0` beats its twin −6.8 % (14/14); the deficit is the 7.1 s batch wait. |
| [**unsaturated_scale_v2**](docs/lineages/unsaturated_scale_v2.md) | `CLOSED` | **`EFFECTS-ARE-REAL-AND-AN-ORDER-OF-MAGNITUDE-SMALLER`.** At 16 (topology, window) environments every v1 effect shrinks 4-10x while p collapses: gnnedge0 -1.62 % (p=0.0023, 13/16), sum costs +2.70 % not +26.69 %. v1's magnitudes were one bursty arrival window. |
| [**unsaturated_scale_v1**](docs/lineages/unsaturated_scale_v1.md) | `CLOSED` | **`NO-LEARNED-ARM-BEATS-REACTIVE-AT-UNSATURATED-SCALE`.** Reactive's capacity never grew with the cluster (80 servers healthy only at 6 servers' 0.46/s), so every 80-server win was "less drowned". At the one healthy 80-server rung all arms tie reactive, `gnn` loses +17 %; cause: a 6.8 s batching tax. |
| [**corpus_matched_v1**](docs/lineages/corpus_matched_v1.md) | `CLOSED` | **`MODEL-CLASS-EDGE-IS-CORPUS-CONTINGENT`.** Corpus held fixed at 1,670: the graph arm wins −31.7 % (16/16) at 40 clients and −17.1 % at 80. At 516: one win, two ties. `best_arm_v1`'s only pointwise win was a corpus effect (+18.7 %); name the corpus in every quote. |
| [**best_arm_v1**](docs/lineages/best_arm_v1.md) | `CLOSED` | **`BEST-ARM-STILL-NOT-ESTABLISHED` — the ladder is SPLIT.** The repaired graph arm beats `mpoff_516` at 40 clients (−14.8 %, 16/16) and 80 (−11.3 %) and loses at 20 (+15.7 %), where BOTH arms lose to reactive. Two wins and a loss does not overturn the clause. Confounded with corpus. |
| [**bipartite_aggr_v1**](docs/lineages/bipartite_aggr_v1.md) | `CLOSED` | **The bipartite penalty is the `sum` aggregation, and the prediction held.** sum costs +13.3 % (p=0.0052) where a task has 47.9 candidates and is not separated where it has 3.55 — the null held at n=32 under AMENDMENT 1 and got weaker. MLP shape explains nothing. Use `mean`. |
| [**bipartite_edge_v1**](docs/lineages/bipartite_edge_v1.md) | `CLOSED` | **BIPARTITE-WORKS-BUT-NOT-BY-EDGE-CONDITIONING.** `gnnedge` beats `gnn` by −20.4 % at 80 srv (16/16) and −28.7 % at 40 clients, and beats reactive −20.8 % (15/16) at an UNSATURATED rung where `gnn` loses. But its zeroed-attribute control matches it (+0.44 %, 8/16): the conv, not the attributes. |
| [**peer_only_v1**](docs/lineages/peer_only_v1.md) | `CLOSED` | **PEERONLY-BEATS-POINTWISE at every cluster size (6–80 srv) and client count (20–80); first UNSATURATED live win vs reactive (−9.3 % at 80 clients, p=0.0097).** But `mpoff`@516 beats reactive there too, the two are NOT separated (B8), and the margin is 99 % **queue**, not peers. |
| [**drainable_serving_config_v1**](docs/lineages/drainable_serving_config_v1.md) | `CLOSED` | **BATCHING-DOES-NOT-EXPLAIN.** With zero batch wait the graph arm is **−1731.86 %** against reactive Knative, 8× worse than the 80 s window it was blamed on — peer-group batching is what keeps the learned arms within an order of magnitude of reactive. `gnn` vs `mpoff` is a function of the window, and TIEs at the windows that serve both arms best. |
| [**pointwise_baseline_v1**](docs/lineages/pointwise_baseline_v1.md) | `PARKED` | **`MLP-CANNOT-BE-SERVED-UNDER-PARTIAL-STATE`.** The MLP scores every edge once per batch, so it cannot carry a prefix-conditioned block: the first arm read 50000/50000 shortest-queue fallbacks. Caught by a one-arm smoke test. This is WHY the record's pointwise arm was always the MP-OFF twin. |
| [**serving_gap_v2**](docs/lineages/serving_gap_v2.md) | `CLOSED` | **NO-GO** — H3 fires on 1 of the 2 corpora it required, though the co-location shortfall does order exactly like the live gap. |
| [**serving_gap_v1**](docs/lineages/serving_gap_v1.md) | `CLOSED` | **NO-GO**, with both registered hypotheses refuted backwards: the graph arm spreads *more* than its twin and is 3–7× *more* load-responsive. Any explanation of the serving gap must start from over-responsiveness, not blindness. |
| [**peer_affinity_warm_v1**](docs/lineages/peer_affinity_warm_v1.md) | `CLOSED` | **NO-WINNING-GNN.** Training on served states did not make message passing transfer: the warm graph arm is 23.9 % slower than the cold one and 10.4 % slower than its own twin (0/16 each). Side finding, unregistered: the warm-trained *pointwise* arm beats Knative uncapped, +11.3 %, 16/16, and finishes sooner. |
| [**scheduler_residence_v1**](docs/lineages/scheduler_residence_v1.md) | `CLOSED` | **RESIDENCE-IS-COLLECTION · LOPSIDEDNESS-DOES-NOT-REPLICATE.** Scheduler wait is 89 % peer-group collection and 0.000 s decode; the three-cell topology lead dies at n = 15 (ρ = +0.041, p = 0.885). |
| [**partial_state_v3**](docs/lineages/partial_state_v3.md) | `CLOSED` | **GENERALISES · SCALE-HELPS.** A size-free rank block (dim 38 → 22) ties v2 live at 6 servers and serves 12 / 24 / 80; at 80 servers the learned arms beat reactive Knative by 28.7 % (`gnn`) and 46.9 % (`mpoff`) on 4/4 topology seeds — the first whole-trace live win in the program, on saturated rungs, relative to reactive; no GNN claim. |
| [**cluster_scale_v1**](docs/lineages/cluster_scale_v1.md) | `CLOSED` | **ASSEMBLY-IS-ARRIVAL-BOUND · REQUIRES-RETRAINED-REPRESENTATION.** Collection falls 6.10 s → 0.73 s as arrivals speed up 13.3×, worth ~5.37 s — but the feature layout hard-caps at 6 hosting nodes and candidate support is 9.58× outside the corpus. |
| [**queue_range_v1**](docs/lineages/queue_range_v1.md) | `CLOSED` | **QUEUE-RANGE-NOT-THE-LEVER.** The served dim-7 column is out of its trained range, and clamping it binds but clears nothing under Holm. The find is the per-term decomposition: one cell loses on batch wait alone. |
| [**serving_stability_v1**](docs/lineages/serving_stability_v1.md) | `CLOSED` | **EARLY-ADVANTAGE-REAL · STABILITY-NOT-THE-LEVER.** Over the first fifth of the trace the learned arms carry less queue than Knative on 3/3 cells and all 91 arms — the first replicated positive in the program — but a decode-time queue guardrail does not recover it. |
| [**offline_live_transfer_v1**](docs/lineages/offline_live_transfer_v1.md) | `CLOSED` | **OFFLINE-SCORE-RESOLVES-WITHIN-RUN-ONLY.** Pooled-z ρ = −0.030 across 96 checkpoints: the offline score ranks epochs, never arms, so the reversal `serving_gap_v1`/`v2` hunted is an arm-average offset. No surrogate among four; the live gate is the cheap instrument. |
| [**drainable_objective_v1**](docs/lineages/drainable_objective_v1.md) | `CLOSED` | **OBJECTIVE-NOT-DELIVERED**, CONFOUNDED-C1: the shaped arms concentrate *more* than the control they were built to repair, so C3/C4/C5 are VOID and latencies tie. The offline/live reversal reproduced on a non-peer label. |
| [**drainable_regime_v1**](docs/lineages/drainable_regime_v1.md) | `CLOSED` | **POINTWISE-BETTER.** The program was gated at a 940× overload; at a defensible load `gnn` is −15.94 % against its own twin (p = 0.0010) and −226 % against reactive Knative (0/16). The x200 GNN-NEEDED reading does not survive. In any rate sweep, scale every policy time constant and add a counter-reading control bar. |
| [**reliability_matched_v1**](docs/lineages/reliability_matched_v1.md) | `CLOSED` | **FAIL.** Matching the corpus removes 87 % of the MLP's collapse burden and the residual is not established at n = 16 (p = 0.113). No unconfounded model-class reliability claim survives. |
| [**mp_ablation_v1**](docs/lineages/mp_ablation_v1.md) | `CLOSED` | **NO_DIFFERENCE_DETECTED** — MP-off matches and directionally beats MP-on, so the credit belongs to the EdgeScorer and the decode, not to graph reasoning. |
| [**link_mp_v1**](docs/lineages/link_mp_v1.md) | `CLOSED` | **NO_DIFFERENCE_DETECTED** on the primary — message passing ties pointwise even on the right graph. The corpus was the real lever (−38 % against Knative, zero collapses across 48 arms), never the model class. |
| [**objective_pivot_v1**](docs/lineages/objective_pivot_v1.md) | `CLOSED` | Phase 1 PASSED (reliability, scope-limited to severe collapse), Phase 2 CLOSED (horizon labels are deterministic chaos), **Phase 3 MEASURED-NEGATIVE** at n = 120 (−0.85 %, p = 0.928, powered). Closing it answered all three CLAUDE.md routes. |
| [**route_a_v1**](docs/lineages/route_a_v1.md) | `FALSIFIED` | **NO-GO.** DAG + distance is genuinely pairwise and still pointwise-optimal. Breaking separability is **necessary but not sufficient** — you also need contention, which is what route B tests. |
| [**program_verdict_v1**](docs/lineages/program_verdict_v1.md) | `CLOSED` | Terminal answer to the D3 fork: **the supervised co-sim path to "GNN > MLP on latency" is closed by measurement.** The reliability/regime win exists on the 30-cell record but is exploratory. P1 (closed-loop objective) is the only remaining path to the latency claim. |
| [**cosim_deepdive_v1**](docs/lineages/cosim_deepdive_v1.md) | `CLOSED` | Does the target's additivity come from the synthetic t=0 snapshot regime? **No — live-visited states are equally additive** (4,400 swept live states, median additive R² 0.99999). The GNN's dispersal edge is a closed-loop property no single-batch regret target can express. |
| [**graph_structure_physics**](docs/lineages/graph_structure_physics.md) | `CLOSED` | The co-sim target is **pointwise-separable** (additive R² 0.988), so a pointwise MLP is the correctly specified model class and no amount of training data lets the GNN beat it. This is what closes CLAUDE.md option 1. |
| [**throughline**](docs/lineages/throughline.md) | `SYNTHESIS` | The program-level synthesis: why a graph-reasoning win looked unavailable on this simulator's supervised targets by construction, that the MP-OFF arm is itself a two-tower pointwise scorer, and the three things that would change the answer — one of which then happened in `peer_affinity_v1`. |
| [**p5b_draw_study**](docs/lineages/p5b_draw_study.md) | `CLOSED` | **Q1 = LOTTERY, Q2 = DRAW-DOMINATED.** The MLP trainer never seeded torch, so every MLP checkpoint before 2026-08-24 is an unreproducible draw. Collapse counts swing 0→26 on the seed alone. **Retires "the MLP collapses 7/30"** and every inference built on it. |
| [**p5b_candidate_relative**](docs/lineages/p5b_candidate_relative.md) | `CLOSED` | **INDETERMINATE — and the indeterminacy is the result.** Kills the mechanism sentence "a pointwise scorer collapses because it cannot condition on its peers": one arm has exactly that conditioning, uses it, and stops collapsing. Resolved by `p5b_draw_study` — it was the training draw. |
| [**gnn_draw_study_v1**](docs/lineages/gnn_draw_study_v1.md) | `CLOSED` | **INDETERMINATE**, and **"the GNN never collapses" is FALSIFIED** — 2 of 8 seeded draws collapse. Direction is clear, the test is under-powered. |
| [**m3_batch_makespan_v1**](docs/lineages/m3_batch_makespan_v1.md) | `CLOSED` | Below both registered thresholds, then **closed permanently** by the argmax-flip diagnostic: when per-branch costs are separable and component choices are free, min-max and min-sum share the same argmin, so re-scoring under makespan was never an independent escape at any fan-out width. |
| [**cache_live_divergence_audit**](docs/lineages/cache_live_divergence_audit.md) | `CLOSED` | Platform reordering: **18/18 collections, BENIGN** (no recache, no asterisk). Dims 9-11 temporal estimate: **8/18, REAL**. The parity verifier now compares by platform identity instead of position. |
| [**serving_speed_v1**](docs/lineages/serving_speed_v1.md) | `CLOSED` | The episode cost was `Data.to()`, not the device — 174× per-call win from moving tensors only. CPU serving is slower than cuda; the cpu default exists for parity. |

## Falsified / failed — the mechanisms that did not work

| Lineage | Status | Outcome |
|---|---|---|
| [**replica_guard_v1**](docs/lineages/replica_guard_v1.md) | `FAILED` | Keep-warm serving guard: K1 `NOT-FIXED` (9434 median 20.7→10.7 s, CD 9.1; seeds still collapse), K2 `TIES` −11.25 % vs CD (p=0.24, same as twin); new starved-client spin on 9119 d3. |
| [**wide_choice_s0_v1**](docs/lineages/wide_choice_s0_v1.md) | `FALSIFIED` | Offline S0 FAIL: live replica pools cap slates at ~2.3–2.9 candidates/task; CD median regret vs exact optimum 13.9 % (`CD-MIDDLE`, falls with width); `xs1load` −1.8 pp vs CD, degrades >3 candidates; W3 33 % complementary. |
| [**route_c_link_transfer_v1**](docs/lineages/route_c_link_transfer_v1.md) | `FALSIFIED` | **FALSIFIED.** The concurrency lever is measured exhausted: even an 8-task, 2-client cap holds the ceiling below 10 %, because link contention charges input over ingress and the large output never touches the fabric. |
| [**dag_fabric_contention_v1**](docs/lineages/dag_fabric_contention_v1.md) | `FALSIFIED` | **NO-GO before any code.** DAG output payloads over the contended link fabric cannot teach the label: the α = 2.0 optimum carries zero link wait in 98–100 % of 204 datasets, so the mechanism is invisible to it. |
| [**topology_transfer_v1**](docs/lineages/topology_transfer_v1.md) | `FAILED` | **RETIRED.** Every arm FAILED and the lineage was never live-gated, because its ablation harness persisted no checkpoints. The pending 14 GPU-h gate is cancelled: a non-binding backbone makes link features label-irrelevant. |
| [**mp_parity**](docs/lineages/mp_parity.md) | `FALSIFIED` | Both arms falsified, gate FAILED on both pre-registered criteria. The residual pays only where interaction exists and costs RTT where the target is additive — which is `graph_structure_physics`' finding arriving from the model side. |
| [**link_contention_v1**](docs/lineages/link_contention_v1.md) | `FALSIFIED` | Per-link capacity over a multi-hop backbone. Gate FAILED on both criteria. **The link controls do not repair (median 0.000)** — genuinely new, and the first escape from the one-integer control — but node-collision coupling still dominates and the magnitude is 14× below the gate. |
| [**network_contention_v1**](docs/lineages/network_contention_v1.md) | `SUPERSEDED` | Shared per-node ingress bandwidth: **the physics works and is opt-in**, but the corpus lever is replica concentration, not bandwidth. One of the four mechanisms in the throughline. |
| [**shallow_longexec_v1**](docs/lineages/shallow_longexec_v1.md) | `FALSIFIED` | The last physics attempt before the paper pivot, and the **fourth independent confirmation** of the throughline: one-integer repair 100%. |
| [**shallow_v1**](docs/lineages/shallow_v1.md) | `SUPERSEDED` | Shallow queues lower the pointwise ceiling but **MISS the coupling gate**. Contains a **retraction**: the coupled(>1%) = 31.0% figure does not reproduce at full corpus size. Read the retraction before citing any number in this node. |
| **contention_v4_v5** | `FALSIFIED` | Deep queues + coupling optimisation — **moved the corpus the wrong way** (additive R² 0.988 → 0.9997). |

## Standing apparatus — baselines, gates and tooling

| Lineage | Status | Outcome |
|---|---|---|
| **contention_v2_v3** | `ACTIVE` | Baseline contention series that v4/v5 is measured against. |
| **sealed_holdout** | `ACTIVE` | The honest generalisation gate. |
| **coupled_trio** | `ACTIVE` | The three-cell coupled comparison. **ECT is not a ceiling and not a distillation teacher** — 0.98–1.13× Knative; see [`docs/hard-stops.md`](docs/hard-stops.md). |
| **encoder_ablation** | `ACTIVE` | Is the graph encoder doing work, or is it the features? |
| **seed_variance** | `ACTIVE` | Seed spread on the contention_v2 GNN. |
| **queue_feature_contract** | `ACTIVE` | `legacy_v0` vs `scale_invariant_v1`. See `docs/adr/0002-two-queue-feature-contracts.md`. |
| **dataset_metadata** | `ACTIVE` | Produces `REGISTRY.json`, `METADATA.json`, `COMPATIBILITY_MATRIX.json`. |

These carry no separate node file. `contention_v4_v5` is listed here too — it is
`FALSIFIED` and has no node, but its entry points are still the only record of it.

| Lineage | Entry points | Datasets | Notes |
|---|---|---|---|
| **contention_v4_v5** | `scripts_cosim/datalab/contention_v4_deepq_cosim.sbatch`, `contention_v5_quick_test.sbatch` | `contention_v4_pilot`, `contention_v5_quick_test` | Deep queues + coupling optimisation — the attempt at giving the GNN real graph structure to exploit. **`FALSIFIED` 2026-08-17: moved the corpus the wrong way** (additive R² 0.988 → 0.9997). See `graph_structure_physics`. |
| **contention_v2_v3** | `important/run_contention_v{2,3}_train_and_live_gate_nohup.sh`, `important/compare_contention_v2_live_gate.py` | `contention_v2{,_verify}`, `contention_v3` | Baseline contention series the v4/v5 work is measured against. Trainers: `train_near_rtt_v2_contention_v{2,3}_dim14_ce_only.py`, `train_mlp_contention_v{2,3}_dim22_batchcache.py`. |
| **sealed_holdout** | `important/run_contention_v2_873_sealed_holdout{,_rebaseline}.sh`, `compare_sealed_live_holdout.py`, `datalab/sealed_holdout_gpu.sbatch` | `contention_v2` | The honest generalisation gate. |
| **coupled_trio** | `important/run_contention_v2_873_coupled_trio.sh`, `chain_coupled_trio_then_rebaseline.sh` | `contention_v2` | See memory note: ECT is not a ceiling. |
| **encoder_ablation** | `important/run_gnn_encoder_ablation.sh`, `compare_encoder_ablation.py` | contention series | Is the graph encoder doing work, or is it the features? |
| **seed_variance** | `scripts_cosim/run_gnn_seed_variance_siv1.sh` | contention_v2 | Uses `train_near_rtt_v2_contention_v2_dim14_ce_only.py`. |
| **queue_feature_contract** | `src/placement/queue_features.py`, `scripts_cosim/test_queue_features.py`, `verify_cache_live_feature_parity.py` | all | `legacy_v0` vs `scale_invariant_v1`. See AGENTS.md. |
| **dataset_metadata** | `scripts_cosim/{extract_dataset_metadata,validate_dataset_collection,compute_compatibility_matrix}.py` | all | Produces `REGISTRY.json`, `METADATA.json`, `COMPATIBILITY_MATRIX.json`. |

Shared core (not a lineage — everything depends on it): `src/placement/`,
`src/policy/{gnn,tabular,knative*,determined,evaluator}/`, `src/executecosimulation.py`,
`src/executesimulation.py`, `scripts_cosim/generate_gnn_datasets_fast.py`,
`src/notebooks/non_unique_lib/`.

## Retired code

Retired code lives in [`archive/`](archive/README.md) — moved with `git mv`, so
`git log --follow` still works. Nothing was deleted. Restore point: tag
`pre-cleanup-2026-08`.

| Lineage | Status | Archive | Files | Outcome |
|---|---|---|---|---|
| **pre_gnn_herosim** | `PAPER` | `archive/pre_gnn_herosim/` | 145 | The original HeROsim proactive-autoscaling paper: XGBoost/GPR demand prediction, Bayesian optimisation over infrastructure, LHS sampling, `scenario-*.sh`, and the HRO/HRC/proactive-Knative policies. Superseded by the GNN co-simulation work; kept because the paper cites it. |
| **regime_b** | `FALSIFIED` | `archive/regime_b/` | 38 | Cold-burst regime with `platform_reuse_v1` physics. Phases 0–3.1 closed the gap 125→31 only by distilling `ect_pull`, and `ect_pull` itself lands at Knative level on the coupled trio — so it was never a ceiling to chase. CLAUDE.md already marks Regime B outdated. |
| **soft_combo** | `FALSIFIED` | `archive/soft_combo/` | 6 | Joint combination scoring (`soft_combo`, `soft_combo_conc`) gave no gain over CE on `oracle_split_v1` (commit `d6f1999`). The **loss functions stay live** in `non_unique_lib/soft_combo_loss.py` — `train_near_rtt.py` still imports them; only the experiment wrappers are archived. |
| **warmth_sparse** | `SUPERSEDED` | `archive/warmth_sparse/` | 110 | The warmth/sparse/skew-merged/hub9 series, plus its regen, repair and health-monitor tooling. Superseded by the contention series, which produces the contention the GNN actually needs. Largest single lineage. |
| **model_sweeps** | `SUPERSEDED` | `archive/model_sweeps/` | 38 | One-off sweeps named after their wandb run (`woven_totem`, `silvery_sun`, `ethereal_lake`, `worthy_bush`, `ssc_trash`, `clean_1230`, `mitrix`) plus `atomic21`, `dim14_1060`, `mega_matrix`, `reviewer_triangle`. **The reason the naming convention changed:** a filename encoding a run name tells you nothing about the hypothesis. |
| **topology_sweeps** | `SUPERSEDED` | `archive/topology_sweeps/` | 32 | `tiered_hub` and `bipartite_coordination` topology experiments. |
| **strategic_merge** | `SUPERSEDED` | `archive/strategic_merge/` | 10 | Merged-corpus training strategy, replaced by the siv1 full-corpus approach. |
| **decode_ablations** | `SUPERSEDED` | `archive/decode_ablations/` | 6 | `seqblend`, `seq_reforward`, pull-decode ablations. The decode paths themselves remain in `src/policy/gnn/seq_decode.py`; only the sweep scripts are archived. |
| **hetero_training** | `SUPERSEDED` | `archive/hetero_training/` | 4 | Heterogeneous-graph GNN training. The **`gnn_hetero` policy stays live** (`executesimulation.py` dispatches it); only the training/caching scripts are archived — nothing invoked them. |
| **exact_rtt** | `SUPERSEDED` | `archive/exact_rtt/` | 2 | Exact-RTT regression objective, replaced by near-RTT + CE. |
| **live_finetune** | `SUPERSEDED` | `archive/live_finetune/` | 3 | Live-trajectory finetuning experiment. |
| **old_scripts** | `SUPERSEDED` | `archive/old_scripts_{idk_big,old_bash}/` | 11 | Pre-`generate_gnn_datasets_fast.py` bash pipeline and one-off state-discrepancy analyses. |

## Conventions

**A lineage is not finished until it has a row in this index and a node under
[`docs/lineages/`](docs/lineages/) with an outcome.** A sweep whose result was never
written down will be re-run by someone in three months.

**A gate tool's own correctness is a recorded fact, not folklore.** When a gate turns out to
have been measuring the wrong thing, it goes in
[`docs/gates/gate-tools.md`](docs/gates/gate-tools.md) — not inside whichever lineage
happened to trip over it. Two of this repo's near-misses came from a tool-level fact being
buried in a lineage narrative.

**A rule that outlives its lineage goes in [`docs/lessons.md`](docs/lessons.md).** The node
records what one investigation found; `lessons.md` records what generalises past it. A
falsified direction also gets a line in [`docs/hard-stops.md`](docs/hard-stops.md), with the
measurement that closed it.

**Do not fork a training script per experiment.** That habit produced 40 near-identical
`train_near_rtt_v2_*.py` wrappers that differed only in cache dir and wandb name. New
experiments get a config, not a copy.

**Do not import from `archive/`.** The live tree is verified closed against it, and that
gate is now a test rather than a block to paste:

```bash
PIPENV_IGNORE_VIRTUALENVS=1 pipenv run python3 -m pytest tests/test_record_hygiene.py -q
```

`tests/test_record_hygiene.py` also enforces the rest of this section mechanically: index
rows stay one line, every node has a row, a row's status matches the node's own header,
cited scripts resolve outside `archive/`, and no committed handovers. The rule was stated
in four places and violated in all four; a check is what actually holds it.

**Session handovers are ephemeral and are not committed.** A handover is one session's
note to the next; it is not a record. Anything in it that is still true a week later
belongs in a lineage node, `docs/lessons.md`, or `docs/gates/gate-tools.md` — put it
there instead. Three committed `HANDOVER*.md` files were retired on 2026-08-27 for
drifting out of agreement with this index while claiming the same facts. If you need to
hand off, write the file outside the repo (the session scratchpad) or leave it untracked.

**One fact, one home.** Before adding a paragraph, find the file that already owns that
fact and edit it. This index existed at 4,995 lines because five files each narrated the
same experiments and drifted apart; the duplication cost more than the writing saved.
