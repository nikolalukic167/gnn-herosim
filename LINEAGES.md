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
| [**exchange_seconds_v1**](docs/lineages/exchange_seconds_v1.md) | `ACTIVE` | Exchange in log1p seconds on top of `fc1load`: X1 `CLOSES-GAP` −0.94 % vs CD (p=0.92, 5/10); X2 `NOT-SEPARATED` −1.34 % vs `fc1load_selfref` (p=0.054, 10/11) but 9461 got worse (+3.9 %); −14.6 % vs self-predict (11/11). |
| [**fullctx_refine_v1**](docs/lineages/fullctx_refine_v1.md) | `ACTIVE` | Training on full-batch context makes the self-refined GNN tie CD from the fast side: F1 `CLOSES-GAP` −1.76 % (p=0.76, 6/11); F2 −2.23 % vs `bc1load_selfref` (10/11, direction only); −13.9 % vs self-predict (12/12). |
| [**backlog_corpus_v1**](docs/lineages/backlog_corpus_v1.md) | `ACTIVE` | Synthetic-backlog corpus does not move the live queue (L1 +0.83 % NOT-SEPARATED, L2 +6.6 % vs CD); the gap is in-batch stacking; self-refine on the same weights ties CD (+1.5 %, p = 0.15). |
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
| [**load_repr_v1**](docs/lineages/load_repr_v1.md) | `CLOSED` | LOAD-HELPS / NARROWS: CD's load terms in seconds (`partial_state_v4`) make the burst-seat GNN −5.3 % vs its twin (11/11) and −8.5 % vs self-predict (10/11); CD still +6.7 % ahead (was +12.4 %). Found and fixed an uncapped-rung serving rank defect (cost < 1.5 %). |
| [**cd_gap_v1**](docs/lineages/cd_gap_v1.md) | `CLOSED` | SCORE-EXPLAINS / CD-FASTER: the burst-seat GNN trails CD because its score cannot load-balance; the label (CD-imitator +13.6 %), decode order and load (+13.8 % at half rate) don't explain it. GNN-seeded CD beats CD −3.8 % (10/10, direction only). |
| [**fresh_topo_burst_v1**](docs/lineages/fresh_topo_burst_v1.md) | `CLOSED` | MP-EDGE-GENERALISES (direction only): on 11 fresh topologies `gnnedge0` beats its MP-OFF twin −4.2 % (11/11, p = 0.001), but ties self-predict (−1.2 %, p = 0.70); MP-OFF loses to it (+2.8 %, 2/11); CD ahead of all. |
| [**selfpredict_burst_v1**](docs/lineages/selfpredict_burst_v1.md) | `CLOSED` | **`GNN-BEATS-SELFPREDICT`, burst-seat bar CD.** Uncapped gnnedge0 beats the self-predict rule −7.25 % (13/13 ckpt; 9/16 env, thin) in the burst seat — not an MP win (ties its MP-OFF twin); the CD greedy stays ahead of both (+12.5 / +21.3 %). |
| [**selfpredict_bar_v1**](docs/lineages/selfpredict_bar_v1.md) | `CLOSED` | **`BAR=SELFPREDICT`.** The rule + a price for unarrived partners at the node the rule would give them beats Knative −19.4/−21.4 % (16/16), the old rule −6.2/−8.5 % and the CD greedy −7.8/−8.9 % (C40 not separated): the programme's best policy and the bar for any learned arm. |
| [**lookahead_mp_v1**](docs/lineages/lookahead_mp_v1.md) | `CLOSED` | **`HAND-COORDINATION-RECOVERS`.** Pricing unarrived partners is real live headroom (oracle −6.6/−8.2 % vs the rule), but a hand self-predict rule gets 93–95 % of it (−6.2/−8.5 %): coordination, not a message-passing lever. P1 never started; that rule is the new bar. |
| [**rollout_imitation_v1**](docs/lineages/rollout_imitation_v1.md) | `CLOSED` | **`RULE-FASTER-LIVE`.** One step of policy improvement: the rollout label is horizon-stable and a pointwise scorer on the rule's own terms beats the rule −12.7% OFFLINE (held-out), but served live is +14–28% SLOWER (R2 FAILS, 0/16 C40). Beats Knative-ECT/random, loses to CD greedy — another offline/live reversal; the rule stays the unbeaten bar. |
| [**joint_burst_v1**](docs/lineages/joint_burst_v1.md) | `CLOSED` | **`NO-GNN-WIN`.** Trained on the served distribution (bursts, loaded states, group-optimum labels): the burst-trained `gnnedge0` beats reactive Knative −12.47 % (15/15) — first learned win in the programme — and beats its own cold twin −7.82 % (J5 HELPS, 15/15), but still loses to the batched greedy +16.83 % (0/15). Necessary, not sufficient; triggers `rollout_imitation_v1`. |
| [**backbone_sparsity_v1**](docs/lineages/backbone_sparsity_v1.md) | `CLOSED` | **`TOO-FEW-UNSATURATED-ENVIRONMENTS` on both variants.** A 250 Mbps backbone saturates reactive on every cell (a load lever, like payload ×3); p = 0.4 hangs reactive on 39/48 cells (a reachability lever). No study; distance cannot be read at this rate. |
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
