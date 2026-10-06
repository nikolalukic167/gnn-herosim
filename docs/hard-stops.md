# Hard stops — falsified, do not revive without new evidence

> **Status:** `REFERENCE` &nbsp;·&nbsp; **Index:** [LINEAGES.md](../LINEAGES.md)
>
> Things measured and closed. Each entry names the measurement that closed it, so reviving
> one is a decision made against evidence rather than by forgetting. Moved verbatim from
> `memory/memory.md` §2 (that file retired 2026-08-27; see git history); the `FALSIFIED` rows in
> [LINEAGES.md](../LINEAGES.md) are the lineage-level counterpart.

## The closed directions

- **Repeat the unchanged eight-request fixed-eviction residency recipes as a GNN-specific win** ([residency_placement_v1](lineages/residency_placement_v1.md), 2026-10-01). Four attempts, 18 checkpoints and 128 audited real-container runs establish no MP benefit. Hand search matches all 72 original test optima and 19/24 mixed-probe optima; vectorized exhaustive search is competitive with neural inference, and both learned arms lose to it on both mixed physical cases. The two-block physical design is exploratory. This stops these recipes and their amortization claim, not explicit joint eviction, larger decision spaces or all memory-aware placement.

- **Train a GNN on the unchanged four-reducer shuffle/fluid-replay recipe** ([shuffle_placement_s0_v1](lineages/shuffle_placement_s0_v1.md), 2026-10-01). The static screen has only 0.959% median headroom, and 48 real-TCP streams show small exploratory search gains while timing calibration fails on 2/4 configurations. The static optimum is not a validated real-network bound. Reopening requires a validated model and fresh evidence of residual placement value, not larger payloads chosen after this read.

- **Train from the tested hidden-arrival one/two-decision online reservation search** ([online_reservation_s0_v1](lineages/online_reservation_s0_v1.md), 2026-09-23: primary and disjoint shortlist retry each gain 0% median over an equally informed hand rollout portfolio, with 5/16 and 2/16 wins; 96 materialized HeROsim runs audit). This stops the tested search and its action labels, not all online callbacks or other future-aware targets.

- **Train a future-action-value GNN from either tested L2D rollout target on the claim that it has two-size headroom over cheap hand dispatch** ([l2d_rollout_headroom_v1](lineages/l2d_rollout_headroom_v1.md), 2026-09-23: the optimistic two-rule rollout gains only 3.55% median at 6×6 and 4.07% at 10×10 over the four-rule hand portfolio; a fresh guarded four-rule continuation gains 6.38% and 4.31%, so both fail the signed ≥5% at both sizes despite expensive complete-workflow search). This stops these continuation policies and the proposed training target, not online uncertainty, other search families or all graph learning.

- **Claim a two-size, both-controls L2D GNN data-efficiency win from the tested fixed-update recipe** ([l2d_learning_efficiency_v1](lineages/l2d_learning_efficiency_v1.md), 2026-09-23: the frozen GNN-1,000/control-3,000 gate passes 1/4 contrasts on seed-302 live rollouts; the fresh GNN-2,000/control-3,000 follow-up passes 3/4 on seed-303, with the 10×10 MP-OFF contrast only +1.12% and bootstrap 95% CI −2.70 to +4.45). The midpoint GNN beats typed MLP with 33% fewer training instances at both sizes, but a message-passing-specific claim needs the MP-OFF bar too. This stop is limited to these rates, sizes, features, seeds and budgets; equal-budget message-passing gains in the parent study remain intact.

- **Promote unchanged converged 6×6 L2D GNN checkpoints as a zero-shot message-passing advantage at larger job-shop sizes** ([l2d_size_transfer_v1](lineages/l2d_size_transfer_v1.md), 2026-09-23: on 100 fresh instances per target and eight paired training seeds, median GNN gain versus MP-OFF is −0.577% at 10×10 and −2.200% at 15×15, positive on 4/8 and 3/8; versus typed MLP it is −0.327% and −0.104%, positive on 3/8 and 4/8). All four registered ≥1% / ≥7/8 bars fail. This stop is limited to inference-only transfer of these checkpoints and features; the within-size 6×6 advantage and target-size retraining remain separate questions.

- **Promote this exact-validation-selected full-GNN proposal portfolio as uniquely useful at its measured CPU decision time** ([proposal_frontier_v1](lineages/proposal_frontier_v1.md), 2026-09-23: 24 audited live runs on eight fresh topologies; full GNN gains only 0.251% median over the time-feasible equal-size new MP-OFF portfolio, positive on 5/8, below the signed ≥2% and ≥7/8 bars). It beats hand multistart by 5.048% median on 8/8, so the stop is this checkpoint, selector, fixed-replica workload and physics recipe's unique GNN value, not learned proposals or GNNs generally.

- **Advance the unchanged four-rule one/two-decision resumed-continuation recipe** ([dag_resume_s0_v1](lineages/dag_resume_s0_v1.md), 2026-09-23: candidate and bounded-reference median gains over matched hand are 0% in both sixteen-workflow phases; 3,677 evaluated continuations and 128 actual-HeROsim runs audit). This stops that finite family under the tested deterministic, fully announced zero-I/O physics. It does not close resumable execution, all stepwise reservations or the true scheduling optimum.

- **Train a GNN from unchanged group-commit prefetch through both access and core pipes** ([g32_peer_prefetch_s0_v1](lineages/g32_peer_prefetch_s0_v1.md), 2026-09-23: finite versus infinite pipes raises exact RTT 32.5–54.9% at the four training-topology medians, but genuinely nonlocal core-link wait is only 5.43–16.54% of finite RTT and clears the signed ≥10% bar on 1/4 topologies; twelve fresh live runs audit). Most contention sits on endpoint access links, a count-shaped mechanism. This stops unfiltered prefetch, not a core-only capacity test.

- **Train a local-move GNN merely because core-only group-commit prefetch creates oracle headroom** ([g32_core_prefetch_s0_v1](lineages/g32_core_prefetch_s0_v1.md), 2026-09-23: median bounded 128-move oracle gain is 13.72% and fresh live core wait is 19.81–21.90% of RTT, but a 16-evaluation route-aware leave-one-topology-out hand control captures 72–100% of the oracle gain). This stops this single-reassignment/swap move pool and physics combination, not coordinated block moves.

- **Train a GNN on same-node peer-edge pair moves as a two-step valley target under core-only prefetch** ([g32_coordinated_valley_s0_v1](lineages/g32_coordinated_valley_s0_v1.md), 2026-09-23: only 18/3,156 pairs are material valleys; their bounded median topology gain is 1.03% versus the signed 8% bar, and the equal-budget route-aware hand control captures 91–100% of the broader pair oracle gain; four fresh live runs audit). This stops this pair pool and seat, not all coordinated multi-step search.
- **Train a GNN from the unchanged platform-start, store-and-forward peer-fabric recipe on four-type 32-task groups** ([g32_peer_fabric_s0_v1](lineages/g32_peer_fabric_s0_v1.md), 2026-09-23: despite high route overlap, actual core-link peer wait is only 0.604–1.060% of group RTT at four training-topology medians, with 0/64 groups above the signed 10% bar; eight fresh live runs confirm only 1.035–1.460% core wait share in the flag-on arms). Platform queues stagger transfers, so the simultaneous-transfer paper overlap is not realized. This stops this transfer-start semantics, not prefetch at group commit or every fabric-coupled workload.
- **Fund a move-value GNN from this four-type, low-payload, fixed-replica 32-task single-reassignment/swap pool** ([g32_fourtype_move_s0_v1](lineages/g32_fourtype_move_s0_v1.md), 2026-09-23: the bounded oracle gains 13.936% at the median of four topology totals and a frozen simple proxy captures under half, but leave-one-topology-out non-GNN 1/2/4-hop hand-feature controls capture 86.26–100% with Ridge and 61.22–100% with ExtraTrees in the same 16-evaluation budget; the required two-topology live manipulation gate audits). The registered residual-information bar fails. This stops training on this workload, capped move pool and target under the measured controls, not other environments or bipartite architectures generally.
- **Claim bipartite value from exact-validation-selected final weights of the 16-plan sampled-slate recipe** ([g32_exact_val_selection_v1](lineages/g32_exact_val_selection_v1.md), 2026-09-23: all 56 sealed live runs audit across eight fresh topologies; full GNN versus peer-only gains −0.640% direct and −0.307% in equal-size portfolios at the median, with 3/8 and 2/8 positives; versus MP-OFF, −0.362% and −0.096%, with 3/8 and 4/8 positives). All four registered +2%/7-of-8 bars fail. Full still beats hand +4.872%, 8/8, but peer-only and MP-OFF also beat hand. This stops this data, target, checkpoint-selection and serving recipe, not new decision targets or bipartite GNNs generally.
- **Repeat the 16-plan sampled-slate CE training recipe to claim bipartite message-passing value in the fixed-replica 32-task seat** ([g32_sampled_slate_v1](lineages/g32_sampled_slate_v1.md), 2026-09-23: all 56 sealed live runs audit across eight fresh topologies; the separately trained full GNN is slower than peer-only by 1.691% median as a direct proposal, 0/8 positive, and by 1.866% in an equal-size portfolio, 1/8 positive; it also loses to MP-OFF by 1.079% and 2.182% respectively). All three selected checkpoints were epoch zero because the capped 16-slate validation sidecar censored out-of-slate decoded plans; the later [exact-validation gate](lineages/g32_exact_val_selection_v1.md) repairs selection and still finds no bipartite gain. This stops this data/target/serving recipe, not new decision targets, proposal mechanisms, or bipartite GNNs in general.

- **Promote the seed-1 `joint_burst_v2` GNN-initialized coordinate-descent hybrid as a win under dynamic autoscaling** ([gnn_seeded_cd_v1](lineages/gnn_seeded_cd_v1.md), 2026-09-22 matched amendment: 16 live 50,000-task runs on four previously opened w0 topologies; at the same exchange weight and six-pass cap, GNN and hand starts split 2–2 and the GNN is 1.69% slower at the median while using about 13–17× more inference time). The earlier 3/4 hand win compared different score weights and is superseded. This stop is specific to policy-dependent replicas; the [fixed-replica successor](lineages/gnn_seeded_cd_fixed_v1.md) passes its fresh hand comparison.
- **Promote the seed-1 fixed-replica GNN-seeded search win over hand as a message-passing win** ([gnn_seeded_cd_mp_ablation_v1](lineages/gnn_seeded_cd_mp_ablation_v1.md), 2026-09-22: the matched separately trained MP-OFF seed beats the GNN seed in 8/8 live 50,000-task pairs by 1.68% median, and beats the hand start in 8/8). The fixed-replica GNN-versus-hand gate passes; this stop concerns graph attribution for this checkpoint and seat, not learned initialization generally.
- **Fund GNN training from the fixed DAG output-cache retention recipe** ([dag_memory_lifetime_s0_v1](lineages/dag_memory_lifetime_s0_v1.md#2026-09-22--corrected-output-memory-screen-and-live-gate-no-go), 2026-09-22: capacity binds on all 16 fresh reference schedules, but the slow feasible reference gains only 6.32% median over 250 ms memory-aware hand search against an 8% bar; hand capture is 55.7%, and all 49 live runs plus 65,536 reference proposals audit). This stops this keep/spill and FIFO-eviction recipe, not rematerialization or all memory-aware DAG scheduling.
- **Train another move-value GNN on the same FIFO keep/spill DAG memory recipe** ([dag_memory_value_learning_v1](lineages/dag_memory_value_learning_v1.md#2026-09-22--matched-training-and-sealed-live-gate), 2026-09-22: the explicitly authorized six-checkpoint exception produced +0.07% median GNN gain over MP-OFF with an interval spanning zero and a 2.75% median loss to 250 ms hand search across 16 sealed workflows and 144 audited live runs). A changed memory mechanism needs a new lineage and new headroom evidence.
- **Train on one-level, per-output inline rematerialization as registered** ([dag_remat_s0_v1](lineages/dag_remat_s0_v1.md#2026-09-22--corrected-one-level-rematerialization-screen-and-live-gate-no-go), 2026-09-22: the corrected fresh gate has 5.90% median feasible-reference headroom below 8%, hand capture 61.7% above its under-half target, and actual recomputation in only 2/16 reference plans; 49 live runs and 3,136 operations audit). This does not close per-consumer-edge recomputation or explicit recompute tasks.

- **Fund GNN training from the fixed critical/lock/setup/block-swap proposal pool on 384-operation reservation DAGs** ([dag_block_stepwise_s0_v1](lineages/dag_block_stepwise_s0_v1.md#2026-09-22--corrected-cpu-screen-and-live-gate-no-go), 2026-09-22: 16 fresh cases yield only 3.05% median feasible-reference gain over 250 ms hand search against an 8% bar; broad blocks beat hand on 0/16, and 52 live runs pass audit). This stops the registered move pool and order, not online reservation decisions or all GNN scheduling.

- **Keep widening static critical/lock/setup/block-swap move pools from a strong workflow start** ([dag_block_proposer_128_s0_v1](lineages/dag_block_proposer_128_s0_v1.md), 2026-09-23: at 128 operations, a 2,193-candidate median pool and up-to-three-round exact reference gain only 0.44% median over the 250 ms strong start on 16 fresh workflows; 49 live runs and 94,183 proposal rechecks pass). The user-signed 8% headroom bar fails. Stop iterating on these local/block move-pool variants; reassess the decision being learned rather than another pool width or scoring model.

- **Scale unchanged guarded one-shot complete-plan imitation on the irregular DAG reservation corpus** ([dag_reservation_learning_v1](lineages/dag_reservation_learning_v1.md#2026-09-22--exploratory-trained-live-result), 2026-09-22: four matched CPU-trained models and 96 actual HeROsim runs; GNN ties MP-OFF at 0.00% median, loses 1.81% to selected hand search, and all 64 raw learned proposals lose the initial-cost guard; 2,359,296 reference proposals and every live resource event audit pass). This stops this proposer and search path, not learned stepwise reservations or all message passing.

- **Fund training from the unchanged 256-operation DAG reservation recipe at its registered 8% bar** ([mixed_dispatch_v1](lineages/mixed_dispatch_v1.md#2026-09-22--schedule-translation-successor-result), 2026-09-22: schedule-preserving initialization and decoder switching leave 5.43% median feasible-reference headroom, interval [3.88%, 6.46%]; all nine controls fit 250 ms, 58 live runs pass, and 2,359,296 reference proposals reproduce). Reservations win 11/16 observed candidate comparisons, so this stops the unchanged funding recipe, not reservation usefulness or all GNN training. The finite reference is not an optimality bound.

- **Train the unchanged four-operation exact-region selector from a 140 ms joint start** ([mixed_dispatch_v1](lineages/mixed_dispatch_v1.md#2026-09-22--exact-region-selection-s0-result), 2026-09-22: on four exploratory inputs, free perfect choice among 128 repairs gives 0.55% median headroom versus full-budget joint search, below 8%; the optimistic path fits 220.15 ms and all 12 live runs complete). This stops this finite pool and serving recipe, not all region sizes, insertion moves or coordinated repair.

- **Fund training from the unchanged 256-operation joint placement/order screen** ([mixed_dispatch_v1](lineages/mixed_dispatch_v1.md#2026-09-22--joint-placement-and-ordering-result), 2026-09-22: the finite reference gains only 0.91% median over the calibration-selected joint-greedy control, below the 8% bar; all five hand arms fit 250 ms and all 48 qualification live runs complete). This stops this search recipe, not the true optimum, larger coordinated repairs or all joint-action learning. A weaker annealed hand arm is not a valid way to restore headroom.

- **Fund training from the unchanged capacity-scaled 256-operation block screen** ([mixed_dispatch_v1](lineages/mixed_dispatch_v1.md#2026-09-22--capacity-scaled-block-search-result), 2026-09-22: four-round structured search finds 5.06% median feasible-reference headroom against an 8% bar; 64 qualification live runs validate baseline/reference/hand plans, including the corrected budget-eligible control). This stops advancement of this recipe, not all block moves or the true scheduling optimum. Timing repairs alone do not meet the headroom prerequisite.

- **Scale the unchanged unguarded outcome-value ready-set policy** ([mixed_dispatch_value_v1](lineages/mixed_dispatch_value_v1.md), 2026-09-22: nine models and 384 real-HeROsim runs; GNN is directionally +1.90% over MP-OFF and +1.68% over hand-graph with intervals crossing zero, loses 8.98% to `srpt:128`, and serves in 11.30–11.45 ms p95). A selective residual over the hand rule is still open; more draws or compilation cannot supply the missing full-policy objective gain.

- **Scale unchanged teacher-trajectory cross-entropy with the hidden16/two-layer ready-set decoder** ([mixed_dispatch_state_v1](lineages/mixed_dispatch_state_v1.md), 2026-09-22: nine models and 384 real-HeROsim runs; GNN ties MP-OFF and hand-graph, loses 9.74% to `srpt:128`, and serves in 15.47–15.69 ms p95). Compiling the event loop can repair latency but cannot create the missing objective or message-passing gain. A successor needs outcome-aware action values or on-policy correction and a residual-MP witness.

- **Scale normalized total-priority-rank imitation with the unchanged direct decoder** ([mixed_dispatch_learning_v1](lineages/mixed_dispatch_learning_v1.md), 2026-09-22: nine models and 384 real-HeROsim runs; GNN is 19.03% slower than `srpt:128`, 0.91% slower than MP-OFF, and 1.59% slower than hand-graph). Full inference fits 5 ms and a posthoc 64-swap refinement remains 3.26–5.12% behind the rule, so timing or a tiny decoder patch cannot rescue this target. This does not stop state-dependent ready-set actions or outcome-aware objectives.

- **Scale the unchanged hidden16/two-layer, 30-epoch soft-target mixed pair selector at the 5 ms / 5% bar** ([mixed_pair_learning_v1](lineages/mixed_pair_learning_v1.md), 2026-09-22: nine models, 384 real-HeROsim runs; zero median GNN gain over MP-OFF, hand-graph and graph1, full GNN path 5.20–5.24 ms p95). This stops this pilot recipe, not all pair moves or all learned search; a timing-only fix cannot supply the missing objective gain.

- **Train a selector over either unchanged mixed pair-refinement portfolio under the registered 2 ms / 5% bar** ([radical_physics_v1](lineages/radical_physics_v1.md#2026-09-22--pair-selector-results-and-live-audit), 2026-09-22: 192 real-HeROsim runs; deep perfect selection gains 6.28% but its free-selection path takes 3.95 ms p95, shallow perfect selection gains only 2.63%). This stops these serving recipes; faster implementations, different neighborhoods and other budgets need new evidence.

- **Repeat the unchanged two-layer mixed-physics imitation/self-critical model with initial-cost guarding and one refinement round** ([mixed_physics_learning_v1](lineages/mixed_physics_learning_v1.md), 2026-09-22: nine models, 768 real-HeROsim runs; zero median GNN gain over MP-OFF/hand-graph controls, 4.97% worse than timed hand search). This stops the recipe, not the setup/lock/power environment or all learned search.

- **Train a selector for the registered twelve-start workflow portfolio at the 5% bar** ([workflow_basin_v1](lineages/workflow_basin_v1.md), 2026-09-22: perfect selection gains 2.58% on qualification and 1.60% in 128 fresh live runs against a calibrated hand selector). This stops this finite portfolio, not learned construction of new schedules.

- **Train a GNN to rank exact single-flip full-workflow RTT values on the fixed 80-operation fixture without first beating compiled exact search** ([workflow_move_value_v1](lineages/workflow_move_value_v1.md), 2026-09-22: all 80 values plus initial ECT cost 0.20 ms p95; 224 fresh live runs confirm exact descent fits 2 ms p95 and beats the GNN hybrid by median 4.04%). This stops this target and family, not multi-step planning or larger instances.

- **Repeat the frozen workflow checkpoints with ECT guarding and probability-weighted proposals at 2 ms** ([workflow_proposal_v1](lineages/workflow_proposal_v1.md), 2026-09-22: 384 live runs on 32 fresh workflows leave the guarded GNN 4.97% worse than timed hand search; no positive GNN proposal count fits the budget). The Knative gain is shared by stronger hand controls. This stops this decoder/budget combination, not guided search generally.

- **Scale the same one-shot workflow imitation recipe solely because a slower teacher has headroom** ([workflow_amortized_v1](lineages/workflow_amortized_v1.md), 2026-09-21: nine trained checkpoints and 3,001 live replays on 256 workflows leave GNN tied with MP-off and 7.15% worse than the best tested hand control within 2 ms). A successor needs a demonstrated full-plan training/validation improvement and a fresh sealed gate; this is not a stop on all workflow GNNs.

- **Treat heterogeneous execution alone as a GNN opportunity in the private-replica three-host fixture** ([graph_unary_screen_v1](lineages/graph_unary_screen_v1.md), 2026-09-21: 60 exploratory graphs and eight live checks establish a real trade-off, but graph-cut expansion leaves at most 0.774% live RTT regret, below the 5% bar). The stop concerns these tested draws, not heterogeneous scheduling generally.

- **Scale the private-platform, three-anchor witness solely by adding tasks** ([graph_size_screen_v1](lineages/graph_size_screen_v1.md), 2026-09-21: non-MP coordinate descent reaches all 12 certified optima at 12/24/48 tasks, with live replay of selected plans; all 48-task optima co-locate 46 tasks). This stops size-only scaling of this fixture, not larger scheduling problems generally.

- **Fund training merely by adding a third host to the nine-task private-platform witness** ([graph_three_host_screen_v1](lineages/graph_three_host_screen_v1.md), 2026-09-21: hand min-sum solves all 16 screened graphs; exact search takes under 1 ms, with complete live sweeps on two preselected graphs). This stop does not extend to larger groups or shared-resource objectives.

- **Train on the private-platform, two-host attractive-exchange witness as evidence of a learned advantage** ([graph_decision_witness_v1](lineages/graph_decision_witness_v1.md), 2026-09-21: cheap hand min-sum and min-cut attain the exhaustive live optimum in all 12 constructed cells). Graph information matters here, but learning has no remaining objective headroom against these controls. This stop is specific to that construction.

One per line. Each names the measurement that closed it; the node carries the method.

- **Promote first-task-only two-hop lookahead from the small uncapped-cycle pilot, or fund GNN training on that pilot alone** ([peer_lookahead_v1](lineages/peer_lookahead_v1.md), 2026-09-21: fresh live validation on 16 infrastructures misses the fixed 5% gain bar against both immediate and peer mass). Reopening requires a materially different environment with verified headroom; this stop does not assert learned-model equivalence.

- **tune link bandwidth or core count to rescue `link_contention_v1`** (null lever + full spectrum swept: n_core 2/4/12 all ≤0.35% mean regret)
- **cite offline regret/acc as evidence a model will place better live** (Arm A: −54% offline regret → +3.5%/+9.9% *worse* live, `logit_tied_rate` rose)
- **cite the 0.88× Kn `sparse_p35`/s42 figure** (irreproducible: current code gives **1.04×** on the same cell/seed/model; the arm that produced it set `GNN_DROP_NODE_EDGES=1`, implemented **nowhere** in the tree today)
- **revive same-node edges without new physics** (Arm B falsified *while trained with them*) · RQ3
- **RQ3b**
- **claim the GNN's live failure is "structural in the joint decode"** (falsified: LQB λ=1.5 no change 275.7M; `argmax_uniq` drove collisions to **0.000** and got *worse* 301.3M)
- **cite ECT as a ceiling or distill teacher for Regime A** (0.98–1.13× Kn, and `contention_v2` is pull-free so `ect_pull`≡`ect`)
- **run `ect_pull` with `ECT_PULL_DISTILL_DIR` on workload-125-225** (561,848 frames/cell ≈ 50GB; fills disk) · cite pre-`25732cf` Regime A tables as current · mix pre/post-serialization in one table · re-eval 873/v5.5 as if it transfers to serialized FilterStore · hub9 decode · claim pull-obs/cosim-retrain/`soft_combo_conc`/hard-CE-distill closed 125→31 · blind retrain scarce-warm 450 · reopen hub9/`total_rtt` primary · HeteroData/Set-Transformer · claim warm/busy v1 init helps FilterStore distill
- **relax α to get a readable route_b pivot rung** (it is a cliff, not a dial: fine sweep on H1 gives α=3.9 binding 204/204 with 80 stuck, α=4.0 stuck 0 and binding **0/204**, byte-identical to the unconstrained anchor — clean counters and "the constraint binds" are mutually exclusive on this grid)
- **raise `replica_server_percentage` to break single-node confinement** (two full 204-dataset probes: confined-task histogram is an identical `{0:102, 2:102}` at 2, 3 and 4 hosting nodes, and stuck drifts slightly *up*; the knob changes how many nodes are eligible, not how the FCFS allocator at `generate_infrastructure.py:625-660` spreads across them)
- **read `greedy_stuck` as a fact about the route_b pivot environment** (it is decoder myopia on **every** rung: over identical candidate sets, caps and ordering, backtracking rescues **458/458** across H0/H1/H2, and every stranded dataset had a feasible plan in its own enumerated sweep)
- **pursue a supervised message-passing win on ANY graph** (mp_ablation_v1 + link_mp_v1, 16 paired seeds each on the binding-backbone corpus: on the old task↔platform graph MP is *harmful* — no-MP beats it +4.50pp, exact Wilcoxon p=0.00107; the `core_v1` link graph repairs exactly that harm — link-MP beats old-graph +4.98pp, p=0.00459; and repaired MP then **ties** no-MP, +0.47pp, p=0.372 — the pointwise ceiling `program_verdict_v1` predicts for any supervised co-sim target. The remaining venue for an MP win is closed-loop training against the live simulator, not a better supervised graph)
- **label a placement by the multi-second horizon return of the live simulator** (objective_pivot_v1 Phase 2 / P3, 300 snapshots × 256 plans at h=10 s: the registered co-primary *fired* — 21.7% of snapshots above 2% regret, both repair controls ~0 — and the chaos control then showed the ranking is random. Across 60 high-signal snapshots, median Spearman ρ(h2,h10) = −0.027 and ρ(h5,h10) = +0.004, and the h=10 optimum sits at rank 120/256 at h=5 against 128/256 for chance. The return is deterministic chaos, not placement quality; the collapsed additive R² of 0.049 is unexplainable variance, not joint structure. Closed-loop policy gradient — which averages expected return over episodes — is unaffected and is the remaining venue)
- **run the residency-hold scaling pilot (hold the node slot across cold start + exec)** (objective_pivot_v1 Phase 2b: its premise is false for co-sim — cold start is **0.000 s in 320 of 328 sampled datasets across all 16 collections**, the 8 exceptions totalling ≤0.206 s, including in `regime_b_cold_burst_v1`. `system_state.replicas` is keyed by task type and warmup warms each replica with its own type, so `sandbox_is_warm` is always true and cold start is never incurred; the cited 38 s is `cnn`/`xavierDla`'s table entry, never an event. Residency therefore equals exec, so the hold is the exec hold that already measured `nodeContentionTime ≡ 0.0`. Firing it needs overlapping replica pools — a corpus physics lever, explicitly stopped)
- **train the served policy by closed-loop policy gradient against the live simulator** (objective_pivot_v1 Phase 3, Amendments A-E: 120 fresh training seeds, REINFORCE with a self-critical baseline and a two-pass N/k replay whose log-probs reproduce the sampler to 1.1e-16, gated on the held-out `bb_core8_bw1p5` fabric. Paired mean **-0.849%**, median -0.846%, **53/120** seeds better than the frozen init, one-sided Wilcoxon **p = 0.928**. **Adequately powered**: at the observed across-run sd of 4.89 pp the registered 3% MDE needs n >= 84 and we ran 120, so a 3% gain would have been visible and the point estimate is negative. The 16-seed pilot's +0.27% median was a draw and was correctly excluded from the primary. What the loop CAN do is move the graph model at all — across-run sd 4.89 pp against the MLP's 0.0325 pp, a 150x trainability asymmetry — but moving is not improving, and at n=120 it moves the wrong way as often as the right one. Reviving this needs a *different* configuration with its own powered tuning stage, registered before its data exists, not another n on lr=1e-4)
- **quote a GNN-vs-MLP latency or reliability number from arms trained on different corpora as a model-class result** (link_mp_v1 pilot 2026-09-03: an MLP trained on the GNN's own `graphs_cache_link_mp_v1_core_v1_dim14` ties it on both backbone fabrics, 4.42–5.02M vs GNN 4.95M on `bb_core8_bw1p5`, so "−44.8% vs −29.2%" was two corpora, not two model classes; reliability_matched_v1 FAIL 2026-09-04: matching the corpus removes 87% of the MLP's collapse burden (107 → 14 cells) and the residual is not established at the registered n=16, rank-sum p = 0.113 at +50%. Any GNN-vs-MLP number must name both arms' training cache; a powered reliability re-run is a separate registration that excludes those 16 draws)
- **grow the route_b DAG corpus to rescue message passing** (`route_b_v1` Phase 2, closed 2026-09-08 on a full retrain, 3 arms × 8 seeds over 204/612/1020 training DAGs). The honest-selector contrast is a **TIE** (median-paired +0.04 pp, p = 0.25) and both GNN arms beat the MLP+prefix baseline, so the framing is **"MP is redundant with the prefix columns, not harmful"** — corpus size is not the lever. Mechanism: the root task is exactly pointwise and the children's ~23 % joint variance is entirely pairwise parent→child co-location, which the prefix columns every arm already encodes. The earlier GAP-PERSISTS reading was a checkpoint-selection artifact and is superseded; the node carries both readings)
- **compare route_b offline decode regret with any live number** (2026-09-06 audit: the live GNN scheduler batches only parent-finished tasks, so a diamond4 is never decoded jointly, the live graph carries no DAG block, cnn/rf have no live candidates, the loader double-counts the one-hot width and all of it is swallowed into shortest-queue; and the offline object — 4-task joint decode under α=2.0 + replica uniqueness on a frozen snapshot — is not the live object even once serving works)
- **serve the DAG output payload over the link fabric to give link contention magnitude** (dag_fabric_contention_v1 paper screen 2026-09-08, no simulation: on the stored route_b `arm_s` sweep, store-and-forward waits on the 800 MB parent→child transfers would pass the 10% manipulation bar at ≤ 100 MB/s, yet the α=2.0 optimum carries zero link wait in 98–100% of 204 datasets at every hosting-node count (0/17 even at 2 hosting nodes) — the label never sees the mechanism, exactly the "optimal plans select against contention" shape of `link_contention_v1`; at 8 tasks the joint optimum cannot avoid waiting in ~50% of datasets but the binding wait is the root's/sink's own access link, a remote-children/parents count one column repairs, while core-link sharing binds at the optimum in only 10–12%. Funnelling the backbone to force core sharing turns the term into the number of remote DAG edges — the hop+coupling block that already closes 0.997 at 8 tasks. Not a tuning miss)
- **explain `peer_affinity_v1`'s offline/live reversal by herding or by queue-blindness in the graph arm** (`serving_gap_v1`, registered and closed NO-GO 2026-09-12 on its own kill bar, 192 live production runs = 3 corpora x 2 arms x 16 seeds, uncapped, S0 reproducing the registered offline contrast to +5.142/+6.532/+9.606pp against +5.14/+6.53/+9.61). **Both hypotheses were refuted in the direction OPPOSITE to the one registered.** H1: the `gnn` arm spreads its node choices MORE evenly than `mpoff` — modal repeat rate -0.083 (p=0.009) and normalised node entropy +0.062 (p=0.018) on x800p3, 0/3 corpora firing. H2: `gnn` is **3-7x MORE** responsive to queue depth than `mpoff` on every corpus (normalised score sensitivity 0.582/0.424/0.770 against 0.169/0.113/0.111, all p<1e-4). The graph arm is the most responsive arm measured in this program, so any future account of the reversal must start from over-responsiveness, not rigidity — and that account is post hoc and needs its own registration. Note what is NOT stopped: reading a decode trace to find a serving defect is the method that produced the platform cap (+21.9% vs Knative, 16/16 seeds) and stays available)
- **model CPU / memory co-tenancy interference (processor-share exec dilation, memory over-commit cliff, type interference matrices) to give a supervised GNN joint structure** (closed by argument 2026-09-09, no code built — `cotenancy_interference_v1` was Track B of the 2026-09-08 plan: exec time is a table lookup (`infrastructure.py:1337`) and nothing on a node dilates, so any such physics is new; but every dilation a task suffers is a function of the multiset of tasks co-resident on its node, and every symmetric function of a multiset is a function of per-type counts (`route_a_v1`'s composition theorem), which the `hetdem`/`krank` pointwise blocks already express — the S2 kill bar closes by construction, and the only overlap grid that makes placed tasks co-run (`route_b_pivot_h2`) carries a separable control that voids S0 on the parent-locality branch (measured 2026-09-08). Reopen only with a mechanism that is NOT node-indexed and binds AT THE OPTIMUM — `dag_fabric_contention_v1` shows the one such candidate does not)
- **cut a supervised corpus from the cluster the model is served on to make message passing transfer live** (`peer_affinity_warm_v1` W0, 2026-09-13: 100 brute-force-labelled datasets from live snapshots of the gate cell, both behaviour sources, support closure PASS — `xavierGpu` in 67 % of datasets, queue column max 590 — and the pointwise-recoverable regret of the warm one-step label is median 0.0 %, 0/100 above 2 %, 88/100 exactly zero. The served label is queue drain + per-pair transfers, a sum of (task, placement) terms; the state mismatch was real and closing it produces labels the pointwise class fits exactly. A different, drainable served regime is a new environment question, not a re-run. **Confirmed on the LIVE gate 2026-09-14 (W1, Amendment 1, rule 6):** 472 + 72 warm datasets, 16 seeds per arm, cell_s7901 — offline `gnn` vs `mpoff` TIE (+0.74 pp, p = 0.083); live capped the warm graph arm is 23.9 % slower than the cold one (0/16) and 10.4 % slower than its MP-OFF twin (0/16, p = 3e-05), uncapped −17.2 % (1/16); vs Knative +1.5 % / −1.9 %. Headline NO-WINNING-GNN in both configurations. Do not re-run with more warm cells or seeds: the live contrasts are at p ≤ 6e-05 on 16 pairs and the offline label is pointwise by W0.b)


## The existing checkpoints on big infrastructure where the baseline is healthy (2026-09-19)

**Closed by:** `unsaturated_scale_v1` S2 live gate — 320 arms, 5 arms × 4 cells × 16
checkpoints at 80 servers / `f4000`, jobs 789489–789795.

**Direction:** "the 80-server wins are real; find a big-infrastructure load where reactive is
not saturated and the repaired graph arm will beat it there too."

**Measurement:** the only 80-server load at which reactive is unsaturated is the same absolute
rate 6 servers runs at (0.46/s, share 0.66; 0.955 at 0.92/s — S1). At that rung every learned
arm is `NOT-SEPARATED` from reactive (`gnnedge0` −6.5 %, p = 0.47; `peeronly` −8.0 %, p = 0.12;
both `mpoff` ±0.2 %) and the full `gnn` **loses** +17.0 % (p = 0.044). The whole margin is a
**6.8 s batch-assembly wait** reactive never pays, identical to R0's because the arrival rate is
identical. Do not re-run with more checkpoints: the losing tail is +14 to +130 % per checkpoint,
not noise. What is NOT stopped: a serving design that does not wait for a batch, and training
at scale — neither has been built.

## The peer-affinity environment at a servable load (2026-09-14)

**Closed by:** `drainable_regime_v1` S1 live gate — uncapped array 34/34, jobs 764836/764837.
(An earlier version of this stop, written on a confounded read, was withdrawn the same day; the
numbers below are from the corrected gate with peer groups assembled and every control holding.)

**Direction:** "the peer-affinity live results are a measurement problem; run the same checkpoints
at an arrival rate where the environment's own pair-indexed cost is a material share of the
objective, and the graph arm's edge will appear."

**It does not.** At x4000 (ρ ≈ 0.16, 0.46 arrivals/s), where peer exchange is **21.19 %** of
`total_rtt` instead of 0.0098 %, cool-down is **0.07 %** instead of 99.89 %, the served dim-7
queue p90 is **5** against the cold-corpus max of 42, and the decoder's batches carry **83,788**
in-batch peer pairs against 10,212 outside: `gnn` vs `mpoff` is **−15.94 %, p = 0.0010, 3/16**
and `gnn` vs reactive Knative is **−226.04 %, 0/16**. `mpoff` is −181.22 %, 0/16.

**What actually happens:** both learned arms *succeed* at the peer objective — rendezvous wait
falls from 47.9 % of exchange time to 9.6 %, a 5× cut — and lose anyway, paying **12.4 s** of
peer-group batch wait the reactive arms do not, **3×** the queue time and **~9×** the autoscaler
churn. The graph arm's loss to its twin is **queue time** (65.19 s against 53.40 s); it carries
*less* peer exchange and *less* rendezvous wait than the pointwise twin.

**Do not propose** re-running the peer-affinity checkpoints at another arrival rate. A learned arm
fitted on states from a 940× overload does not transfer to a cluster with headroom; that is a
**training-distribution** problem, and the supervised route to fixing it is closed by
`peer_affinity_warm_v1` W0.b.

**Amended 2026-09-14 (`drainable_debug_v1`): the premise above is false for the COLD checkpoints
this gate served, and the stop is narrowed accordingly.** The cold T1b corpus states are the
generator's own seeded queue draws (`shallow_pois2`, `deepvar_uniform0_12`, `deepvar_pois4`;
dim-7 max **42** over 516 datasets) — it was the *warm* corpus that was cut from served states,
and the gate *trace*, not the corpus, that carried the 940× overload. So "fitted on overload
states" describes `peer_affinity_warm_v1`'s arms, not these. What stands unchanged: do not re-run
the **warm** checkpoints at another rate (closed at p ≤ 6e-05), and do not re-run **any** arm at a
new rate without scaling every policy time constant and adding a control bar that reads the arms'
own counters — three reads of this very gate were confounded exactly that way. A registered
ladder that carries those controls is permitted and is what `drainable_debug_v1` Phase 2 is.

**Separately closed: `GNN_PREFIX_PLATFORM_CAP=1` in a drainable regime.** Three of sixteen capped
`gnn` seeds deadlock the simulator — identical simulated clock at death across 24 GB, 64 GB and
256 GB and a 4× runtime range, memory growing while time stands still. The same seeds complete
uncapped. The cap concentrates placement until a client's reachable servers are memory-full and
the starved-client retry loop takes over. The cap is for a cluster whose capacity mask cannot
bind; it is not a serving default.

## Not a stop: a closed-form shaped supervised label (scope note, 2026-09-14)

Filed with `drainable_objective_v1`'s registration so the next reader does not close it by
pattern-match against the two neighbouring stops. **Neither covers a label of the form
`sweep_rtt(plan) + V × f(state, plan)` where `f` is a closed form.**

* The **horizon-return** stop closed a label that was the live simulator's **rollout return** over
  h = 10 s of trace. Its killing control was rank stability across horizon lengths (median Spearman
  ρ(h2,h10) = −0.027; the h10 optimum at rank 120/256 at h5), and `docs/lessons.md` L32 names the
  mechanism: such a label measures *"the decision plus everything the simulator did afterwards."*
  A closed form runs no simulator forward and has no such amplification. **The control is still
  owed** — L32 makes rank stability non-optional for anything in this family — and is inherited as
  stability under a change of the shaping coefficient rather than of a horizon.
* The **closed-loop policy-gradient** stop closed REINFORCE against the live simulator. A
  supervised label is not that, and inherits none of its apparatus; its reopen clause ("a
  *different* configuration with its own powered tuning stage") governs policy gradient, not
  relabelling.
* The **warm-corpus** stop closed **one-step** labels cut from served states (W0.b: pointwise
  recoverable, median regret 0.0 %). A shaped label is not one-step; its own W0.b-shaped question
  is asked as that lineage's A2 read.

**What a shaped label does inherit, and it is binding:** it is a *label* lever, not a GNN lever.
Any term expressible in per-platform counts and their squares is inside registered count
competitor v2 (R² 1.0000 median, `peer_affinity_v1` A6) and by the `route_a_v1` composition
theorem a pointwise scorer handed those columns expresses it exactly. **No GNN-vs-MLP or
GNN-vs-MP-OFF claim may be founded on one.** And it still ends with a live gate (rule 6).

## The queue-externality shaped label, as constructed (2026-09-15)

`drainable_objective_v1`, CLOSED **OBJECTIVE-NOT-DELIVERED**. The scope note above stands —
a closed-form shaped label is still not covered by the horizon-return or policy-gradient
stops — but **this instance of one is closed, and the reason is not the latency number.**

**What is stopped, exactly.** The label
`L_V(plan) = rtt(plan) + V · Σ_p (λ_p/2)·[(B_p + A_p)² − B_p²]`
at **V = 1**, **λ = 0.46 arrivals/s**, the **measured** per-item drain clock, trained on
T1b's own 516-parent corpus and split at lr 2e-3 with T1b's recipe, decoded `masked_topo`
at a 16 s window and gated at the x4000 drainable cell. Do not re-run this configuration.

**Why, and this is the part that matters.** It is not that the charge was priced and the
decoder disagreed; it is that **the charge never reached the decoder's behaviour.** The
registered behaviour control asked that the shaped arms stop taking replicas deeper than
the shallowest: bar ≤ 18 %, unshaped control 35.5 %, shaped arms **37.8–42.2 %** — every
shaped arm concentrates *more* than the thing it was built to repair. The live latencies
are consequently ties with the unshaped control (`mpoff` −0.16 %, p = 0.638; `gnn` −2.55 %,
p = 0.441) and lose to reactive Knative 0/16 and 0/15. Phase A had already measured the
label agreeing with the stream on only 33.3 / 42.5 / 34.5 % of states with real choice
against a 60 % bar, so the gate is consistent with the screen — through a mechanism that
sits upstream of latency.

**What is NOT stopped.** Whether *any* non-myopic supervised label helps. This one was
never delivered, so it is no evidence against the family. **A successor is admissible and
owes three things before it is worth a gate:**

1. **C1 first.** Measure that the decoder's placement depth actually moves under the new
   label, on served states, before spending a live gate on latency. A label that does not
   change behaviour cannot be tested for whether the behaviour helps.
2. **An account of why its term survives the decode.** This one was a per-platform cost the
   sequential decoder could see and did not follow; say what is different.
3. **The inherited constraints, unchanged:** the rank-stability control over its own
   coefficient, the count-competitor concession (label lever, never a GNN-vs-MLP claim),
   and rule 6.

**Two side facts from the same gate, recorded so they are not rediscovered.** One shaped
`gnn` seed **deterministically livelocks** the simulator (sim t = 4,293 s, 99.4 % CPU, zero
log growth, reproduced byte-for-byte) where its T1b twin at the same seed and cell runs in
54.02 s — retraining on a shaped label can produce an unservable checkpoint, the same class
as `GNN_PREFIX_PLATFORM_CAP`'s 3/16 deadlocks. And the **offline/live reversal reproduced
on a label with nothing to do with peer affinity**: zero-overlap 10.0 % offline win for the
graph arm, 6.65 % live loss (p = 0.015). Whatever drives that reversal is not a property of
the peer-affinity objective.

## The decode-time queue guardrail, as configured (2026-09-15)

`serving_stability_v1`, CLOSED **STABILITY-NOT-THE-LEVER**.

**What is stopped:** masking a candidate replica whose queue depth exceeds
`K × (shallowest candidate for the same task + 1)` at decode time, with **K = 3**, relaxed
when it would strand a task, on the V = 1 checkpoints at the x4000 drainable rung, 16 s
window, uncapped, Q = 100, cells s7901 / s9001 / s9002. Do not re-run this configuration.

**Why.** The guardrail is not broken and not unsafe — it bound on **84–90 %** of decisions and
produced **0 hangs**, where `GNN_PREFIX_PLATFORM_CAP` deadlocks 3/16 seeds in this same regime.
It simply does not deliver: **0/3 cells beat reactive Knative** (−35.2 %, −5.1 %, −216.3 %),
and its paired effect runs the wrong way across cells — **+36.0 %** on s7901, **+1.5 %** on
s9001, **−13.6 %** on s9002, which is the cell where the arms were *least* stable. "Stabilise
the runaway arm and it recovers" predicts the opposite of what s9002 did.

**Disclosed:** the S3-d bar named an unpaired test on seeds paired by construction; under the
matched test the outcome would be HELPS-NOT-ENOUGH rather than STABILITY-NOT-THE-LEVER. The
registered verdict stands. **S3-c is 0/3 under either test**, so nothing here beats reactive.

**What is NOT stopped** is an agenda, not a stop: it lives with its lineage, in
[`serving_stability_v1`](lineages/serving_stability_v1.md) under **What is NOT stopped**.

**Inherited, unchanged:** no GNN-vs-pointwise claim may be founded on a queue guardrail — the
pointwise twin gained at least as much as the graph arm throughout, exactly as the count
theorem predicted before the gate ran.

**put the served queue column back inside its trained range to recover the learned arms' early advantage** (`queue_range_v1`, CLOSED 2026-09-15, 285 live arms = 3 cells × 3 serving arms × 2 policies × 16 seeds, one commit, `plain` reproducing `serving_stability_v1`'s unguarded medians to three decimals). The premise is **true and measured**: `legacy_v0`'s dim7 has a trained range of 0–42 (the adaptive divisor is 1.0 in 516/516 datasets, so the feature is the raw queue depth, candidate p50 12), and live it reaches **38** on `cell_s7901` and **113** (peak 202) on `cell_s9002`, absent in deciles 1–2 on every cell and present mid-trace, with severity ordering perfectly against the loss to reactive (−6.7 % where the column never leaves range, −111 % at 38, −183 % at 113). **The fix binds and does nothing.** Clamping dim7 at 42 and dim13 at 0.44 drives the out-of-range share from 0.107/0.164/0.197 to exactly **0.000**; paired Wilcoxon with Holm over the registered family of 6 clears **nothing**, the largest effect anywhere is 0.5 s on a 74 s arm (**0.7 %**), and on the worst-violating cell the fix is significantly **worse** (−0.367 s, 3/15, p = 0.011). Pinning the divisor is an exact no-op by construction and measured as one (+0.000 s, 0 seeds moved, 6/6 combinations). Do not revive as a *serving* change. What is **not** closed: `scale_invariant_v1` as a **training** contract (untested — this lineage retrained nothing); the queue blow-up itself, which is real on 2 of 3 cells and whose cause is unknown; and the **COMPRESSED** failure mode (an inflated divisor squeezing candidate differences to nothing), which never occurred here — the divisor was 1.0 in **100 %** of batches on every arm — and is therefore untested rather than refuted.

**explain the learned arms' loss to reactive by head-of-line blocking in the scheduler** (`scheduler_residence_v1` R0, CLOSED 2026-09-16, 95 live arms across 3 cells at n = 13–15). The scheduler is a single serial process and a ready task that is not a peer of the batch being collected does wait out that batch's 16 s window, so the mechanism is real — it is just **9–11 % of the cost**. Decomposed per task and reconciled against `averageWaitTime` to **six decimal places** with zero uncovered tasks: head-of-line **0.59–0.76 s**, collection **6.10–6.13 s (89 %)**, placement **0.000 s**. The instrument is bit-identical to the control on all 95 shared arms. Two things this settles: **the GNN's inference costs no simulated time at all**, so none of the loss is decode cost; and `queue_range_v1`'s "batch wait" label was right — the 6.87 s is peer-group assembly at 0.46 arrivals/s, where a 10-task group takes ~21.7 s to co-arrive, and `drainable_serving_config_v1` has already swept the only knob on it (the window; 0 s = −1731 %, 16 s the interior optimum). The remaining lever is the arrival rate, which is a corpus/environment question, not a serving one.

**predict a cell's queue blow-up from its reachability structure** (`scheduler_residence_v1` R1/R3, CLOSED 2026-09-16, 135 live arms over 20 minted topology draws, 15 readable). On three cells the separation looked decisive and had a mechanism: the only cell without a blow-up had every client reaching ≥ 2 servers where both bad cells had a client reaching exactly 1, and a clients-per-server imbalance of 3 against 8 and 7, with the *mean* fan-out separating nothing. On **15 independent draws from the same generator** it carries no information: `min_reachable_servers` vs the cell's excess queue over its own reactive arm is **ρ = +0.041, p = 0.885** against a registered bar of \|ρ\| ≥ 0.60 and the wrong sign; the second statistic is +0.029; the pointwise control is +0.020. Do not revive either statistic without a *different* mechanism. **Still open, and now sized:** the excess ranges from **−29.2 s to +51.6 s** across draws that differ in one config field, and nothing measured explains it.

**serve the v1/v2-contract checkpoints on a cluster larger than 6 hosting nodes** (`cluster_scale_v1`, CLOSED 2026-09-16, jobs 769363/769390/769426). *Qualified the same day by `partial_state_v3` (CLOSED): the stop is about the fixed-pad representation, not the models — checkpoints retrained under the size-free `partial_state_v3` contract served 12 / 24 / 80 hosting nodes on 140/142 arms without a raise and beat reactive Knative at 24 and 80 servers on 4/4 topology seeds (saturated rungs, relative statistic). Do not serve a v1/v2 checkpoint above 6 nodes; retrain under v3 instead.* Two independent hard limits compound: (1) `KRANK_WIDTH = 6` in `src/policy/tabular/reduced_features.py:252` — the partial-state feature block encodes candidate-hosting nodes in a **fixed-width rank-ordered pad of six**, and `krank_node_order` raises rather than truncate; every `gnn` arm at 24 and 80 servers FAILED on this guard, correctly. (2) S0.d: candidates per task scale with reachable servers, so at 24 servers the median is **14.18** (2.83× the corpus max of 5) and at 80 it is **47.92** (9.58×), out-of-support everywhere. **Scaling this axis requires a retrained representation**: widening the pad changes `PARTIAL_STATE_FEATURE_DIM` and invalidates every cached graph and checkpoint. A separate axis — thinning reachability to hold candidate count while adding capacity — is untested and pushes toward the starved-client spin (5 of 20 draws hang already at full reachability). **The mechanism itself is confirmed**: peer-group collection falls 6.10 → 2.20 → 0.73 s as arrivals speed 0.46 → 1.84 → 6.14 /s and is worth ~5.37 s of net latency on the best cell, so the retrain is priced.

## The 6-server client-rung headline as quoted (2026-09-20)

**Direction:** "a bipartite message-passing arm beats reactive Knative by −20.8 % (15/16) at an
unsaturated 6-server / 40-client rung" (`bipartite_edge_v1` → `best_arm_v1`, and `peeronly`'s
−9.3 % at 80 clients).

**What closed it** (`unsaturated_edge_v1`, 1,338 live arms on 16 environments per rung): the
number was an unpaired ratio of medians over four cells, two of them saturated (queue share 0.87,
0.80). Paired per cell the same data reads +4.6 / +75.8 / +9.7 / −36.8 % (loses three of four);
on 16 admissible environments every learned arm loses to Knative — `gnnedge0` +6.68 % (0/14,
disclosed), `peeronly` +8.36 % (0/16), `mpoff` +11.53 % (0/16) at 40 clients; `gnnedge0` +14.40 %
(0/16) at 80 clients. The decisions are good (37 % less time in the system after placement, and
−6.82 % vs the pointwise twin on 14/14) and the 7.1 s peer-group wait is the whole deficit.

**Do not restart** a "learned arm beats Knative at 6 servers" claim with these checkpoints.
The window is not the lever either (below); the arm's genuine gain is −1.05 s of peer exchange
per task and its shorter queue and rendezvous are relocated scheduler wait.

## The peer-group batch window as a lever (2026-09-20)

**Direction:** "the learned arms lose only because they wait ~7 s to assemble a peer group;
shorten the window and the placement quality shows."

**What closed it** (`batch_window_edge_v1`, 189 live arms, 2 / 4 / 8 / 16 s on the screen
topology): the wait falls 7.15 → 1.66 s and the median vs Knative stays +7.70 → +8.77 %, with
the platform queue (4.8 → 8.2 s) and rendezvous (1.25 → 3.3 s) absorbing exactly what the
scheduler released. Batching relocates waiting; it does not remove it. Removing it outright is
closed too (`drainable_serving_config_v1`, −1731 %). The untried lever is a decoder that places
each arrival immediately conditioned on the partners already placed.

## "A graph model learns co-location worth quoting" at the 6-server rungs (2026-09-20)

**Direction:** "the graph arm's genuine gain is co-location (−1.05 s of exchange per task); a
decoder that keeps that gain without the group wait would beat Knative."

**What closed it** (`peer_greedy_live_v1`, 95 live arms on the 16 environments per rung): a
two-line rule with the same information — queue drain in seconds plus the exchange the physics
charges to partners already placed — beats reactive Knative **−12.96 % (C40) / −16.01 % (C80),
16/16 each**, with no wait and no constant; served in `gnnedge0`'s own seat (16 s peer-group
batching, greedy in task-id order) it beats `gnnedge0` **−13.18 % (16/16)** at C80 and −8.26 %
(14/14, disclosed) at C40. The exchange-off twin ties Knative, so the margin is co-location, and
the queue gets *shorter* under it (8.66 → 7.80 s), so concentration is not the cost.

**Do not restart** a model-class claim at these rungs against Knative or random: the bar is the
rule (`peer_greedy_network`), served in the same configuration. A learned arm that does not beat
it has learned less than a greedy on the physics it was trained to approximate. The no-wait
decoder (the "untried lever" above) is now worth building only against that bar, and G5 says
removing the wait is worth ~8 % on top of whatever it learns.

## Unarrived-partner lookahead as a message-passing lever (2026-09-24)

**Direction:** "a GNN that also sees partners not yet arrived can beat its pointwise twin — the
served graph never contains an out-of-batch partner, so friend-of-friend lookahead is the one
graph-specific mechanism never tested here."

**What closed it** (`lookahead_mp_v1`, live, the 16 + 16 `unsaturated_edge_v1` environments,
per-arrival seat): the headroom is real — an oracle knowing each unarrived partner's realised node
beats `peer_greedy_network` −6.56 % / −8.18 % (15/16 each) — but a hand rule with no learning,
`peer_greedy_selfpredict_network` (the partner's node = the rule's own argmin for it now), beats the
rule −6.19 % / −8.53 % and recovers 95 % / 93 % of the oracle's gain. The gain is coordination the
rule can compute for itself, not prediction a graph must learn.

**Do not restart** a lookahead-for-message-passing lineage (out-of-batch partner nodes in the
served graph, horizon-aware peer labels) at this physics without a mechanism the self-predict rule
cannot express. The bar for any learned placement at these rungs is now
`peer_greedy_selfpredict_network`, not `peer_greedy_network`.
