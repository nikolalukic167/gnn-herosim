# LINEAGES.md — the index of what is known

**Read this before starting work.** It is the one entry point to this repo's research
record: every lineage, every gate-tool correction, every transferable rule, and every
closed question. `simulation_data/REGISTRY.json` does the same for datasets.

This file is an **index only**. It carries a status and a one-line outcome per lineage;
the full record lives in the node file each row links to. Nothing here is a summary you
may cite — cite the node.

## The graph

| | |
|---|---|
| [`docs/lineages/`](docs/lineages/) | **One node per lineage.** Standing, entry points, datasets, and the full dated record. Indexed below. |
| [`docs/lessons.md`](docs/lessons.md) | **The transferable rules** — 350 of them, newest first. What generalises past any one lineage. |
| [`docs/hard-stops.md`](docs/hard-stops.md) | **Falsified — do not revive without new evidence.** Each entry names the measurement that closed it. |
| [`docs/gates/gate-tools.md`](docs/gates/gate-tools.md) | **Corrections to the gates themselves.** Deliberately not filed under a lineage. |
| [`docs/notes/`](docs/notes/) | Design notes on physics and features that outlive any lineage (warmth model, storage contention, separability, corpus regen). |
| [`docs/adr/`](docs/adr/) | Decision records for choices with two live answers (warmth physics, queue contracts, the mandatory placement sweep). |
| [`CONTEXT.md`](CONTEXT.md) · [`PARITY.md`](PARITY.md) · [`CO_SIMULATION_GUIDE.md`](CO_SIMULATION_GUIDE.md) | Vocabulary · cross-venue comparability · the co-sim pipeline. |

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
| [**objective_pivot_v1**](docs/lineages/objective_pivot_v1.md) | `ACTIVE` | **Current work. Program pivot registered 2026-08-28: stop engineering the environment, change the training objective.** **Phase 1 CLOSED 2026-08-29 — PASS.** The 16-seeded-draw reliability gate ran at the pinned commit `c08aa7e` (all 8 new arms provenance-verified, 240/240 cells): GNN severe-collapse burden is stochastically smaller than the frozen MLP's — primary rank-sum **p = 0.00143** at +50%, must-hold **p = 0.00045** at +100% (GNN clean in 16/16 draws vs MLP up to 23 collapsed cells). Secondary also passes: mean margin vs Knative negative in **16/16** draws (p = 1.5e-05). **Registered scope limit: severe collapse only** — at +30% two GNN draws still carry real burden, so the claim is "tight distribution with a tail vs a lottery", never "never collapses" (`gnn_draw_study_v1` falsified that). Operational: the gate lost 191/240 tasks twice to an exhausted per-account `/home` quota (0-byte SLURM logs, `df` misleading); freed by deleting the PARKED route_b dataset dirs (588 datasets, 55G, none registered). **Claim REWORDED 2026-08-29 after `mp_ablation_v1`:** the burden result stands exactly as measured, but the attribution to *graph-awareness* does not — disabling message passing leaves the edge intact and directionally improves it, so the credit belongs to the scoring/decode architecture (encoder + masked-softmax EdgeScorer), not graph reasoning. Any write-up must report the control alongside. **Phase 2 CLOSED 2026-09-01 — P3 fired on the letter, and the signal is CHAOS.** The registered pilot ran clean (300 snapshots × 256 plans at h=10 s, 76,800 mini co-sims, zero failures) and co-primary (b) fired: 21.7% of snapshots above 2% regret (bar 15%; binomial p = 1.5e-33 vs the 3.3% t=0 base rate), with **both** repair controls at ~0 (node-count 0.009, link 0.016) — the first mechanism in the program to escape both, and additive R² collapsed 0.988 → 0.049. A post-hoc chaos control then re-ranked the same plans at h=2 and h=5 on the high-signal cells: median Spearman ρ = −0.027 / +0.004 and the h=10 optimum sits at rank 120/256 at h=5 against 128/256 for chance — i.e. **the horizon return is deterministic chaos, not placement quality**, and the low R² is unexplainable variance rather than joint structure. Horizon labels are dead as supervised/DAgger targets (hard-stop filed; transferable rule in `docs/lessons.md`). Phase 3 = P1 closed-loop training goes straight to policy gradient — unaffected, since it averages *expected* return over episodes, and the measured chaos (~1.5% mean, 3.2% p90 of total RTT) is now the registered per-episode variance its paired-seed CRN baseline must cancel  and its episode budget must be sized from. **Phase 3 Increment 1 (2026-09-01): GO** — 48 episodes, 3 backbone cells, all three registered temperatures clear both bars (exploration cost +0.05%/+1.01%/+6.24%; within-cell paired sd 0.0029/0.0039/0.0053, an order of magnitude under the 5% bar). T = 0.1 buys **17.3% exploration for +0.05% RTT**, so the cheap temperature is not a degenerate argmax clone; on `cell02_p35` sampling **beats** argmax by 0.75% on all 5 seeds, direct evidence the frozen policy is not at a local optimum. The registered sizing rule returned n >= 1 and is named as a **registration defect** awaiting a signed amendment. **Increment 2 (2026-09-02): the closed-loop trainer is BUILT, no arm trained yet** — REINFORCE with a self-critical baseline, two-pass `N/k` replay (unbiasedness measured, not asserted), CL-GNN and CL-MLP sharing one loop through the inherited scheduler. Pass 2 reproduces pass 1 to **1.1e-16** on the real checkpoint and gradient reaches 40/40 parameters; episodes are bit-reproducible from their seed and unchanged by concurrency. Two defects fixed in build that would have produced a plausible curve rather than an error (advantage clip specified in the wrong units; a sidecar requirement that would have blocked both MLP arms, since MLP checkpoints carry their contract inside the `.pt`). **AMENDMENT D signed 2026-09-02** fixes the budget: (D1) the old rule sized on *evaluation* noise (frozen policy, different sampling seed) when the replication unit for a training claim is the **training run** — that substitution is what produced n >= 1; (D2) **floor of 16 paired training seeds per trained arm**, matching this program's own Phase 1 / mp_ablation / link_mp precedent and costing ~190 CPU-h against the ~500 CPU-h anchor; (D3) a pre-registered tuning budget on cells disjoint from the gate (train cell01/02/04 and dev cell03/05 of `bbrob_bb_core4_bw0p5`, gate = all 5 cells of `bbrob_bb_core8_bw1p5`), because an unforgiving kill criterion pointed at an untuned learning rate converts a tuning failure into a program-closing false negative. The kill criterion, the five arms, the 3% MDE and the paired primary statistic are deliberately untouched. **PHASE 3 GATE RAN 2026-09-02 (job 734064) — NOT ESTABLISHED, and the kill criterion does NOT fire.** Held-out `bb_core8_bw1p5` (unseen fabric, unseen cells), 35 arms x 5 cells, all argmax. Primary CL-GNN minus Frozen-GNN over 16 training seeds: mean **+0.82%**, median **+0.27%**, **8/16** better, exact Wilcoxon **p = 0.372**, per-seed spread -11.5% to +10.8%. Median > 0 so P1 does not freeze; nothing is claimed either — the registration's binary did not anticipate 'positive but not significant', and this is that. Secondary CL-MLP minus Frozen-MLP: mean -0.053%, 1/16 better, p ~ 1.0. **The well-powered finding is the asymmetry, not the latency:** identical loop, objective, budget, grid and seeds move the two model classes by sds differing **180x** (5.84 pp GNN vs 0.033 pp MLP) — closed-loop training *can move* the graph model and *cannot move* the pointwise one, which is a trainability claim and explicitly not 'is improved' (the GNN moves the wrong way in 8/16). Absolute standings on the unseen fabric: Frozen-GNN **-44.8% vs Knative**, Frozen-MLP -29.2%. Achieved power, with the across-run sd finally measured: **n >= 119** needed for the registered 3% MDE, ~7.4x the D2 floor and over the ~500 CPU-h anchor. Two instrument defects found by the run and fixed/named: `analyze_gate.py` printed the program-closing kill language for a *secondary* arm (now requires `--primary`), and **Amendment D3's tuning used one seed per config**, so lr=1e-4 was selected on noise — a defect in the amendment itself. The last open path is neither closed nor confirmed at n=16. **AMENDMENT E signed 2026-09-02 (120 FRESH seeds, the original 16 excluded from the primary because the decision to extend followed a look at them; hyperparameters NOT re-tuned; the kill criterion given the 'positive but not significant' branch it lacked). FINAL 2026-09-03: MEASURED-NEGATIVE — P1 FREEZES.** Gate job 734369, held-out `bb_core8_bw1p5`, n=120: paired mean **-0.849%**, median **-0.846%**, **53/120** better, one-sided Wilcoxon **p = 0.928** (seeded 200k sign-flip). **Adequately powered** — at the observed sd 4.89 pp the registered 3% MDE needs n >= 84 and we ran 120, so a 3% gain would have been visible and the point estimate is negative. E1 vindicated by what it excluded: the pilot's 16 seeds read +0.27%, the 120 fresh read -0.85%. **E5 secondary stands: the trainability asymmetry is 150x** (CL-GNN sd 4.89 pp vs CL-MLP 0.0325 pp) — the same loop moves the graph model and cannot move the pointwise one, which is trainability and explicitly NOT latency. Serving-path reproducibility exact across both gates (all three deterministic arms delta = 0.000000000). Frozen-GNN remains **-44.8% vs Knative** on the unseen fabric. Third instrument defect found and fixed: E1 registered n=120 against an analyzer that refused above n=22 (2^120 enumeration), so the gate FAILED at analysis with all 615 episodes intact; the null is now sampled rather than enumerated, verified against scipy, written before any per-seed number was seen. **All three CLAUDE.md routes are now answered.** |
| [**reliability_matched_v1**](docs/lineages/reliability_matched_v1.md) | `CLOSED` | **FAIL 2026-09-04 (job 735692) — the severe-collapse reliability edge is NOT established once both model classes train on the same corpus.** Registered 2026-09-03 before any gate episode; statistics inherited verbatim from the signed Phase 1 registration. Venue control CLEAN (4 frozen GNN arms re-gated at HEAD, 0 collapses, matching the frozen table). Primary +50%: matched-MLP counts `2,0,0,0,0,0,0,0,0,0,1,0,0,0,11,0` vs GNN all-zero, rank-sum **p = 0.1127** against α = 0.05; must-hold +100% p = 0.2414. The +30% row (p = 0.0222) is DESCRIPTIVE ONLY — Phase 1 pre-committed it out of the verdict and reading it now would be threshold-shopping. **Two facts to quote together:** corpus matching removes **87% of the MLP's collapse burden** (107 collapsed cells across Phase 1's fabric-blind group → **14** here; 9/16 unclean draws → 3/16), and what remains is a real tail this n cannot resolve — post-hoc power P(clear α) ≈ **0.12 at n=16**, 0.73 at n=32, 0.97 at n=48. **Registration defect, named:** n=16 was inherited along with the statistic, but it was calibrated against a 107-cell burden and is not powered against a 14-cell one. A powered re-run is a SEPARATE registration and must exclude these 16 draws from its primary (the decision to extend follows a look at them). |
| [**mp_ablation_v1**](docs/lineages/mp_ablation_v1.md) | `CLOSED` | **NO_DIFFERENCE_DETECTED (2026-08-29) — and every line points the other way.** Paired 16-seed control on `objective_pivot_v1`'s claim, registered before the run. Primary paired exact Wilcoxon on mean margin vs Knative: **p = 0.05066** against α = 0.05 — the bar missed by 0.00066, so the registered verdict is a null and **stands**. Direction is unambiguous and opposite to the hypothesis: mean paired difference **−5.63 pp with MP-OFF better** (13/16 seeds), MP-OFF better on severe collapse in 2 pairs and worse in 0, +30% collapse `[0,8,0,0,10,2…]` → `[0,0,0,0,3,0…]`, and on the backbone cells where the latency edge lives MP-OFF is **−34.4% vs −25.1% (Δ −9.3 pp)**. Message passing over the CURRENT graph (bipartite + same-node) does not carry the edge and looks harmful. **Consequence, as registered: the Phase 1 claim is reworded away from "graph-aware"** — with the GIN skipped the model is an encoder plus a masked-softmax `EdgeScorer`, so the credit belongs to the scoring/decode architecture, not graph reasoning. Cannot distinguish "MP is unhelpful" from "MP is over the WRONG graph" (shared-link contention is unrepresentable in the current edges); building the link-aware graph and re-running this ablation is the decisive next experiment. Scorer bug found and fixed mid-scoring (inverted co-primary direction; verdict unaffected, regression test added). |
| [**link_mp_v1**](docs/lineages/link_mp_v1.md) | `CLOSED` | **NO_DIFFERENCE_DETECTED on the primary (2026-08-31) — and the secondaries resolve the mechanism.** The registered rematch of `mp_ablation_v1` with the link-aware graph (`core_v1`) on a binding-backbone corpus (1,675 datasets, frozen by Amendment 2), 3×16 seeded arms, 960 gate cells, zero task failures, fully unattended through the auto-score stage. Primary lgon-vs-lgmpoff: +0.47 pp, p = 0.372 — link-graph MP does not beat no-MP. But S1: lgon beats old-graph MP by **+4.98 pp (p = .0046)**, and context: no-MP beats old-graph MP by **+4.50 pp (p = .0011)** — so **old-graph message passing is measurably harmful, the link graph repairs exactly that harm, and repaired MP ties the pointwise ceiling** — the outcome `program_verdict_v1` predicts for a pointwise-separable target. The supervised MP question is closed on both graphs; MP moves to the closed-loop phase, over `core_v1` if retained. Buried lede: the binding-backbone **corpus** is the largest lever ever measured — every family lands −33…−38% vs Knative (deployed fabric-blind model: −25.1%) with **zero collapses across all 48 arms × 20 cells at every threshold**, the first all-clean reliability table in the repo. Promotion (2026-08-31): `models/gnn-linkmp-lgon-s8.pt` (median lgon seed, −38.0% vs Kn, self-describing `core_v1` sidecar) is the new reference GNN checkpoint — rule and caveats in the node. **Exploration pilot 2026-09-03 (4 seeds, unregistered): the corpus-matched MLP this lineage never trained ties the promoted GNN on both bbrob fabrics** (core8: MLP seeds 4.42–5.02M vs GNN 4.95M; core4: −44…−52% vs GNN −49%; fabric-blind MLP −29%/−26%), zero collapse cells in 40 — the standing GNN-vs-MLP latency gap on those cells was the corpus. Record in the node. |
| [**route_b_env_pivot_v1**](docs/lineages/route_b_env_pivot_v1.md) | `PARKED` | **PARKED 2026-08-28** (user decision; closing entry in the node). Ladder at parking: **H0 VOID-TIE-INDETERMINATE / H1 FAIL / H2 VOID / H3 VOID — no PIVOT-CANDIDATE.** Both overlap rungs' paired separable controls are MEASURED non-additive, so S0 as registered cannot license a read there — "**the screen could not measure it**", NOT "no exploitable joint structure" (Arm S's bars were never read on H2/H3; the same grid shape passed all four bars in probe). Parked because `program_verdict_v1` closes the supervised objective for any environment and `route_b_v1` stage 2 measured a GNN losing to pointwise-plus-prefix at memorization — even a PIVOT-CANDIDATE would feed a closed objective. AMENDMENT 4 stays drafted, unsigned. The pre-2026-08-28 running narrative this row used to carry lives in the node, which is its one home. |
| [**route_b_v1**](docs/lineages/route_b_v1.md) | `PARKED` | **Stage 1 PASS** — contention + coupling produces the non-pointwise structure five mechanisms and route A could not. **Stage 2 NO-GO-PREPROBE** (2026-08-26): a GNN cannot beat pointwise-plus-prefix on this environment even at memorization. **⚠ Probe 2026-09-03: that abort's premise fails a convergence check** — A1 ran 40 epochs; at 300 its train regret goes 28.45% → 2.0–2.9%, 4–5× better than the competitor and below the greedy floor (1 seed, far outside the draw spread). Held-out ordering still favours pointwise (MP-OFF 11.1% < MLP+prefix 15.4% < GNN 18.6–26.2%), and MP is load-bearing on this DAG graph (4× fit-capacity gap). Record in the node. Forked to `route_b_env_pivot_v1` — **that fork was PARKED 2026-08-28**, so this lineage parks with it; both measured results stand. |
| [**route_c_link_transfer_v1**](docs/lineages/route_c_link_transfer_v1.md) | `REGISTERED` | Screen registered 2026-08-26 before generation; **name is reserved and is only claimed if the screen passes.** Asks whether an environment where link waiting is a material share of RTT resists a fairly-armed pointwise competitor. **Superseded in priority by `objective_pivot_v1` (2026-08-28)** — registration stands, ungenerated, not scheduled. |
| [**siv1_full_corpus**](docs/lineages/siv1_full_corpus.md) | `ACTIVE` | First real live-gate FAILED, then **SUPERSEDED the same day** — it measured an uncommitted code diff, not the model. The synced-code re-gate (job 709163) wins **5/5 on `workload-150-100` and `workload-175-100`**, 2W/1T/2L on `workload-125-225`. Corrected-cache retrain (job 709234) is ungated. |
| [**trainer_determinism_v1**](docs/lineages/trainer_determinism_v1.md) | `ACTIVE` | The seed fix reached **1 of 4 trainers**. Three defect classes now — newest: `prepare_graphs_cache` seeded 42 at module import and clobbered every GNN draw's seed. `tests/test_trainer_determinism.py` covers every trainer; run it before training anything you intend to gate. |

## Closed — answered, do not re-run

| Lineage | Status | Outcome |
|---|---|---|
| [**route_a_v1**](docs/lineages/route_a_v1.md) | `FALSIFIED` | **NO-GO.** DAG + distance is genuinely pairwise and still pointwise-optimal. Breaking separability is **necessary but not sufficient** — you also need contention, which is what route B tests. |
| [**program_verdict_v1**](docs/lineages/program_verdict_v1.md) | `CLOSED` | Terminal answer to the D3 fork: **the supervised co-sim path to "GNN > MLP on latency" is closed by measurement.** The reliability/regime win exists on the 30-cell record but is exploratory. P1 (closed-loop objective) is the only remaining path to the latency claim. |
| [**cosim_deepdive_v1**](docs/lineages/cosim_deepdive_v1.md) | `CLOSED` | Does the target's additivity come from the synthetic t=0 snapshot regime? **No — live-visited states are equally additive** (4,400 swept live states, median additive R² 0.99999). The GNN's dispersal edge is a closed-loop property no single-batch regret target can express. |
| [**graph_structure_physics**](docs/lineages/graph_structure_physics.md) | `CLOSED` | **The co-sim target is pointwise-separable, so a pointwise MLP is the correctly specified model class and the GNN cannot beat it by training.** Additive R² 0.988 → 1.00000 across collections. Deep queues as a coupling lever: FALSIFIED — the lever runs backwards. The finding is closed; its **M4 diagnostic stays live** and is the standing separability gate every later lineage is measured with. |
| [**throughline**](docs/lineages/throughline.md) | `SYNTHESIS` | Cross-lineage synthesis: four mechanisms, one collapse. **In this simulator, coupling is either count-shaped or negligible** — and the negligible half is demonstrated, not assumed. |
| [**p5b_draw_study**](docs/lineages/p5b_draw_study.md) | `CLOSED` | **Q1 = LOTTERY, Q2 = DRAW-DOMINATED.** The MLP trainer never seeded torch, so every MLP checkpoint before 2026-08-24 is an unreproducible draw. Collapse counts swing 0→26 on the seed alone. **Retires "the MLP collapses 7/30"** and every inference built on it. |
| [**p5b_candidate_relative**](docs/lineages/p5b_candidate_relative.md) | `CLOSED` | **INDETERMINATE — and the indeterminacy is the result.** Kills the mechanism sentence "a pointwise scorer collapses because it cannot condition on its peers": one arm has exactly that conditioning, uses it, and stops collapsing. Resolved by `p5b_draw_study` — it was the training draw. |
| [**gnn_draw_study_v1**](docs/lineages/gnn_draw_study_v1.md) | `CLOSED` | **INDETERMINATE**, and **"the GNN never collapses" is FALSIFIED** — 2 of 8 seeded draws collapse. Direction is clear, the test is under-powered. |
| [**m3_batch_makespan_v1**](docs/lineages/m3_batch_makespan_v1.md) | `CLOSED` | Below both registered thresholds, then **closed permanently** by the argmax-flip diagnostic: when per-branch costs are separable and component choices are free, min-max and min-sum share the same argmin, so re-scoring under makespan was never an independent escape at any fan-out width. |
| [**cache_live_divergence_audit**](docs/lineages/cache_live_divergence_audit.md) | `CLOSED` | Platform reordering: **18/18 collections, BENIGN** (no recache, no asterisk). Dims 9-11 temporal estimate: **8/18, REAL**. The parity verifier now compares by platform identity instead of position. |
| [**serving_speed_v1**](docs/lineages/serving_speed_v1.md) | `CLOSED` | The episode cost was `Data.to()`, not the device — 174× per-call win from moving tensors only. CPU serving is slower than cuda; the cpu default exists for parity. |

## Falsified / failed — the mechanisms that did not work

| Lineage | Status | Outcome |
|---|---|---|
| [**topology_transfer_v1**](docs/lineages/topology_transfer_v1.md) | `FAILED` | Changed the win condition from per-plan accuracy to inductive generalization across topology sizes — and **all four arms FAILED**, including `gnn_topo`, the only arm that ever had backbone topology in the graph. ⚠ Never live-gated; checkpoints were not persisted until the 2026-08-21 unblock. Retired 2026-09-01: the pending ~14 GPU-h partial gate is cancelled — link_mp_v1 measured that this corpus's non-binding 1000 MB/s backbone makes link features label-irrelevant, so the gate could not have answered its question. |
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
| **queue_feature_contract** | `src/placement/queue_features.py`, `scripts_cosim/test_queue_features.py`, `verify_cache_live_feature_parity.py` | all | `legacy_v0` vs `scale_invariant_v1`. See CLAUDE.md. |
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

**Do not import from `archive/`.** The live tree is verified closed against it. Re-run
the gate after any move:

```bash
# no live file may reference an archive-only filename
python3 - <<'PY'
import subprocess, pathlib, re
all_f = subprocess.check_output(["git","ls-files"], text=True).split()
live  = [f for f in all_f if not f.startswith("archive/")]
arch  = {pathlib.Path(f).name for f in all_f if f.startswith("archive/")
         and f.endswith((".py",".sh",".sbatch"))}
arch -= {pathlib.Path(f).name for f in live}
bad = []
for f in live:
    if not f.endswith((".py",".sh",".sbatch")): continue
    for ln, line in enumerate(pathlib.Path(f).read_text(errors="ignore").split("\n"), 1):
        if "archive/" in line: continue
        for n in arch:
            if re.search(r'(?<![\w.-])'+re.escape(n)+r'(?![\w])', line):
                bad.append(f"{f}:{ln} -> {n}")
print("\n".join(bad) or "CLEAN"); print("broken:", len(bad))
PY
```

Known-benign hits: six comments in `src/placement/{orchestrator,simulation}.py` that
name `executeinitial.py` while explaining why a guard exists. They are prose about a
code path that no longer exists, not references to it.

**Session handovers are ephemeral and are not committed.** A handover is one session's
note to the next; it is not a record. Anything in it that is still true a week later
belongs in a lineage node, `docs/lessons.md`, or `docs/gates/gate-tools.md` — put it
there instead. Three committed `HANDOVER*.md` files were retired on 2026-08-27 for
drifting out of agreement with this index while claiming the same facts. If you need to
hand off, write the file outside the repo (the session scratchpad) or leave it untracked.

**One fact, one home.** Before adding a paragraph, find the file that already owns that
fact and edit it. This index existed at 4,995 lines because five files each narrated the
same experiments and drifted apart; the duplication cost more than the writing saved.
