# Hard stops — falsified, do not revive without new evidence

> **Status:** `REFERENCE` &nbsp;·&nbsp; **Index:** [LINEAGES.md](../LINEAGES.md)
>
> Things measured and closed. Each entry names the measurement that closed it, so reviving
> one is a decision made against evidence rather than by forgetting. Moved verbatim from
> `memory/memory.md` §2 (that file retired 2026-08-27; see git history); the `FALSIFIED` rows in
> [LINEAGES.md](../LINEAGES.md) are the lineage-level counterpart.

**tune link bandwidth or core count to rescue `link_contention_v1`** (null lever + full spectrum swept: n_core 2/4/12 all ≤0.35% mean regret) · **cite offline regret/acc as evidence a model will place better live** (Arm A: −54% offline regret → +3.5%/+9.9% *worse* live, `logit_tied_rate` rose) · **cite the 0.88× Kn `sparse_p35`/s42 figure** (irreproducible: current code gives **1.04×** on the same cell/seed/model; the arm that produced it set `GNN_DROP_NODE_EDGES=1`, implemented **nowhere** in the tree today) · **revive same-node edges without new physics** (Arm B falsified *while trained with them*) · RQ3 · **RQ3b** · **claim the GNN's live failure is "structural in the joint decode"** (falsified: LQB λ=1.5 no change 275.7M; `argmax_uniq` drove collisions to **0.000** and got *worse* 301.3M) · **cite ECT as a ceiling or distill teacher for Regime A** (0.98–1.13× Kn, and `contention_v2` is pull-free so `ect_pull`≡`ect`) · **run `ect_pull` with `ECT_PULL_DISTILL_DIR` on workload-125-225** (561,848 frames/cell ≈ 50GB; fills disk) · cite pre-`25732cf` Regime A tables as current · mix pre/post-serialization in one table · re-eval 873/v5.5 as if it transfers to serialized FilterStore · hub9 decode · claim pull-obs/cosim-retrain/`soft_combo_conc`/hard-CE-distill closed 125→31 · blind retrain scarce-warm 450 · reopen hub9/`total_rtt` primary · HeteroData/Set-Transformer · claim warm/busy v1 init helps FilterStore distill · **relax α to get a readable route_b pivot rung** (it is a cliff, not a dial: fine sweep on H1 gives α=3.9 binding 204/204 with 80 stuck, α=4.0 stuck 0 and binding **0/204**, byte-identical to the unconstrained anchor — clean counters and "the constraint binds" are mutually exclusive on this grid) · **raise `replica_server_percentage` to break single-node confinement** (two full 204-dataset probes: confined-task histogram is an identical `{0:102, 2:102}` at 2, 3 and 4 hosting nodes, and stuck drifts slightly *up*; the knob changes how many nodes are eligible, not how the FCFS allocator at `generate_infrastructure.py:625-660` spreads across them) · **read `greedy_stuck` as a fact about the route_b pivot environment** (it is decoder myopia on **every** rung: over identical candidate sets, caps and ordering, backtracking rescues **458/458** across H0/H1/H2, and every stranded dataset had a feasible plan in its own enumerated sweep) · **pursue a supervised message-passing win on ANY graph** (mp_ablation_v1 + link_mp_v1, 16 paired seeds each on the binding-backbone corpus: on the old task↔platform graph MP is *harmful* — no-MP beats it +4.50pp, exact Wilcoxon p=0.00107; the `core_v1` link graph repairs exactly that harm — link-MP beats old-graph +4.98pp, p=0.00459; and repaired MP then **ties** no-MP, +0.47pp, p=0.372 — the pointwise ceiling `program_verdict_v1` predicts for any supervised co-sim target. The remaining venue for an MP win is closed-loop training against the live simulator, not a better supervised graph) · **label a placement by the multi-second horizon return of the live simulator** (objective_pivot_v1 Phase 2 / P3, 300 snapshots × 256 plans at h=10 s: the registered co-primary *fired* — 21.7% of snapshots above 2% regret, both repair controls ~0 — and the chaos control then showed the ranking is random. Across 60 high-signal snapshots, median Spearman ρ(h2,h10) = −0.027 and ρ(h5,h10) = +0.004, and the h=10 optimum sits at rank 120/256 at h=5 against 128/256 for chance. The return is deterministic chaos, not placement quality; the collapsed additive R² of 0.049 is unexplainable variance, not joint structure. Closed-loop policy gradient — which averages expected return over episodes — is unaffected and is the remaining venue) · **run the residency-hold scaling pilot (hold the node slot across cold start + exec)** (objective_pivot_v1 Phase 2b: its premise is false for co-sim — cold start is **0.000 s in 320 of 328 sampled datasets across all 16 collections**, the 8 exceptions totalling ≤0.206 s, including in `regime_b_cold_burst_v1`. `system_state.replicas` is keyed by task type and warmup warms each replica with its own type, so `sandbox_is_warm` is always true and cold start is never incurred; the cited 38 s is `cnn`/`xavierDla`'s table entry, never an event. Residency therefore equals exec, so the hold is the exec hold that already measured `nodeContentionTime ≡ 0.0`. Firing it needs overlapping replica pools — a corpus physics lever, explicitly stopped) · **train the served policy by closed-loop policy gradient against the live simulator** (objective_pivot_v1 Phase 3, Amendments A-E: 120 fresh training seeds, REINFORCE with a self-critical baseline and a two-pass N/k replay whose log-probs reproduce the sampler to 1.1e-16, gated on the held-out `bb_core8_bw1p5` fabric. Paired mean **-0.849%**, median -0.846%, **53/120** seeds better than the frozen init, one-sided Wilcoxon **p = 0.928**. **Adequately powered**: at the observed across-run sd of 4.89 pp the registered 3% MDE needs n >= 84 and we ran 120, so a 3% gain would have been visible and the point estimate is negative. The 16-seed pilot's +0.27% median was a draw and was correctly excluded from the primary. What the loop CAN do is move the graph model at all — across-run sd 4.89 pp against the MLP's 0.0325 pp, a 150x trainability asymmetry — but moving is not improving, and at n=120 it moves the wrong way as often as the right one. Reviving this needs a *different* configuration with its own powered tuning stage, registered before its data exists, not another n on lr=1e-4) · **quote a GNN-vs-MLP latency or reliability number from arms trained on different corpora as a model-class result** (link_mp_v1 pilot 2026-09-03: an MLP trained on the GNN's own `graphs_cache_link_mp_v1_core_v1_dim14` ties it on both backbone fabrics, 4.42–5.02M vs GNN 4.95M on `bb_core8_bw1p5`, so "−44.8% vs −29.2%" was two corpora, not two model classes; reliability_matched_v1 FAIL 2026-09-04: matching the corpus removes 87% of the MLP's collapse burden (107 → 14 cells) and the residual is not established at the registered n=16, rank-sum p = 0.113 at +50%. Any GNN-vs-MLP number must name both arms' training cache; a powered reliability re-run is a separate registration that excludes those 16 draws) · **grow the route_b DAG corpus to rescue message passing** (route_b_v1 Phase 2, 2026-09-07: learning curve 204/612/1020 training DAGs against one fixed 204-parent held-out block, 3 arms × 8 seeds — registered read GAP-PERSISTS; at the registered last-epoch checkpoints the GNN *without* message passing generalizes best at every rung (rung 3 median-paired p=0.039, mean-paired p=0.023, 7/8 seeds) — but MP-ON overfits from epoch ~60 against a censored val metric; fixing the checkpoint-selection metric and fully retraining both arms 8 seeds each (SIGNED OFF 2026-09-08, supersedes this reading) confirms a TIE by design, not by accident — median-paired D = +0.04pp, p = 0.25 (was +0.36pp/p=0.039 at the broken selector) — and both GNN arms now beat the MLP+prefix baseline. So the stop is about corpus size, and the framing should read **"MP is redundant with the prefix columns, not harmful."** The MP-on/off gap is flat (−1.7/−2.0/−1.4 pp at the broken selector; ~0 at the honest one), and the GNN's ≈1% train regret never converts to a held-out MP-on/off gap either way. Mechanism, corrected 2026-09-07: the root task is exactly pointwise (≈(queue+1)×7.63 s of warmup writes or a 39 s node-write-contention term), while the children carry ~23% joint variance that is entirely pairwise parent→child co-location (4.78 s vs 9.54 s) — which the prefix columns every arm consumes already encode, so MP has nothing left to learn; physics tweaks that raise the cap-constrained contention statistic halve the value-level joint share and collapse the optimum onto one node) · **compare route_b offline decode regret with any live number** (2026-09-06 audit: the live GNN scheduler batches only parent-finished tasks, so a diamond4 is never decoded jointly, the live graph carries no DAG block, cnn/rf have no live candidates, the loader double-counts the one-hot width and all of it is swallowed into shortest-queue; and the offline object — 4-task joint decode under α=2.0 + replica uniqueness on a frozen snapshot — is not the live object even once serving works) · **serve the DAG output payload over the link fabric to give link contention magnitude** (dag_fabric_contention_v1 paper screen 2026-09-08, no simulation: on the stored route_b `arm_s` sweep, store-and-forward waits on the 800 MB parent→child transfers would pass the 10% manipulation bar at ≤ 100 MB/s, yet the α=2.0 optimum carries zero link wait in 98–100% of 204 datasets at every hosting-node count (0/17 even at 2 hosting nodes) — the label never sees the mechanism, exactly the "optimal plans select against contention" shape of `link_contention_v1`; at 8 tasks the joint optimum cannot avoid waiting in ~50% of datasets but the binding wait is the root's/sink's own access link, a remote-children/parents count one column repairs, while core-link sharing binds at the optimum in only 10–12%. Funnelling the backbone to force core sharing turns the term into the number of remote DAG edges — the hop+coupling block that already closes 0.997 at 8 tasks. Not a tuning miss) · **explain `peer_affinity_v1`'s offline/live reversal by herding or by queue-blindness in the graph arm** (`serving_gap_v1`, registered and closed NO-GO 2026-09-12 on its own kill bar, 192 live production runs = 3 corpora x 2 arms x 16 seeds, uncapped, S0 reproducing the registered offline contrast to +5.142/+6.532/+9.606pp against +5.14/+6.53/+9.61). **Both hypotheses were refuted in the direction OPPOSITE to the one registered.** H1: the `gnn` arm spreads its node choices MORE evenly than `mpoff` — modal repeat rate -0.083 (p=0.009) and normalised node entropy +0.062 (p=0.018) on x800p3, 0/3 corpora firing. H2: `gnn` is **3-7x MORE** responsive to queue depth than `mpoff` on every corpus (normalised score sensitivity 0.582/0.424/0.770 against 0.169/0.113/0.111, all p<1e-4). The graph arm is the most responsive arm measured in this program, so any future account of the reversal must start from over-responsiveness, not rigidity — and that account is post hoc and needs its own registration. Note what is NOT stopped: reading a decode trace to find a serving defect is the method that produced the platform cap (+21.9% vs Knative, 16/16 seeds) and stays available) · **model CPU / memory co-tenancy interference (processor-share exec dilation, memory over-commit cliff, type interference matrices) to give a supervised GNN joint structure** (closed by argument 2026-09-09, no code built — `cotenancy_interference_v1` was Track B of the 2026-09-08 plan: exec time is a table lookup (`infrastructure.py:1337`) and nothing on a node dilates, so any such physics is new; but every dilation a task suffers is a function of the multiset of tasks co-resident on its node, and every symmetric function of a multiset is a function of per-type counts (`route_a_v1`'s composition theorem), which the `hetdem`/`krank` pointwise blocks already express — the S2 kill bar closes by construction, and the only overlap grid that makes placed tasks co-run (`route_b_pivot_h2`) carries a separable control that voids S0 on the parent-locality branch (measured 2026-09-08). Reopen only with a mechanism that is NOT node-indexed and binds AT THE OPTIMUM — `dag_fabric_contention_v1` shows the one such candidate does not) · **cut a supervised corpus from the cluster the model is served on to make message passing transfer live** (`peer_affinity_warm_v1` W0, 2026-09-13: 100 brute-force-labelled datasets from live snapshots of the gate cell, both behaviour sources, support closure PASS — `xavierGpu` in 67 % of datasets, queue column max 590 — and the pointwise-recoverable regret of the warm one-step label is median 0.0 %, 0/100 above 2 %, 88/100 exactly zero. The served label is queue drain + per-pair transfers, a sum of (task, placement) terms; the state mismatch was real and closing it produces labels the pointwise class fits exactly. A different, drainable served regime is a new environment question, not a re-run. **Confirmed on the LIVE gate 2026-09-14 (W1, Amendment 1, rule 6):** 472 + 72 warm datasets, 16 seeds per arm, cell_s7901 — offline `gnn` vs `mpoff` TIE (+0.74 pp, p = 0.083); live capped the warm graph arm is 23.9 % slower than the cold one (0/16) and 10.4 % slower than its MP-OFF twin (0/16, p = 3e-05), uncapped −17.2 % (1/16); vs Knative +1.5 % / −1.9 %. Headline NO-WINNING-GNN in both configurations. Do not re-run with more warm cells or seeds: the live contrasts are at p ≤ 6e-05 on 16 pairs and the offline label is pointwise by W0.b)

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

**What is NOT stopped, and one of these is now the most promising open question in the record:**

1. **Why the early advantage is lost.** The same lineage measured, on **3/3 cells and every one
   of 91 learned arms**, that the arms beat reactive over the first fifth of the trace by
   3.4–4.1 s of queue. That is the first learned-beats-reactive reading here that survives
   replication. **Something destroys a real advantage**, and no measurement yet says what.
2. **Any non-queue stabiliser** — admission control, load-aware batch sizing, or giving the
   model a queue feature inside its trained range (the live column is ~300× out of range,
   `legacy_v0`).
3. **A tuned or adaptive K.** K = 3 was registered untuned and is a carried limitation; an
   adaptive variant that no-ops where the arms are already stable is a different experiment
   and needs its own registration.

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

## Relabelling or re-serving the burst-seat GNN to reach the CD greedy (2026-09-25)

**Direction:** "`gnnedge0` trails the CD greedy in the burst seat because of its label, its decode
order, its serving candidate set or the load it was gated at. Fix that one and it reaches CD."

**What closed it** (`cd_gap_v1`, live, the `fresh_topo_burst_v1` fresh topologies):
- **The label:** the same architecture trained on CD's own plans (4 seeds) agrees with CD 17.8 %
  offline and trails CD live +13.6 % (0/11), tying the sweep-trained model (+0.6 %, p = 0.52).
- **Decode order:** revising its plan with its own score recovers 8.9 % of CD's refine gain.
- **Load:** at arrivals ×2 slower, with every policy time constant scaled, the gap is +13.8 % (0/10).
- **Out-of-batch blindness:** worth ~3.4 % (blind-CD vs CD, direction only).

What the score cannot express is CD's load balancing: queue drain in seconds plus in-batch committed
service.

**Do not restart** a relabel, imitation, decode-order or load-ladder attempt on this architecture
without a representation change that carries per-platform committed load into the score. What did
work is using the GNN plan as a seed for CD's refine passes (−3.8 % vs CD, direction only), which is a
learned-seed result, not a message-passing one.

**That representation change has now been made** (`load_repr_v1`, `partial_state_v4`: replica backlog
and in-batch committed service in seconds). It cuts the gap from +12.4 % to +6.7 % and beats its zeroed
twin −5.3 % (11/11). A further attempt starts from `v4load`, not from `gnnedge0`.
