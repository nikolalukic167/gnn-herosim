# Learned lessons — the transferable rules

> **Status:** `REFERENCE` &nbsp;·&nbsp; **Index:** [LINEAGES.md](../LINEAGES.md)
>
> 68 rules extracted from the experiment record, newest first. A lineage node in
> [`docs/lineages/`](lineages/) tells you what one investigation found; this file tells you
> what generalises past it.
>
> **Adding a rule:** only when it would have changed a decision *before* it was learned.
> A rule that just restates a lineage outcome belongs in that lineage's node, not here.
>
> **One rule, one `##` section.** There used to be two append points — a flat bullet list and
> a run of `##` sections — both receiving entries on the same days. On 2026-09-16 every rule
> became a section and the 212 un-marked entries, which indexed checkpoints and caches rather
> than stating rules, moved to
> [`docs/lessons-archive/2026-06-artifact-inventory.md`](lessons-archive/2026-06-artifact-inventory.md).

## A corpus-matched win is not a win against the best arm you have (2026-09-17)

`peer_only_v1` B2. Registered as a corpus-matched contrast, exactly as
[the worst-matched-arm rule](#-a-model-class-comparison-is-only-as-good-as-its-worst-matched-arm--check-the-training-corpus-of-every-baseline-before-quoting-a-gap-2026-09-03)
demands: `peeronly` and `mpoff`, both trained on the 1,654-dataset corpus, same seeds, same
cells. It fired — `peeronly` beats its MP-OFF twin **−15.68 %, 16/16, p = 0.0004** at 80
servers, and beats reactive Knative **−41.4 %, 16/16**, the program's first arm to do both.
Then the unmatched comparison: the same arm against `mpoff` trained on the **516**-dataset
corpus reads **+10.40 %, 0/16, p = 0.0004**. B1 explains it — more data made *every* arm
slower live (6/6 arm × rung `CORPUS-DOES-NOT-HELP`), and `mpoff` most of all, **+35.49 %**.
The matched baseline had been degraded by the very lever the match was controlling for.

⇒ **Corpus-matching removes a confound in one direction and can manufacture a win in the
other.** A matched contrast answers "does this component help *on this corpus*"; it does not
answer "is this the best scheduler we have". Report both, always: the matched paired test as
registered, and the winner against the best-performing arm in the program at that operating
point, named with its corpus. If the second one is not in the node, the first will be quoted
as if it were. The 2026-09-03 rule and this one are the same instrument read in both
directions, and neither is sufficient alone.

## ⛔ A SEPARATION ON THREE CELLS IS A COINCIDENCE GENERATOR — AND THE COST OF CHECKING IT IS ONE AFTERNOON (2026-09-16)

`scheduler_residence_v1` R1 found the one cell of three without a queue blow-up separated from the other two **with no overlap on two statistics at once** — every client can reach ≥ 2 servers against a client that can reach exactly 1 on both bad cells, and a clients-per-server imbalance of 3 against 8 and 7 — where the *mean* fan-out separated nothing (3.45 / 3.05 / 3.50). The cells' configs are byte-identical in **149 of 150** flattened fields; only the topology seed differs, so there was no other candidate. It fired a bar registered with a NEGATIVE expectation, it had a concrete mechanism (a client with one reachable server cannot be spread; a server every client can reach is a queue sink), and it was **wrong**. On 15 independent draws from the same generator: **ρ = +0.041, p = 0.885** against a bar of \|ρ\| ≥ 0.60, wrong sign, second statistic +0.029, pointwise control +0.020. Not a weak effect — no effect. ⇒ **With three units, "strictly outside the range of the other two" is the expected outcome for roughly a third of candidate statistics by chance, and testing two of them at once makes it likelier, not less.** This is the third time in this program: `serving_stability_v1`'s stability mechanism (3 aggregates lined up; pooled-z within cells ρ = +0.088, two cells significant in *opposite* directions), a 3-aggregate over-read retracted the same week, and now this. **Before a three-cell ordering is written into a node as a mechanism, price the replication** — here it was 108 arms, ~90 minutes, and it converted a false lead into a closed question. ⚠ And mint the replication units from the *same generator* with the selection made on the **independent** variable before any of them is served: choosing units after seeing the outcome is choosing the answer.

## ⛔ A RATIO CONTROL IS DEGENERATE WHEN THE PRIMARY IS NULL — GIVE IT A FLOOR OR IT WILL REPORT "CONFOUNDED" FOR "NOTHING THERE" (2026-09-16)

`scheduler_residence_v1` R3 registered a control that is exactly right in shape: if the structure predicts the *reactive baseline's* queue at least 0.75× as strongly as it predicts the learned arm's excess, the effect belongs to the environment and the primary reads CONFOUNDED whatever its ρ. That rule exists because `drainable_objective_v1`'s C0 and `peer_affinity_v1`'s H5 both died of a control that moved with the treatment. It then fired on a primary ρ of **+0.041**: the threshold it computes is 0.75 × 0.041 = **0.031**, so any control ρ above noise trips it, and the gate printed `CONFOUNDED-ENVIRONMENT` for what is plainly a **null**. ⇒ **Any control expressed as a fraction of the primary's magnitude needs an absolute floor on that magnitude, below which it reports NOT-APPLICABLE rather than CONFOUNDED** — otherwise the one verdict that means "a real effect was obscured" gets printed for the case where there was no effect to obscure, and a reader who trusts the verdict line draws the opposite conclusion from the data. ⚠ The fix is not to move the ratio: a null and a confound are different findings and the instrument must be able to say which, so the floor is a third branch, not a retuning.

## ⛔ AN OUT-OF-RANGE INPUT CAN BE REAL, LARGE, AND ORDERED WITH THE OUTCOME, AND STILL NOT BE CAUSAL — CLAMP IT AND MEASURE (2026-09-15)

`queue_range_v1` asked why the learned arms beat reactive Knative over the first fifth of a trace and lose afterwards. The queue column they rank candidates with is `legacy_v0`'s dim7, whose trained range is 0–42 (measured over 516 datasets: the adaptive divisor is 1.0 in **every** one, so the feature is literally the raw queue depth, candidate p50 12). Live, three cells gave a textbook-perfect causal story: the column peaks at **2** on the cell that nearly ties reactive (−6.7 %), **38** on the cell that loses 111 %, and **113** on the cell that loses 183 %; deciles 1–2 are in range on **all three**, which is exactly where the arms win; and the violation is absent early and present late. Severity ordered with the loss across every cell. **Clamping the column to 42 at serve time — a change proven to bind, driving the out-of-range share from 0.107/0.164/0.197 to exactly 0.000 — moved latency by at most 0.5 s on a 74 s arm (0.7 %), cleared nothing under Holm, and was significantly WORSE on the cell with the worst violation** (−0.367 s, 3/15, p = 0.011). ⇒ **A feature that is out of its trained range is a defect worth fixing and is not thereby an explanation; the clamp is cheap, so run it before building a story on the correlation.** The trap is specific: an out-of-range input and a bad outcome share a common cause — load — so they order together on every cell without either causing the other. Here the deep queues put the column out of range *and* cost the latency, and removing the first did not touch the second. ⚠ Note what the null does **not** cover: the divisor was measured at 1.0 in **100 % of batches on every arm**, so the *compression* failure mode (an inflated divisor squeezing real differences to nothing) never occurred and remains untested rather than refuted.

## ⛔ DECOMPOSE A LIVE DEFICIT BY TERM AS WELL AS BY TRACE POSITION — THE CELL WHERE THE POLICY WORKS IS THE ONE THAT NAMES THE TAX (2026-09-15)

`serving_stability_v1` decomposed the loss by decile and found the early advantage. `queue_range_v1` decomposed the same arms by **cost term** and found something the decile view could not show: on `cell_s9001` the learned arm **wins every term it was designed to win** — queue −2.67 s, peer exchange −0.85 s, peer rendezvous −1.85 s against reactive — **and still loses the cell, by +1.43 s, entirely on batch wait (+6.87 s)**. Remove that one number and the arm beats reactive by ~5.4 s (−26 %). Batch wait is a **flat ~6.8 s on all three cells** regardless of whether placement succeeds or collapses, so it was invisible in every aggregate: on the two bad cells it is 16–24 % of a deficit dominated by a queue blow-up, and only on the cell where placement works does it stand out as the whole story. Autoscaler churn looked like the same kind of culprit (2.2–6.6× reactive everywhere) and is not — it is **+5,930 events on the best cell and +3,455 on the worst**, i.e. anti-ordered with the loss. ⇒ **Run the per-term table across a family that contains at least one cell where the policy does its job; a constant tax hides inside every aggregate and only surfaces where nothing else is wrong.** The corollary for this program: "the learned arms lose to reactive" is at least two different failures — a flat batching tax and a queue blow-up on some cells — and a single intervention was never going to close both.

## ⛔ A CORPUS CAPTURED BEFORE THE AUTOSCALER HAS BUILT THE CLUSTER DOES NOT DESCRIBE THE CLUSTER'S ACTION SPACE — AND A SERVING GAP CAN BE A PROPERTY OF THE RESOURCE MIX, NOT OF THE POLICIES (2026-09-13)

`peer_affinity_v1`'s co-sim infrastructure **contains** `xavierGpu` and `xavierDla` platforms, 9 of each per dataset; in **516 of 516** datasets neither is ever a replica, because the state is captured after a short warmup. Live, the same cell under 450,729 tasks scales up until `xavierGpu` *is* one, so **7.60 % of live candidates are of a platform type the checkpoint never ranked** — identical for every arm and seed, invisible to a parity check (which runs on co-sim states), and invisible in the topology (the rows are there in both venues). The damage is not mainly the unseen one-hot: `node_caps = α × max candidate demand on the node`, so ONE candidate with 8× the corpus's largest demand switches the capacity mask off for the whole batch — **the roomiest node goes from holding 5.2 tasks in the cache to 32.5 live and back to 3.6 when the type is removed**, and 46.4 % of live node caps exceed anything in training. Removing that one type from every arm then **erased the lineage's headline offline/live sign reversal** (x800 p3: POINTWISE-BETTER −10.67 % → TIE −0.28 %) **and simultaneously reversed its one deployable win** (`gnn` vs reactive Knative +21.9 % → −14.0 %, 0/16 seeds), because it costs the reactive baseline ~6 % and the concentrating learned arm ~55 %. ⇒ **Enumerate the candidate set the corpus actually contains, compare it to the one the live cluster offers, and vary the resource mix before attributing a serving gap to a model class.** Two numbers catch it in minutes and neither was ever computed: the set of platform types appearing as candidates in the cache, and the distribution of `node_caps` live against the cache's maximum. ⚠ And note what this does *not* license: the fix is not "serve only what you trained on", because the restriction removes real capacity — the clean experiment keeps the platform and brings its demand into the corpus's range, so the cap is not inflated while capacity is unchanged.

## ⛔ A MODULE THAT CONFIGURES LOGGING AT IMPORT TIME DISABLES EVERY LATER CONFIGURATION, AND THE BILL ARRIVES AS DISK EXHAUSTION (2026-09-13)

The incident — an import-time `basicConfig` that silently discarded the simulator's own log
configuration, ~390 MB of stderr per arm, and 128 of 306 gate arms dying with an unwritable
traceback — is recorded once, in
[`docs/gates/gate-tools.md`](gates/gate-tools.md) (2026-09-13), with the fix, the pinning
tests and the archival protocol. Three rules generalise past it:

1. **A library must never call `basicConfig` at import.** An application's own configuration
   must pass `force=True` if it cannot control what was imported first.
2. **Before launching an array, multiply arms × per-arm log bytes against quota headroom.**
   A gate that writes more than it measures will kill itself.
3. **When a long job dies with no traceback, check the quota first** —
   `getfattr -n ceph.quota.max_bytes ~` against `du -sh ~`. Both `df` and `sacct` mislead,
   and an unwritable traceback is indistinguishable from a silent kill.

⚠ And discard the partial gate. 38 of 102 arms had landed, each individually valid, and
re-running only the missing 64 would have split one gate across two commits — the
cheaper-looking option is the one that produces a number nobody can defend.

## ⛔ A MECHANISM THAT REPRODUCES THE HEADLINE EFFECT IS NOT THE CAUSE — RANK IT ACROSS THE CORPORA THE EFFECT IS GRADED ON (2026-09-13)

The `peer_affinity_v1` audit found a real serving defect (the live queue column ~300× out of the trained range and non-monotone past a crossover), and a probe that scales that column on held-out co-sim data **reproduced the offline/live sign flip**: the registered +5.14 pp contrast went to −3.46 pp (p = 0.025), with the graph arm degrading 2.5× more than its pointwise twin — and the architecture gives a clean reason, since only message passing carries platform features into task embeddings at all. Every box ticked: calibrated (k = 1 reproduced the published number to the decimal), mechanistic, sign-correct, asymmetric in the predicted direction. **It is not the cause.** The effect it had to explain is graded across three corpora, and running the probe on all three put the crossover in the **inverse** order: the only corpus whose offline sign flips is the only one that did **not** reverse live, while the two that did reverse keep a *positive* contrast at every scale tested. One corpus could not tell them apart; three could, and the discriminating run cost 40 minutes on four CPU tasks. ⇒ **When the target is an ordered or graded effect, a candidate mechanism must reproduce the ORDER, not the sign on one rung — register that prediction before running it, and run every rung.** The trap is that a single-rung reproduction is maximally convincing exactly when it is least informative: a small edge is the one most easily erased by any common degradation, so the rung that flips first is usually just the rung with the smallest effect. ⚠ And keep the two results apart when writing up: the defect measurement still stands on its own (at live queue magnitudes the registered MP edge is not significant on **any** corpus, so that reading is regime-dependent) — it just is not the explanation it was registered as.

## ⛔ A SERVING-PARITY CHECK RUN ON CO-SIM STATES CANNOT SEE AN OUT-OF-MANIFOLD INPUT — COMPARE THE SERVED FEATURE RANGE AGAINST THE TRAINED ONE, SEPARATELY (2026-09-12)

`peer_affinity_v1` built the strongest parity check this repo has: the live scheduler run on held-out co-sim datasets, asserting the served graph equals the cache graph *attribute by attribute*, the decoded plan equals the offline report's, and the engine RTT equals the sweep row — 34/34 datasets, and it is genuinely what makes the live path trustworthy as **code**. It is also structurally incapable of catching the defect that was there: the co-sim states **are** the training manifold, so every input it compares is in range by construction. Measured afterwards: `legacy_v0`'s dim-7 divisor is `min(max(1, p90 over all platforms), 100)` and that p90 is **0 in 516 of 516 training datasets** (only 3.7 % of platforms ever hold a queue), so the divisor is 1.0 everywhere in training and the column is the raw depth, 0–42. Live it is 1.0 too — until the cluster warms up, at which point the divisor jumps and the column **falls 574 → 19.6 while the real queue keeps growing, 620 → 804**. Non-monotone in the quantity it represents, ~300× out of range on the production trace, on every arm of every live gate in the lineage, and invisible to a parity check that only ever sees idle clusters. ⇒ **Ship two checks, not one: (a) the paths agree on states where the answer is known, and (b) the distribution of every served feature, live, sits inside the range the cache contains.** (b) costs one traced run and a histogram; here it was never run, and `docs/adr/0002` had already written down what it would have found. Corollary: a contract that is "kept bit-exact for old checkpoints" must be **refused for new corpora** in code, not in prose — this corpus was cached under `legacy_v0` two weeks after the ADR said new training uses `scale_invariant_v1`.

## ⛔ WHEN A POLICY IS TRAINED TO MINIMISE A COST INDEXED ON ONE AXIS, SPREADING ITS WORK ALONG THAT SAME AXIS DESTROYS THE SAVING — SPREAD ALONG A DIFFERENT ONE (2026-09-11)

`peer_affinity_v1`'s live gate tied because the learned arms halved per-task service (peer transfer 5.39 s → 2.85 s) and gave it all back to concurrency (15.3 → 5.4 tasks in service at once). Two fixes aimed at the obvious culprit — the decoder's capacity cap resets every batch, so it is blind to the queue already standing on a node — made it **worse**: seeding the cap with the standing queue cost **−181 %** of baseline total RTT and a soft per-candidate queue penalty **−30 %**, because peer exchange is charged **per node pair**, so anything that spreads a peer group across *nodes* deletes exactly the saving the checkpoint was trained to produce (`averageCommunicationsTime` went straight back to the reactive baseline's 5.4 s). The axis the cost was **not** indexed on was the platform: a node's memory was capped, a platform was not, and a platform is one FIFO queue — 12.4 % of live batches placed their entire plan on a single platform. Capping tasks-per-platform at 1 took the GNN from −1.5 % to **+21.9 % against reactive Knative, 16/16 seeds, p = 3e-05**. ⇒ **Before adding a spreading constraint, ask which index the objective is written over: spreading along that index cancels the objective, spreading along an orthogonal one is nearly free.** Corollary for this simulator: `queueTime` (= `arrived − scheduled`) is not an independent cost, it is the accumulation of other tasks' service, so cutting mean service and cutting concurrency pull the same number in opposite directions and a total-RTT read alone cannot tell them apart — compute `throughput × service` (tasks in service at once) and read it beside the total.

## ⛔ AN ABORT ON A MEMORIZATION STATISTIC IS ONLY VALID IF BOTH ARMS ARE AT THEIR FIT CEILING — VERIFY CONVERGENCE BEFORE READING IT (2026-09-03)

`route_b_v1` stage 2 aborted on "the GNN cannot out-fit the pointwise competitor on its own training split" — A1 28.45% vs A2 19.34% train decode regret. A1 had run **40 epochs** and its own node recorded convergence as unverified. Same arm, same cache, same split, same decoder, 300 epochs: **2.0–2.9%**, i.e. 4–5× *better* than the competitor and below the greedy-on-true-marginals floor, a move an order of magnitude outside the arm's recorded draw spread (σ 13.57 pp). The whole NO-GO rested on a number that was still falling. ⇒ **A memorization comparison needs a convergence check on every arm before it is read — train loss still descending is not a fit ceiling** — and the cheap insurance is to save last-epoch weights next to the val-selected ones (`NEAR_RTT_SAVE_FINAL=1`), because a val-selected checkpoint cannot answer "can this arm fit its training data at all?" once validation has plateaued. ⚠ Note what did *not* change: on held-out parents the pointwise arms still win (MP-OFF 11.1% < MLP+prefix 15.4% < GNN 18.6–26.2%), so the corrected reading is "fit ceiling favours the GNN, generalization favours pointwise" — an abort on the wrong statistic hid a more interesting result rather than a favourable one.

## ⛔ INHERITING A STATISTIC DOES NOT INHERIT ITS POWER — RE-DERIVE n FOR THE EFFECT THE NEW EXPERIMENT EXPECTS (2026-09-04)

`reliability_matched_v1` took its threshold, statistic, α and **n=16** verbatim from the signed Phase 1 registration, which was the right call for everything except the n. Phase 1's n=16 was ~fully powered against the fabric-blind MLP's **107** collapsed cells; this gate's entire premise was that matching the training corpus would shrink that burden, and it did — to **14** cells, 3/16 unclean draws instead of 9/16. Post-hoc power against the burden that actually remained: **0.12 at n=16**, 0.73 at n=32, 0.97 at n=48. The gate returned a FAIL that is a genuinely under-powered null, and the one number that could have prevented it — a plausible *matched* effect size — was derivable at registration time from the hypothesis itself. ⇒ **When reusing a signed design on a new comparison, re-derive the sample size from the effect the new comparison predicts, and say so in the registration.** Copying n across is the same defect as tuning at one seed (2026-09-03), reached from the opposite direction: there the instrument was too noisy for the grid, here it was too small for the effect. ⚠ And note what the null still bought: the *burden* comparison (107 → 14 cells) is descriptive, needs no α, and is the more publishable half of the result.

## ⛔ A MODEL-CLASS COMPARISON IS ONLY AS GOOD AS ITS WORST-MATCHED ARM — CHECK THE TRAINING CORPUS OF EVERY BASELINE BEFORE QUOTING A GAP (2026-09-03)

The standing "GNN −44.8% vs MLP −29.2% vs Knative" on the bbrob cells compared a GNN trained on the binding-backbone corpus against an MLP trained fabric-blind on `full_corpus_siv1`. `link_mp_v1` had already measured the corpus as a ~13 pp lever on the GNN and still never trained a pointwise arm on it. A 4-seed pilot did (`experiments/link_mp_v1_mlp.yaml`, ~1 CPU-hour): the corpus-matched MLP lands at 4.42–5.02M against the GNN's 4.95M on `bb_core8_bw1p5` and −44…−52% against −49% on `bb_core4_bw0p5`, zero collapses in 40 cells. **The whole model-class gap was corpus.** ⇒ Before any "A beats B" number leaves a node, list each arm's training cache next to it; if they differ, the comparison measures the caches. This is the model-side twin of the separability result — on a pointwise-separable target the pointwise model is correctly specified, so it reaches the same ceiling once it sees the same data — and it was reachable for an hour of CPU at any point in the last four days.

## ⛔ A TUNING STAGE MUST BE POWERED LIKE THE COMPARISON IT FEEDS (2026-09-03)

`objective_pivot_v1` Phase 3 selected a learning rate from a grid run at **one seed per configuration**, then discovered at the pilot that the across-run sd was 4.9-5.8 pp. The winner had won on a single +10.6% draw that later turned out to sit near the top of a distribution centred on zero; the +8.5% it showed on held-out dev cells was the same draw evaluated twice, not independent confirmation. Every downstream run then spent its budget at a hyperparameter chosen by noise. **Before selecting anything from a grid, measure the seed variance of the selection statistic — if it exceeds the gaps between grid rungs, the grid is measuring draws.** A dev-cell evaluation does NOT rescue this: it controls for memorisation, not for the training draw, and the two failure modes look identical in a single run.

## ⛔ CHECK THE INSTRUMENT CAN EXECUTE THE n BEFORE SIGNING IT (2026-09-03)

Phase 3's Amendment E registered n = 120 with an exact Wilcoxon, against an analyzer that raised above n = 22 because 2^n enumeration is infeasible. The guard was correct and fired correctly — **at the analysis step, after all 615 gate episodes had been computed.** The artifacts survived, so only the analysis was re-run, but the failure mode is general: a registration fixes a number, a test and an alpha, and nobody runs the analysis path on synthetic data at the registered n first. **Dry-run the full analysis on fake data shaped like the real thing before signing the number.** It costs seconds and it is the only check that catches a registration its own tooling cannot carry out.

## ⛔ ANY TRAJECTORY- OR HORIZON-RETURN LABEL NEEDS A RANK-STABILITY CONTROL, AND IT IS NOT OPTIONAL (2026-09-01)

A label built by letting a simulator run forward from a decision measures *the decision plus everything the simulator did afterwards*, and in a discrete-event system those are separable only by test. P3 pre-registered co-primaries, an n, a horizon calibration, and two repair controls — and **passed all of them on pure noise**: 21.7% of snapshots above the 2% regret bar (binomial p = 1.5e-33 vs the t=0 base rate), node-count repair 0.009 and link repair 0.016 (both bars < 0.5), additive R² collapsed 0.988 → 0.049. Every number said 'new mechanism that escapes both known degeneracies'. The unregistered control that settled it took **one extra sweep at two shorter horizons**: re-rank the same 256 plans at h=2 and h=5 and correlate against h=10 — median ρ = −0.027 / +0.004, the h=10 optimum at rank 120/256 against 128/256 for chance. ⇒ **Before reading any horizon/return label, verify the ranking survives a change of horizon length.** If it does not, the label is not a stable property of the (state, action) pair, so it cannot transfer to another trace, seed or horizon — and no repair control will catch this, because noise has no node-count or link-load shape either. ⚠ Note which way the two facts point: the repair controls reading ~0 is what made the result look strongest, and it was the *same* fact as the chaos. A mechanism that escapes every known degeneracy deserves a suspicion budget, not only a celebration. ⚠ And the finding does not transfer to closed-loop RL, which estimates an *expected* return over episodes: there the same chaos is per-episode variance to average through (measured: ~1.5% mean, 3.2% p90 of total RTT), which is what a paired-seed common-random-numbers baseline exists to cancel — size the episode budget from it.

## ⛔ MATCH THE TRAINING CORPUS'S NETWORK FABRIC TO THE SERVING FABRIC BEFORE TOUCHING ARCHITECTURE (2026-08-31)

The largest single lever ever measured in this repo is not a model change — it is training on a corpus whose backbone bandwidth actually *binds*. link_mp_v1's 48 arms (3 families × 16 seeds), all trained on the binding-backbone corpus, land at **−33.5% to −38.5% vs Knative with zero severe collapses in all 48×20 cells**, against **−25.1%** for the deployed checkpoint trained fabric-blind — a **~13pp corpus effect**, where no architecture change (MP on/off, graph choice, residuals, decode guards) has ever moved more than ~5pp. The converse killed a whole lineage silently: topology_transfer's 1000 MB/s corpus made link bandwidth non-binding, so link features were **label-irrelevant** — the labels literally contained no signal the features could explain, and no training run could reveal that. ⇒ **Before any architecture experiment, check that the physics the new inputs describe actually moves the training labels** (is the resource binding? does ablating the feature change the label?): a fabric the labels never price is a fabric the model provably cannot learn, and months of architecture iterations sat on top of exactly that. Full record in `docs/lineages/link_mp_v1.md`.

## ⛔ TURNING A SHARING FLAG OFF IS NOT A ONE-VARIABLE ABLATION — READ THE ALLOCATOR FIRST (2026-08-28)

The obvious diagnosis for the amended H2's S0 VOID was to re-run the control with `replica_overlap` **off**, since overlap is the only structural difference from the rungs whose controls passed. "Same grid, one flag" is what it looks like in the preset diff. It is not what it is. Overlap is what makes the allocator's FCFS walk *harmless*: with sharing on, all four task types draw the same pool (8 or 9, every type on both hosting nodes); with it off, `assigned_platforms` becomes exclusive and early types eat the pool. **Measured on a 24-dataset probe covering both arms:** `per_server=4` gives `(dnn1,dnn2,rf,cnn)` = **(8, 4, 1, 1)** with three of four types confined to one node, and `per_server=5` gives **(9, 4, 0, 1)** — `rf` starves to **zero replicas**, so warmup auto-resolve finds no warm replica and 12/24 datasets die as `System state capture FAILED`. One flag moved pool size, per-type asymmetry and node confinement together. ⇒ **Before proposing an ablation on a grid key, trace what else consumes it in the generator** — a flag that gates an *exclusivity check* changes every downstream allocation, not just the thing it is named after. ⚠ And check the ablation is still *informative* where it does run: the surviving arm's sweeps are **32 rows** against H2's 1,680/3,024, i.e. the H0/H1 regime (16/64) that already **passes** S0 — so a pass there would have re-measured the rungs that already work, not implicated overlap. An ablation can be both runnable and incapable of answering the question. The probe cost **5 seconds**; the corpus it replaced would have cost ~30 minutes and produced an uninterpretable number.

## ⛔ PROBE THE GATE THAT VOIDS THE BARS, NOT ONLY THE BARS (2026-08-28)

An amendment that changes a grid gets probed on the bars it is trying to rescue, because those are the numbers in dispute. AMENDMENT 3 did exactly that: it probed the proposed `per_server` 4/5 pair at 204/204 and **all four of S1–S4 passed**, which is what carried the sign-off. Nobody probed **S0**, the VOID gate that decides whether any of those four may be read — and when the registered rung's paired separable control was finally generated, S0 **failed by 24×** (`optimistic` 0.4853 against a `≤ 0.02` bar, 59.5% max regret, independent verifier agreeing to 1e-9 on all 612 cells). Every bar the probe measured is now unreadable. The control cost **31.5 minutes** to generate; it would have stopped the amendment before sign-off. ⇒ **Before signing an amendment, ask which gate can void the numbers the probe is about to produce, and probe that one too.** A probe covering only the disputed bars is measuring the half of the outcome that cannot kill it. ⚠ And note the asymmetry that hid it: S1–S4 are read on Arm S, which the probe generated; S0 is read on a *paired control* the probe never had a reason to build.

## ⛔ A SEPARABILITY CONTROL THAT READS EXACTLY ZERO UNCONSTRAINED CAN STILL FAIL BADLY ONCE THE CONSTRAINT BINDS (2026-08-28)

`score_route_b_contention.py` asserts that on separable physics `r_exact_band["optimistic"] == 0` **exactly, under any feasibility restriction** — so an unconstrained zero reads like proof the control arm is separable. It is not. The amended H2 control reads `optimistic` **0.0000, max 0.00%** at α=None and **0.4853, max 59.5%** at the registered primary α=2.0, on the same 204 datasets, verified to 1e-9 by an independent solver. The failure sits entirely in datasets where the *cap* excludes the componentwise-argmin plan; where uniqueness alone excludes it (87 of 204 at α=None, from `replica_overlap`) the surrogate still recovers the true optimum in every one. ⇒ **Read a control at every α the rung will be read at.** An unconstrained zero is necessary, not sufficient, and it is the reading most likely to be taken as sufficient because it is the one the theorem names. ⚠ **And do not attribute a control's degradation to constraint tightness without a passing control at equal or greater tightness** — the obvious story here was "H2 squeezes harder", and it is false: H1's control **passes** at `cw_infeas` 0.963 with 3.3 feasible rows while H2's **fails** at a looser 0.877 with 1,677.

## ⛔ UNDER A UNIQUENESS CONSTRAINT, THE CANDIDATE POOL MUST HOLD THE TASK COUNT — CHECK IT PER ARM AT REGISTRATION (2026-08-27)

A brute-force sweep that requires **globally distinct** resources across tasks (`generate_brute_force_placement_combinations`) has zero valid plans the instant `|pool| < |tasks|`, no matter how many candidates each individual task has. The route_b pivot registered H3 as "H2 + `dag_instances=2`" — doubling the joint decision from 4 tasks to 8 — while `replica_overlap` collapses every task type onto **one** pool of `per_server × n_hosting_nodes` slots, sized 2 and 4. **H3 therefore generates 0/204 on both arms, and this was discovered only when someone finally probed it, one rung after the same defect took H2 to 102/204.** The rung had a fixed order, a fresh seed block, a derived skip threshold and a registered α — everything except the one inequality that decides whether it produces data at all. ⇒ **At registration, for every arm, write down the pool size and the task count and compare them.** It is one line of arithmetic, it is invisible in every other registered quantity, and the probe that settles it costs seconds — 24 datasets in 3.1s here, against a rung that was otherwise headed for a cluster job. And read the skip reason: `uniqueness_exhausted` and `too_many_combinations` are different diseases and the threshold is not the cure for the first.

## ⛔ ATTRIBUTE A SQUEEZE TO THE GRID KEY THAT ACTUALLY CREATES IT — HISTOGRAM IT, DON'T REASON ABOUT IT (2026-08-27)

"Widening `replica_configs` would loosen the very squeeze the rung exists to create" was written into a lineage node **and** a findings attachment, cited as a reason not to take the one option that could rescue a bar, and was **false**. The squeeze was `server_node_counts=[4] × replica_server_percentage=0.5` ⇒ exactly **2 hosting nodes**; `per_server` sets how many platform slots sit on those two nodes and cannot change how many nodes there are. Measured hosting-node histogram `{2: 204}` on every rung **and** on the widened probe, with componentwise infeasibility going *up* (0.91) as the pool widened. The same file already carried a 2026-08-27 CORRECTION for reading `server_node_counts=[4]` as "4 hosting nodes" — **the same misreading, uncaught, propagated into a second decision.** ⇒ **A claim about what a lever does to a scarcity is a histogram, not an argument.** Print the distribution of the quantity you believe is scarce, per arm, before you let the belief block an option.

## ⛔ SIZE A PER-DATASET COMPETITOR AGAINST THE GRID'S SMALLEST SWEEP BEFORE SIGNING OFF THE BAR (2026-08-27)

A bar and a grid can each be sound and still be incompatible **only in the product**, which is invisible until a rung is generated. The route_b pivot's S2 kill bar reads a per-dataset `t1x` fit; the saturation guard refuses any fit with `n_rows < 2 × n_params`, and `t1x` has 41 parameters ⇒ **needs ≥ 82 sweep rows**. The bars were calibrated on a pilot corpus with 256–1248-row sweeps; the rung's own scarcity squeeze — the thing that creates the contention under test — takes sweeps to **16 and 64**, so `t1x` was refused on **204/204 datasets in both arms of both rungs** and the kill bar had no denominator. ⇒ **Whenever a screen fits a competitor per-dataset, compute `2 × n_params` and compare it to the smallest sweep the grid can produce, at registration time.** Tightening an environment shrinks its sweeps; the measuring instrument does not shrink with it. Full incident in GATE TOOLS (2026-08-27, "the bars vs the grid"). ⚠ **But do not generalise a refusal past the arm it applies to** — see the next rule; two of the three bars reported as blocked by this were blocked by a tool, and their own column sets fit fine.

## ⛔ A REFUSAL IS NOT AN ABORT — SCOPE IT TO THE STATISTIC IT REFUSES (2026-08-27)

A guard that refuses one unfittable arm must cost you *that arm*, not every statistic computed in the same pass. `route_b_coefficient_transfer.py` raised on the first per-dataset saturation refusal, and it was concluded from the crash that the whole screen was unmeasurable on the grid. **Wrong on two of three bars.** S4's bar block is 7 parameters and fits in every arm — it was unreadable only because an unrelated 21-parameter arm refused in the same loop. S3's statistic is a *pooled* fit over every row of every dataset with one shared coefficient set, which cannot interpolate per-dataset at all, and it was gated behind a per-dataset "did everything fit?" condition that does not apply to it. ⇒ **When a fail-loud guard blocks a reading, check the width of the specific column set the BAR names before concluding the corpus cannot support it** — the crash names the arm that refused, not the arms that would have fitted. And when you fix it: tolerate exactly the refusal (its own exception class, never a bare `except RuntimeError`, which also swallows "no feasible rows" and "non-positive optimum"), report `n_fitted`/`n_saturated` per arm, and return `null` — never a zero, which reads as "the competitor closes nothing", the opposite of "we refused to let it interpolate".

## ⛔ CHECK A LEVER'S ADDITIVE/INTERACTION SCALING BEFORE BUILDING IT — AND ONLY TO RULE IT OUT (2026-08-18)

A physics lever only creates coupling if it moves the **interaction** term faster than the **additive** one. Deep queues failed because additive `depth × exec_time` grew with the lever and interaction `added_in_batch × exec_time` did not. Link bandwidth failed the opposite way: store-and-forward charges every hop a full transmission, so additive `hops × T` and interaction `crossings × T` **both** scale as 1/bandwidth ⇒ ratio invariant, measured **0.0100 → 0.0088** across a 3× cost change, contention stuck at ~1% of the link cost at any bandwidth. Two lines of algebra would have predicted both. ⚠ **But the test only rules levers OUT.** On the hub↔mesh sweep `wait/transfer` got the ranking backwards — n_core=4 had the *lowest* ratio (0.0072) and the *highest* regret (0.35%/7.2%) — because the ratio is measured on **optimal** plans, which select *against* contention, while regret is a property of the spread across the whole sweep. Bandwidth-invariance ⇒ stop; a favourable ratio ⇒ no information.

## ⛔ THE COLLISION TERM MASKS EVERY NEW MECHANISM — ISOLATE WITH `--spread-plans-only` (2026-08-18)

`added_in_batch` dominates every corpus in this repo, so a headline gate cannot tell a new mechanism's coupling from the collision term it rides on. New control restricts every metric to plans placing each task on a **distinct node**: node-occupancy excess is then identically zero (the column that killed four mechanisms becomes a constant) and `added_in_batch` is zero too, while link-style contention *survives* because it acts between different destinations. Result that settled `link_contention_v1`: baseline additive R² = **1.00000 exactly, 0.00% regret, 100% of datasets** ⇒ **once collisions are removed the base physics has literally no coupling left**, so there is no reservoir waiting for a better lever. It is an isolation control, **not** a corpus gate — it discards ~2/3 of the sweep. Two tests pin both halves of the contract.

## ⛔ THE REPAIR CONTROL MUST MATCH THE MECHANISM BEING TESTED (2026-08-18)

`one_integer_repair_frac` uses **node-occupancy excess**, which is *structurally blind* to link contention — two tasks on different nodes sharing a core segment read exactly zero there. Shipping `link_contention_v1` against the stock gate would have returned a **false PASS**. Added `--gate-link-repair` (busiest link `k1`, top-2 `k2`, total sharing `excess`). Generalisation: **before gating a new physics, ask what the existing repair column cannot see, and add the column that can.** And test the gate itself — `--gate-coupled-fraction` was recommended as primary while being incapable of failing; the new controls ship with 11 tests.

## ⚠ A COST THAT PRICES ONLY *REMOTENESS* PUSHES THE OPTIMUM TO THE LOCAL CORNER (2026-08-18)

First `link_contention_v1` pilot ran on stock `shallow_v1` (`per_client >= 1`) and the backbone made the corpus **more** separable (regret 2.51% off → 1.07%/1.09% on), because most tasks could run on their own source node and never touch the network. Same shape as `netc_scarce_v1`, where penalising co-location pushed the optimum toward the corner greedy already picked. Fix is `per_client = 0` (`netc_multihop_v1`), varying **only** that — unlike `netc_hotspot_v1`, which moved `replica_server_percentage` and `per_client` together. **Check what fraction of the optimum even uses the resource you are pricing before generating a corpus.**

## ⛔ M1 / `--gate-coupled-fraction` IS THE WRONG GATE — IT CANNOT FAIL (2026-08-18)

M1's marginal greedy scores each option as `min RTT over all joint plans with task t there` ⇒ it has **oracle access to the joint sweep**, so it is not a pointwise model and cannot fail once the candidate set is small. Measured: a corpus where the additive fit picked a suboptimal plan **75% of the time at 21.5% mean regret** still reported M1 coupled = **0%** — the pre-registered gate would have **rejected the only config showing headroom**. Retroactive scope: `--gate-coupled-fraction` appears in **no committed script/sbatch/log**, so **no corpus was ever accepted or rejected by it** — but it was recommended as primary in this file (0.48.0) and in `LINEAGES.md`, and shallow_v1's since-retracted "31.0% coupled" justified the whole shallow lever. ⇒ **Gate on M4 `additive_choice_regret_rel` / `additive_choice_is_optimal`** (the additive fit is literally `PointwiseEdgeMLP`'s expressive class). New `--gate-additive-argmin-regret`, which **argparse-errors unless `--gate-one-integer-repair` is also given**.

## ⛔ THE ONE-INTEGER CONTROL — MANDATORY NEXT TO ANY REGRET NUMBER (2026-08-18)

Re-take the additive fit's argmin after handing it **one** scalar (per-plan node-occupancy excess). If that repairs the regret, the "coupling" is a count-like feature an MLP learns from one extra input, and a GNN win would only mean the baseline was under-featurised. **Now first-class in `separability_diagnostic.py`** (`one_integer_repair_frac`, printed with a `!! DEGENERATE` banner at ≥0.5, gated by `--gate-one-integer-repair`). **Three independent mechanisms have now collapsed this way** — `added_in_batch` (base physics, +1.10pp), deep queues (diluted by depth), node-ingress bandwidth (75% repaired at its only high-regret setting) ⇒ this is looking like a property of **this class of scheduling problem**, not of any one experiment. That throughline + the reusable diagnostic is a legitimate paper contribution on its own.

## ⛔ THE WORKLOAD DRAW WAS UNSEEDED — NO CORPUS IS REPRODUCIBLE FROM ITS SEED (2026-08-18)

`generate_workload_templates` drew task source nodes from the **global** `random.randint`, with **no `random.seed()` anywhere** in the generator. Two runs of the same grid with the same seeds gave different workloads, RTTs and sweeps. Infrastructure was always deterministic (`network_maps`/`replica_placements`/`queue_distributions` byte-identical) — only the workload moved. **Matched A/B arms were therefore impossible**, and `node_contention_v3`'s "metrics disagreed and were underpowered" 12-ds pilot was built this way. Fixed: local `random.Random(--workload-seed)`, default 42. **`placements.jsonl` row order is still nondeterministic and that is benign** (parallel completion order) — compare sweeps as **sets**, never with `diff`. ⇒ **Never use an existing corpus as a bit-identity baseline for a physics change**; build the baseline arm from the same tree as the treatment arm.

## ⛔ SCARCITY BY COUNT AND TOPOLOGY FUNNELLING BOTH BACKFIRE — OVERLAP IS THE VARIABLE (2026-08-18)

Penalising co-location only creates a joint decision if tasks cannot cheaply avoid each other. Free-spreading check (min premium `θ*` admitting a task→distinct-node matching, Hall's condition, **no simulation**): `θ*=0` in **92%** of baseline datasets — each task's favourite node was already distinct. Cutting links (`netc_scarce_v1`) and funnelling to hubs (`netc_funnel_v1`) both **reduced** mean pairwise candidate-node overlap (0.93 → **0.36** → **0.14** of 4 tasks) because tasks anchored to different clients get *more disjoint* sets as they shrink ⇒ θ*=0 rose to **100%**. The blocker on the fix was `generate_infrastructure.py`'s `replica_server_pct = max(server_pct, 0.6)` floor (now overridable via `preinit.replica_server_percentage` / `--replica-server-percentage`). ⇒ **Measure candidate-set OVERLAP, not candidate count**; and **vary one thing** — `netc_hotspot_v1` moved `replica_server_percentage` **and** `per_client` 1→0 at once, so its "cliff at 0.15" was really "client-local replicas removed".

## ⛔ OFFLINE REGRET IS NOT A LIVE PREDICTOR, AND THE RESIDUAL'S SIGN FLIPS WITH COUPLING (2026-08-17, 15/15 final)

Arm A (gated GIN residual, `x = x0 + mp_gate·gin(x0, ei)`) cut offline test greedy regret **0.5682 → 0.2621s (−54%)**, top-5 0.0346 → 0.0239, `mp_gate` learned **1.08** (>init ⇒ it leans on MP *more*). Live it **split by config**: `sparse_p25` 1.14→**1.18×** Kn (+3.5%), `sparse_p35` 1.02→**1.12×** (+9.9%), but **`sparse_p25_skew` 0.84→0.80× (−4.3%), p99 71.0→63.1s**. Gate verdict unchanged: **PRIMARY 1/3, TAIL 1/3 FAILED**; paired wins identical to baseline (GNN 3/15 · MLP 10/15 · Kn 2/15); SUM 1.05→1.12×. ⇒ **A −54% offline gain bought nothing overall and cost RTT on the two near-pointwise configs, while paying only on the one config with real coupling** — exactly the separability entry's prediction (extra capacity buys ~1pp of interaction variance and blurs the ~99pp pointwise signal). **⚠ Correction to the mid-session read: `logit_tied_rate` is NOT the discriminator** — it rose on *all three* configs (0.5097→0.5542, 0.5374→0.5807, 0.5338→0.5511) including the one that improved. The sign of the effect tracks **the config's coupling**, not the tie rate. ⇒ **Never promote a checkpoint on offline regret/acc alone, and never gate a graph-capacity change on a pointwise-separable config** — measure per-config with M1 coupled-fraction alongside.

## ⛔ A RESULT WHOSE CODE NO LONGER EXISTS IS NOT A RESULT (2026-08-17)

The 0.47.0 headline "0.88× Kn on `sparse_p35`/s42" is **not reproducible** — the full gate gives **1.04×** for the identical cell/seed/model/config/decoder. Proof the code changed, without the old source: that arm ran `GNN_DROP_NODE_EDGES=1`, a variable implemented **nowhere** in the tree today; had it been a no-op then, the arm would have been the unfixed baseline (~276M), but it measured 22.34M ⇒ the flag worked then and the file was rewritten after (`GNN_DROP_NODE_EDGES` → opt-in `GNN_MP_NODE_EDGES`), with only the final state committed in `2a591ed`. **Root enabler: `run_provenance` records neither the git commit nor `OMP_NUM_THREADS`/`MKL_NUM_THREADS`** (ablation ran 4 threads, gate 6) — and with `logit_tied_rate ≈ 0.54`, FP reduction-order differences can flip argmax on half the decisions. **Add both fields to provenance; until then treat any cross-sweep GNN comparison as suspect and re-measure under one runner.**

## ⛔ ONE MODEL DEFINITION, AND MAKE ARCH FLAGS SELF-DESCRIBING (2026-08-17)

`train_near_rtt.py` no longer declares its own `TaskPlacementGNN` — it **imports** the canonical one from `src/policy/gnn/gnn_model.py` (−84 lines of duplicate class; training behaviour preserved via new ctor args `dropout`/`post_gin_dropout`/`normalize_platform_inputs`). Two copies is what caused the 25:1 flood; a second latent drift was already queued behind it (the trainer had a `platform_input_norm` LayerNorm the server lacked — absent from the deployed ckpt, so `strict=True` never fired, but one training run from silently serving wrong). **Pattern that prevents recurrence:** the residual carries a learnable scalar `mp_gate`, so it appears in the state_dict and a residual ckpt **cannot** strict-load into a non-residual model (or vice versa) — silent architecture drift becomes a loud `RuntimeError`; what weights can't encode (`mp_node_edges`) goes in the existing `<model>.contract.json` sidecar, which `load_gnn_model` reads back, and a **stale `GNN_MP_NODE_EDGES` that disagrees with the sidecar now fails loud** instead of being ignored. Candidate-restricted same-node edges measured on the full corpus: **1428 → 20.6 mean edges (25.9× → 0.37× bipartite)**, 80% of graphs retain ≥1 pair, 20% correctly degenerate to bipartite-only.

## ⛔ DATALAB: TRAINING PORTS, SERVING DOES NOT — and never `rm -rf` the repo dir (2026-08-17)

Remote `gnn-herosim` sat on divergent history where `constants.py` lives in **`src/motivational/`**, local has **`src/placement/`** ⇒ `train_near_rtt.py` runs on both (its imports happen to be satisfied) but `executesimulation.py` `ImportError`s ⇒ **the live gate cannot be moved to datalab without a real branch sync.** It was also still serving the **pre-fix** `gnn_model.py` (unconditional `node_edge_index` concat = the 12.4× regression) because earlier syncs were file-by-file rsync — **code must travel by git, only data by rsync.** **Safe re-sync tactic (data survives):** `git fetch` → `git checkout --force -B <branch> origin/<branch>` → `git clean -fd`; `clean` skips **ignored** files, so all 65G of `simulation_data/` + 7.3G `logs/` + gitignored ckpts survive. A clone-and-move-data-dirs plan would have silently dropped `src/notebooks/models/*.pt`, `src/notebooks/logs/`, and 673M of wandb runs under `src/legacy/` — **gitignored data is scattered inside the source tree, not confined to data dirs.** Verify by diffing `git ls-files` against a throwaway clone. ⚠ **`--partition=GPU-rtx6000` is unusable**: the `gnn` env's torch 2.5.1+cu121 has no kernel image for those cards; both arms died at weight init. Use `GPU-a40,GPU-a100s,GPU-l40s,GPU-a100` (the existing `full_corpus` sbatch omitted rtx6000 deliberately).

## ⚠ the pre-registered gate's own criteria are the bar, and a losing config can still be the interesting one (2026-08-17)

MP-parity gate: **PRIMARY 1/3, TAIL 1/3 — FAILED**, yet the single config the GNN wins (`sparse_p25_skew`: GNN **0.84×** Kn / MLP **2.27×**, p99 **71.0s vs 498.5s**) is where the pre-registered *collision cliff* actually reproduces — on skew, not the `sparse_p35` where the n=1 probe saw it. Aggregate `SUM` (GNN 1.05× / MLP 0.93×) **hides** this because it is dominated by the two large-RTT configs. ⇒ report per-config wins **and** the declared criteria, never the SUM alone; the honest claim is *bounded collision-robustness on skewed queues*, not a general win. **Process note: cells were twice dismissed as "record completion" and twice held the only GNN win — score every cell before concluding.**

## ⛔ THE CO-SIM TARGET IS POINTWISE-SEPARABLE — the MLP was correctly specified all along (2026-08-17)

New **M4** block in `scripts_cosim/separability_diagnostic.py` fits `rtt(plan) ≈ μ + Σ_t f_t(plan[t])` by least squares over every plan in `placements.jsonl` — exactly what `PointwiseEdgeMLP` can express. Additive R²: `contention_v2` **0.98812** (additive argmin = true optimum in **81%** of ds), `contention_v4_pilot` **0.99973** (**100%**), `contention_v5_quick_test` **0.99973** (97%), `highq_safe` **1.00000** (**100%**). ⇒ **No amount of data or architecture work could have made a GNN win.** Reframes the whole `mp_parity` line: the GNN didn't lose to a bug, it lost because nothing non-pointwise was left to learn. **The entire interaction is one integer** — adding a collision-count column takes v2 0.98812→0.99912 (+1.10pp); on v4/v5 it is +0.03pp. Mechanism (`src/placement/scheduling_cost.py:108-132`): every term of `current_work + queue_work + cold_start + exec_time + comm_time + network` depends on `(task, platform)` alone **except `added_in_batch`**. Network latency is a static table lookup paid once per task (`infrastructure.py:985-993`), never congestible. **Co-located platforms share NOTHING** — the only `capacity` in `infrastructure.py` is disk cache; `memoryRequirements` is never enforced. Measured RTT split on a real optimum: queue **1.330s (~95%)**, network 0.0719s (5.1%), exec 0.0239s, cold-start and pull **exactly 0**. ⚠ **Trap: platform ids are globally unique, so `platform_only_r2 == additive_r2` by construction** — the node/platform split says *where* the separable signal sits, NOT whether topology matters; use the network share of RTT for that.

## ⛔ DEEP QUEUES ARE THE WRONG LEVER — confirmed on all 899 `contention_v2` sweeps (2026-08-17)

Queue depth **predicts** separability, monotonically: shallowest quartile (depth 27.6) → additive R² **0.97822**, collision gain **+1.986pp**; middle (35.5) → 0.99211/+0.672pp; deepest (50.8) → **0.99803/+0.181pp**. `corr(depth, additive_r2) = +0.256`, `corr(depth, collision_gain) = −0.259`. **Coupling is 11× weaker in the deepest quartile.** Arithmetic reason: additive term `depth × exec_time` grows with depth; interaction `added_in_batch × exec_time` does **not** ⇒ depth dilutes the only coupling there is. This is why `contention_v4/v5` landed at 0.9997 — **the contention series' core lever was backwards.** Inverse works: `shallow_v1` grid (depths ~0–8, stock dnn1/dnn2, otherwise v2 axes, 200 ds) → coupled(>1%) **31.0%** vs v2 **7.1%**, coupled(>5%) **19.6%** vs 2.4%, additive-argmin optimal **62%** vs 92%, pointwise worst-case regret **48.4%** vs 3.6%. **First corpus where the node-collision column outexplains the platform one.**

## ⚠ MEAN ADDITIVE R² IS THE WRONG GATE STATISTIC — the target is bimodal (2026-08-17)

`shallow_v1` median R² **0.99990** against mean **0.96074** — most datasets stay perfectly separable and *all* the structure sits in a large minority, so averaging hides exactly what matters. Also n-sensitive: the same corpus read 0.93838 at n=12 and 0.96074 at n=168 (would **fail** a 0.95 gate at full size while passing at pilot size). **Use the coupled fraction (M1 regret >1%) instead** — new `--gate-coupled-fraction`, which cleanly separates the arms (shallow **0.167 PASS** / baseline **0.000 FAIL**). Both gates live in `separability_diagnostic.py`; run on a ~30-ds pilot **before** generating a full corpus.

## ⛔ A SHARED RESOURCE ONLY COUPLES IF PLACED TASKS HOLD IT LONG ENOUGH TO OVERLAP (2026-08-17)

Built `node_contention_v3` — `Node.compute_slots` (`simpy.Resource`) that co-located platforms contend for, opt-in via `--compute-slots-per-node` / `config.nodes.compute_slots_per_node`, **default None = bit-identical `node_disk_v2`**. Matched 12-ds pilots: metrics disagreed and n was underpowered (additive R² 0.98847 vs 0.98584 wrong way; additive-argmin optimality 75% vs 92% right way) — **but `nodeContentionTime` was exactly 0.0 on EVERY placed task in every dataset.** Placed tasks never overlap: exec is ~0.024s and the seeded backlogs drain first, so slot contention only serialized the backlogs — a per-`(task, node)` **additive** term. Candidates that *would* satisfy the rule: memory held across the whole residency (cold start + exec, cold starts reach **38s**), or a slot held for a replica's warm lifetime. Guarded by `scripts_cosim/test_node_contention.py` (9 tests). **Non-obvious trap found building it:** queue depth is seeded as a compressed warmup backlog (`Platform.virtual_warmup_total_time`) drained in a **single** `env.timeout`, NOT as `queue.items` — wrapping only the per-task execution path left contention at exactly 0 while bypassing ~95% of RTT.

## ⛔ CACHE LABELS ≠ SWEEP OPTIMA — silently invalidates any ablation (2026-08-17)

`prepare_graphs_cache.py` labels graphs from `optimal_result.json` (`sample.placement_plan`), which is **not guaranteed** to be the `placements.jsonl` minimum. On fresh `shallow_v1`: **176/200 match, 24/200 (12%) do NOT**, mean regret of a wrong label **15.78%**, max **92.5%**, 0 absent. Symptom that exposed it: the ablation reported **0/30 coupled test datasets** (greedy baseline regret 0.00% mean/p90/max) against a corpus-level M1 coupled(>1%) of **31.0%** — the ablation is self-consistent *within* the corrupted label space but disconnected from the sweep. **This is `gnn_necessity_separability.md` §1's open item reproduced on a new corpus, unfixed since 2026-08-04.** ⇒ The 2026-08-17 `shallow_v1` ablation (`gnn_base` 91.7%/0.90% beating pointwise 90.0%/2.68%, and `gnn_node` re-falsified) is **NOT established** — both arms measured on contaminated labels. **Always verify label provenance before running or citing an ablation.**

## ⚠ `queue_feature_contract` in METADATA was a hardcoded guess (2026-08-17)

`extract_dataset_metadata.py` had `queue_contract = "legacy_v0"  # Default assumption` while `compute_compatibility_matrix.py` decides **what may be trained together** from that field. Now read from the collection's graph cache `metadata.json`. **The contract is NOT single-valued per collection** — `contention_v2` has both `graphs_cache_contention_v2_873_v5.5` (no field ⇒ legacy_v0) and `…_873_v5.7_siv1_dim14` (scale_invariant_v1); on conflict the extractor records `legacy_v0` and **warns**. A cache with no contract field predates it and *is* legacy_v0 — skipping such caches hides real conflicts. `shallow_v1` now sits alone in group `scale_invariant_v1_node_disk_v2_4task`, unmixable with the 15 `legacy_v0` collections.

## ⚠ new-corpus runbook gaps (2026-08-17)

(1) The generator does **not** emit `system_state_captured_unique.json`; `prepare_graphs_cache.py` hard-requires it and dies with `FileNotFoundError: Missing system_state_captured_unique.json for ds_00000`. Run `refresh_optimal_full_stats.py --base-dir <collection> --rewrite-ssc` first — same trap recorded for `contention_v2` in June. (2) New task types need a `workload_nofs-<type>` param in the sampled space (`combinations_simple_mapping.pkl`), not just `wsc`/`prewarm`/`replicas` (now auto-synthesized). (3) Two loud-failure fixes: `prepare_workloads` silently dropped apps missing from the sample mapping; `calculate_workload_stats` returned `average_rps` while `flatten_workloads` reads `stats['rps']`, so an empty workload died as a bare `KeyError: 'rps'`. (4) `contention_v5_quick_test` has **3/38 ds with no `placements/` at all** (violates the mandatory-JSONL rule); `highq_safe`'s sampled sweep can't satisfy M1's strictness — M1 needs relaxing before batch size >8.

## ⛔ TRAIN/SERVE MESSAGE-PASSING MISMATCH — the real cause of the GNN's 12× live failure (2026-08-16)

Three *different* architectures were in play. **Offline ablation** `scripts_cosim/gnn_necessity_ablation.py` (the RQ3 "GNN>MLP" evidence): GIN with a **residual** (`x = x0 + gin(x0, ei)`) **and candidate-restricted** same-node edges (`synth_node_edges(candidates_only=True)`) — its own comments say full same-node edges "flood the GIN and oversmooth". **Training** `train_near_rtt.py:270`: `self.gin(x, data.edge_index)` — bipartite only, **never references `node_edge_index`**, no residual. **Serving** `src/policy/gnn/gnn_model.py`: bipartite **+ ALL** same-node edges, no residual. Measured on `graphs_cache_contention_v2_873_v5.7_siv1_dim14`: **1428 same-node vs 57 bipartite = 25:1 flood**; on the deployed ckpt **87.5% of argmax decisions flip** (480 tasks / 420 flips) and the chosen platform's dim7 is 1.25× deeper. Co-located platforms get averaged, erasing dim7 — the one feature that separates them. Live `sparse_p35`/s42 (ladder `gnn_encoder_ablation_20260816/`, seed 42, full-corpus siv1 ckpt): control **276.0M / 10.93× Kn / qvsmin 4,007 / collide 0.411** → **drop same-node edges 22.34M / 0.88× Kn / p99 213 / qvsmin 98 / collide 0.335**, i.e. **beats Knative and ECT with no retrain**. No-MP is 22.44M (0.89×) ⇒ once the flood is gone, bipartite MP is ~neutral, so rung B is still a real GNN. **This is why "offline accuracy is decoupled from live placement" — offline was scored on a different graph than serving used.** Root cause of the drift: `train_near_rtt.py` runs training at import so it can't be imported as a library; the model class was copy-pasted and diverged. Fixed: `GNN_MP_NODE_EDGES` (default **off** = parity) + guard `scripts_cosim/test_train_serve_mp_parity.py`. **Any future GNN arch change must land in BOTH files.**

## ⛔ DECODE GUARDS ARE NOT THE GNN LEVER (2026-08-16)

Same ladder, same cell. `GNN_LQB_LAMBDA=1.5` → **275.7M (10.92×)**, i.e. *no change* vs control 276.0M — the λ·log1p(4000)≈12 penalty is swamped by corrupted logit margins (`logit_chosen_minus_minq` mean **+229.9**, `confident_worse_queue_rate` 0.95). `GNN_DECODE_MODE=argmax_uniq` drove `collision_batch_rate` to **exactly 0.000** (550,851 tasks enforced, 81 relaxed) and got **WORSE: 301.3M (11.93×)**, qvsmin 4,508. ⇒ **intra-batch collisions were a symptom, not the cause**; a decode guard cannot rescue an inverted ranking. `GNN_QUEUE_FILTER_MAX_DELTA=8` does reach 21.80M (0.86×) but only by overriding the model (qvsmin→4). Retires the 0.45.0/0.46.0 conclusion that the failure is "structural in the joint 4-task argmax".

## ⛔ ECT IS NOT A CEILING AND NOT A TEACHER FOR REGIME A (2026-08-16)

First-ever run of `--knative_network_ect` on the coupled trio (seed 42, `node_disk_v2`): `sparse_p35` **25.87M (1.02× Kn)**, `sparse_p25` **16.79M (0.98×)**, `sparse_p25_skew` **4.95M (1.13×)** — vs MLP 0.78×/0.84×/2.73×. A greedy that knows queue length, exec time, cold start, comm and network latency *exactly* performs at Knative level ⇒ better myopic cost estimation is not the lever, and there is **no ECT headroom to distill** (Regime B's 125→31 does not carry over: `contention_v2` has 0/900 datasets with any cold start or pull, so `expected_completion_with_filterstore_pull` degenerates to plain `ect`). **Do NOT infer "no headroom ⇒ pivot to scenario design"** — the MLP already beats both. ⚠ `ECT_PULL_DISTILL_DIR` dumps a PyG frame **per decision**: on `workload-125-225` that is 561,848 frames/cell (~5.1GB in 15min across 3 cells, ~50GB/cell projected) — keep harvesting opt-in (`HARVEST_DISTILL=1`).

## ⛔ CORPUS SIZE IS NOT THE GNN LEVER (2026-08-15/16)

3.04× more training data (873 → **2,651** graphs; merged `contention_v2`/`v3`/`v4_pilot` + `1060_warmth_v2` + `sparse_warmth_v2` + `highq_safe`, `scale_invariant_v1`/dim14, cache `graphs_cache_full_corpus_siv1_dim14`) improved the GNN **offline** and changed **nothing live**. Offline test acc **55.7% → 65.3%** (val 62.6→66.3, task_acc 89.4%) — a real gain, well outside the ±4.2 seed noise in the entry below. Live 45-cell trio (`full_corpus_siv1_coupled_trio_20260815/`, datalab **685484**): **GNN 0/15, SUM 12.40× Kn** (was 15.07× at 873) — marginal, still catastrophic. **MLP 10/15, SUM 0.93× Kn.** Decisive: `chosen_queue_vs_min` **mean 3,329** across all 15 GNN cells (median chosen ~1,100 vs min-available ~0–50), collisions 0.31–0.48 — i.e. **queue-blindness is unchanged despite the offline gain**. ⇒ **Offline accuracy and live placement quality are decoupled for the joint decode; stop generating/merging corpora as a GNN fix.** Both ckpts served under a **verified-identical** `scale_invariant_v1` contract (grepped from cell logs), so this is **not** a train/serve mismatch. wandb `gnn-full-corpus-siv1-aug2026` (GNN `v06uk8gb`, MLP `5k7phldu`).

## ⚠ datalab repo has divergent history + a regressed `BASE_DIR` (2026-08-15)

Remote `gnn-herosim` sits on its own older history (`a9c217a` mlp), so a file can differ from local **even when local `git status` is clean** — transitively-imported modules must be synced too (hit `CACHE_VERSION` and `split_by_parent_three_way` ImportErrors before a full `src/{policy,placement,notebooks}/` rsync fixed it). The KG's known `run_simulation.py` `BASE_DIR` gotcha (see entry below) had **regressed on datalab** to hardcoded `/root/projects/my-herosim` — killed all 30 cells instantly with `PermissionError`. Fixed by rsync; **re-check after any remote-side edit**.

## ⚠ golden feature-parity test was blind to queue features (2026-08-13)

`verify_cache_live_feature_parity.py`'s default fixture (`regime_b_cold_burst_v1_oracle_split_cosim/ds_00003`) has **8 queue keys all at 0**, so dim7/dim13 were 0 on both sides and any cache↔live divergence in them passed. Added `--queue-depth-scale` (synthesizes a 0/1/7/23/61/150 depth spread ×scale) — at ×400 legacy dim7 reads **28–600** and dim13 **up to 570** vs v1's 0.007–1.0 and 1.7–4.4. Also surfaced a **pre-existing, unrelated asymmetry**: the cache decides dims 9–11 estimation per *snapshot* (`if temporal_state:`) while live decides per *platform* (queue>0 and no recorded remainder), so a queued platform with a zero remainder diverges (~0.08). Real snapshots pair depth with a running task, so the fixture now synthesizes temporal state alongside depth; **the asymmetry itself is still open**. 6/6 layout×contract parity combos PASS.

## 🏆 siv1 5-SEED GATE: MLP BEATS KNATIVE 13/15 CELLS + p99 3/3 (2026-08-13, datalab job 664414)

The seed-42 smoke replicates across seeds 42–46. `contention_v2_873_v5.7_siv1_coupled_trio_20260813/` (30 ML cells on CPU-amd; Knative arm **symlinked** from the v5.5 trio sweep — policy-independent, identical configs/seeds/workload/`node_disk_v2`), `manifest.json` + `compare.json` written. **Seed-averaged `mlp/kn`: `sparse_p35` 0.74× (5/5 cells), `sparse_p25` 0.79× (4/5), `sparse_p25_skew` 0.90× (4/5); SUM 0.775×, paired wins MLP 13/15 · Kn 2/15 · GNN 0/15.** **Tail: MLP wins p99 3/3** (108.4 vs Kn 183.4 · 102.8 vs 130.0 · 144.9 vs 254.1) and p90 2/3 (loses skew 22.4 vs 21.6). `sparse_p35` is the tightest cell (0.71–0.80× every seed). The 2 losses are `sparse_p25` s45 (1.07×) and `sparse_p25_skew` s43 (**2.14×**, the entire ±2.17M MLP std in that config) ⇒ an occasional tail blow-up survives; claim "beats Knative on average + on p99", **not** "uniformly". **GNN siv1 remains catastrophic: 0/15, SUM 15.07× Knative** (p90/p99 ~10× worse).

## ⛔ GNN's live collapse is NOT the queue features and NOT model quality (2026-08-13) — **the "structural in the joint decode" half is FALSIFIED**

> **Correction (2026-09-16).** The negative half of this rule stands: the collapse is not input scaling and not accuracy. Its positive conclusion does **not** — [`docs/hard-stops.md`](hard-stops.md) records *"claim the GNN's live failure is 'structural in the joint decode'"* as falsified, because LQB λ = 1.5 changed nothing (275.7M) and `argmax_uniq` drove collisions to 0.000 and got **worse** (301.3M). Read the diagnosis below as history, not as a live explanation.

Ran init-seed variance with the canonical-parent split held fixed (new `NEAR_RTT_TRAIN_SEED`, default 42; the split's own `random_state=42` untouched), n=3 per contract. **Test acc: `legacy_v0` 55.5% ±4.2 (52.7/53.4/60.3) vs `scale_invariant_v1` 53.9% ±2.5 (51.1/55.0/55.7); ranges overlap, greedy regret 0.880s vs 0.980s.** So the deployed v5.5 GNN's 60.3% was the **top of its own seed distribution**, and the siv1 GNN's "regression" to 55.7% is **within seed noise** — the features neither help nor hurt the GNN offline. Since siv1 fixed the pointwise MLP decisively (0.78× Kn) but left the GNN 15× worse, the GNN's failure is **structural in its joint decode**, not an input-scaling or accuracy problem: it is argmax over 4 tasks per batch and its `intra_batch_platform_collisions` *rose* 0.179→0.236 while `chosen_queue_vs_min` median went 734→3676. Next GNN lever must target the decode/coupling, not features or more offline accuracy. Checkpoints: `models/near-rtt-v2-cv2-873-{siv1,legacy}-dim14-ce-only-seed{43,44}.pt`.

## 🎯 siv1 LIVE (seed-42 smoke, superseded by the 5-seed gate above)

First post-`25732cf` win for an ML policy. `…_v5.7_siv1_coupled_trio_smoke_20260813/`, Kn s42 symlinked from the v5.5 sweep, `num_tasks` **561848 identical** in all runs (win is not dropped work). **MLP siv1 vs Knative:** rtt `0.747× / 0.748× / 0.747×` (p25 12.86M vs 17.21M · p35 18.89M vs 25.25M · skew 3.26M vs 4.36M), p99 `0.40× / 0.60× / 0.43×`, p90 lower 3/3, avgQueue `22.8/33.6/5.8` vs Kn `30.6/44.9/7.7`; **loses p50** on p25/p35 (2.83 vs 0.70 · 4.64 vs 0.73). vs MLP v5.5: rtt **0.16–0.29×**, p99 **0.05–0.07×**. Not a degenerate shortest-queue: `chosen_queue_vs_min` **median rose** 7→20 / 9→36 / 0→0 while the **tail collapsed** p95 3911→522 / 12668→980 / 1356→259 — i.e. the model still deviates deliberately but no longer saturates. **GNN siv1 got WORSE than GNN v5.5** on all 3 (rtt 236.9M vs 118.0M · 332.7M vs 282.9M · 77.2M vs 49.4M; qvsmin median 492→1909 / 734→3676 / 210→1554) and remains 14–19× Knative. ⚠ **Confounded**: siv1 features *and* fresh weights (test acc 60.3→55.7, greedy_regret 0.848→0.911s); not isolated. Pointwise-vs-joint asymmetry is the leading hypothesis (scale-free dim7 directly fixes per-edge ranking; the GNN's joint argmax over 4 tasks also raised `intra_batch_platform_collisions` 0.179→0.236 on p35). Seed 42 only — **not** the 5-seed pre-registered gate.

## ⚠ two contract-plumbing defects found by the smoke, both fixed (2026-08-13)

(1) **provenance lied for `mlp_batch`**: `build_run_provenance` runs before `MLPBatchScheduler.set_models` adopts the ckpt contract, so the MLP results record `legacy_v0` although inference served `scale_invariant_v1` (proved by re-running the adopt block: trained `siv1`, declared `''` ⇒ resolved `siv1`; the runner exports no `QUEUE_FEATURE_CONTRACT`). Fixed via `apply_mlp_checkpoint_queue_feature_contract()` called **before** provenance; GNN was already correct (`load_gnn_model` → sidecar, recorded `siv1` in both `env` and resolved). **Older MLP result JSONs' `queue_feature_contract` field is unreliable — trust the ckpt key.** (2) `train_near_rtt.py` `_refresh_queue_dependent_platform_features` / `_queue_norm_from_values` still hardcoded legacy cap-100 + `ratio/5.0`, bypassing `queue_features.py`; now delegated. **Did not affect the retrain**: that path runs only when `PHASE_B_CHECKPOINT_METRIC == "seq_reforward_regret"`, which requires `TRAIN_INIT_CHECKPOINT` and non-CE-only (log shows `seq_reforward=0.0000s`, 21×).

## ✅ scale_invariant_v1 shipped + retrained, offline cost ≈ 0 (2026-08-13)

Fix for the OOD entry below. **Single source of truth** `src/placement/queue_features.py` (`queue_depth_norm` / `usage_ratio_feature`) replaces the duplicated formulas in `prepare_graphs_cache.py`, `feature_builder.py`, `seq_decode.py`, `gnn_snapshot_inference.py`. Two **versioned contracts**: `legacy_v0` (bit-exact history, cap 100 + `ratio/5`) and `scale_invariant_v1` (dim7 divisor **uncapped** ⇒ 400× deeper infra gives identical features; dim13 `log1p(ratio)/log1p(5)`). Contract travels with the artifact: cache `metadata.json`, MLP ckpt dict key, GNN **`.contract.json` sidecar** (bare state_dict), `run_provenance`; mismatch raises `QueueFeatureContractMismatchError` (absent key ⇒ `legacy_v0`, so v5.5 ckpts still serve). Validation: 28 contract tests; `legacy_v0` recache **numerically identical** to deployed `…873_v5.5` on all shared dims+labels; parity verifier passes 6/6 (contract × depth-scale 0/1×/live) — at live scale legacy dim7 reads **28–600** vs v1 **0.007–1.0**. Cache `graphs_cache_contention_v2_873_v5.7_siv1_dim14` (`CACHE_VERSION` 5.7, new `--platform-feature-dim 14` keeps `dim22` inference layout for apples-to-apples). **Retrain (wandb `gnn-queue-scale-invariant-aug2026`, identical splits/seed/hparams):** GNN val **62.6%** (v5.5 61.1%) test **55.7%** (60.3%) task_acc 85.9% (87.2%) greedy_regret 0.911s (0.848s); MLP val edge **91.6%** (91.4%) test **87.3%** (88.3%), early-stop e44. ⇒ offline ≈ unchanged, so any live delta is attributable to the feature scaling, not to model quality.

## ⛔ LIVE FAILURE IS QUEUE-FEATURE OOD, NOT LABELS (2026-08-13)

Models only see platform **queue depth (count)** at dim 7 (`raw_q / queue_norm`) and derived `usage_ratio` at dim 13 (`(raw_q/target_concurrency)/5`); task `queueTime` is **never** an input. `queue_norm = clamp(p90(depths), 1, **100**)` in *both* `prepare_graphs_cache.py:813` and `feature_builder.py:86-94` — formula parity holds, **magnitude parity does not**. Training (`contention_v2`): per-dataset p90 depth **26–70**, cap **binds 0/200** → dim7 ∈ ~[0,2], dim13 ~1.6. Live (`sparse_p35` s42 GNN): `averageExecutionTime` **0.035s** with `averageQueueTime` **503s** ⇒ chosen-platform depth ≈ **14.5k tasks**; cap always binds → dim7 ≈ **145** (~100× OOD), dim13 ≈ **580** (~360× OOD, **no cap at all**). No clipping ⇒ **silent** shift, not a loud failure. Behavioural proof from `*.decode_stats.json` `chosen_queue_vs_min` (n=117): **GNN** median **+411** / p95 **+6399** extra queued tasks vs the shortest available platform (worst `client_heavy_p50` 748 / `sparse_p35` p95 18373), **MLP** median **+7** / p95 **+1232** — matches MLP's better live p50 (0.69s vs 68s). Knative wins because its rule is scale-free. The cap is the only train/serve asymmetry and provably a **no-op on the training corpus**, so removing it is parity-preserving; dim13 has no normalizer and needs one (⇒ recache + retrain).

## ⛔ contention_v4_deepq pilot FAILED the coupling gate (2026-08-13)

Deep pre-seeded queues **destroy** coupling instead of creating it. 27 ds on datalab (job **663969**, CPU-amd, 60–290s/ds, best_rtt 13–116s vs `contention_v2` 1.8–16.4s): greedy == optimum **27/27**, regret **0.00%**, coupled>1% **0.0%** (vs `contention_v2` **7.1%**); best **colliding** plan is **−6.3%** *better* than best spread ⇒ the grid would teach **more** piling. Mechanism: depths sampled **176–648** per platform, so cross-platform depth differences swamp the marginal cost of one co-located task (v2: 26–55). Coupling needs marginal self-interference ≳ inter-platform spread. Corollary: **`contention_v2` labels already punish piling** — pile penalty (all tasks on one platform vs best spread) mean **+333%** / median **+455%**, and 0/900 have any cold start or pull. Labels are not the defect. Corpus kept at `simulation_data/gnn_datasets_4tasks_contention_v4_pilot/` (9.2M, JSONL intact) as the negative record; **do not train on it**. Probe: `/tmp/colocation_cost.py`.

## ⚠ per-cell SLURM execution skips whole-sweep provenance (2026-08-13)

Sweep runners write `manifest.json` (phase 0) and `compare.json` (last phase), but the datalab path submits **one SLURM task per cell** via `run_sealed_holdout_one.sh`, so neither phase ever runs and only `results/` is rsynced back — the sweep dir arrives with no physics/commit/ckpt-hash record and no compare. Fixed: manifest writer extracted to `scripts_cosim/important/write_sweep_manifest.py` (CLI, fail-loud on missing model/workload/empty seeds, refuses silent overwrite without `--force`), callable from the per-cell path; `run_contention_v2_873_sealed_holdout.sh` now calls it with overridable `MANIFEST_KIND`/`MANIFEST_NOTE`/`MANIFEST_EXTRA_JSON` so a re-baseline cannot inherit the original's identity. Old inline heredoc also had dead code (built `configs` wrong, then overwrote it with a hardcoded list). **After any datalab sweep: write the manifest + run compare locally.**

## ⚠ warmth_physics CONFOUND (2026-08-13)

Live sim default is **`platform_reuse_v1`**, not `node_disk_v2`. `HEROSIM_WARMTH_PHYSICS` unset on the `regime_b_transfer_gate3` cell → total_rtt **1.019e9** vs **9.30M** with explicit `node_disk_v2` (**~110×**), because v1 serializes cold pulls per node when the platform has a `previous_task`. **Every launcher must export `HEROSIM_WARMTH_PHYSICS` explicitly.** Now enforced: `build_run_provenance()` in `executesimulation.py` writes `run_provenance` (physics + source `config|env|default`, defer-cold-init, batch/decode/feature-layout/model-path env) into every result JSON and prints a banner; `HEROSIM_REQUIRE_EXPLICIT_PHYSICS=1` fails loud on implicit physics (`ImplicitWarmthPhysicsError`). Helpers `describe_warmth_physics` / `require_explicit_warmth_physics` in `src/placement/warmth.py`. **All 26 pre-2026-08-13 sweeps have `warmth_physics: null`** — physics is not recoverable from those JSONs, only from the launcher script; do not mix them with post-provenance sweeps.

## ⚠ pull-serialization re-baseline `25732cf` (2026-08-12)

Commit moved the image-pull timeout **inside** the node FilterStore hold in `KnativeAutoscaler.initialize_replica`. Before: N co-located cold pulls run **parallel**; after: **serialize per node**. Identical cell (`sparse_p25` / knative / seed 42 / `node_disk_v2` / workload-125-225): total_rtt **6.90M → 17.21M (2.50×)**, carried entirely by `averageQueueTime` **12.23s → 30.57s**, not by per-task pull time. Intentional for Regime B FilterStore realism, but **pre-2026-08-12 Regime A sweeps are NOT comparable to HEAD** (incl. `contention_v2_live_gate_20260615`, `contention_v2_873_v5.5_sealed_holdout_20260806`). Historical dirs kept as the pre-serialization record; re-baseline lands in `…_sealed_holdout_rebaseline_20260813/`.

## An INDETERMINATE model-class contrast on a small corpus is not evidence of redundancy

**Measured twice, most sharply 2026-09-11 (`peer_affinity_v1` T1 → T1b).** The same contrast, same
code, same held-out block, differing only in training-corpus size and seed count:

| training datasets | seeds | `gnn` vs `mpoff` (only `PeerConv` differs) |
|---|---|---|
| 136 | 8 | +2.08 pp, p = 0.15 — INDETERMINATE, written up as "message passing is not the lever" |
| 482 | 16 | **+5.14 pp, p = 0.001, 13/16 seeds — GNN-NEEDED** |

The first read was measuring the corpus, not the architecture. The tell is the **per-arm gain from
more data**: the higher-capacity arm gained −5.09 pp where its ablated twin gained −2.43 pp and a
pointwise arm −3.24 pp. A model class whose extra capacity is genuinely redundant does not improve
faster than its ablation when you feed it more.

**The rule.** Before writing "architecture X is redundant" off a null or indeterminate contrast:
(a) vary the corpus size and report each arm's delta, not just the contrast; (b) check the power —
at n = 8 with a 3 pp per-seed sd, a 2 pp effect is invisible, and "we could not resolve it" is the
honest reading; (c) state which checkpoint every arm was read at, because an arm with more capacity
overfits harder and a last-epoch read will flatter its ablation (`route_b_v1` Phase 2, and again in
`peer_affinity_v1` T1b where the two arms **tie** at last epoch and separate at the selected one).

Related: corpus was also the dominant lever in `link_mp_v1` and `reliability_matched_v1` — but there
it erased an apparent GNN edge, and here it revealed one. The lesson is the same either way: **name
both arms' corpus before naming the model class.**

## Screen the label's regime before building the corpus (`peer_affinity_warm_v1`, 2026-09-13)

A corpus is a week of cluster time; the question "can a pointwise scorer recover this label?" is one
statistic on 100 datasets. `peer_affinity_warm_v1` registered that screen (W0.b, `mean_tied` regret
at the training rung) with its NO-GO bar *before* any corpus, and the served regime answered it in
a day: 0.0 % median, 0/100 above 2 %. The W1 corpus, 32 training seeds and a live gate never ran.
Two corollaries from the same day: (a) a co-sim label whose horizon is a long queue drain has a
per-plan cost set by the simulator's periodic processes, not by the physics — stretch the
autoscaler tick (`COSIM_AUTOSCALER_RECONCILE_INTERVAL`, label-invariant) or the sweep OOMs; and
(b) any candidate subsampling for a capped rung must check cap feasibility of the *draw* with the
scorer's own rule (`cap_feasible`), or the cache refuses the dataset after the sweep is paid for.
And never trust a non-empty `placements.jsonl`: `placement_metadata.json.sweep_complete` is the
completeness fact, and a driver that records `truncated` and continues is a silent failure.


## Read a finished run's curves against their chance floor (2026-09-14/15)

W&B draws every logged scalar at the same size, so a metric that *cannot move* under the running
objective and one that carries the result look identical. Audit of `peer_affinity_warm_v1` W1 run
`xnjb91ic`: of ~34 per-step scalars, **25 were structurally dead or constant**, and three of the
nine live ones read as findings and were not — `task_acc` "starting at 43 %" is chance on a corpus
averaging 2.79 candidates per task, `ce` "climbing to 11" is the plan NLL falling back *through*
the 10.26 uniform-scorer floor, and `regret_greedy` in raw seconds means nothing without the
random-plan regret beside it. ⇒ **A `task_acc`, `acc` or `ce` quoted without its floor is not a
result.** `scripts_cosim/read_training_curves.py <wandb/run-dir>` classifies dead / constant /
guard / live, prints each live curve against the floor the run recorded, names the selected epoch
and how much training ran after it, and flags memorisation. It reads the on-disk `*.wandb`
transaction log, so it works on an unsynced or killed run, and it can be piped to a machine that
does not have it (`cat tool.py | ssh host 'python3 - <run-dir>'`).

Worked example, `drainable_objective_v1` job 766897 (2026-09-15): the 32 shaped-label runs looked
broken on the charts — `val/ce` rising after epoch ~60, `val/acc` ≈ 10 %, regret swinging
52 ↔ 290 s between adjacent epochs. Against the floors they are healthy and *better behaved than
their own control*: chance CE is 9.89 and the runs end at 7.6–7.8 while the T1b control ends at
**10.7, above chance**; chance graph accuracy is 8.0e-5, so 10 % is ~1,300× chance against the
control's 4.2 % peak. The per-epoch regret swing is decode noise present in both. The one real
observation the floors leave standing is a clean arm separation, and the one trap they expose is
that the arm with the **higher** `task_acc` (0.85 vs 0.80) has the **worse** plan regret — score
the plan, never the per-task argmax.

Corollary on selection: the tool names the selected epoch because "the curve got worse" is usually
irrelevant. These runs select at epoch 16–34 of 300 (88–91 % of training runs after it) and the
T1b control at 70 (76 % after). That is wasted compute, not a defect — but it means **the last
epoch is not the run**, and a two-arm contrast read at last epoch can invert against the same
contrast at the selected checkpoint (measured in `route_b_v1` Phase 2, and in `peer_affinity_v1`
T1b in the other direction).

## An offline score can say "worse than it was" without being able to say "worse than that one" (2026-09-15)

`offline_live_transfer_v1` measured both readings of the same number on the same checkpoints
and they came out opposite, which is why a decade of "it wins offline and loses live" in this
record read as a paradox.

**Across runs it carries nothing.** 96 checkpoints from three families, each with both its
selected-checkpoint held-out score and its own live latency at a matched cell: Spearman
**−0.030**, p = 0.772, at ~80 % power for ρ = 0.30. Not underpowered — measured zero. The
same 96 points read **−0.525, p < 0.0001** when pooled *raw*, and that number is the famous
"offline/live reversal": it is the gap between two arm averages, not a relationship. Two
group means, or three corpus rungs, cannot be told apart from noise-plus-offset. **Two
registered lineages (`serving_gap_v1`, `serving_gap_v2`) closed NO-GO hunting a mechanism for
a pattern that needs none.**

**Within a run it carries a lot.** Gating each run's offline-selected checkpoint against its
own last epoch — weights that already exist, no retraining — the selected one is **13.10 %**
(`gnn`) and **16.42 %** (`mpoff`) faster live, p = 0.028 and p = 0.0005.

⇒ **Use an offline curve to choose an epoch. Never use an offline number to choose between
runs, seeds, arms or model classes** without showing, on that corpus, that it ranks
checkpoints by live outcome. The two uses feel like one number and are not.

Three practical corollaries:

* **The cheap check is cheap.** Any program that gates N checkpoints already owns N paired
  (offline, live) points. Correlate them **z-scored within each (family, arm) cell** — raw
  pooling re-imports the group offsets that make the paradox — before trusting an offline
  screen to order anything.
* **A positive control belongs in the read.** Ours correlated live latency against live queue
  time and returned ρ = 1.0000 in all three families, which is what makes "we measured zero"
  different from "our pipeline is broken".
* **Two group means are not a trend.** When a pattern is built from two or three aggregates,
  price in that a constant offset reproduces it exactly, and go find the per-unit version of
  the same question before registering a lineage to explain it.

## A single-seed live number measures whether that seed hit an excursion (2026-09-15)

`offline_live_transfer_v1` R3. Two checkpoints of one recipe — same corpus, same split, same
lr, **differing only in the training seed** — serve the same 50,000-task trace at **43.99 s and
74.76 s**. Decile the trace by arrival and the difference is not a drift:

* deciles 1–3 carry **0.2 %** of the gap — the seeds are indistinguishable at the start;
* **decile 6 alone carries 50.5 %**, the worse seed's mean queue hitting **185.1 s against the
  better seed's 28.8 s**, a 6.4× excursion over its own baseline;
* deciles 7–10 **recover**, decile 8 going negative.

**A 1.7× headline difference — larger than any architecture effect this program has chased —
is one traffic jam's worth of queue spread over 50,000 tasks.** ⇒ **Never quote a single-seed
live latency as a property of a checkpoint**, and size seed counts against the excursion, not
against the effect you hope to see. It also explains why no offline statistic predicts live
outcome here (R1, R2): the excursion belongs to the interaction between a policy and a trace,
not to the model, so nothing measured on captured states can see it coming.

Two corollaries for reading any such gap:

* **Decile it before naming a mechanism.** "Compounds" and "one excursion that recovers" clear
  the same threshold and are different physics; a runaway closed loop does not come back.
* **A shape summary is not a shape.** This read's own RISING/FLAT/FALLING descriptor called a
  single-decile spike with full recovery "RISING". Print the curve.

## The learned arms beat reactive early and lose it later — the first replicated positive (2026-09-15)

`serving_stability_v1` S1. Over the first fifth of a 50,000-task drainable trace, the learned
arms carry **3.4–4.1 s less queue than reactive Knative**, and it holds on **3 of 3 cells and
every one of 91 learned arms** (43/43 `gnn`, 48/48 `mpoff`) across two topologies that had
never been looked at. The registered expectation was **UNCERTAIN** — this was the bar most
likely to evaporate and it did the opposite.

**Why it matters more than any offline number in this record:** every previous "the learned
arm is worse" reading was a whole-trace aggregate, and a whole-trace aggregate cannot tell
*never had an advantage* from *had one and lost it*. These models are not bad at placement.
They are good at it and something destroys that later in the trace. ⇒ **Decompose a live gate
by trace position before concluding an arm is worse at its job.** The same table also shows
the loss is 2.4–3.9× concentrated rather than uniform.

**What it is not.** Stabilising their queues does not recover it: a decode-time guardrail that
bound on 84–90 % of decisions, with zero deadlocks, still lost to reactive on 0/3 cells, and
**helped +36 % on one cell while hurting 13.6 % on the cell where the arms were least stable**.
So the advantage is real, its destruction is real, and the queue-runaway explanation is
refuted. The open question — what actually destroys it — is now the strongest lead in the
record, and it starts from a measured phenomenon rather than a hunch.

Corollary on aggregates, for the third time this day: across the three cells, instability and
the gap to reactive line up almost perfectly. Within cells, across seeds, they do not
(pooled-z ρ = +0.088, p = 0.40, two cells significant in **opposite** directions). Three
aggregates are still not a trend.

## Size a rate ladder from the drain achieved in the regime being run (2026-09-16)

`cluster_scale_v1` S0. The registration used 2.83 tasks/s on 6 servers to size the scaled
rungs — but that drain comes from the **landed overloaded gate**, where the autoscaler had
long since built every replica it was ever going to build. From a cold start the realised
per-server drain is **~0.077 tasks/s**, about **6× lower**, so holding ρ ≈ 0.16 at 6.139
arrivals/s needed ~480 servers, not 80. The scaled rungs were 25× and 49× the baseline's
reactive queue instead of the registered [0.33, 3.0]× band.

⇒ **Measure the drain from the regime's own cold-start ramp, not from an overloaded steady
state where the cluster has already built out.** The two differ by the autoscaler's lifetime
of work, and a capacity model built from the final state predicts a cluster that can drain
orders of magnitude more than the cold start actually sees.

## An arm name must carry every axis it varies (2026-09-16)

`cluster_scale_v1` S0.b, job 769390. The arm name was `${CELL}__${KIND}` with no rung tag,
so all three rungs resolved to the same summary path and the f300 rung exited on the
idempotence guard — `"exists, not re-running"`, 1 s per task — having measured nothing. The
guard did exactly its job against the wrong key, and nothing failed loud.

⇒ **Every axis the gate varies must appear in the arm name.** If a gate varies workload,
topology seed, and policy, the arm name carries all three. An idempotence guard tests "did
this arm already run", and if two arms share a name the second one silently inherits the
first's file.

## The out-of-support prior is a prior, not a measurement — and a bar naming a failure class is read by cause (2026-09-16)

`partial_state_v3`. Three times this record had measured a served model failing when its
inputs left the corpus's support (`xavierGpu`, the queue column, S0.d's candidate counts),
so the registered expectation for serving a 6-server checkpoint on 12 / 24 / 80 servers was
**DEGRADES**. Measured live on 4 topology seeds per rung: the arms' standing relative to
reactive Knative **improves** with cluster size, to −28.7 % / −46.9 % at 80 servers from
9.6× the corpus's candidates per task. The prior was wrong here and it would have been
cheaper to believe it — one afternoon of cluster time bought the first whole-trace live win
in the program. ⇒ **"It is out of distribution" is a reason to register a bar, not to skip
the gate.** The size-free encoding made the question *askable*; only the live gate answered it.

Two things that came with it, both cheap and both general. **Generalisation across cluster
size can only be read live** — co-simulation brute-forces every placement, so labels at 24
servers (14¹⁰ plans) do not exist; the live gate is the instrument, at ~2 min per arm.
And **a bar that names a failure class (a raise, a refusal) must be read by cause, not by
count**: the first cut of the read printed the class it was written for on two arms that
died of a tail livelock at the memory cap — see `docs/gates/gate-tools.md`, 2026-09-16.
