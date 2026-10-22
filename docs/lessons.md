# Learned lessons — the transferable rules

> **Status:** `REFERENCE` &nbsp;·&nbsp; **Index:** [LINEAGES.md](../LINEAGES.md)
>
> Transferable rules from the experiment record, newest first: 51 bullets, then one `##` section per rule.
> Status entries and artifact inventories live in [`lessons-archive/`](lessons-archive/) (last moved 2026-10-12). A lineage node in
> [`docs/lineages/`](lineages/) tells you what one investigation found; this file tells you
> what generalises past it. Moved verbatim from `memory/memory.md` §3 (that file retired 2026-08-27; see git history).
>
> **Adding a rule:** only when it would have changed a decision *before* it was learned.
> A rule that just restates a lineage outcome belongs in that lineage's node, not here.

## Rules, 2026-09-03 → 2026-09-17 (bullet era)

- **A CORPUS-MATCHED WIN IS NOT A WIN AGAINST THE BEST ARM YOU HAVE (2026-09-17):** ⇒ **Corpus-matching removes a confound in one direction and can manufacture a win in the other.** A matched contrast answers "does this component help *on this corpus*"; it does not answer "is this the best scheduler we have". Report both, always: the matched paired test as registered, and the winner against the best-performing arm in the program at that operating point, named with its corpus. If the second one is not in the node, the first will be quoted as if it were. The 2026-09-03 rule and this one are the same instrument read in both directions, and neither is sufficient alone. Evidence and figures: [peer_only_v1](lineages/archive/peer_only_v1.md).

- **⛔ A SEPARATION ON THREE CELLS IS A COINCIDENCE GENERATOR — AND THE COST OF CHECKING IT IS ONE AFTERNOON (2026-09-16):** `scheduler_residence_v1` R1 found the one cell of three without a queue blow-up separated from the other two **with no overlap on two statistics at once** — every client can reach ≥ 2 servers against a client that can reach exactly 1 on both bad cells, and a clients-per-server imbalance of 3 against 8 and 7 — where the *mean* fan-out separated nothing (3.45 / 3.05 / 3.50). The cells' configs are byte-identical in **149 of 150** flattened fields; only the topology seed differs, so there was no other candidate. It fired a bar registered with a NEGATIVE expectation, it had a concrete mechanism (a client with one reachable server cannot be spread; a server every client can reach is a queue sink), and it was **wrong**. On 15 independent draws from the same generator: **ρ = +0.041, p = 0.885** against a bar of \|ρ\| ≥ 0.60, wrong sign, second statistic +0.029, pointwise control +0.020. Not a weak effect — no effect. ⇒ **With three units, "strictly outside the range of the other two" is the expected outcome for roughly a third of candidate statistics by chance, and testing two of them at once makes it likelier, not less.** This is the third time in this program: `serving_stability_v1`'s stability mechanism (3 aggregates lined up; pooled-z within cells ρ = +0.088, two cells significant in *opposite* directions), a 3-aggregate over-read retracted the same week, and now this. **Before a three-cell ordering is written into a node as a mechanism, price the replication** — here it was 108 arms, ~90 minutes, and it converted a false lead into a closed question. ⚠ And mint the replication units from the *same generator* with the selection made on the **independent** variable before any of them is served: choosing units after seeing the outcome is choosing the answer.

- **⛔ A RATIO CONTROL IS DEGENERATE WHEN THE PRIMARY IS NULL — GIVE IT A FLOOR OR IT WILL REPORT "CONFOUNDED" FOR "NOTHING THERE" (2026-09-16):** ⇒ **Any control expressed as a fraction of the primary's magnitude needs an absolute floor on that magnitude, below which it reports NOT-APPLICABLE rather than CONFOUNDED** — otherwise the one verdict that means "a real effect was obscured" gets printed for the case where there was no effect to obscure, and a reader who trusts the verdict line draws the opposite conclusion from the data. ⚠ The fix is not to move the ratio: a null and a confound are different findings and the instrument must be able to say which, so the floor is a third branch, not a retuning. Evidence and figures: [drainable_objective_v1](lineages/archive/drainable_objective_v1.md), [scheduler_residence_v1](lineages/archive/scheduler_residence_v1.md), [peer_affinity_v1](lineages/peer_affinity_v1.md).

- **⛔ AN OUT-OF-RANGE INPUT CAN BE REAL, LARGE, AND ORDERED WITH THE OUTCOME, AND STILL NOT BE CAUSAL — CLAMP IT AND MEASURE (2026-09-15):** ⇒ **A feature that is out of its trained range is a defect worth fixing and is not thereby an explanation; the clamp is cheap, so run it before building a story on the correlation.** The trap is specific: an out-of-range input and a bad outcome share a common cause — load — so they order together on every cell without either causing the other. Here the deep queues put the column out of range *and* cost the latency, and removing the first did not touch the second. ⚠ Note what the null does **not** cover: the divisor was measured at 1.0 in **100 % of batches on every arm**, so the *compression* failure mode (an inflated divisor squeezing real differences to nothing) never occurred and remains untested rather than refuted. Evidence and figures: [queue_range_v1](lineages/archive/queue_range_v1.md).

- **⛔ DECOMPOSE A LIVE DEFICIT BY TERM AS WELL AS BY TRACE POSITION — THE CELL WHERE THE POLICY WORKS IS THE ONE THAT NAMES THE TAX (2026-09-15):** ⇒ **Run the per-term table across a family that contains at least one cell where the policy does its job; a constant tax hides inside every aggregate and only surfaces where nothing else is wrong.** The corollary for this program: "the learned arms lose to reactive" is at least two different failures — a flat batching tax and a queue blow-up on some cells — and a single intervention was never going to close both. Evidence and figures: [queue_range_v1](lineages/archive/queue_range_v1.md), [serving_stability_v1](lineages/archive/serving_stability_v1.md).

- **⛔ A CORPUS CAPTURED BEFORE THE AUTOSCALER HAS BUILT THE CLUSTER DOES NOT DESCRIBE THE CLUSTER'S ACTION SPACE — AND A SERVING GAP CAN BE A PROPERTY OF THE RESOURCE MIX, NOT OF THE POLICIES (2026-09-13):** ⇒ **Enumerate the candidate set the corpus actually contains, compare it to the one the live cluster offers, and vary the resource mix before attributing a serving gap to a model class.** Two numbers catch it in minutes and neither was ever computed: the set of platform types appearing as candidates in the cache, and the distribution of `node_caps` live against the cache's maximum. ⚠ And note what this does *not* license: the fix is not "serve only what you trained on", because the restriction removes real capacity — the clean experiment keeps the platform and brings its demand into the corpus's range, so the cap is not inflated while capacity is unchanged. Evidence and figures: [peer_affinity_v1](lineages/peer_affinity_v1.md).

- **⛔ A MODULE THAT CONFIGURES LOGGING AT IMPORT TIME DISABLES EVERY LATER CONFIGURATION, AND THE BILL ARRIVES AS DISK EXHAUSTION (2026-09-13):** `logging.basicConfig` is a no-op once the root logger has a handler. `prepare_graphs_cache.py` calls it at **module import**, and loading a prefix-conditioned checkpoint imports that module — so `start_simulation`'s request for "`ERROR` only, on stdout" was discarded in silence and every per-event `logging.info` in the simulator went to **stderr at INFO**: ~390 MB of log for one 450,729-task arm. A 102-arm gate wrote ~40 GB, took the account to 251 G of a 250 GiB quota, and arms began dying with **exit 1 and an empty traceback, because the traceback could not be written either**. Nothing in the job's own output named the cause: `sacct` showed `FAILED 1:0` with `MaxRSS` 4.5 GB against a 24 GB request, and the `.err` ended mid-run on an ordinary event line. ⇒ **Three rules.** (1) A library must never call `basicConfig` at import; an application's own configuration must use `force=True` if it cannot control what was imported first. (2) Before launching an array, multiply `arms x per-arm log bytes` and compare it to the quota headroom — a gate that writes more than it measures will kill itself, and this is now the second incident where an exhausted quota presented as an inexplicable job death (the first time the logs were 0 bytes, this time 390 MB each). (3) When a long job dies with no traceback, check `getfattr -n ceph.quota.max_bytes ~` against `du -sh ~` **before** reading anything else — `df` and `sacct` both mislead, and an unwritable traceback is indistinguishable from a silent kill. ⚠ And discard the partial gate: 38 of 102 arms had landed, each individually valid, and re-running only the missing 64 would have split one gate across two commits — the cheaper-looking option is the one that produces a number nobody can defend.

- **⛔ A MECHANISM THAT REPRODUCES THE HEADLINE EFFECT IS NOT THE CAUSE — RANK IT ACROSS THE CORPORA THE EFFECT IS GRADED ON (2026-09-13):** ⇒ **When the target is an ordered or graded effect, a candidate mechanism must reproduce the ORDER, not the sign on one rung — register that prediction before running it, and run every rung.** The trap is that a single-rung reproduction is maximally convincing exactly when it is least informative: a small edge is the one most easily erased by any common degradation, so the rung that flips first is usually just the rung with the smallest effect. ⚠ And keep the two results apart when writing up: the defect measurement still stands on its own (at live queue magnitudes the registered MP edge is not significant on **any** corpus, so that reading is regime-dependent) — it just is not the explanation it was registered as. Evidence and figures: [peer_affinity_v1](lineages/peer_affinity_v1.md).

- **⛔ A SERVING-PARITY CHECK RUN ON CO-SIM STATES CANNOT SEE AN OUT-OF-MANIFOLD INPUT — COMPARE THE SERVED FEATURE RANGE AGAINST THE TRAINED ONE, SEPARATELY (2026-09-12):** ⇒ **Ship two checks, not one: (a) the paths agree on states where the answer is known, and (b) the distribution of every served feature, live, sits inside the range the cache contains.** (b) costs one traced run and a histogram; here it was never run, and `docs/adr/0002` had already written down what it would have found. Corollary: a contract that is "kept bit-exact for old checkpoints" must be **refused for new corpora** in code, not in prose — this corpus was cached under `legacy_v0` two weeks after the ADR said new training uses `scale_invariant_v1`. Evidence and figures: [peer_affinity_v1](lineages/peer_affinity_v1.md).

- **⛔ WHEN A POLICY IS TRAINED TO MINIMISE A COST INDEXED ON ONE AXIS, SPREADING ITS WORK ALONG THAT SAME AXIS DESTROYS THE SAVING — SPREAD ALONG A DIFFERENT ONE (2026-09-11):** ⇒ **Before adding a spreading constraint, ask which index the objective is written over: spreading along that index cancels the objective, spreading along an orthogonal one is nearly free.** Corollary for this simulator: `queueTime` (= `arrived − scheduled`) is not an independent cost, it is the accumulation of other tasks' service, so cutting mean service and cutting concurrency pull the same number in opposite directions and a total-RTT read alone cannot tell them apart — compute `throughput × service` (tasks in service at once) and read it beside the total. Evidence and figures: [peer_affinity_v1](lineages/peer_affinity_v1.md).

- **⛔ AN ABORT ON A MEMORIZATION STATISTIC IS ONLY VALID IF BOTH ARMS ARE AT THEIR FIT CEILING — VERIFY CONVERGENCE BEFORE READING IT (2026-09-03):** ⇒ **A memorization comparison needs a convergence check on every arm before it is read — train loss still descending is not a fit ceiling** — and the cheap insurance is to save last-epoch weights next to the val-selected ones (`NEAR_RTT_SAVE_FINAL=1`), because a val-selected checkpoint cannot answer "can this arm fit its training data at all?" once validation has plateaued. ⚠ Note what did *not* change: on held-out parents the pointwise arms still win (MP-OFF 11.1% < MLP+prefix 15.4% < GNN 18.6–26.2%), so the corrected reading is "fit ceiling favours the GNN, generalization favours pointwise" — an abort on the wrong statistic hid a more interesting result rather than a favourable one. Evidence and figures: [route_b_v1](lineages/archive/route_b_v1.md).

- **⛔ INHERITING A STATISTIC DOES NOT INHERIT ITS POWER — RE-DERIVE n FOR THE EFFECT THE NEW EXPERIMENT EXPECTS (2026-09-04):** ⇒ **When reusing a signed design on a new comparison, re-derive the sample size from the effect the new comparison predicts, and say so in the registration.** Copying n across is the same defect as tuning at one seed (2026-09-03), reached from the opposite direction: there the instrument was too noisy for the grid, here it was too small for the effect. ⚠ And note what the null still bought: the *burden* comparison (107 → 14 cells) is descriptive, needs no α, and is the more publishable half of the result. Evidence and figures: [reliability_matched_v1](lineages/archive/reliability_matched_v1.md).

- **⛔ A MODEL-CLASS COMPARISON IS ONLY AS GOOD AS ITS WORST-MATCHED ARM — CHECK THE TRAINING CORPUS OF EVERY BASELINE BEFORE QUOTING A GAP (2026-09-03):** ⇒ Before any "A beats B" number leaves a node, list each arm's training cache next to it; if they differ, the comparison measures the caches. This is the model-side twin of the separability result — on a pointwise-separable target the pointwise model is correctly specified, so it reaches the same ceiling once it sees the same data — and it was reachable for an hour of CPU at any point in the last four days. Evidence and figures: [link_mp_v1](lineages/archive/link_mp_v1.md).

- **⛔ A TUNING STAGE MUST BE POWERED LIKE THE COMPARISON IT FEEDS (2026-09-03):** `objective_pivot_v1` Phase 3 selected a learning rate from a grid run at **one seed per configuration**, then discovered at the pilot that the across-run sd was 4.9-5.8 pp. The winner had won on a single +10.6% draw that later turned out to sit near the top of a distribution centred on zero; the +8.5% it showed on held-out dev cells was the same draw evaluated twice, not independent confirmation. Every downstream run then spent its budget at a hyperparameter chosen by noise. **Before selecting anything from a grid, measure the seed variance of the selection statistic — if it exceeds the gaps between grid rungs, the grid is measuring draws.** A dev-cell evaluation does NOT rescue this: it controls for memorisation, not for the training draw, and the two failure modes look identical in a single run.

- **⛔ CHECK THE INSTRUMENT CAN EXECUTE THE n BEFORE SIGNING IT (2026-09-03):** Phase 3's Amendment E registered n = 120 with an exact Wilcoxon, against an analyzer that raised above n = 22 because 2^n enumeration is infeasible. The guard was correct and fired correctly — **at the analysis step, after all 615 gate episodes had been computed.** The artifacts survived, so only the analysis was re-run, but the failure mode is general: a registration fixes a number, a test and an alpha, and nobody runs the analysis path on synthetic data at the registered n first. **Dry-run the full analysis on fake data shaped like the real thing before signing the number.** It costs seconds and it is the only check that catches a registration its own tooling cannot carry out.

## Rules, 2026-08-13 → 2026-09-01 (bullet era)

- **⛔ ANY TRAJECTORY- OR HORIZON-RETURN LABEL NEEDS A RANK-STABILITY CONTROL, AND IT IS NOT OPTIONAL (2026-09-01):** A label built by letting a simulator run forward from a decision measures *the decision plus everything the simulator did afterwards*, and in a discrete-event system those are separable only by test. P3 pre-registered co-primaries, an n, a horizon calibration, and two repair controls — and **passed all of them on pure noise**: 21.7% of snapshots above the 2% regret bar (binomial p = 1.5e-33 vs the t=0 base rate), node-count repair 0.009 and link repair 0.016 (both bars < 0.5), additive R² collapsed 0.988 → 0.049. Every number said 'new mechanism that escapes both known degeneracies'. The unregistered control that settled it took **one extra sweep at two shorter horizons**: re-rank the same 256 plans at h=2 and h=5 and correlate against h=10 — median ρ = −0.027 / +0.004, the h=10 optimum at rank 120/256 against 128/256 for chance. ⇒ **Before reading any horizon/return label, verify the ranking survives a change of horizon length.** If it does not, the label is not a stable property of the (state, action) pair, so it cannot transfer to another trace, seed or horizon — and no repair control will catch this, because noise has no node-count or link-load shape either. ⚠ Note which way the two facts point: the repair controls reading ~0 is what made the result look strongest, and it was the *same* fact as the chaos. A mechanism that escapes every known degeneracy deserves a suspicion budget, not only a celebration. ⚠ And the finding does not transfer to closed-loop RL, which estimates an *expected* return over episodes: there the same chaos is per-episode variance to average through (measured: ~1.5% mean, 3.2% p90 of total RTT), which is what a paired-seed common-random-numbers baseline exists to cancel — size the episode budget from it.

- **⛔ MATCH THE TRAINING CORPUS'S NETWORK FABRIC TO THE SERVING FABRIC BEFORE TOUCHING ARCHITECTURE (2026-08-31):** ⇒ **Before any architecture experiment, check that the physics the new inputs describe actually moves the training labels** (is the resource binding? does ablating the feature change the label?): a fabric the labels never price is a fabric the model provably cannot learn, and months of architecture iterations sat on top of exactly that. Full record in `docs/lineages/archive/link_mp_v1.md`. Evidence and figures: [link_mp_v1](lineages/archive/link_mp_v1.md).

- **⛔ TURNING A SHARING FLAG OFF IS NOT A ONE-VARIABLE ABLATION — READ THE ALLOCATOR FIRST (2026-08-28):** The obvious diagnosis for the amended H2's S0 VOID was to re-run the control with `replica_overlap` **off**, since overlap is the only structural difference from the rungs whose controls passed. "Same grid, one flag" is what it looks like in the preset diff. It is not what it is. Overlap is what makes the allocator's FCFS walk *harmless*: with sharing on, all four task types draw the same pool (8 or 9, every type on both hosting nodes); with it off, `assigned_platforms` becomes exclusive and early types eat the pool. **Measured on a 24-dataset probe covering both arms:** `per_server=4` gives `(dnn1,dnn2,rf,cnn)` = **(8, 4, 1, 1)** with three of four types confined to one node, and `per_server=5` gives **(9, 4, 0, 1)** — `rf` starves to **zero replicas**, so warmup auto-resolve finds no warm replica and 12/24 datasets die as `System state capture FAILED`. One flag moved pool size, per-type asymmetry and node confinement together. ⇒ **Before proposing an ablation on a grid key, trace what else consumes it in the generator** — a flag that gates an *exclusivity check* changes every downstream allocation, not just the thing it is named after. ⚠ And check the ablation is still *informative* where it does run: the surviving arm's sweeps are **32 rows** against H2's 1,680/3,024, i.e. the H0/H1 regime (16/64) that already **passes** S0 — so a pass there would have re-measured the rungs that already work, not implicated overlap. An ablation can be both runnable and incapable of answering the question. The probe cost **5 seconds**; the corpus it replaced would have cost ~30 minutes and produced an uninterpretable number.

- **⛔ PROBE THE GATE THAT VOIDS THE BARS, NOT ONLY THE BARS (2026-08-28):** An amendment that changes a grid gets probed on the bars it is trying to rescue, because those are the numbers in dispute. AMENDMENT 3 did exactly that: it probed the proposed `per_server` 4/5 pair at 204/204 and **all four of S1–S4 passed**, which is what carried the sign-off. Nobody probed **S0**, the VOID gate that decides whether any of those four may be read — and when the registered rung's paired separable control was finally generated, S0 **failed by 24×** (`optimistic` 0.4853 against a `≤ 0.02` bar, 59.5% max regret, independent verifier agreeing to 1e-9 on all 612 cells). Every bar the probe measured is now unreadable. The control cost **31.5 minutes** to generate; it would have stopped the amendment before sign-off. ⇒ **Before signing an amendment, ask which gate can void the numbers the probe is about to produce, and probe that one too.** A probe covering only the disputed bars is measuring the half of the outcome that cannot kill it. ⚠ And note the asymmetry that hid it: S1–S4 are read on Arm S, which the probe generated; S0 is read on a *paired control* the probe never had a reason to build.
- **⛔ A SEPARABILITY CONTROL THAT READS EXACTLY ZERO UNCONSTRAINED CAN STILL FAIL BADLY ONCE THE CONSTRAINT BINDS (2026-08-28):** `score_route_b_contention.py` asserts that on separable physics `r_exact_band["optimistic"] == 0` **exactly, under any feasibility restriction** — so an unconstrained zero reads like proof the control arm is separable. It is not. The amended H2 control reads `optimistic` **0.0000, max 0.00%** at α=None and **0.4853, max 59.5%** at the registered primary α=2.0, on the same 204 datasets, verified to 1e-9 by an independent solver. The failure sits entirely in datasets where the *cap* excludes the componentwise-argmin plan; where uniqueness alone excludes it (87 of 204 at α=None, from `replica_overlap`) the surrogate still recovers the true optimum in every one. ⇒ **Read a control at every α the rung will be read at.** An unconstrained zero is necessary, not sufficient, and it is the reading most likely to be taken as sufficient because it is the one the theorem names. ⚠ **And do not attribute a control's degradation to constraint tightness without a passing control at equal or greater tightness** — the obvious story here was "H2 squeezes harder", and it is false: H1's control **passes** at `cw_infeas` 0.963 with 3.3 feasible rows while H2's **fails** at a looser 0.877 with 1,677.
- **⛔ UNDER A UNIQUENESS CONSTRAINT, THE CANDIDATE POOL MUST HOLD THE TASK COUNT — CHECK IT PER ARM AT REGISTRATION (2026-08-27):** A brute-force sweep that requires **globally distinct** resources across tasks (`generate_brute_force_placement_combinations`) has zero valid plans the instant `|pool| < |tasks|`, no matter how many candidates each individual task has. The route_b pivot registered H3 as "H2 + `dag_instances=2`" — doubling the joint decision from 4 tasks to 8 — while `replica_overlap` collapses every task type onto **one** pool of `per_server × n_hosting_nodes` slots, sized 2 and 4. **H3 therefore generates 0/204 on both arms, and this was discovered only when someone finally probed it, one rung after the same defect took H2 to 102/204.** The rung had a fixed order, a fresh seed block, a derived skip threshold and a registered α — everything except the one inequality that decides whether it produces data at all. ⇒ **At registration, for every arm, write down the pool size and the task count and compare them.** It is one line of arithmetic, it is invisible in every other registered quantity, and the probe that settles it costs seconds — 24 datasets in 3.1s here, against a rung that was otherwise headed for a cluster job. And read the skip reason: `uniqueness_exhausted` and `too_many_combinations` are different diseases and the threshold is not the cure for the first.
- **⛔ ATTRIBUTE A SQUEEZE TO THE GRID KEY THAT ACTUALLY CREATES IT — HISTOGRAM IT, DON'T REASON ABOUT IT (2026-08-27):** "Widening `replica_configs` would loosen the very squeeze the rung exists to create" was written into a lineage node **and** a findings attachment, cited as a reason not to take the one option that could rescue a bar, and was **false**. The squeeze was `server_node_counts=[4] × replica_server_percentage=0.5` ⇒ exactly **2 hosting nodes**; `per_server` sets how many platform slots sit on those two nodes and cannot change how many nodes there are. Measured hosting-node histogram `{2: 204}` on every rung **and** on the widened probe, with componentwise infeasibility going *up* (0.91) as the pool widened. The same file already carried a 2026-08-27 CORRECTION for reading `server_node_counts=[4]` as "4 hosting nodes" — **the same misreading, uncaught, propagated into a second decision.** ⇒ **A claim about what a lever does to a scarcity is a histogram, not an argument.** Print the distribution of the quantity you believe is scarce, per arm, before you let the belief block an option.
- **⛔ SIZE A PER-DATASET COMPETITOR AGAINST THE GRID'S SMALLEST SWEEP BEFORE SIGNING OFF THE BAR (2026-08-27):** A bar and a grid can each be sound and still be incompatible **only in the product**, which is invisible until a rung is generated. The route_b pivot's S2 kill bar reads a per-dataset `t1x` fit; the saturation guard refuses any fit with `n_rows < 2 × n_params`, and `t1x` has 41 parameters ⇒ **needs ≥ 82 sweep rows**. The bars were calibrated on a pilot corpus with 256–1248-row sweeps; the rung's own scarcity squeeze — the thing that creates the contention under test — takes sweeps to **16 and 64**, so `t1x` was refused on **204/204 datasets in both arms of both rungs** and the kill bar had no denominator. ⇒ **Whenever a screen fits a competitor per-dataset, compute `2 × n_params` and compare it to the smallest sweep the grid can produce, at registration time.** Tightening an environment shrinks its sweeps; the measuring instrument does not shrink with it. Full incident in GATE TOOLS (2026-08-27, "the bars vs the grid"). ⚠ **But do not generalise a refusal past the arm it applies to** — see the next rule; two of the three bars reported as blocked by this were blocked by a tool, and their own column sets fit fine.
- **⛔ A REFUSAL IS NOT AN ABORT — SCOPE IT TO THE STATISTIC IT REFUSES (2026-08-27):** A guard that refuses one unfittable arm must cost you *that arm*, not every statistic computed in the same pass. `route_b_coefficient_transfer.py` raised on the first per-dataset saturation refusal, and it was concluded from the crash that the whole screen was unmeasurable on the grid. **Wrong on two of three bars.** S4's bar block is 7 parameters and fits in every arm — it was unreadable only because an unrelated 21-parameter arm refused in the same loop. S3's statistic is a *pooled* fit over every row of every dataset with one shared coefficient set, which cannot interpolate per-dataset at all, and it was gated behind a per-dataset "did everything fit?" condition that does not apply to it. ⇒ **When a fail-loud guard blocks a reading, check the width of the specific column set the BAR names before concluding the corpus cannot support it** — the crash names the arm that refused, not the arms that would have fitted. And when you fix it: tolerate exactly the refusal (its own exception class, never a bare `except RuntimeError`, which also swallows "no feasible rows" and "non-positive optimum"), report `n_fitted`/`n_saturated` per arm, and return `null` — never a zero, which reads as "the competitor closes nothing", the opposite of "we refused to let it interpolate".
- **⛔ CHECK A LEVER'S ADDITIVE/INTERACTION SCALING BEFORE BUILDING IT — AND ONLY TO RULE IT OUT (2026-08-18):** A physics lever only creates coupling if it moves the **interaction** term faster than the **additive** one. Deep queues failed because additive `depth × exec_time` grew with the lever and interaction `added_in_batch × exec_time` did not. Link bandwidth failed the opposite way: store-and-forward charges every hop a full transmission, so additive `hops × T` and interaction `crossings × T` **both** scale as 1/bandwidth ⇒ ratio invariant, measured **0.0100 → 0.0088** across a 3× cost change, contention stuck at ~1% of the link cost at any bandwidth. Two lines of algebra would have predicted both. ⚠ **But the test only rules levers OUT.** On the hub↔mesh sweep `wait/transfer` got the ranking backwards — n_core=4 had the *lowest* ratio (0.0072) and the *highest* regret (0.35%/7.2%) — because the ratio is measured on **optimal** plans, which select *against* contention, while regret is a property of the spread across the whole sweep. Bandwidth-invariance ⇒ stop; a favourable ratio ⇒ no information.
- **⛔ THE COLLISION TERM MASKS EVERY NEW MECHANISM — ISOLATE WITH `--spread-plans-only` (2026-08-18):** ⇒ **once collisions are removed the base physics has literally no coupling left**, so there is no reservoir waiting for a better lever. It is an isolation control, **not** a corpus gate — it discards ~2/3 of the sweep. Two tests pin both halves of the contract. Evidence and figures: [link_contention_v1](lineages/archive/link_contention_v1.md).
- **⛔ THE REPAIR CONTROL MUST MATCH THE MECHANISM BEING TESTED (2026-08-18):** `one_integer_repair_frac` uses **node-occupancy excess**, which is *structurally blind* to link contention — two tasks on different nodes sharing a core segment read exactly zero there. Shipping `link_contention_v1` against the stock gate would have returned a **false PASS**. Added `--gate-link-repair` (busiest link `k1`, top-2 `k2`, total sharing `excess`). Generalisation: **before gating a new physics, ask what the existing repair column cannot see, and add the column that can.** And test the gate itself — `--gate-coupled-fraction` was recommended as primary while being incapable of failing; the new controls ship with 11 tests.
- **⚠ A COST THAT PRICES ONLY *REMOTENESS* PUSHES THE OPTIMUM TO THE LOCAL CORNER (2026-08-18):** First `link_contention_v1` pilot ran on stock `shallow_v1` (`per_client >= 1`) and the backbone made the corpus **more** separable (regret 2.51% off → 1.07%/1.09% on), because most tasks could run on their own source node and never touch the network. Same shape as `netc_scarce_v1`, where penalising co-location pushed the optimum toward the corner greedy already picked. Fix is `per_client = 0` (`netc_multihop_v1`), varying **only** that — unlike `netc_hotspot_v1`, which moved `replica_server_percentage` and `per_client` together. **Check what fraction of the optimum even uses the resource you are pricing before generating a corpus.**

- **⛔ M1 / `--gate-coupled-fraction` IS THE WRONG GATE — IT CANNOT FAIL (2026-08-18):** M1's marginal greedy scores each option as `min RTT over all joint plans with task t there` ⇒ it has **oracle access to the joint sweep**, so it is not a pointwise model and cannot fail once the candidate set is small. Measured: a corpus where the additive fit picked a suboptimal plan **75% of the time at 21.5% mean regret** still reported M1 coupled = **0%** — the pre-registered gate would have **rejected the only config showing headroom**. Retroactive scope: `--gate-coupled-fraction` appears in **no committed script/sbatch/log**, so **no corpus was ever accepted or rejected by it** — but it was recommended as primary in this file (0.48.0) and in `LINEAGES.md`, and shallow_v1's since-retracted "31.0% coupled" justified the whole shallow lever. ⇒ **Gate on M4 `additive_choice_regret_rel` / `additive_choice_is_optimal`** (the additive fit is literally `PointwiseEdgeMLP`'s expressive class). New `--gate-additive-argmin-regret`, which **argparse-errors unless `--gate-one-integer-repair` is also given**.
- **⛔ THE ONE-INTEGER CONTROL — MANDATORY NEXT TO ANY REGRET NUMBER (2026-08-18):** Re-take the additive fit's argmin after handing it **one** scalar (per-plan node-occupancy excess). If that repairs the regret, the "coupling" is a count-like feature an MLP learns from one extra input, and a GNN win would only mean the baseline was under-featurised. **Now first-class in `separability_diagnostic.py`** (`one_integer_repair_frac`, printed with a `!! DEGENERATE` banner at ≥0.5, gated by `--gate-one-integer-repair`). **Three independent mechanisms have now collapsed this way** — `added_in_batch` (base physics, +1.10pp), deep queues (diluted by depth), node-ingress bandwidth (75% repaired at its only high-regret setting) ⇒ this is looking like a property of **this class of scheduling problem**, not of any one experiment. That throughline + the reusable diagnostic is a legitimate paper contribution on its own.
- **⛔ THE WORKLOAD DRAW WAS UNSEEDED — NO CORPUS IS REPRODUCIBLE FROM ITS SEED (2026-08-18):** `generate_workload_templates` drew task source nodes from the **global** `random.randint`, with **no `random.seed()` anywhere** in the generator. Two runs of the same grid with the same seeds gave different workloads, RTTs and sweeps. Infrastructure was always deterministic (`network_maps`/`replica_placements`/`queue_distributions` byte-identical) — only the workload moved. **Matched A/B arms were therefore impossible**, and `node_contention_v3`'s "metrics disagreed and were underpowered" 12-ds pilot was built this way. Fixed: local `random.Random(--workload-seed)`, default 42. **`placements.jsonl` row order is still nondeterministic and that is benign** (parallel completion order) — compare sweeps as **sets**, never with `diff`. ⇒ **Never use an existing corpus as a bit-identity baseline for a physics change**; build the baseline arm from the same tree as the treatment arm.
- **⛔ SCARCITY BY COUNT AND TOPOLOGY FUNNELLING BOTH BACKFIRE — OVERLAP IS THE VARIABLE (2026-08-18):** Penalising co-location only creates a joint decision if tasks cannot cheaply avoid each other. Free-spreading check (min premium `θ*` admitting a task→distinct-node matching, Hall's condition, **no simulation**): `θ*=0` in **92%** of baseline datasets — each task's favourite node was already distinct. Cutting links (`netc_scarce_v1`) and funnelling to hubs (`netc_funnel_v1`) both **reduced** mean pairwise candidate-node overlap (0.93 → **0.36** → **0.14** of 4 tasks) because tasks anchored to different clients get *more disjoint* sets as they shrink ⇒ θ*=0 rose to **100%**. The blocker on the fix was `generate_infrastructure.py`'s `replica_server_pct = max(server_pct, 0.6)` floor (now overridable via `preinit.replica_server_percentage` / `--replica-server-percentage`). ⇒ **Measure candidate-set OVERLAP, not candidate count**; and **vary one thing** — `netc_hotspot_v1` moved `replica_server_percentage` **and** `per_client` 1→0 at once, so its "cliff at 0.15" was really "client-local replicas removed".

- **⛔ OFFLINE REGRET IS NOT A LIVE PREDICTOR, AND THE RESIDUAL'S SIGN FLIPS WITH COUPLING (2026-08-17, 15/15 final):** Arm A (gated GIN residual, `x = x0 + mp_gate·gin(x0, ei)`) cut offline test greedy regret **0.5682 → 0.2621s (−54%)**, top-5 0.0346 → 0.0239, `mp_gate` learned **1.08** (>init ⇒ it leans on MP *more*). Live it **split by config**: `sparse_p25` 1.14→**1.18×** Kn (+3.5%), `sparse_p35` 1.02→**1.12×** (+9.9%), but **`sparse_p25_skew` 0.84→0.80× (−4.3%), p99 71.0→63.1s**. Gate verdict unchanged: **PRIMARY 1/3, TAIL 1/3 FAILED**; paired wins identical to baseline (GNN 3/15 · MLP 10/15 · Kn 2/15); SUM 1.05→1.12×. ⇒ **A −54% offline gain bought nothing overall and cost RTT on the two near-pointwise configs, while paying only on the one config with real coupling** — exactly the separability entry's prediction (extra capacity buys ~1pp of interaction variance and blurs the ~99pp pointwise signal). **⚠ Correction to the mid-session read: `logit_tied_rate` is NOT the discriminator** — it rose on *all three* configs (0.5097→0.5542, 0.5374→0.5807, 0.5338→0.5511) including the one that improved. The sign of the effect tracks **the config's coupling**, not the tie rate. ⇒ **Never promote a checkpoint on offline regret/acc alone, and never gate a graph-capacity change on a pointwise-separable config** — measure per-config with M1 coupled-fraction alongside.
- **⛔ A RESULT WHOSE CODE NO LONGER EXISTS IS NOT A RESULT (2026-08-17):** The 0.47.0 headline "0.88× Kn on `sparse_p35`/s42" is **not reproducible** — the full gate gives **1.04×** for the identical cell/seed/model/config/decoder. Proof the code changed, without the old source: that arm ran `GNN_DROP_NODE_EDGES=1`, a variable implemented **nowhere** in the tree today; had it been a no-op then, the arm would have been the unfixed baseline (~276M), but it measured 22.34M ⇒ the flag worked then and the file was rewritten after (`GNN_DROP_NODE_EDGES` → opt-in `GNN_MP_NODE_EDGES`), with only the final state committed in `2a591ed`. **Root enabler: `run_provenance` records neither the git commit nor `OMP_NUM_THREADS`/`MKL_NUM_THREADS`** (ablation ran 4 threads, gate 6) — and with `logit_tied_rate ≈ 0.54`, FP reduction-order differences can flip argmax on half the decisions. **Add both fields to provenance; until then treat any cross-sweep GNN comparison as suspect and re-measure under one runner.**
- **⛔ ONE MODEL DEFINITION, AND MAKE ARCH FLAGS SELF-DESCRIBING (2026-08-17):** `train_near_rtt.py` no longer declares its own `TaskPlacementGNN` — it **imports** the canonical one from `src/policy/gnn/gnn_model.py` (−84 lines of duplicate class; training behaviour preserved via new ctor args `dropout`/`post_gin_dropout`/`normalize_platform_inputs`). Two copies is what caused the 25:1 flood; a second latent drift was already queued behind it (the trainer had a `platform_input_norm` LayerNorm the server lacked — absent from the deployed ckpt, so `strict=True` never fired, but one training run from silently serving wrong). **Pattern that prevents recurrence:** the residual carries a learnable scalar `mp_gate`, so it appears in the state_dict and a residual ckpt **cannot** strict-load into a non-residual model (or vice versa) — silent architecture drift becomes a loud `RuntimeError`; what weights can't encode (`mp_node_edges`) goes in the existing `<model>.contract.json` sidecar, which `load_gnn_model` reads back, and a **stale `GNN_MP_NODE_EDGES` that disagrees with the sidecar now fails loud** instead of being ignored. Candidate-restricted same-node edges measured on the full corpus: **1428 → 20.6 mean edges (25.9× → 0.37× bipartite)**, 80% of graphs retain ≥1 pair, 20% correctly degenerate to bipartite-only.
- **⛔ DATALAB: TRAINING PORTS, SERVING DOES NOT — and never `rm -rf` the repo dir (2026-08-17):** Remote `gnn-herosim` sat on divergent history where `constants.py` lives in **`src/motivational/`**, local has **`src/placement/`** ⇒ `train_near_rtt.py` runs on both (its imports happen to be satisfied) but `executesimulation.py` `ImportError`s ⇒ **the live gate cannot be moved to datalab without a real branch sync.** It was also still serving the **pre-fix** `gnn_model.py` (unconditional `node_edge_index` concat = the 12.4× regression) because earlier syncs were file-by-file rsync — **code must travel by git, only data by rsync.** **Safe re-sync tactic (data survives):** `git fetch` → `git checkout --force -B <branch> origin/<branch>` → `git clean -fd`; `clean` skips **ignored** files, so all 65G of `simulation_data/` + 7.3G `logs/` + gitignored ckpts survive. A clone-and-move-data-dirs plan would have silently dropped `src/notebooks/models/*.pt`, `src/notebooks/logs/`, and 673M of wandb runs under `src/legacy/` — **gitignored data is scattered inside the source tree, not confined to data dirs.** Verify by diffing `git ls-files` against a throwaway clone. ⚠ **`--partition=GPU-rtx6000` is unusable**: the `gnn` env's torch 2.5.1+cu121 has no kernel image for those cards; both arms died at weight init. Use `GPU-a40,GPU-a100s,GPU-l40s,GPU-a100` (the existing `full_corpus` sbatch omitted rtx6000 deliberately).
- **⚠ the pre-registered gate's own criteria are the bar, and a losing config can still be the interesting one (2026-08-17):** MP-parity gate: **PRIMARY 1/3, TAIL 1/3 — FAILED**, yet the single config the GNN wins (`sparse_p25_skew`: GNN **0.84×** Kn / MLP **2.27×**, p99 **71.0s vs 498.5s**) is where the pre-registered *collision cliff* actually reproduces — on skew, not the `sparse_p35` where the n=1 probe saw it. Aggregate `SUM` (GNN 1.05× / MLP 0.93×) **hides** this because it is dominated by the two large-RTT configs. ⇒ report per-config wins **and** the declared criteria, never the SUM alone; the honest claim is *bounded collision-robustness on skewed queues*, not a general win. **Process note: cells were twice dismissed as "record completion" and twice held the only GNN win — score every cell before concluding.**

- **⛔ THE CO-SIM TARGET IS POINTWISE-SEPARABLE — the MLP was correctly specified all along (2026-08-17):** New **M4** block in `scripts_cosim/separability_diagnostic.py` fits `rtt(plan) ≈ μ + Σ_t f_t(plan[t])` by least squares over every plan in `placements.jsonl` — exactly what `PointwiseEdgeMLP` can express. Additive R²: `contention_v2` **0.98812** (additive argmin = true optimum in **81%** of ds), `contention_v4_pilot` **0.99973** (**100%**), `contention_v5_quick_test` **0.99973** (97%), `highq_safe` **1.00000** (**100%**). ⇒ **No amount of data or architecture work could have made a GNN win.** Reframes the whole `mp_parity` line: the GNN didn't lose to a bug, it lost because nothing non-pointwise was left to learn. **The entire interaction is one integer** — adding a collision-count column takes v2 0.98812→0.99912 (+1.10pp); on v4/v5 it is +0.03pp. Mechanism (`src/placement/scheduling_cost.py:108-132`): every term of `current_work + queue_work + cold_start + exec_time + comm_time + network` depends on `(task, platform)` alone **except `added_in_batch`**. Network latency is a static table lookup paid once per task (`infrastructure.py:985-993`), never congestible. **Co-located platforms share NOTHING** — the only `capacity` in `infrastructure.py` is disk cache; `memoryRequirements` is never enforced. Measured RTT split on a real optimum: queue **1.330s (~95%)**, network 0.0719s (5.1%), exec 0.0239s, cold-start and pull **exactly 0**. ⚠ **Trap: platform ids are globally unique, so `platform_only_r2 == additive_r2` by construction** — the node/platform split says *where* the separable signal sits, NOT whether topology matters; use the network share of RTT for that.
- **⛔ DEEP QUEUES ARE THE WRONG LEVER — confirmed on all 899 `contention_v2` sweeps (2026-08-17):** Queue depth **predicts** separability, monotonically: shallowest quartile (depth 27.6) → additive R² **0.97822**, collision gain **+1.986pp**; middle (35.5) → 0.99211/+0.672pp; deepest (50.8) → **0.99803/+0.181pp**. `corr(depth, additive_r2) = +0.256`, `corr(depth, collision_gain) = −0.259`. **Coupling is 11× weaker in the deepest quartile.** Arithmetic reason: additive term `depth × exec_time` grows with depth; interaction `added_in_batch × exec_time` does **not** ⇒ depth dilutes the only coupling there is. This is why `contention_v4/v5` landed at 0.9997 — **the contention series' core lever was backwards.** Inverse works: `shallow_v1` grid (depths ~0–8, stock dnn1/dnn2, otherwise v2 axes, 200 ds) → coupled(>1%) **31.0%** vs v2 **7.1%**, coupled(>5%) **19.6%** vs 2.4%, additive-argmin optimal **62%** vs 92%, pointwise worst-case regret **48.4%** vs 3.6%. **First corpus where the node-collision column outexplains the platform one.**
- **⚠ MEAN ADDITIVE R² IS THE WRONG GATE STATISTIC — the target is bimodal (2026-08-17):** `shallow_v1` median R² **0.99990** against mean **0.96074** — most datasets stay perfectly separable and *all* the structure sits in a large minority, so averaging hides exactly what matters. Also n-sensitive: the same corpus read 0.93838 at n=12 and 0.96074 at n=168 (would **fail** a 0.95 gate at full size while passing at pilot size). **Use the coupled fraction (M1 regret >1%) instead** — new `--gate-coupled-fraction`, which cleanly separates the arms (shallow **0.167 PASS** / baseline **0.000 FAIL**). Both gates live in `separability_diagnostic.py`; run on a ~30-ds pilot **before** generating a full corpus.
- **⛔ A SHARED RESOURCE ONLY COUPLES IF PLACED TASKS HOLD IT LONG ENOUGH TO OVERLAP (2026-08-17):** Built `node_contention_v3` — `Node.compute_slots` (`simpy.Resource`) that co-located platforms contend for, opt-in via `--compute-slots-per-node` / `config.nodes.compute_slots_per_node`, **default None = bit-identical `node_disk_v2`**. Matched 12-ds pilots: metrics disagreed and n was underpowered (additive R² 0.98847 vs 0.98584 wrong way; additive-argmin optimality 75% vs 92% right way) — **but `nodeContentionTime` was exactly 0.0 on EVERY placed task in every dataset.** Placed tasks never overlap: exec is ~0.024s and the seeded backlogs drain first, so slot contention only serialized the backlogs — a per-`(task, node)` **additive** term. Candidates that *would* satisfy the rule: memory held across the whole residency (cold start + exec, cold starts reach **38s**), or a slot held for a replica's warm lifetime. Guarded by `scripts_cosim/test_node_contention.py` (9 tests). **Non-obvious trap found building it:** queue depth is seeded as a compressed warmup backlog (`Platform.virtual_warmup_total_time`) drained in a **single** `env.timeout`, NOT as `queue.items` — wrapping only the per-task execution path left contention at exactly 0 while bypassing ~95% of RTT.
- **⛔ CACHE LABELS ≠ SWEEP OPTIMA — silently invalidates any ablation (2026-08-17):** ⇒ The 2026-08-17 `shallow_v1` ablation (`gnn_base` 91.7%/0.90% beating pointwise 90.0%/2.68%, and `gnn_node` re-falsified) is **NOT established** — both arms measured on contaminated labels. **Always verify label provenance before running or citing an ablation.** Evidence and figures: [shallow_v1](lineages/archive/shallow_v1.md).
- **⚠ `queue_feature_contract` in METADATA was a hardcoded guess (2026-08-17):** `extract_dataset_metadata.py` had `queue_contract = "legacy_v0"  # Default assumption` while `compute_compatibility_matrix.py` decides **what may be trained together** from that field. Now read from the collection's graph cache `metadata.json`. **The contract is NOT single-valued per collection** — `contention_v2` has both `graphs_cache_contention_v2_873_v5.5` (no field ⇒ legacy_v0) and `…_873_v5.7_siv1_dim14` (scale_invariant_v1); on conflict the extractor records `legacy_v0` and **warns**. A cache with no contract field predates it and *is* legacy_v0 — skipping such caches hides real conflicts. `shallow_v1` now sits alone in group `scale_invariant_v1_node_disk_v2_4task`, unmixable with the 15 `legacy_v0` collections.
- **⚠ new-corpus runbook gaps (2026-08-17):** (1) The generator does **not** emit `system_state_captured_unique.json`; `prepare_graphs_cache.py` hard-requires it and dies with `FileNotFoundError: Missing system_state_captured_unique.json for ds_00000`. Run `refresh_optimal_full_stats.py --base-dir <collection> --rewrite-ssc` first — same trap recorded for `contention_v2` in June. (2) New task types need a `workload_nofs-<type>` param in the sampled space (`combinations_simple_mapping.pkl`), not just `wsc`/`prewarm`/`replicas` (now auto-synthesized). (3) Two loud-failure fixes: `prepare_workloads` silently dropped apps missing from the sample mapping; `calculate_workload_stats` returned `average_rps` while `flatten_workloads` reads `stats['rps']`, so an empty workload died as a bare `KeyError: 'rps'`. (4) `contention_v5_quick_test` has **3/38 ds with no `placements/` at all** (violates the mandatory-JSONL rule); `highq_safe`'s sampled sweep can't satisfy M1's strictness — M1 needs relaxing before batch size >8.

- **⛔ TRAIN/SERVE MESSAGE-PASSING MISMATCH — the real cause of the GNN's 12× live failure (2026-08-16):** Three *different* architectures were in play. **Offline ablation** `scripts_cosim/gnn_necessity_ablation.py` (the RQ3 "GNN>MLP" evidence): GIN with a **residual** (`x = x0 + gin(x0, ei)`) **and candidate-restricted** same-node edges (`synth_node_edges(candidates_only=True)`) — its own comments say full same-node edges "flood the GIN and oversmooth". **Training** `train_near_rtt.py:270`: `self.gin(x, data.edge_index)` — bipartite only, **never references `node_edge_index`**, no residual. **Serving** `src/policy/gnn/gnn_model.py`: bipartite **+ ALL** same-node edges, no residual. Measured on `graphs_cache_contention_v2_873_v5.7_siv1_dim14`: **1428 same-node vs 57 bipartite = 25:1 flood**; on the deployed ckpt **87.5% of argmax decisions flip** (480 tasks / 420 flips) and the chosen platform's dim7 is 1.25× deeper. Co-located platforms get averaged, erasing dim7 — the one feature that separates them. Live `sparse_p35`/s42 (ladder `gnn_encoder_ablation_20260816/`, seed 42, full-corpus siv1 ckpt): control **276.0M / 10.93× Kn / qvsmin 4,007 / collide 0.411** → **drop same-node edges 22.34M / 0.88× Kn / p99 213 / qvsmin 98 / collide 0.335**, i.e. **beats Knative and ECT with no retrain**. No-MP is 22.44M (0.89×) ⇒ once the flood is gone, bipartite MP is ~neutral, so rung B is still a real GNN. **This is why "offline accuracy is decoupled from live placement" — offline was scored on a different graph than serving used.** Root cause of the drift: `train_near_rtt.py` runs training at import so it can't be imported as a library; the model class was copy-pasted and diverged. Fixed: `GNN_MP_NODE_EDGES` (default **off** = parity) + guard `scripts_cosim/test_train_serve_mp_parity.py`. **Any future GNN arch change must land in BOTH files.**
- **⛔ DECODE GUARDS ARE NOT THE GNN LEVER (2026-08-16):** Same ladder, same cell. `GNN_LQB_LAMBDA=1.5` → **275.7M (10.92×)**, i.e. *no change* vs control 276.0M — the λ·log1p(4000)≈12 penalty is swamped by corrupted logit margins (`logit_chosen_minus_minq` mean **+229.9**, `confident_worse_queue_rate` 0.95). `GNN_DECODE_MODE=argmax_uniq` drove `collision_batch_rate` to **exactly 0.000** (550,851 tasks enforced, 81 relaxed) and got **WORSE: 301.3M (11.93×)**, qvsmin 4,508. ⇒ **intra-batch collisions were a symptom, not the cause**; a decode guard cannot rescue an inverted ranking. `GNN_QUEUE_FILTER_MAX_DELTA=8` does reach 21.80M (0.86×) but only by overriding the model (qvsmin→4). Retires the 0.45.0/0.46.0 conclusion that the failure is "structural in the joint 4-task argmax".
- **⛔ ECT IS NOT A CEILING AND NOT A TEACHER FOR REGIME A (2026-08-16):** First-ever run of `--knative_network_ect` on the coupled trio (seed 42, `node_disk_v2`): `sparse_p35` **25.87M (1.02× Kn)**, `sparse_p25` **16.79M (0.98×)**, `sparse_p25_skew` **4.95M (1.13×)** — vs MLP 0.78×/0.84×/2.73×. A greedy that knows queue length, exec time, cold start, comm and network latency *exactly* performs at Knative level ⇒ better myopic cost estimation is not the lever, and there is **no ECT headroom to distill** (Regime B's 125→31 does not carry over: `contention_v2` has 0/900 datasets with any cold start or pull, so `expected_completion_with_filterstore_pull` degenerates to plain `ect`). **Do NOT infer "no headroom ⇒ pivot to scenario design"** — the MLP already beats both. ⚠ `ECT_PULL_DISTILL_DIR` dumps a PyG frame **per decision**: on `workload-125-225` that is 561,848 frames/cell (~5.1GB in 15min across 3 cells, ~50GB/cell projected) — keep harvesting opt-in (`HARVEST_DISTILL=1`).
- **⛔ CORPUS SIZE IS NOT THE GNN LEVER (2026-08-15/16):** 3.04× more training data (873 → **2,651** graphs; merged `contention_v2`/`v3`/`v4_pilot` + `1060_warmth_v2` + `sparse_warmth_v2` + `highq_safe`, `scale_invariant_v1`/dim14, cache `graphs_cache_full_corpus_siv1_dim14`) improved the GNN **offline** and changed **nothing live**. Offline test acc **55.7% → 65.3%** (val 62.6→66.3, task_acc 89.4%) — a real gain, well outside the ±4.2 seed noise in the entry below. Live 45-cell trio (`full_corpus_siv1_coupled_trio_20260815/`, datalab **685484**): **GNN 0/15, SUM 12.40× Kn** (was 15.07× at 873) — marginal, still catastrophic. **MLP 10/15, SUM 0.93× Kn.** Decisive: `chosen_queue_vs_min` **mean 3,329** across all 15 GNN cells (median chosen ~1,100 vs min-available ~0–50), collisions 0.31–0.48 — i.e. **queue-blindness is unchanged despite the offline gain**. ⇒ **Offline accuracy and live placement quality are decoupled for the joint decode; stop generating/merging corpora as a GNN fix.** Both ckpts served under a **verified-identical** `scale_invariant_v1` contract (grepped from cell logs), so this is **not** a train/serve mismatch. wandb `gnn-full-corpus-siv1-aug2026` (GNN `v06uk8gb`, MLP `5k7phldu`).

- **⚠ per-cell SLURM execution skips whole-sweep provenance (2026-08-13):** Sweep runners write `manifest.json` (phase 0) and `compare.json` (last phase), but the datalab path submits **one SLURM task per cell** via `run_sealed_holdout_one.sh`, so neither phase ever runs and only `results/` is rsynced back — the sweep dir arrives with no physics/commit/ckpt-hash record and no compare. Fixed: manifest writer extracted to `scripts_cosim/important/write_sweep_manifest.py` (CLI, fail-loud on missing model/workload/empty seeds, refuses silent overwrite without `--force`), callable from the per-cell path; `run_contention_v2_873_sealed_holdout.sh` now calls it with overridable `MANIFEST_KIND`/`MANIFEST_NOTE`/`MANIFEST_EXTRA_JSON` so a re-baseline cannot inherit the original's identity. Old inline heredoc also had dead code (built `configs` wrong, then overwrote it with a hardcoded list). **After any datalab sweep: write the manifest + run compare locally.**


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

## Prefix-free accuracy drifts down as the selector improves (2026-10-12)

The user read the W&B `val/acc` and `val/task_acc` panels of the accel, scale_160, r1_attribution and slate_fix runs as "terrible". They are not
the runs' decision. Every one of these runs is teacher-forced (`NEAR_RTT_PARTIAL_STATE_EDGES=1`, so `TEACHER_FORCED`, `train_near_rtt.py:270`), and for
those runs `evaluate()` zeroes the prefix block before the per-task forward (`_prefix_free_prefix_block`, `:462`, called at `:1221`). `val/task_acc`,
`val/task_acc_choice`, `val/acc`, `val/regret_greedy` and `val/regret_topk` score every task as if **no partner had been placed**, against **one**
label index (`data.y`) although the loss accepts any tied-optimal plan. The served decision, which the checkpoint is selected on, is the
prefix-conditioned masked-topo decode: `val/regret_masked_topo` (`ranking_checkpoint_metric`, `:1428`), with `val/mt_task_acc_choice` and
`val/mt_plan_exact` as its accuracies. The run summary says so (`readout/decision_metrics`, `readout/prefix_free_metrics`).

**Mechanism.** The label co-locates a task with its placed partners. A model that learns to use the partner columns (exchange seconds, peer mass)
puts its evidence for the label's choice into exactly the inputs the diagnostic zeroes. So as the selector improves, the empty-prefix
forward drifts away from the label. Curve sweep of 2026-10-12 (16 runs read in full, a 261-run pool; per-family best → final in the table below):
- prefix-free `task_acc` falls 1–8 pp from its early peak on the GNN arms, and 7–11 pp on `mlp-same`;
- the prefix-free accuracies are **positively** rank-correlated with the selector's regret (ρ +0.2 … +0.8): lower accuracy goes with *better* decisions;
- `mt_plan_exact` rises 0.40 → 0.60–0.64 with ρ −0.6 … −0.9 to the regret;
- the prefix-free `regret_greedy` *rises* while `regret_masked_topo` falls (slate_fix seed 2: 6.4 → 9.3 s at its worst, against 3.86 → 0.77 s).

The one arm whose prefix-free accuracy does not drift, r1_attribution `twin-raw`, is also the one whose regret barely moves (4.3 → 4.2 s, about 0.33 of
the random plan). A flat prefix-free curve is the worse sign.

**Floors** (`baseline/val_*` in each summary, written by `_chance_baselines`; the plan NLL floor is for the any-of-K whole-plan CE):

| family (val split) | cand./task | chance task_acc | chance whole-plan acc | chance val/ce | random-plan regret | prefix-free task_acc best → final | selected regret (share of random) |
|---|---|---|---|---|---|---|---|
| accel_replica_v1, capped (438) | 3.92 | 0.313 | 0.044 | 4.66 | 22.66 s | 0.58–0.62 → 0.53–0.60 | 0.61–0.94 s (0.03–0.04) |
| slate_fix_v1/v2, uncapped (438) | 5.03 | 0.290 | 0.039 | 5.23 | 24.17 s | 0.59 → 0.52–0.53 | 0.77–0.96 s (0.03–0.04) |
| scale_160_v1 (451) | 4.18 | 0.274 | 0.034 | 4.89 | — | 0.53–0.56 → 0.46–0.54 | 0.83–1.01 s (0.05–0.06) |
| r1_attribution_v1 (805) | 2.87 | 0.478 | 0.109 | 3.47 | — | 0.73–0.76 → 0.62–0.75 | 0.83–0.98 s engineered GNN / twin / set-transformer, 1.15–1.23 s mlp-same, 1.7–2.3 s raw, 4.2–4.4 s twin-raw |

Guards held on every run in these families: every val dataset decoded (438, 451 or 805 of the split) and mapped 100 %. `val/ce` at selection was 1.36–1.38 against the 5.23 floor
(uncapped). GNN vs twin and seed vs seed differ by less than the per-epoch noise (±0.15 s against a step sd of 0.08–0.24 s).
`val/acc` moves in steps of 1/438. Its identical peaks across runs (151/438 = 0.3447) are quantisation, not a ceiling.

⇒ **On a teacher-forced run, read `regret_masked_topo` (with its `_frac`), `mt_plan_exact` and `mt_task_acc_choice`. Treat the prefix-free
accuracies as a probe of the empty-prefix scorer, never as the run's accuracy, and expect them to fall.** For the same checkpoint (accel g4 seed 1,
best) the two families read prefix-free task_acc 52.4 % / acc 24.4 % against served choice accuracy 73.5 % / plan-exact 62.6 %.

**Exception (2026-10-14): sampled sweeps.** Without the full sweep (`NEAR_RTT_VAL_EXACT_REGRET=0`, datasets whose `placements.jsonl` header says
`exhaustive=false`) a decoded plan absent from the sidecar is charged the constant worst regret, so `regret_masked_topo`, its median, `regret_topk` and
`regret_oracle_topk` track sidecar coverage, not plan quality, and checkpoint selection on them can land on an early epoch. Since 301f293f the trainer does not log
them on such teacher-forced runs (`NEAR_RTT_LOG_UNSERVED_METRICS=1` restores them). Read `mt_task_acc_choice`, `mt_plan_exact` and val `ce`, gate on `-final.pt`
and check val `ce` for overfitting. The uncut_groups_s0_v1 Record ("the served checkpoint is `-final.pt`") owns the measurements.

**Two defects in `scripts_cosim/read_training_curves.py` found by this sweep** (S6 is fixing them):
1. **Direction of the `mt_*` accuracies.** The SELECTION block takes the minimum of `val/mt_plan_exact`, `val/mt_task_acc_choice` and
   `val/task_acc_choice` as their "peak", so it prints "peaks at epoch 0, not 96 — the selector is not optimising it" for metrics that in fact rise to
   their maximum near the selected epoch (slate_fix seed 2: `mt_plan_exact` 0.395 @ 0 → 0.614 @ 99). These are higher-is-better.
2. **Floor of `train/ce`.** The reader compares `train/ce` with `baseline/val_chance_ce`. With `NEAR_RTT_FULL_CONTEXT_CE_WEIGHT=1.0` the training loss
   is the prefix any-of-K plan NLL **plus** a full-context plan NLL (`train_near_rtt.py:626–635`, `:1086`), so its uniform-scorer floor is about twice
   `chance_ce`. That is ≈ 10.5 uncapped, not 5.23. `train/ce` and `val/ce` (prefix term only, `:1224`) are different quantities and must not be
   read as a train/val gap in either direction.

Not a defect, still a trap: several W&B run names occur 3–4 times in `gnn-slate-fix-v1` (relaunches; only one per name carries
`best_val_regret_masked_topo`), and the slate_fix_v2 seeds 3–4 log into that same project. Group by run ID before overlaying curves. And clean
offline curves are not live evidence: these runs' 0.03–0.04 of random-plan regret still lost live (`slate_fix_v1` +2.6 / +5.3 % vs
`cd_exactS_ib0`; `offline_live_transfer_v1`).

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

## Name every path on `git add` — the scope is not the problem, the wildcard is (2026-09-18)

`peer_only_v1`. Twice in one lineage a broad `git add` swept another session's uncommitted
work into a commit that had nothing to do with it. First `a11b042` took a 350-bullet →
67-section `lessons.md` refactor, because the staging ran `git hash-object -w` over
*working-tree* files. Then `6629ee7` took nine `literature_reeval_l2d` sbatch scripts,
`dag_fabric_ceiling_probe.py` and a task-duration decomposition script — untracked files
belonging to a different session — because the add was written as `git add -A scripts_cosim/`.

The second one is the instructive one, because it looked careful. It **was** scoped: to one
directory, the directory the work was in. Scoping a wildcard does not make it a list. `-A`
and `-u` mean "whatever is here", and on a shared checkout what is here includes work that is
not yours and that you cannot see in a diff of your own changes. ⇒ **Pass explicit paths to
`git add`, always. Never `-A`, never `-u`, not even confined to a subdirectory.**

Recovery, if it has already happened and been pushed: `git rm --cached` the foreign paths and
commit that. It returns every file to untracked with its contents untouched on disk, and it
does not rewrite a history someone else may have pulled. Do **not** reach for a revert or a
rebase — the files were never theirs to lose, only yours to un-take.

One further trap when a file is shared and both sides have uncommitted edits: reconstruct the
staged blob from `HEAD` plus your hunk (`git show HEAD:path` → edit → `git hash-object -w` →
`git update-index --cacheinfo`), never from the working tree. And say so out loud, because the
other session's working tree will not contain your hunk — if they commit wholesale, your
addition is lost, and only they can reconcile that.

## Swapping a module to add a capability changes more than that capability — train the disabled twin

`bipartite_edge_v1` replaced a bipartite `GIN` with a `BipartiteEdgeConv` so that message
passing could see the 5-column task↔platform `edge_attr` the `GIN` has no `edge_dim` for. The
new arm beat the old by **−20.4 %** at 80 servers (16/16) and **−28.7 %** at 40 clients, and
took the arm from *losing* to reactive Knative (+8.44 %) to beating it by **−20.8 %** at an
unsaturated rung. Every part of that is real — except the −20.8 % vs Knative, later shown to be an unpaired
ratio of medians that reverses when paired (`unsaturated_edge_v1`); the arm-vs-arm contrasts stand.

The obvious write-up — "edge-conditioning repairs the bipartite stage" — is **wrong**, and only
one thing caught it: a second arm, `gnnedge0`, trained identically with the attributes
**zeroed**. Same module, same parameter count, same depth, same aggregation. It reads
**+0.44 %, 8/16, p = 0.8361** against the treatment at the unsaturated rung, and it beats
reactive on **16/16** — one seed *better* than the treatment. The gain is the conv (`sum` →
`mean` aggregation, a different MLP shape); the attributes contribute a marginal extra that
clears its bar at one rung on 11/16 and is a flat null at the other.

**The rule.** When a change is expressed as "swap module A for module B so the model can now
see X", the contrast A-vs-B answers *"is B better"*, never *"does X help"* — B differs from A in
every way its author happened to write it. Train B with X disabled and make that the primary
contrast. It costs one more arm and it is the difference between a mechanism and a coincidence.

**How to spot that you need it:** if you cannot state the change as a single flag on one module,
you are swapping modules, and the disabled twin is not optional. `mp_residual` (one learnable
gate on an unchanged `GIN`) needed no twin; `mp_bipartite_edge_conv` did.

**Corollary for the sidecar.** The disabled twin is usually **weight-invisible** — zeroing a
tensor leaves no parameter behind — so the two arms' checkpoints load into each other in
silence. Its flag must be in the `.contract.json`, on the serving whitelist, and wired into
*every* loader, or the control is served as the treatment and the two arms become one.

## A `sum` over a variable-sized neighbourhood is a scale bug, not an architecture choice

`peer_only_v1` measured a message-passing stage costing **−18.94 %** live at 80 servers and
**nothing at all** at 6 (p = 0.61). That absence survived five registered attempts at a
mechanism — over-smoothing, a residual repair, a rank-correlation probe, a queue-discrimination
probe, edge-conditioning — before the answer turned out to be one word in a constructor.

`torch_geometric.nn.models.GIN` aggregates with **`sum`**. In this graph a task aggregates over
its **candidate platforms**, and that set is not a fixed size: it is whatever its client can
reach, measured at **3.55 candidates/task at 6 servers and 47.92 at 80** (`cluster_scale_v1`
S0.d). A `sum` over it therefore produces activations that scale with cluster size, on a model
fitted where the set was ~3.55 wide. A `mean` does not. Swapping one for the other is worth
**+13.26 % (p = 0.0052)** at 80 servers and **nothing** at 6 — the scale-dependence the
hypothesis predicted (`bipartite_aggr_v1`).

**The rule.** Before choosing or accepting a message-passing aggregator, ask what the
neighbourhood is and **whether its size varies across the deployments you will serve**. If it
does, `sum` makes the layer's output a function of deployment size rather than of the state you
meant to encode, and the failure appears only where the set is large — so it reads as "the model
does not generalise to bigger clusters" rather than as an aggregator bug. Default to `mean` (or
normalise) for any neighbourhood whose cardinality is a property of the environment.

**How to spot it before it costs five lineages.** A defect that is **present at one operating
point and absent at another** is evidence about the *axis*, not just noise. Measure what varies
along that axis first — here, candidates per task, which was already in the record — and prefer
a hypothesis that **predicts the existing null** over one that merely explains the effect.
Over-smoothing explained the penalty; only the aggregator predicted where it would vanish.

**Corollary for testing it.** `aggr` changes no parameter, so a `sum` and a `mean` checkpoint
are byte-compatible and load into each other in silence. It needs a sidecar key, a serving
whitelist entry and every loader — and before believing a null, check that the two arms are not
bit-identical. See [[herosim-swap-a-module-train-the-disabled-twin]].

## Matching a confound is a measurement at one level, not a control (2026-09-18)

`best_arm_v1` compared a graph arm trained on 1,670 datasets with a pointwise arm trained on
516 and closed carrying "confounded with corpus" as a clause it could not discharge. The
successor matched the corpus and read the contrast again. At 1,670 the graph arm won two rungs
of three and lost none; at **516 it won one and tied two**. Same architectures, same bars, same
cells, same seeds — **the matched verdict depends on the level you matched at.**

So "we controlled for X" is only ever "we measured it at this value of X". A single matched
comparison is one point, and a confound whose *size* varies is not removed by matching, only
relocated into the choice of level. Two consequences, both cheap:

- **Match at more than one level whenever the levels exist** — here the second level cost 16
  training runs and 192 gate arms, against a claim that would otherwise have read as general.
  Where a second level is genuinely unavailable, say the claim is at one level, in the node
  head, not in a clause 40 KB down.
- **Measure the confound itself, as its own registered contrast.** Holding X fixed tells you
  nothing about how much X was worth. The same arm at the two corpora read **+18.71 / +23.81 /
  +8.64 %** — bigger at one rung than the entire cross-corpus gap it had produced, which is what
  turned "the pointwise arm wins at 20 clients" into "the 516 corpus wins at 20 clients".

The trap this avoids is subtle and one-directional: a matched contrast **feels** like it has no
confound left, so it gets quoted without the scope its unmatched parent would have carried. The
parent's clause is honest about being unfinished; the child's silence is not.

And match the **split**, not the count: rebuilding "516" from `--base-dirs` yields 1,670 in
this tree, so the matched arm's training config pins the split artifact and its sbatch asserts
`num_datasets == 516` and the 386/96/34 split before a single epoch runs
(`docs/lineages/archive/corpus_matched_v1.md`).

## Scaling arrivals with servers is not load-matching — sweep the baseline alone first (2026-09-19)

The server ladder held arrivals-per-server at R0's ratio (f4000 at 6 servers → f300 at 80, both
×13.3) and called the rungs comparable. Reactive's queue share went 63 → 94 → 98 → 99 % and its
elapsed 22 → 611 s, and for three lineages every 80-server number was read as an arm beating
reactive by 30–47 %. `unsaturated_scale_v1` S1 then swept reactive *alone* across load at 80
servers: it is unsaturated only at the **same absolute rate** 6 servers runs at, and at twice
that rate it is already 95 % queue. Thirteen times the servers had bought the baseline no
throughput, so the ladder had pushed it 13× past a ceiling that never moved — every "win" was
"less drowned". At the one healthy 80-server rung nothing learned is separated from it.

**Why:** "load per server" assumes the baseline's capacity scales with servers. Whether it does
is a property of the baseline (here, most likely the FCFS replica allocator) and has to be
measured, not assumed; a saturated baseline turns any arm's queue advantage into a headline.

**How to apply:** before reading any arm against a baseline at a new cluster size, run the
baseline alone across an arrival ladder and select the rung by a rule signed in advance (here:
smallest factor with queue share ≤ 0.80). A rung where the baseline is ≥ 90 % queue is read
relative-only and labelled saturated, never as "faster than". 24 reactive arms, ~40 s each,
would have reframed three lineages.

## An effect size measured on four environments is not an effect size (2026-09-19)

`unsaturated_scale_v1` read five quantities on 4 environments. `unsaturated_scale_v2` re-read
the same five, same rung, same checkpoints, same bars, on 16 — and **every one shrank by 4-10x
while its p-value fell by one to three orders of magnitude**:

| read | 4 environments | 16 environments |
|---|---|---|
| `gnnedge0` vs reactive | -6.46 % (p = 0.47) | **-1.62 % (p = 0.0023)** |
| `peeronly` vs reactive | -7.97 % (p = 0.12) | **-1.23 % (p = 0.0013)** |
| `gnn` vs reactive | +16.98 % (p = 0.044) | **+2.31 % (p = 0.0052)** |
| `sum` vs `mean` | +26.69 % (p = 0.039) | **+2.70 % (p = 0.0019)** |

Both halves matter. The **directions all replicated** — no sign flipped, and the mechanism
(`sum` over a variable-sized neighbourhood) held. The **magnitudes were inflated four- to
tenfold**, because with 4 environments the checkpoint statistic's sd is 14-29 pp and the median
of 16 draws from that is dominated by whichever environments happen to be in the set. Here all
four happened to be the same arrival window, and the burstiest one.

**Why:** a small environment set does not merely widen the interval around a stable point
estimate — on this apparatus the point estimate itself is a function of the set, because
(checkpoint x environment) interaction is *all* of the variance and none of it is a stable
per-checkpoint quality. A significant result from a 4-environment read is evidence of a
direction and says almost nothing about a size.

**How to apply:** quote a direction from a small design and a magnitude only from a large one;
size the environment count from a measured sd before believing any effect size (the sd scales
as 1/sqrt(n_environments), so the arithmetic is one line); and put a **design-validation bar**
in the registration — the delivered sd against a projected one — so that "the design did not
buy the power it was built for" is a signed outcome rather than a later apology. This is
`docs/lessons.md`'s "never quote a 4-checkpoint number" again, one level up: the same failure
recurs on every axis that is sampled rather than enumerated.

## Pair on the environment before you take a median (2026-09-20)

A median over cells of an arm's time, divided by a median over cells of the baseline's time, is
not a comparison of the arm with the baseline: the two medians come from different cells. When the
cells differ in load — reactive at 18 s on one and 64 s on another — the ratio can say "−20 %" for
an arm that loses three of four cells, because the median ignores the size of the catastrophic
cell while the baseline's slow cell pulls its median up. `unsaturated_edge_v1` found the
programme's standing 6-server headline was exactly this: +4.6 / +75.8 / +9.7 / −36.8 % per cell
read as −20.3 %, 15/16.

**Why:** a 16-checkpoint read that fires at p < 0.01 feels like a result, and every downstream
lineage inherited the reader without re-deriving what its statistic pairs on.

**How to apply:** compute the relative difference against the reference on the same environment
first, then take the median over environments (`checkpoint_stats`); for arm-vs-arm, pair raw
elapsed on environment and checkpoint (`pair_checkpoint_stats`). Print the per-cell values next
to any collapsed number the first time a reader is used. And screen every cell for baseline
saturation on every ladder — a ratio of medians is most wrong exactly where a screen would have
dropped a cell. See [[herosim-effect-sizes-from-four-environments-are-inflated-2026-09-19]].

## A scheduler that holds tasks moves their queueing into its own wait — decompose before crediting a placement (2026-09-20)

`unsaturated_edge_v1` read a learned arm as "37 % less time in the system after the decision"
and the 7.1 s it waits to assemble a peer group as the whole deficit. `batch_window_edge_v1`
then shortened the window to 2 s: the wait fell to 1.7 s and the platform queue and rendezvous
rose by the same amount, total unchanged. A task held in the scheduler is not in a platform
queue and is not waiting for a partner at the platform — the same seconds are booked under a
different name.

**Why:** the per-task decomposition (`wait`, `queue`, `peer_exchange`, `rendezvous`) is exact,
and every bucket is a real wait, but only one of them — exchange, here −1.05 s from co-location
— is caused by the placement; the others are caused by *when* the placement was made.

**How to apply:** before crediting a policy with a shorter queue or rendezvous, ask whether it
holds tasks somewhere the decomposition does not call "queue". Vary the holding time (the window)
and see which buckets move together; only the buckets that do not move with it are placement.
See [[herosim-headline-was-an-unpaired-statistic-2026-09-20]].

## The bar for a learned placement policy is a rule with the same information, never the reactive baseline alone (2026-09-20)

For nine lineages the learned arms were read against Knative, random and each other, and the
programme's question stayed "does a GNN beat Knative". `peer_greedy_live_v1` put a two-line rule
on the same gate — Knative's candidate set scored in seconds, plus the exchange the physics
charges to partners already placed — and it beat Knative by 13–16 % where every learned arm
lost, and beat the graph arm by 8–13 % served in the graph arm's own configuration. Nothing in
the record could have said this earlier, because nothing in the record had asked.

**Why:** "beats Knative" and "beats a rule that uses the model's inputs" are different claims,
and only the second is a claim about learning. A learned arm can lose to Knative in an
environment that rewards its information (this one) or beat Knative in one where a rule would
beat it by more; the reactive baseline cannot distinguish the two. The rule is also cheap: no
training, no corpus, no train/serve contract, an afternoon of gate time.

**How to apply:** before a learned arm is gated, write the greedy that uses the same features on
the physics the simulator charges, register it as an arm, and read the learned arm against it
paired on environment (and, for a deterministic rule, broadcast over checkpoints). If the greedy
wins, the paper's result is the greedy and the model's task is to beat it. Keep the rule
constant-free where the physics allows (here the exchange saved is compared with the queue
added, both in seconds) so that it cannot be tuned into or out of a win. See
[[herosim-hand-rule-beats-reactive-and-the-graph-arm-2026-09-20]].

## An environment lever that changes service time is a load lever first — screen the lever, not the baseline (2026-09-20)

Three levers were registered to move the exchange-vs-queue ratio (payload ×3 / ×10, a 4× slower
backbone, burst arrivals). Every one of them also moved reactive Knative's load, and at the
study's operating point (6 servers, 0.46 arrivals/s, reactive queue share 0.43–0.63) there was
no headroom: payload ×3 and the 250 MB/s backbone put reactive at a queue share of 0.95–0.99 on
every candidate cell, and bursts of 10 hung reactive on 22 of 48 cells and the batch path on a
topology reactive survived. Only the levers that *lowered* service time (payload ×0.1) or moved
structure without service time (bursts, barely) produced a readable design.

**Why:** a lever is registered for what it does to the ratio; what it does to the load is
discovered on the screen, and "unknown is not a pass" turns most of a sweep into "no design at
this rate" — which is a finding about the operating point, not about the ratio.

**How to apply:** before registering a sweep upward from a healthy-but-tight operating point,
compute the lever's effect on reactive's service time and check it against the headroom
(here: 0.80 − 0.63 = 0.17 of share); if the lever will saturate, register the matched-load
design (the rate that holds reactive's share) as the lineage itself, with rate named as a
confound, rather than discovering it one screen at a time. Screen every lever value on its own;
never reuse the parent study's topologies. See
[[herosim-environment-levers-are-load-levers-2026-09-20]].

## A disclosed partial read from an in-flight gate is not a preview of the final direction (2026-09-20)

`joint_burst_v1`'s J5 (does training on the served distribution help vs. the cold corpus)
briefly read the *opposite* of its final answer. Mid-gate, with only a few checkpoints
resolved out of 16 per arm, the disclosed decomposition table showed the cold control
reading faster per-task than the burst-trained arms. Once the last block finished — mostly
end-of-trace OOM hangs in the burst-trained checkpoints' arms, resolved by rerunning under
the watchdog — the same table reversed: burst-trained `gnnedge0` 9.46 s vs cold 10.25 s
elapsed, and J5 fired `SERVED-DISTRIBUTION-TRAINING-HELPS` at 15/15.

**Why:** a gate's checkpoints do not finish in a random order — the ones that hang (OOM
growth, spin) are disproportionately absent from an early disclosed subset, and which
checkpoints hang is not independent of the arm or the environment. A partial "disclosed"
read is a valid substitute for an UNREADABLE registered slot only once the run is complete;
mid-run, it is a biased sample, not a preview.

**How to apply:** report an in-flight gate's intermediate numbers as status ("here's what's
in so far"), never as a finding, and flag explicitly when a decomposition or comparison
table might still flip once the last block lands. Only the read taken after every block
completes (or every remaining arm's hang is confirmed and disclosed) is evidence. See
[[herosim-joint-burst-v1-closed-2026-09-20]].

## An oracle's headroom is not a learner's headroom — gate a hand convention on the same signal first (2026-09-24)

An oracle that is handed a hidden quantity (where an unarrived partner will run, a future
arrival, a realised cost) measures what KNOWING it is worth. That is two things at once:
*prediction* (information the decision genuinely lacks) and *coordination* (every decision
agreeing on a convention). Only the first needs a learner. `lookahead_mp_v1` found −6.6 % / −8.2 %
of oracle headroom over the peer-greedy rule, and a two-step hand guess got only 2–7 % of it — the
reading that would have funded a new graph representation. The cheapest control, *replace the
oracle's knowledge with the incumbent rule's own forecast of it* ("the partner will go where the
rule would send it now"), recovered 93–95 %.

**How to apply:** after any oracle/headroom screen, and before representation or training work,
register a blocking hand control that computes the hidden quantity from the incumbent policy
itself, with a recovered-fraction bar (0.80 was used). A weak hand guess failing is not evidence
that learning is needed; only the strongest cheap control failing is. See the node for the
measurements and `docs/hard-stops.md` for the direction it closed.

## Threads above the cpuset's cores inflate torch time about 200× — set threads ≤ `nproc` and log it (2026-10-14)

When datalab switched SMT off on every CPU-amd node, each 8-CPU job's cpuset fell to 4 physical cores under 6 torch threads, and one Arm 1 job slowed about 200×
(py-spy: the main thread spins in libgomp inside the backward; no I/O, memory or cgroup throttling). It would have biased only the learned arms of a decision-time-charged
gate. **Rule:** every timed or calibrated run sets torch/OMP threads ≤ the in-job `nproc` (after SMT), uses the same thread count in calibration and cells, and logs
`nproc`, threads, cpuset, `lscpu` model and "Thread(s) per core" per run. A calibration with threads > `nproc` is void, and a change of SMT state or CPU model between
calibration and cells is a venue change. The `uncut_groups_s0_v1` Record ("Arm 1 relaunched after datalab switched SMT off") owns the measurement and the venue rules.

## A decision charge is felt by every task in the batch — report the felt share, not the per-plan share (2026-10-14)

When decision time is charged as simulated time per batch, every one of the batch's `n_b` tasks waits that batch's charge `c_b`. The share of latency it costs is the
**felt share** `Σ_batches n_b · c_b ÷ Σ_tasks latency`, not `Σ c_b ÷ Σ latency`, which understates it wherever batches are large. The per-plan figure is the **pooled** share. State which one a "charge share" is, and
base any bound on the exact felt share. The `uncut_groups_s0_v1` Record ("Felt share: definition, hand F measured
exactly, bounds recomputed") owns the definition and the measurements.

## A pilot direction from 3 topologies and one window can flip sign at confirmation — a 3 × 1 pilot is not a "small design" (2026-10-20)

`uncut_groups_s0_v1`'s server-count ladder chose 128 servers for the FL size confirmation.
At 128 servers × 160 clients it read FL gnn vs `cd_ib0_inflightfull` at heavy load over the
whole run as −14.2 % (mean over 3 topologies, 3/3 ahead, window g0, seeds 1–2; `ladder/read_v1/ladder_read.txt` md5
`b1e33b19`). The registered confirmation, on 12 fresh topologies and windows g0–g3 with the
same two seeds, read **+4.65 % (0/12)** on that test (`size128/reads/read_s128_primary.json`
md5 `dfd3a5ca`). The direction flipped. The registered prior, written from the pilot,
expected −8 to −15 % and put P(SIZE-128-CONFIRMED) at about 0.70. The other three tests
kept their pilot direction (pilot −29.8 / −53.1 / −15.6 %; confirmation −19.0 / −46.4 /
−27.9 %).

**Why:** a 3-topology, one-window pilot samples neither of the two things that move the
heavy whole-run statistic. One is the topology draw. The other is the scale-from-zero
opening, which differs by window and by training seed (seed 1 alone read +12.3 % at 128 and
−0.66 % at 64). Three topologies agreeing in sign is what one draw with a common shift looks
like. It is not three replicates of the effect.

**How to apply:** AGENTS.md says to quote a DIRECTION from a small design and a MAGNITUDE
only from a large one. A 3-topology × 1-window pilot is below that small design for any
statistic that the opening dominates: the heavy rung, and every whole-run test. Use such a
pilot to choose and order the work, never as the basis of a prior on direction. Write the
registered prior for those tests as if the pilot had not been run. The paper's pilot table
reports the pilot row beside the confirmation that overturned it.
