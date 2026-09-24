# throughline — SYNTHESIS

> **Status:** `SYNTHESIS` &nbsp;·&nbsp; **Index:** [LINEAGES.md](../../LINEAGES.md) &nbsp;·&nbsp; **Record spans:** 2026-08-18 → 2026-09-09

**Outcome.** Cross-lineage synthesis: four mechanisms, one collapse. **In this simulator, coupling is either count-shaped or negligible** — and the negligible half is demonstrated, not assumed. **Closed as a question 2026-09-09:** a graph-reasoning win over a pointwise scorer is not available on this simulator's supervised targets — by the structure of what the simulator computes, not by a shortfall of data, epochs or architecture; the 2026-09-09 section below says why in plain language and what would change the answer.

**Related:** [graph_structure_physics](graph_structure_physics.md) · [network_contention_v1](network_contention_v1.md) · [link_contention_v1](link_contention_v1.md) · [shallow_longexec_v1](shallow_longexec_v1.md)

> **Split note.** This node was carved out of a single 4,995-line `LINEAGES.md` on
> 2026-08-27; the section bodies below are byte-for-byte as written. An *above* / *below*
> pointer inside one may refer to a section that now lives in another node — the
> **Related** links say which.

## Record

Newest first; the sections themselves are in chronological order below.

- [Is a graph-reasoning win possible in this simulator? Answered, in plain language (2026-09-09)](#is-a-graph-reasoning-win-possible-in-this-simulator-answered-in-plain-language-2026-09-09)
- [The throughline — four mechanisms, one collapse (2026-08-18)](#the-throughline-four-mechanisms-one-collapse-2026-08-18)

---

### The throughline — four mechanisms, one collapse (2026-08-18)

**`shallow_longexec_v1` is UNBLOCKED and is the fourth confirmation.** The config gap its
note described is closed — `sample_loader.ensure_workload_params` synthesizes the missing
`workload_nofs-{cnn,rf}`, and the grid now generates cleanly (cnn 3.086 s on rpiCpu vs rf
0.004 s, a 730× exec-time contrast). Gated locally at n=16 per arm:

| arm | additive R² | argmin regret mean/max | argmin optimal | **one-integer repair** | +col optimal |
|---|---|---|---|---|---|
| `shallow_longexec_v1` | 0.9330 | 3.05% / 37.0% | 62% | **100%** (n=6) | 81% |
| ...+ 0.5 MB/s ingress | 0.9290 | 0.99% / 4.8% | 56% | **86%** (n=7) | 81% |

Every dataset where the pointwise fit picked a suboptimal plan was repaired by one scalar,
and optimal-recovery rises 62%→81% / 56%→81% from that single column. *(Honest wrinkle: the
augmented fit's mean regret is higher (5.24%/4.57%) because the extra column shifts the
argmin on some datasets that were already optimal — the degeneracy claim rests on the
repair of actual failures and the optimal-recovery jump, not on mean regret.)*

**So five structurally different attempts to inject coupling all end the same way:**

| # | mechanism | how it failed |
|---|---|---|
| 0 | `added_in_batch` (base physics) | +1.10 pp — "a column you hand an MLP" |
| 1 | execution-slot pool (`node_contention_v3`) | no interaction at all — `nodeContentionTime` ≡ 0.0 |
| 2 | deep queues (`contention_v4_v5`) | interaction diluted by depth; R² moved the wrong way |
| 3 | node-ingress bandwidth (`network_contention_v1`) | R² scales continuously, but at its only high-regret setting one integer repairs 75% |
| 4 | long-exec task types (`shallow_longexec_v1`) | one integer repairs 86-100% |
| 5 | per-link capacity (`link_contention_v1`) | **first mechanism to escape the one-integer control** — isolated on spread plans the node column repairs 0% and link scalars only 11-22% — but the effect is 0.08-0.10% regret against a 5% gate, and bandwidth is a null lever (wait/transfer 0.0100 → 0.0088 across a 3× cost change) |

**Whenever coupling in this simulator shows up with teeth, it is capturable by one
count-like feature.** That now looks like a property of this class of scheduling problem
rather than of any single experiment — and it is the reusable finding, together with the
diagnostic that detects it before any GPU-hours are spent.

**`link_contention_v1` sharpens that statement rather than breaking it.** It is the first
mechanism to produce coupling a scalar count cannot capture — the escape was real, and the
reason it worked is now understood: its contended object (a link) has more identities than
the destination-node count, so two tasks that share *no* node can still contend. But the
resulting effect has no teeth (0.08-0.10% regret), so the observed rule survives in its
stronger form:

> **In this simulator, coupling is either count-shaped or negligible.** Five mechanisms, two
> failure modes, and the second one is now demonstrated rather than assumed — the isolation
> control shows base physics is additive to R² = 1.00000 exactly once collisions are removed,
> so there is no reservoir of non-count coupling waiting to be uncovered by a better lever.

The practical corollary for anyone extending this: **check the additive/interaction scaling of
a proposed lever before building it.** Deep queues failed because the additive term grew with
the lever and the interaction term did not; link bandwidth failed because both grow together
and the ratio is invariant. A lever only helps if it moves interaction *faster* than additive
— that is a two-line calculation, and it would have predicted both outcomes.

⚠ **The scaling test rules levers out; it does not rank the survivors.** Tested against the
hub↔mesh sweep above, `wait / transfer` got the ordering backwards — the configuration with
the lowest ratio had the highest regret, because the ratio is measured on optimal plans, which
select *against* contention, while regret is a property of the spread across the whole sweep.
A bandwidth-invariant ratio is decisive evidence to stop; a favourable ratio is not evidence
to proceed.

**The corrected gate is wired into the tool, not just written down here.**
`separability_diagnostic.py` gained `--gate-additive-argmin-regret` (primary) which
**argparse-errors unless `--gate-one-integer-repair` is supplied**, plus a first-class
`one_integer_repair_frac` in the M4 block and a `!! DEGENERATE` banner at ≥0.5. Verified:
both failure modes fire and the tool exits 1. `--gate-coupled-fraction`'s help text now
carries the deprecation and the reason.

**2026-08-18 additions from `link_contention_v1`,** both covered by
`scripts_cosim/test_link_repair_control.py` (11 tests) so the gate is not itself untested —
the `--gate-coupled-fraction` episode was a gate nobody had exercised:

- **`--gate-link-repair`** — repair columns for the busiest link load (`k1`), the top-2 loads
  (`k2`), and total link-sharing excess (`excess`). Needed because the existing node-collision
  control is *structurally blind* to link contention and would have returned a false PASS on
  any per-link mechanism.
- **`--spread-plans-only`** — the isolation control described above. Reusable for any future
  mechanism: it answers "is there coupling here that is not the collision term?", which the
  headline gate cannot, because the collision term dominates every corpus in this repo.

---

### Is a graph-reasoning win possible in this simulator? Answered, in plain language (2026-09-09)

**The question.** The program's goal was a GNN that places serverless tasks better than a
pointwise model (an MLP that scores each task–platform pair on its own). Every route has now
been run to a measured end, and two more environment levers were killed on paper on
2026-09-08 (`dag_fabric_contention_v1`, and CPU/memory co-tenancy interference in
`docs/hard-stops.md`). This section records the answer and the reason, so the question is
not reopened by forgetting.

**The answer.** On this simulator, for any target labelled by brute-forcing one batch of
placements, **a pointwise scorer that is told what is already on each machine can express
the exact cost. A graph neural network has no additional fact to work with.** That is a
statement about what the simulator computes, not about training. It was measured six
independent ways before it was understood as a rule, and the rule then predicted the two
2026-09-08 screens correctly before any code was written.

**Why, in simple language.**

1. *How the simulator prices a task.* A task's time is a table lookup for its platform
   (`executionTime[type][platform]`, `src/placement/infrastructure.py:1337`) plus waits: its
   platform's queue, the machine's shared disk during an image pull, the machine's ingress
   pipe or a network link if those are switched on, and — for a DAG child — whether its
   parent ran on the same machine. Nothing dilates: two tasks sharing a machine never make
   each other's *work* slower, only each other's *start* later.
2. *What those waits depend on.* Every one of them is indexed by the **machine** (or by
   the task's own platform). A wait on machine X depends on *how many tasks of each kind*
   are on X — never on *which* specific task, and never on where the tasks on machine Y
   went. The one exception is a network link, which two tasks on *different* machines can
   share.
3. *Why that closes the door.* "How many tasks of each kind are on my machine" is a handful
   of counts. Hand those counts to a pointwise model as extra input columns and it can
   reproduce the wait exactly — the formal version is that any symmetric function of a
   set of co-residents is a function of their per-type counts (`route_a_v1` composition
   theorem; `route_b_env_pivot_v1`'s `hetdem`/`krank` blocks are exactly those columns).
   What a GNN adds — looking at the other tasks' *candidate options* while scoring this one
   — only pays when the cost depends on the joint pattern in a way counts do not capture.
   Here it never does. Measured: one occupancy integer repairs 75–100% of the coupling in
   four mechanisms; the `--spread-plans-only` control leaves base physics additive to
   R² = 1.00000 exactly; the extended pointwise competitor closes the env-pivot corpus at
   0.892; and on the DAG corpus the prefix columns already encode the only joint term
   (parent→child co-location, ~23% of variance), so message passing ties without them
   (p = 0.25 at honest checkpoint selection).
4. *The exception, closed twice.* Links are the only shared thing not indexed by a machine.
   Their coupling survives the count control (`link_contention_v1`: link-repair medians
   0.000) but is tiny (≤ 0.35% regret), cannot be made material on the input payload
   (`route_c_link_transfer_v1`: concurrency capped at 2 by one client and a diamond), and
   on the output payload the best plan simply routes around the shared cable — zero link
   wait at the α = 2.0 optimum in 98–100% of 204 datasets (`dag_fabric_contention_v1`). The
   label never sees it.
5. *And the model that "won" was pointwise.* With message passing off, the GNN codebase's
   model is two per-entity encoders and a 2-layer edge MLP trained with the same
   candidate-set cross-entropy as `PointwiseEdgeMLP` (`src/policy/gnn/gnn_model.py:403-430,
   179-200`; `train_near_rtt.py:379` vs `train_mlp.py:100`) — a two-tower pointwise scorer.
   Its edge over the tabular MLP on the DAG corpus (−1.76 pp, n = 8 pilot; draft
   registration in `route_b_v1`) is a parametrisation/feature-layout result between two
   pointwise models, and MP-ON adds nothing to it.

**Is it "impossible"?** Within this simulator family and a supervised single-batch target:
yes, by construction — the target lies inside the class "pointwise + per-machine counts",
and a GNN cannot beat a correctly specified model on its own class; it can only tie or
overfit. Outside that: the closed-loop objective (`objective_pivot_v1` Phase 3) is not
closed by theorem, only measured negative at one configuration and n = 120; its remaining
venue is a differently-configured, separately-powered run, and the 150× trainability
asymmetry shows the graph model *can* be moved by it — it just did not move the right way.

**What would change the answer.** One of:
- a resource that is **shared across machines and not routable around** — a cost the best
  plan cannot dodge that depends on the joint pattern of *which* machines were chosen, not
  on any machine's counts (the `dag_fabric_contention_v1` node states the bar: core-link
  binding at the optimum in ≥ 25% of datasets on some substrate);
- **per-task heterogeneity that counts cannot summarise** at the batch sizes the sweep can
  enumerate (per-instance demand spread was tried at 4 tasks; demand moments closed it);
- a **different objective** — a multi-step return whose ranking is stable (the h = 10 s
  horizon return was chaos), or a closed-loop configuration with its own tuning stage.
**Checked against seven other simulators on 2026-09-09** (faas-sim, LEAF, iFogSim2,
EdgeCloudSim, PureEdgeSim, YAFS, EdgeSimPy, plus NoServer — `docs/notes/simulator_landscape_pureedgesim_noserver.md`
§19): none supplies any of the three in kind. Their non-avoidable link contention sits on
trees, which makes it a per-link count; their processor sharing is a per-node count; the one
per-instance attribute (EdgeCloudSim's sampled lengths) is a sorted fixed-width column at
enumerable batch sizes. Porting is not recommended on that evidence.

Each of these is a change to what the simulator computes or optimises. None is a change to
the model, the data volume, or the training recipe — those levers are measured exhausted
(`docs/lessons.md`: corpus size is not the GNN lever; capacity buys ~1 pp of interaction
variance and blurs the pointwise signal).

**Why this does not contradict the scheduling-GNN literature (2026-09-09).** A reviewer will ask
how Decima (Mao et al., SIGCOMM 2019), Placeto / GAP (device placement, 2019–), and "Learning
to Dispatch" (Zhang et al., NeurIPS 2020, GIN over the job-shop disjunctive graph) report GNN
wins if graph reasoning has nothing to compute here. Three ingredients recur in those results,
and this program removed each on purpose — references from memory, not re-verified (the
literature API was rate-limited on 2026-09-09):

1. *The graph is the job, not the placement.* Their message passing runs over a large,
   variable, input-dependent structure (a Spark DAG, a TensorFlow op-graph with tensor sizes on
   edges, a disjunctive graph) and computes **lookahead** — remaining work downstream of a
   stage, i.e. a learned critical-path estimate, the quantity HEFT hand-computes. Our DAG is a
   fixed diamond of four typed tasks: its structure is a constant per type and the one variable
   relation (parent→child co-location) is a single prefix column.
2. *The objective is long-horizon RL, not a one-step label.* They optimise job completion time
   or makespan over hundreds of sequential decisions; the sequence model earns its keep in
   credit assignment. Our supervised target is the brute-force optimum of one batch, which lies
   inside "pointwise + counts" by the theorem above; our one closed-loop attempt
   (`objective_pivot_v1` Phase 3, n = 120) moved the graph model 150× more than the MLP and not
   in the right direction — consistent with RL-for-scheduling needing its own tuning campaign,
   not with a graph effect.
3. *The baseline is a heuristic, not a matched pointwise learner.* Decima beats FIFO/fair/SJF,
   Placeto beats human experts and an RNN placer, L2D beats dispatch rules. Where a "no message
   passing" ablation exists it usually also removes the aggregate features, confounding the two.
   This program ran the control the literature skips — the T1 contract gives the MLP every
   prefix/count column the GNN can compute — and the graph component tied every time.

Add the problem class: JSSP and device placement are QAP-shaped over many co-decided items
with long decision sequences where greedy is provably poor; ours is 4–8 tasks on ≤ 6 nodes,
small enough that a prefix-conditioned pointwise scorer reaches the optimum. The published
wins are therefore attributable to **lookahead over rich job structure + a long-horizon
objective + weak baselines**, and the honest framing of our result is the complement: on batch
placement against a fairly-featured pointwise competitor, message passing adds nothing. A
HeROsim program that resembled the literature would need large variable DAG workloads with
per-instance sizes, a scheduling *sequence* scored by end-to-end job time, RL training, HEFT and
Knative as baselines — **and** the matched-MLP control; the prior from this record is that the
control bites again, because "remaining work downstream" is a per-stage scalar.

**What the evidence does support for the paper** (`program_verdict_v1` P6): the terminal
negative with its reusable diagnostics (`separability_diagnostic.py`, the one-integer and
spread-plans controls, `dag_fabric_ceiling_probe.py`); the codebase's learned scheduler
beating Knative by −44.8% on the held-out fabric (tied by a corpus-matched MLP, so quoted as
a corpus result); the throughline itself as a property of this class of scheduling problem;
and, if signed and run, the two-tower-vs-tabular pointwise contrast — never written as
"GNN > MLP".

## 2026-09-11 — the throughline has a measured exception (SUPERSEDED IN SCOPE 2026-09-12: the exception is OFFLINE ONLY)

The section above argues that on this simulator's supervised targets a graph-reasoning win over a
pointwise-plus-counts scorer is unavailable by construction, and names three things that would
change the answer. **One of them has now happened, and it was the environment change.**

`peer_affinity_v1` puts the cost on **pairs of task instances** (continuous exchange volumes, no
commit order) under a binding capacity cap — not on machines, so the count theorem this document
leans on does not reach it. On that environment, at 482 training datasets and 16 seeds, message
passing beats its own MP-OFF twin by **+5.14 pp (p = 0.001, 13/16 seeds)** on held-out decode regret.
That contrast is the cleanest in the program: same architecture, features, decoder, checkpoint
selector and seeds, differing only in whether `PeerConv` runs.

**The correction this forces on the document's method, not just its conclusion.** The same contrast,
on the same code, read **+2.08 pp, p = 0.15 at 136 training datasets** and was written up as
"message passing is not the lever". It was not measuring the architecture; it was measuring the
corpus. The per-arm gain from 3.5× data separates cleanly — MP-ON −5.09 pp, MP-OFF −2.43 pp,
pointwise-plus-lookahead −3.24 pp. **An INDETERMINATE model-class contrast at n = 8 on a small
corpus is not evidence of redundancy**, and this record has now made that mistake twice
(`route_b_v1` Phase 2's checkpoint-selection artifact was its cousin). Before concluding that an
architecture is redundant, vary the corpus and read the per-arm delta.

What does **not** change: the argument still holds for every **node-indexed** target in this
program, which is every option-1 and route-B corpus. The exception is specific to a cost indexed by
instance pairs, and it comes with three caveats that must travel with it — the edge is at the
selected checkpoint (the arms tie at last epoch, MP-ON having memorised the training split to
0.00 % regret), the live replay gate agrees in direction but not significance (+3.78 pp, p = 0.13),
and the GNN-vs-MLP comparisons carry a selector asymmetry favouring the GNN while GNN-vs-MP-OFF
carries none. Full record: `docs/lineages/peer_affinity_v1.md`, entries T1 and T1b.

---

## 2026-09-12 — the exception is offline only, and it reverses live

The section above records a measured graph-reasoning win and, correctly for what was known then,
carries "the live replay gate agrees in direction but not significance" as a caveat. **That caveat
is now superseded by a stronger measurement, and it points the other way.** The scope of the
exception narrows to the supervised target; it does not reach serving.

**What was measured.** The rung was extended (a third peer partner at 800 MB) and both rungs were
served live on matched 450,729-task production traces, 16 seeds per arm, with a live path proven
bit-identical to the offline evaluator on 34/34 held-out datasets. Ordered by peer-graph density:

| corpus | OFFLINE `gnn` − `mpoff` | LIVE `gnn` − `mpoff` |
|---|---|---|
| x200 p2 | +5.14 pp (p = 0.001) | +2.37 % (p = 0.23) |
| x800 p2 | +6.53 pp (p = 0.013) | −4.42 % (p = 0.074) |
| x800 p3 | +9.61 pp (p = 0.0003) | **−10.67 % (p = 0.018)** |

**The two venues are anti-correlated and monotone in density.** Every increment that makes message
passing look better on the supervised target makes it worse on the live stream, and the live column
crosses zero between the first and second rung. So the earlier "direction but not significance" note
was the last point before a sign flip, not an under-powered version of the offline result. The live
peer term inverts with it: the MP-OFF twin achieves *lower* `totalPeerExchangeTime` on the stream in
every condition measured, i.e. the graph arm is worse at its own objective when it is served.

**Three explanations were registered in advance and all failed** (`serving_gap_v1`, `serving_gap_v2`,
both CLOSED NO-GO the same day, 192 live production runs). Herding: refuted backwards — the graph arm
spreads its node choices *more* than its twin. Queue blindness: refuted backwards — it is **3–7×
more** load-responsive, p < 1e-4 on every corpus. Group splitting: fires on 1 of the 2 corpora the
bar required, though the shortfall orders exactly like the live gap. **The reversal is measured and
unexplained.**

**What this does and does not change.**

- The count-theorem argument is untouched: it still holds for every node-indexed target here.
- The instance-pair exception is **real and offline**. Quote it as an offline result about the
  supervised target, never as a claim about a served scheduler.
- **No measurement in this program has a graph-reasoning arm beating both its pointwise twin and the
  reactive baseline.** That remains the open question, and it is now known to be open in a sharper
  way than "we have not tried": the offline and live answers disagree in sign on the same weights.
- The one deployable result of the arc is a **serving** fix, not a modelling one: a per-platform cap
  in the masked decoder takes the graph arm from −1.5 % to **+21.9 % against reactive Knative,
  16/16 seeds, p = 3.1e-05**, and helps the pointwise twin about as much (+20.2 %).

**Method correction this forces, on top of the corpus-size one above.** An offline model-class
contrast on this simulator is not evidence about live behaviour *even when the serving path is
proven bit-identical to the offline evaluator*. The missing ingredient is cross-batch state: offline
every batch is scored into the state it was labelled in; live, 45,375 batches land on each other's
queues. Relabelling on realistic loaded snapshots does not close that gap either — measured, 4,400
live-visited states, additive R² 0.99999 (`cosim_deepdive_v1`). Full record:
`docs/lineages/peer_affinity_v1.md`, `serving_gap_v1.md`, `serving_gap_v2.md`.
