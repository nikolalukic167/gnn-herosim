# cluster_scale_v1 — the axis every load rung held fixed

**Status:** `REGISTERED` (2026-09-16). Every bar below is a module constant in
`scripts_cosim/cluster_scale_v1_read.py`, committed **before** any arm runs. Amend by dated
amendment only.

**Parents:** `scheduler_residence_v1` (CLOSED — the scheduler-side wait is **89 % peer-group
collection**, and collection is arrival-rate-bound), `drainable_regime_v1` (the load ladder
that held the cluster fixed), `queue_range_v1` (the per-term decomposition),
`drainable_serving_config_v1` (the window sweep, closed).

---

## The claim under test

Every rung of every load ladder in this program moved **one** thing: the timestamp scale.
`drainable_regime_v1`'s own registration says it "changes no physics, no payload, no
topology… the only moving part is the timestamp scale." The cluster stayed at **20 clients /
6 servers** throughout. Arrival rate and utilisation therefore moved together and could never
be separated.

That coupling is why no rung could ever be good. Two different things drive the two costs:

* **Utilisation** drives queueing. A busy cluster buries the peer mechanism the checkpoints
  optimise (at the landed gate it was 0.0098 % of the number being scored).
* **Absolute arrival rate** drives peer-group assembly. At 0.46 arrivals/s a 10-task peer
  group takes **~21.7 s** to co-arrive, and `scheduler_residence_v1` R0 measured the
  consequence directly: **6.10–6.13 s of collection per task, 89 % of the scheduler-side
  wait**, against 0.59–0.76 s of head-of-line and **0.000 s** of decode.

With the cluster fixed, the only way to cut utilisation was to slow arrivals, which *raises*
assembly cost. The ladder could buy one or the other, never both.

> **The claim:** hold ρ ≈ 0.16 and raise the arrival rate by adding capacity. Assembly cost is
> arrival-bound and service time is not, so collection should fall from ~6.1 s toward ~0.5 s
> while queueing stays where it is.

## The measurement that constrains this before anything runs

Measured 2026-09-16 on the arms' own training corpus
(`graphs_cache_drainable_objective_v1_v1`, 516 datasets, 5,160 task slots):

| | p50 | p90 | p99 | **max** |
|---|---|---|---|---|
| **candidates per task** | 3.0 | 4.0 | 5.0 | **5** |
| candidates per dataset | 28 | 32 | — | 38 |

**The decoder has never been shown more than five candidates for a task.** Candidates are the
reachable replicas of that task's type, and with `per_server: 1` every server hosts every
type — so candidates per task scales with reachable servers. At 6 servers it is ~3; at 24 it
would be ~12; at 80, ~40.

**So the arithmetic says, before a single arm runs, that scaling servers densely takes the
served candidate set 2–8× outside anything the corpus contains.** That is the `xavierGpu`
audit shape (7.60 % of live candidates out of corpus) except far larger and affecting every
decision rather than some. It is recorded here as the reason S1 is designed as a *bounded,
possibly-confounded probe* rather than a claim, and as the reason **S0 carries no checkpoints
at all**.

## The rungs

ρ = A / (0.4717 × servers), from the landed drain of **2.83 tasks/s on 6 servers**. Workloads
already exist; no trace is regenerated.

| rung | workload | arrivals/s | servers | nominal ρ | 10-task group assembles in |
|---|---|---|---|---|---|
| **R1** (baseline) | `drainable_f4000_n50000` | 0.460 | **6** | 0.163 | ~21.7 s |
| **R2** | `drainable_f1000_n50000` | 1.842 | **24** | 0.163 | ~5.4 s |
| **R3** | `drainable_f300_n50000` | 6.139 | **80** | 0.163 | ~1.6 s |

Cells are derived from `cell_s9001_f4000_pg16` by changing **only** `nodes.server_nodes.count`,
asserted field-by-field, and **4 topology seeds per rung** so a rung is never one draw
(`scheduler_residence_v1` R3: 5 of 20 draws hang on every policy, so budget **25 % attrition**).

**Every policy time constant is held fixed** — window 16 s, poll 1 ms, batch size 10 —
because they are the thing whose *cost* is being measured. `drainable_regime_v1`'s rule says a
rate sweep must scale them; that rule exists so a policy stays self-similar across rungs, and
here self-similarity is precisely what would hide the effect: R0 measured that the collector
**exits the moment the group is complete**, so a fixed window is an upper bound that binds less
as arrivals speed up. Scaling it would rescale the answer to zero by construction. This
departure is deliberate, registered, and the realised collection time is measured rather than
assumed.

## Bars

**S0.a — load match (blocking).** A rung that did not hold ρ is measuring load, not scale.
Reactive `knative_network` mean queue time at R2 and R3 must lie within
`S0_RHO_BAND = [0.33, 3.0]` × R1's. Outside it, that rung is **VOID**.

**S0.b — the primary, and the mechanism.** Median GNN **collection** time (the instrument
built for `scheduler_residence_v1` R0, which reconciles with `averageWaitTime` to six decimal
places) must fall **monotonically** across the readable rungs and reach
`S0_COLLECTION_MAX_S = 2.0` at the fastest one, from a measured 6.10–6.13 s at R1.
⇒ **`ASSEMBLY-IS-ARRIVAL-BOUND`**. Otherwise **`ASSEMBLY-DOES-NOT-SCALE`** and the whole axis
closes for one afternoon of CPU, with no checkpoint loaded and nothing retrained.
**Registered expectation: POSITIVE.** The mechanism is arithmetic and R0 measured its
ingredient; this bar is placed to be falsifiable, not because it is in doubt.

**S0.c — feasibility (blocking for S1, not for S0).** Median wall-clock per arm at the fastest
readable rung ≤ `S0_MAX_WALLCLOCK_MIN = 45`. `create_first_replica` was 59.3 % of runtime and
scales with candidate count, so an 80-server rung may simply be unaffordable — which is an
answer, recorded as `RUNG-UNAFFORDABLE`, not a failure.

**S0.d — candidate support, computed not simulated (blocking for S1's readability).**
Median and max candidates per task at each rung, against the corpus maximum of **5**.
`S0_CORPUS_CANDIDATE_MAX = 5`. A rung above it is marked `OUT-OF-SUPPORT` **in advance**, and
S1 on that rung is reported `CONFOUNDED-CANDIDATE-SUPPORT` whatever its latency.

**S1 — the live gate (rule 6), on the existing checkpoints.**
Per readable rung: 4 topology seeds × (1 reactive + 4 `gnn` + 4 `mpoff`).

* `S1_MIN_CELLS = 3` readable cells per rung, `S1_ALPHA = 0.05`, `S1_HOLM_N = 3` (rungs).
* **Primary:** `gnn` median elapsed vs reactive's, paired by topology seed, Wilcoxon
  signed-rank (the arms share seeds). **`SCALE-HELPS`** if the arm's deficit shrinks
  monotonically across rungs **and** the fastest readable rung is not worse than R1's.
* **`S1_CONFOUND` (blocking, registered in advance):** any rung marked `OUT-OF-SUPPORT` by
  S0.d reports `CONFOUNDED-CANDIDATE-SUPPORT`. **A clean positive there needs a retrain to
  confirm; a clean negative is still informative, because it cannot be blamed on the
  confound.** Stated now so neither outcome can be re-interpreted later.
* **`S1_DIFFICULTY` — pre-declared covariate, and the point of writing it down now.**
  `scheduler_residence_v1` R3 found, *post hoc and not significantly* (ρ = −0.446, p = 0.095,
  n = 15), that the learned arms beat reactive on exactly the topology draws where **reactive
  itself is slow** — 3 of 15 draws, margins −20.3 / −24.0 / −14.0 s, against 0 of 3 on the
  cells this family has always used. **Difficulty is defined here, before the data exists, as
  reactive's own median elapsed on that cell**, and `S1_DIFFICULTY_MIN_ABS_RHO = 0.50` with
  the registered sign **negative**. It is reported beside the primary under the same Holm
  family, and it is the first time this program tests that claim rather than noticing it.

`mpoff` is carried throughout and **no GNN-vs-pointwise claim may be founded here** — cluster
size is policy-agnostic and must move both.

## Declared in advance

* **Seeds 1, 2, 4, 5.** Seed 3's checkpoint deterministically livelocks the simulator on
  `cell_s7901`; excluded by name rather than risked on fresh draws.
* **25 % cell attrition budgeted** (`scheduler_residence_v1` R3: 5 of 20 draws hang on every
  policy including reactive, the documented starved-client spin). `--time` is set to ~3× a
  servable arm so a hang is identified in a third of the wall clock.
* **Nothing is retrained.** If S1 reads `CONFOUNDED-CANDIDATE-SUPPORT` on every readable rung,
  the honest outcome is that this axis needs a corpus built at the served cluster size — a
  separate registration, and one that must first read `peer_affinity_warm_v1` (training on
  served states is CLOSED, NO-WINNING-GNN) to say why it would differ.

## Not in scope

The batch window (swept, closed); removing batching (closed, −1731 %); the queue column
(closed); retraining; sparser reachability as a way to hold candidate count down while adding
capacity — a real alternative, noted here so it is not lost, and a separate registration
because it changes the topology distribution rather than its size.

## Record

### 2026-09-16 — S0.d read, no simulation: **BOTH SCALED RUNGS ARE OUT-OF-SUPPORT**

`simulation_data/cluster_scale_v1/s0d_cells.json`. 12 cells minted (3 rungs × 4 topology
seeds), each verified to differ from the base in exactly the two declared fields.

| rung | servers | mean candidates/task | min | vs corpus max of 5 | support |
|---|---|---|---|---|---|
| R1 | **6** | **3.55** | 1 | **0.71×** | **IN-SUPPORT** |
| R2 | 24 | **14.18** | 9 | **2.83×** | **OUT-OF-SUPPORT** |
| R3 | 80 | **47.92** | 38 | **9.58×** | **OUT-OF-SUPPORT** |

The predicted arithmetic holds exactly. **Every rung that adds capacity takes the served
candidate set outside anything the corpus contains**, and at R3 the *minimum* candidate count
(38) is nearly 8× the corpus *maximum* (5). By the bar registered above, S1 on either scaled
rung reports `CONFOUNDED-CANDIDATE-SUPPORT` whatever its latency.

**This is a general statement about the axis, not about these three rungs.** Drain is
proportional to replicas and candidates are the *reachable* replicas, so any lever that raises
capacity raises the candidate count with it — unless reachability is thinned at the same rate,
which is the separate registration noted under *Not in scope* and which pushes straight into
the starved-client spin (`scheduler_residence_v1` R3 measured **5 of 20** draws hanging on
every policy already, and thinning reachability is what makes that worse).

**So: the cluster cannot be scaled and these checkpoints kept in distribution.** Scaling this
axis requires a retrain. That was registered as a possible S1 outcome; S0.d establishes it
before a single arm runs, for free.

**What still runs, and why it is worth it.** S0 asks whether peer-group assembly cost actually
falls with the arrival rate — the mechanism, and the thing that decides whether the axis is
worth a retrain at all. **That question needs no in-distribution checkpoint:** collection
happens strictly before any decode and never consults the model
(`src/policy/gnn/scheduler.py:433-457`), so collection time is model-independent. The `gnn` arm
in S0 is an **instrument**, and no latency number from it is a quality claim.



### 2026-09-16 — S0 read: **RUNGS-NOT-LOAD-MATCHED**, and a hard structural blocker

Job 769363, 14 of 24 arms. Two arms lost to the budgeted starved-client attrition
(`cs6s9004`, both policies), and **every `gnn` arm above 6 servers FAILED, loudly and for the
same reason** (below).

**S0.a — load match: FAILS on both scaled rungs, so R2 and R3 are VOID.**

| rung | servers | arrivals/s | reactive queue | vs R1 | realised throughput |
|---|---|---|---|---|---|
| R1 | 6 | 0.460 | **12.17 s** | — | 0.460 /s |
| R2 | 24 | 1.842 | **308.86 s** | **25×** | 1.829 /s |
| R3 | 80 | 6.139 | **601.25 s** | **49×** | 5.550 /s |

The band was [0.33, 3.0]×. This is 25× and 49×. **The rungs were not held at ρ ≈ 0.16 — they
are close to saturation**, and the capacity model in the registration is wrong.

**Why, and it is worth recording.** The sizing used **2.83 tasks/s on 6 servers**, the drain
`drainable_regime_v1` measured — but that figure comes from the *landed overloaded gate*, where
the autoscaler had long since built every replica it was ever going to build. From a cold start
the realised per-server drain is **~0.077 tasks/s**, about **6× lower**. Holding ρ ≈ 0.16 at
6.139 arrivals/s therefore needs on the order of **480 servers**, not 80. ⇒ **Size a rate ladder
from the drain the cluster actually achieves in the regime being run, not from a drain measured
in a different one.** Throughput does scale roughly with servers (0.46 → 1.83 → 5.55 /s); the
error was purely in the constant.

**The blocker, which makes the sizing error moot for these checkpoints.** Every `gnn` arm at 24
and 80 servers died on:

```
ValueError: krank_node_order: 7 nodes exceed the registered pad width 6
```

`src/policy/tabular/reduced_features.py:252` — `KRANK_WIDTH = 6`, *"registered pad width R = the
route_b grid's max node count"*. The partial-state feature block encodes candidate-hosting nodes
in a **fixed-width, rank-ordered pad of six**, and `krank_node_order` raises rather than
truncate. **So the served feature layout cannot represent a cluster with more than six
candidate-hosting nodes at all.** The guard is correct and it fired correctly; it is a property
of the trained representation, not a bug.

Two independent structural facts now stand against this axis for these checkpoints, and they
compound rather than substitute:

1. **Candidate support** (S0.d, free): 14.2× and 47.9× candidates per task against a corpus
   maximum of 5.
2. **Representation width** (S0, measured): a hard cap of **6** hosting nodes in the feature
   layout, enforced by a raise.

**The cluster-scale axis requires a retrained representation, not just a retrained model.**
Widening the pad changes `PARTIAL_STATE_FEATURE_DIM` and therefore every cached graph and every
checkpoint in the program.

---

## Amendment 1 (2026-09-16) — decouple the mechanism test from the capacity test

**Signed before the S0.b arms run.** No bar moves; S0.a's verdict and R2/R3's VOID stand.

S0.b — *does peer-group assembly cost fall when arrivals speed up?* — is the question that
decides whether this axis is worth a retrained representation, and **it did not need the
scaled rungs at all.** Collection is a property of the arrival process and the peer-group rule:
`_collect_task_batch` waits for a task's peer-group members to arrive and **exits the moment
the group is complete** (`src/policy/gnn/scheduler.py:443`), strictly before any decode, never
consulting the model. **So collection time can be measured at 6 servers, inside the pad width,
by varying only the workload.**

**S0.b-amended:** 6 servers, 4 topology seeds, three workloads — `f4000` / `f1000` / `f300`
(0.460 / 1.842 / 6.139 arrivals/s). 12 arms. Bars unchanged:
`S0_COLLECTION_MAX_S = 2.0` at the fastest rung, falling monotonically, from the 6.10–6.13 s
measured at f4000.

**Disclosed: these rungs are overloaded** (6 servers cannot drain 6.139 arrivals/s), so their
**latency numbers are meaningless and are not read** — S0.a already VOIDed load-matching and
this amendment does not resurrect it. The measurement is collection time alone, and its
mechanism is arrival-driven by construction. If collection does **not** fall, the axis is closed
on its own mechanism and no retrain is ever justified; if it does, the axis is worth exactly as
much as a new representation costs.
