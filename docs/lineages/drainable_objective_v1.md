# drainable_objective_v1 — does a label that charges the externality beat one that does not?

**Status:** `REGISTERED` 2026-09-14 — every bar below is signed before any datum exists.

**Parent:** [`drainable_debug_v1`](drainable_debug_v1.md). Its D1 read fired H-MYOPIA on all three
state sources: the T1b checkpoints make **better one-step plans than reactive Knative everywhere**
(median regret 0.35 % against 16.06 % on the reactive arm's own states) and still lose the x4000
live gate by **−226.04 %**. Its D2 read found the excess queue is **cross-batch** — only 9.6–22.9 %
of it sits behind a task the arm's own decode placed — and that the learned arms take a replica
**18–43 tasks deeper than the shallowest** at p95 in ~36 % of the placements where a choice
existed, buying 0.7 pp of peer co-location. The parent's own words: *"the supervised target
rewards exactly the behaviour that loses live … Knative is not a better scorer. It is a worse
scorer with a better externality."*

The parent named the lever — **the objective** — and registered nothing for it. This node is that
registration. It also carries a defect the parent did not look for, found while designing this one
and stated in full below, because it changes what "the objective" even means here.

## The two things this lineage claims are wrong with the label, in order

### 1. The label prices a standing queue on the wrong clock (a physics defect, not myopia)

The cold corpus the T1b checkpoints were trained on compresses a seeded backlog with
`seed_virtual_warmup` (`src/placement/infrastructure.py:692-757`):

```
total_time = cold_start + count × (execution + comm)
```

Live, a queued task also pays its **peer transfers** — seconds each at 200 MB payloads — plus the
source→platform latency. The repository already knows this and says so at
`src/placement/live_snapshot_seed.py:195-200`: the exec+comm formula *"under
HEROSIM_PEER_EXCHANGE=1 understates a deep queue's drain ~100x"*, which is why a snapshot that
measured its own drain (`live_audit.platform_queue_drain_seconds`) replays that clock instead.
**Captured (warm, D1) states therefore run on the live clock; the cold T1b corpus does not.**

Motivating measurement taken while designing this node (disclosed here as motivation; the
registered bar is A1, below, which recomputes it with a committed tool):

| per queued task ahead of the batch | median | p90 | n |
|---|---|---|---|
| T1b corpus states (`system_state_captured_unique.json`, 150 datasets) | **0.38 s** | 1.18 s | 1,472 placements |
| live x4000 captures (D1 snapshots, `queue_drain_seconds / queue_length`) | **3.6–5.5 s** | 10.4–12.1 s | 3 sources |

A model taught that twenty queued tasks cost ~8 s, served where they cost ~100 s, will go deeper
than the shallowest replica — which is exactly the behaviour D2 measured, and it needs no myopia
to explain it. **Shaping a mis-clocked label would trade the shaped term off against a wrong
base**, so the clock is fixed first and gets its own control arm (V = 0) in the gate.

### 2. The label charges nothing for what the batch does to later arrivals (the myopia proper)

With one-at-a-time FIFO platforms (`infrastructure.py:1240-1251`), work added to a platform
lengthens its busy period and every later arrival routed there waits behind it. For a platform
holding backlog `B_p` seconds, a batch adding `A_p` seconds, and a fluid arrival rate `λ_p`, the
extra waiting inflicted on **future** arrivals is `(λ_p/2)·[(B_p+A_p)² − B_p²]`. The one-step
sweep charges the batch's own queueing and nothing beyond it; the reactive count-`min` avoids the
term by construction without ever computing it.

**The label under test:**

```
B_p       = q_p · c_p                                   # backlog before the batch, seconds
A_p(plan) = n_p(plan) · c_p                             # work this plan adds, same clock
L_V(plan) = rtt_liveclock(plan) + V · Σ_p (λ_p/2)·[(B_p+A_p)² − B_p²]
```

`c_p` is the measured per-item drain for that (task type, platform type) from A1; `λ_p` is the
trace's per-type arrival rate at the gate rung divided by the replicas of that type in the state;
`rtt_liveclock` is the sweep RTT of a corpus **generated with the measured drain**, so the base is
the simulator's own number and not an additive reconstruction. `V ∈ {0, 0.5, 1, 2}`, with `V = 0`
the clock-only control and `V = 1` nominal (the fluid term at its physical value, no fitted gain).

## Why no existing hard stop covers this

Checked against each neighbouring stop, with the sentence that binds it:

* **The horizon-return stop** (`objective_pivot_v1` Phase 2; `hard-stops.md`). That label was the
  **rollout return** of the live simulator over h = 10 s — *"total RTT over batch AND horizon
  tasks"* — and it died because its **ranking did not survive a change of horizon length**
  (median Spearman ρ(h2,h10) = −0.027, the h10 optimum at rank 120/256 at h5). `docs/lessons.md`
  L32 names the mechanism: a label built by letting a simulator run forward measures *"the
  decision plus everything the simulator did afterwards."* **`L_V` runs no simulator forward.** It
  is a closed form in `(q_p, c_p, n_p, λ_p)`, deterministic, and monotone in the work a plan piles
  on a platform. The stop's mechanism cannot apply — but its **control** is inherited anyway, as
  A2's rank-stability bar, because L32 makes that control non-optional for anything in this family.
* **The closed-loop policy-gradient stop** (Phase 3: paired −0.849 %, p = 0.928, n = 120,
  adequately powered). Its reopen clause demands *"a different configuration with its own powered
  tuning stage."* This lineage is **supervised**; it trains no policy against the live simulator
  and inherits none of that apparatus.
* **The warm-corpus stop** (`peer_affinity_warm_v1` W0.b: the served **one-step** label is
  pointwise-recoverable, median regret 0.0 %, 0/100 above 2 %). That closed *one-step* labels on
  served states. `L_V` at V > 0 is not one-step. Its own W0.b-shaped question is A2.
* **The arrival-rate stop**, as amended by the parent: a re-run at another rate needs every policy
  time constant scaled and a counter-read control bar. This lineage does not change the rate — it
  runs at the parent's x4000 rung, its 16 s window and its Q = 100 — and carries the counter bars
  regardless (C0, C1).

## What this lineage concedes before it starts: it is a LABEL lever, not a GNN lever

`Σ_p[(B_p+A_p)² − B_p²]` expands into per-platform counts, their squares and their cross terms.
That is precisely the registered **count competitor v2** — *"per-(platform, type) counts, their
squares, and the platform occupancy square — the second-order sufficient statistics of any
symmetric function of a platform's co-resident multiset"* — which `peer_affinity_v1` Amendment A6
measured to **R² 1.0000 median** on exactly this simulator's serialisation. By the composition
theorem (`route_a_v1`), a pointwise scorer handed those columns expresses `L_V` exactly.

**So the registered prediction is that `mpoff` gains as much as `gnn`** (bar C5), the primary
contrast is shaped-vs-unshaped at matched architecture (C3), and the headline is
learned-vs-reactive (C4). **No GNN-vs-MLP claim is made or permitted from this lineage.** A
GNN-NEEDED reading at C5 would contradict the theorem and is to be reported as an anomaly to
investigate, never as a headline.

## Why the label side should differ from D5, which already tried the decode side

D5 (parent) swept `GNN_PREFIX_CONCURRENCY_PENALTY ∈ {0, 0.5, 1, 2}` on seed 1 and closed
**16.4 %** of the gap at 1 while going **−40.6 %** at 2 — the decode-time twin of `V`. Four
structural differences, stated now so the comparison is honest either way:

| | D5 penalty (`prefix_serving.py:476-554`) | `L_V` |
|---|---|---|
| granularity | **node**, summed over its platforms | **platform** — the FIFO server that actually serialises |
| reference | queue **imbalance** vs the least-loaded candidate node ⇒ **identically zero on a uniformly deep cluster** | absolute backlog in seconds |
| units | dimensionless share × the batch's own logit spread | seconds of inflicted wait |
| the batch's own contribution | none: `depth` is fixed at batch start, commits do not increment it | `A_p` is the plan's own added work, and the term is quadratic in it |
| the scorer | re-ranks scores fitted to the mis-clocked one-step label | the scorer is **fitted to** `L_V`, on the corrected clock |

That is a reason to expect a different result, not a guarantee of one. **The over-correction D5
found at 2 is registered as a real risk**, which is why `V` is a ladder and why A2 refuses to
train a `V` whose plan ranking is unstable.

## Bars

Sign convention: regret % = 100 · (plan − optimum) / optimum, medians over datasets or seeds.

### Phase A — reads only, no training. Bars committed with the read tools, before their data.

| bar | statistic | threshold | what it decides |
|---|---|---|---|
| **A1** | live per-item drain (from the three D1 capture files) ÷ the `seed_virtual_warmup` formula on the same (task type, platform type), median | ≥ **3.0×** ⇒ CLOCK-DEFECT | whether Phase B regenerates the corpus on the measured clock. Emits the drain table itself. |
| **A2** | on the 151 D1 states (already on the live clock), relabel every enumerated sweep row at each `V`: fraction of states-with-real-choice where the **shortest-queue** plan's regret ≤ the `gnn` checkpoint plan's regret, at V = 1 | ≥ **60 %** ⇒ LABEL-AGREES-WITH-STREAM | whether the shaped label *prefers the behaviour that wins live*. This is the cheapest possible falsification of the whole idea and it costs no training. |
| **A2-rank** | median Spearman of the full plan ranking between V = 1 and V = 2, and between V = 1 and V = 0.5 | ≥ **0.80** on both | the inherited horizon-chaos control. A `V` that fails it is **not trained** — the label would not be a stable property of the (state, action) pair. |
| **A3** | Spearman between the closed-form `(λ_p/2)[(B+A)²−B²]` predicted for a batch and the **realized** queue time later tasks provably spent behind that batch's tasks, over ≥ 5,000 batches per arm, on D2's six retained raw results | ≥ **0.50** | whether the fluid term tracks the externality that actually occurred. The fitted slope is reported as a calibration of `λ_p` and is **not** used to pick the gated `V` — V = 1 is fixed here, in advance. |

A NO-GO on any Phase A bar is recorded and **Phase B and C still run** (rule 6): an offline screen
orders the work, it does not close it. The single exception is A2-rank, which gates *which V values
are trained*, not whether the lineage proceeds; if V = 1 itself fails A2-rank, the gate runs with
V = 0.5 substituted and the substitution is stated in every sentence quoting the result.

### Phase C — the live gate. This is what closes the lineage.

x4000 (`drainable_f4000_n50000.json`), `cell_s7901_f4000_pg16.json` (16 s window — the parent's
best-for-both-arms configuration), Q = 100, **uncapped** (`GNN_PREFIX_PLATFORM_CAP=1` deadlocks
3/16 seeds in a drainable regime), `HEROSIM_PEER_EXCHANGE=1`, server-only replicas, `node_disk_v2`,
16 seeds per learned arm, no `--seed`.

| bar | contrast | fires when |
|---|---|---|
| **C0** | assembly control, each learned arm's own counters | mean batch size ≥ 4 **and** incomplete peer-group batches ≤ 20 % of `prefix_batches`; otherwise that arm is **CONFOUNDED** and not read. Absent counters fail loud. |
| **C1** | behaviour control: `chosen_queue_vs_min` on two raw-retained seeds per arm | the V = 1 arms place above the shallowest legal replica in ≤ **18 %** of choosable placements (half of T1b's 35.5 %); otherwise the label did not do what its name says and C2/C3 are read as CONFOUNDED |
| **C2** | V = 0 vs the T1b checkpoints at this cell (53.45 s `gnn`, 51.32 s `mpoff`) | median latency lower **and** ≥ 12/16 seeds below the T1b median ⇒ **CLOCK-WAS-A-DEFECT** |
| **C3** (primary) | V = 1 vs V = 0, same architecture | median latency lower, Mann-Whitney p < 0.05, ≥ 12/16 ⇒ **LABEL-HELPS** |
| **C4** (headline) | V = 1 arms vs `knative_network` (25.95 s) | median below Knative's **and** ≥ 12/16 seeds below ⇒ **LEARNED-BEATS-REACTIVE** — which no measurement in this program has yet produced at a drainable load |
| **C5** | V = 1 `gnn` vs V = 1 `mpoff` | registered prediction **TIE**; see the concession above |

**Outcomes.** C3 ∧ C4 ⇒ OBJECTIVE-WAS-THE-LEVER. C3 ∧ ¬C4 ⇒ LABEL-HELPS-NOT-ENOUGH, with the
residual decomposed from the raw seeds. ¬C3 ⇒ OBJECTIVE-NOT-THE-LEVER, written into
`docs/hard-stops.md` with the exact configuration (the `V` ladder, the corrected clock, the cell,
the window, the rung) so it cannot be revived by forgetting. Every number is quoted with its load
factor and window, per the parent's standing rule.

## Carried limitations, stated before the data

* **The learning rate is not re-tuned.** Every arm runs at T1b's lr = 2e-3. Re-selecting it now
  would be choosing a hyperparameter with knowledge of the outcome; a powered tuning stage is a
  separate registration. This is the same limitation Phase 3 of `objective_pivot_v1` carried, and
  it is disclosed for the same reason.
* **Seed budget is asymmetric by design:** 16 seeds at V = 0 and V = 1 (the gated arms), 8 at
  V = 0.5 and V = 2 (offline dose-response only, never gated, never quoted live).
* **The cell bounds the effect size.** 53–58 % of this trace's tasks have exactly **one** legal
  replica, so more than half of it cannot distinguish two schedulers at all; and within-arm seed
  spread at this rung is larger than the between-arm gap (`gnn_s1` 48.60 s vs `gnn_s2` 73.78 s),
  which is why every live contrast here is 16 seeds and paired on nothing less.
* **`λ_p` is a fluid approximation** of a discrete arrival process on a cluster whose replica set
  moves under the autoscaler. A3 measures how well it tracks reality; it is not assumed.

## Amendment to the parent, filed with this registration

`drainable_debug_v1`'s summary sentence — *"the model is not the problem"* — is narrower than its
own data. On the states with real choice its restricted table reads 43.96 / 53.80 / 82.47 %
checkpoint regret against the faithful one-step optimum: **closer to optimal than the reactive
rule on every source, which is what D1c tested, but not at the optimum.** The clock defect above is
the candidate explanation and A1/C2 are its test. The parent node carries this amendment verbatim.

## 2026-09-14 — Phase A engineering landed, and one design check that matters

The label is implemented in one home, `scripts_cosim/drift_label.py` (23 tests), and reaches
training through `prepare_graphs_cache.py --label-objective rtt|rtt_drift:<V>` plus
`--label-arrival-rate` and `--label-drain-table`. The config travels in the environment
(`NEAR_RTT_LABEL_OBJECTIVE`, `NEAR_RTT_LABEL_ARRIVAL_RATE`, `HEROSIM_BACKLOG_DRAIN_TABLE`) because
the JSONL parse fans out over a `ProcessPoolExecutor` and a `run_experiment.py` config can only
set env; the parent resolves the config before any dataset is read, so a bad combination fails
there rather than inside a worker whose traceback the pool swallows. The corrected clock is
`Platform.seed_virtual_warmup` reading the same table, so a corpus and its label cannot disagree.
Which label built a cache is in `metadata.json`, travels into the checkpoint sidecar as
`label_objective`, and reaches a live result through the `checkpoint_mp_config` whitelist and the
provenance env list — the whitelist being the thing that makes a new sidecar key *silently* inert
otherwise.

**Controls, all green.** With the flag unset, a full cache build of the 34 held-out datasets is
**identical to the pre-change code on every graph tensor, every tied-optimal label set and every
partial-state context**, and `dataset_ids.pkl`, `optimal_rtt.pkl` and `rtt_chunk_0.pkl` match by
md5. (`graphs.pkl` is not byte-reproducible across two runs of the *same* code, so the comparison
is on contents; that was verified before it was relied on.) `tests/test_trainer_determinism.py` is
17/17 with a new case pinning bit-identical weights at a fixed seed with the shaped label declared,
and the suite is 831 passed.

**The design check, and it changes nothing in the plan but confirms its order.** Building the same
34 datasets at `V = 1` on each clock:

| clock | datasets whose one-step optimum MOVED | shaped optimum ÷ one-step optimum (median) |
|---|---|---|
| cold, `execution + comm` (the corpus's own) | **0 / 34** | 1.01× |
| live, a provisional flat 4.5 s/item | **32 / 34** | **4.38×** |

**On the clock the T1b corpus was built with, the shaped label is inert** — the term is ~1 % of
the label and never moves the argmin, because a backlog priced at 0.38 s/item makes `B` and `A`
small and the term is quadratic in them. It only bites once the backlog is priced at what it
actually costs. So the two defects are not independent levers to be tried in either order: **fixing
the clock is a precondition for the shaping term to exist at all**, which is why Phase B
regenerates the corpus first and why `V = 0` is the control arm rather than an afterthought.

Disclosed as engineering, not as a bar: the 4.5 s figure is a flat provisional stand-in for the
A1 read's per-(task type, platform type) table, chosen as the midpoint of the live captures'
3.6–5.5 s medians. A1 replaces it with the measured table before any corpus is generated.

## 2026-09-14 — Phase A read: A1 FIRES hard, **A2 DOES NOT FIRE**

Jobs 766266 (A1) / 766267 (A2), chained; A3 separately. Everything below is at the gate
rung's arrival rate, λ = 0.46 tasks/s.

### A1 — CLOCK-DEFECT, by three orders of magnitude

Nine qualifying (task type, platform type) cells over the three D1 capture sources
(781–1,351 observations each). **Median ratio 784.5× against a 3.0× bar.**

| cell | formula s/item | measured s/item | ratio |
|---|---|---|---|
| `dnn1`/`rpiCpu` | 0.0052 | 7.950 | **1536×** |
| `rf`/`xavierCpu` | 0.0058 | 4.584 | 785× |
| `dnn2`/`rpiCpu` | 0.1707 | 7.950 | 47× |
| `cnn`/`xavierCpu` | 0.7078 | 4.584 | 6.5× |
| `cnn`/`rpiCpu` | 3.0881 | 7.950 | 2.6× |

The spread across cells is the point. The live drain is **4.3–8.0 s per queued item on a
platform regardless of what type is queued**, because most of it is the peer transfers and
latency the formula omits, which are per task *instance*. The corpus charges between
0.005 s and 3.1 s depending on type. **So the corpus does not merely under-price a
backlog — it prices it as if a queued task's cost were its own compute, when live it is
dominated by a term the formula does not contain at all.**

Two things the read found on the way, both recorded rather than only survived:

* **The live cluster queues cells the cold corpus never registers.** A2's first attempt on
  the measured table died on `(dnn2, pynqFpga)`, which appears in no x200 dataset's
  `replica_placements`. Same shape as the 2026-09-13 `xavierGpu` audit. A1 now emits the
  full cross product and reports `platform_types_live` against `platform_types_in_corpus`.
* **The emitted table is the offset form**, `formula(t,p) + max(0, measured(p) − mean_t
  formula(t,p))`, not the flat measured value: the measurement is per platform (the
  snapshot does not record which types are queued), and assigning it flat would erase the
  per-type heterogeneity the simulator does model — `cnn` on `rpiCpu` costs 3.09 s of
  execution against `dnn1`'s 0.003 s, and a label indifferent to which type goes where
  would be wrong in a new way. Chosen after seeing the ratios; it changes no bar (A1's is
  the ratio) and the flat table is emitted alongside.

### A2 — LABEL-DOES-NOT-AGREE on all three sources. **This is the registered falsification, and it landed.**

Median regret against each state's own optimum *under the label in the column*, on the
states with real choice:

| source (n with choice) | V | shortest-queue | `gnn` | `mpoff` |
|---|---|---|---|---|
| `knb` (3) | 0 | 85.20 % | 43.96 % | 81.42 % |
| | **1** | **49.95 %** | **22.65 %** | 61.00 % |
| | 2 | 42.78 % | 16.75 % | 50.04 % |
| `gnn` (40) | 0 | 75.05 % | 53.80 % | 49.85 % |
| | **1** | **48.41 %** | **39.97 %** | 34.13 % |
| | 2 | 40.90 % | 35.24 % | 34.58 % |
| `mpoff` (29) | 0 | 82.11 % | 82.47 % | 73.56 % |
| | **1** | **44.07 %** | 34.41 % | 32.31 % |
| | 2 | 37.57 % | 26.73 % | 26.49 % |

**Reversal rate at V = 1: 33.3 % / 42.5 % / 34.5 % against a 60 % bar → LABEL-DOES-NOT-AGREE
on every source.** A2-rank is **STABLE** everywhere (median Spearman 0.888–0.985 between
V = 1 and V = 2, 0.967–0.996 between V = 1 and V = 0.5, bar 0.80), so every V in the ladder
is trainable and the horizon-chaos failure mode is absent — this label *is* a stable
property of the (state, action) pair, which the one it replaces was not.

**Direction confirmed, magnitude insufficient.** The shaped label likes the reactive plan
much more than the one-step label does: shortest-queue regret falls 85.2 → 50.0 % on the
reactive source and 75.1 → 48.4 % on the `gnn` source. It just likes the checkpoint's plan
more as well, and the ordering never flips. **The externality term as specified does not
make the supervised target prefer the behaviour that wins the stream.**

Caveat carried: the `knb` source has only **3** states with real choice out of 50 (its
median whole-group plan space is 8 rows, as D1 recorded), so its 33.3 % is one state in
three. The `gnn` (40) and `mpoff` (29) sources carry the weight, and they agree.

### What this does and does not decide

A2 is an **offline** read. Under rule 6 it orders the work and does not close it, and the
registration said so before the number existed: *"A NO-GO on any Phase A bar is recorded
and Phase B and C still run."* Two independent reasons that is the right call here rather
than a formality:

1. **A1 fired, and the clock fix is untested live.** V = 0 — the live-clock corpus with the
   one-step label — is a control arm whose own question A2 does not touch. The record's
   standing explanation for the drainable loss is that the arms go 18–43 tasks deeper than
   the shallowest replica; A1 says they were taught a backlog costs ~1/800th of what it
   does. Whether fixing that alone moves the live gate is bar C2, and nothing offline
   answers it.
2. **A2 measures agreement with *shortest-queue*, not with the stream.** The reactive rule
   is what wins live on this cell, but "the shaped label ranks the reactive plan first" is
   a proxy for "an arm trained on the shaped label places better", and the program has a
   standing stop against exactly that inference in the other direction (`hard-stops.md`:
   do not cite offline regret as evidence about live placement).

Phase B therefore proceeds: corpus generation launched on the measured clock (jobs 766268
train, 766269 held-out), four caches, and the 96 training runs. **The registered
expectation is now explicitly a negative one for C3** — A2 predicts the shaped label will
not help — and that prediction is recorded here, before the gate, so the gate can confirm
or refute it rather than be read backwards afterwards.
