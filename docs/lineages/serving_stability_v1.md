# serving_stability_v1 — the learned arms place better and control worse

**Status:** `REGISTERED` (2026-09-15). Every bar below is signed **before** its data. Amend by
dated amendment only.

**Parents:** `drainable_regime_v1` (both learned arms lose to reactive at a defensible load),
`offline_live_transfer_v1` (CLOSED — the offline score cannot rank arms; a single-seed live
number measures whether that seed hit an excursion).

---

## The claim under test

Every reading in this program has treated the learned arms' loss to reactive Knative as *being
worse at placement*. The scoping read below says that is the wrong description:

> **The learned arms beat reactive Knative while queues are shallow, and lose once their own
> queues run away. Reactive's queues cannot run away, because `min(queue)` is self-stabilising
> by construction. The learned arms have no such mechanism.**

If that is right, the lever is **not** a better placement score — three lineages have now
failed at that — but a **serving-time stabiliser** that keeps the arms in the regime where they
already win. That is a different intervention, it needs no retraining, and it is cheap to gate.

## The scoping measurement that motivates this, disclosed, with what it burns

Read 2026-09-15 **before these bars were written**, no bars of its own
(`scripts_cosim/jam_probe_shape_read.py`, output `simulation_data/jam_probe/shape.json`).
Reactive `knative_network` against two V = 1 `gnn` checkpoints, all three reproducing their
gate latencies exactly (25.9459 / 43.9905 / 74.7555 s), on
`drainable_f4000_n50000` at `cell_s7901_f4000_pg16`.

**BURNED and excluded from every bar below:** `gnn` seeds **8** and **14**, and the reactive
arm, **on that trace-and-cell combination**.

| decile | reactive queue | `gnn` s14 queue | excess | share |
|---|---|---|---|---|
| 1 | 22.412 | **16.285** | **−6.128** | −0.049 |
| 2 | 12.517 | **9.727** | **−2.790** | −0.022 |
| 3 | 14.213 | 27.216 | +13.003 | 0.104 |
| 4 | 12.892 | 26.414 | +13.522 | 0.109 |
| 5 | 14.534 | 44.937 | +30.403 | 0.244 |
| 6 | 17.081 | 28.806 | +11.724 | 0.094 |
| 7 | 21.836 | 41.684 | +19.848 | 0.159 |
| 8 | 22.385 | 45.707 | +23.322 | 0.187 |
| 9 | 18.755 | 28.792 | +10.038 | 0.081 |
| 10 | 20.944 | 32.442 | +11.498 | 0.092 |

Three things in that table:

1. **The loss is not uniform.** The worst decile carries **0.244** of the excess for the good
   seed and **0.387** for the bad one, against 0.100 for a uniform loss; the top three carry
   0.591 and 0.749 against 0.300.
2. **Deciles 1 and 2 are negative — the graph arm is AHEAD of reactive**, by 27 % and 22 % of
   queue time, and it happens for **both** checkpoints including the bad one. No measurement in
   this program's record has previously shown a graph arm ahead of reactive on any slice.
3. **Reactive's queue is bounded and the arms' are not.** Across all ten deciles reactive spans
   12.5–22.4 s (**1.8×**); `gnn` s14 spans 9.7–44.9 (**4.6×**); `gnn` s8 spans 9.4–185.1
   (**19.7×**).

This read has no bars, n = 2 checkpoints, n = 1 trace, n = 1 cell. It orders this lineage and
decides nothing in it.

## Why no existing stop covers this

* **It is a serving change, not a label or a corpus.** The count-theorem concession, the
  horizon-return stop, the closed-loop PG stop and the warm-corpus stop all govern what a model
  is *trained on*. Nothing here retrains anything.
* **It is not the platform cap.** `GNN_PREFIX_PLATFORM_CAP=1` is closed as a serving default —
  it **deadlocks 3/16 seeds** in a drainable regime (`drainable_regime_v1` Amendment 2) — and it
  caps *per-platform task count within a batch*, which is a concentration limit with no
  reference to queue state. The guardrail here is on **queue depth relative to the shallowest
  candidate in the same batch**, so it is a no-op when queues are even, whatever the
  concentration. **That difference is exactly what could be wrong**, so the deadlock control
  below is a blocking bar, not a footnote.
* **It is not `drainable_serving_config_v1`.** That varied the batch *window* and found the
  graph arm cannot decode singletons. The window is held fixed here at 16 s.

## Bars

Replication is across **cells** — independent topology and placement-seed draws (`cell_s9xxx`)
— not across traces, because the drainable traces are timestamp rescales of one event sequence
and are therefore not independent draws. Three cells: **C1 = s7901** (burned only for `gnn`
seeds 8 and 14), **C2** and **C3** = two fresh `s90xx` cells given the identical `f4000` +
`pg16` treatment. Trace, window (16 s), Q = 100 and uncapped are fixed throughout.

### S0 — engineering controls (must pass, or the reads are VOID)

* **Per-decile queue is computed in the run's own summary step** from `taskResults`, and the
  ~780 MB raw is discarded. 99 arms × 780 MB is not a storage plan, and a read that can only
  afford two captures is how this lineage's parent ended at n = 2.
* **The derived C2/C3 configs differ from their base cell only in the `f4000` + `pg16` fields
  that C1 differs from *its* base by.** Asserted field-by-field, not eyeballed.
* Every arm reproduces its own recorded latency where one exists.

### S1 — does the early advantage replicate? (the surprising one)

Per arm and seed: mean queue time over deciles 1–2, minus reactive's on the same cell.

* **Bar:** the median over seeds is **< 0**, in **≥ 12/16 seeds**, on **≥ 2 of 3 cells**, for
  the `gnn` arm ⇒ **EARLY-ADVANTAGE-REAL**.
* **Registered expectation: uncertain, and this is the bar most likely to fail.** It rests on
  two checkpoints on one cell. It is placed first because it is the claim that would change what
  this program believes, and therefore the one that most deserves a chance to evaporate.
* The `mpoff` arm is read identically and reported. A result on `gnn` alone is **not** a graph
  claim: by the count theorem this lineage may found no GNN-vs-pointwise claim of any kind.

### S2 — do the arms fail to stabilise?

Per arm and seed: (max over deciles of mean queue) ÷ (min over deciles of mean queue).

* **Bar:** reactive's ratio ≤ **2.5** AND each learned arm's median ratio ≥ **2×** reactive's,
  on **≥ 2 of 3 cells** ⇒ **ARMS-DO-NOT-STABILISE**.
* Scoping values for scale, not bars: reactive 1.8, `gnn` s14 4.6, `gnn` s8 19.7.

### S3 — the live gate that closes the lineage (rule 6)

**The guardrail.** At decode time, a candidate replica is masked when its queue depth exceeds
`S3_K × (shallowest candidate depth in this batch + 1)`, with `S3_K = 3`, **and the mask is
relaxed whenever it would leave a task with no candidate** — the existing
`decode_relax_on_stuck` path. It is a no-op when queues are even. `S3_K` is **not tuned**; 3 is
registered in advance and a carried limitation.

| bar | fires when |
|---|---|
| **S3-a deadlock control** (blocking) | **0/16** seeds hang on each cell. Any hang ⇒ the guardrail is **not a serving default**, reported with the count, and S3-c is read as CONFOUNDED. The platform cap deadlocked 3/16 here; assume nothing. |
| **S3-b bind control** (blocking) | the arm's own counters show the mask bound on ≥ **5 %** of decisions. Below that it did nothing and the gate is VOID — `drainable_regime_v1` lost three reads to policies that did not do what their name said. |
| **S3-c primary** | guarded `gnn` vs `knative_network`: median latency below AND ≥ **12/16** seeds below, on ≥ 2 of 3 cells ⇒ **LEARNED-BEATS-REACTIVE**, which no measurement in this program has produced at a drainable load. |
| **S3-d paired** | guarded vs unguarded, same seeds, Mann-Whitney p < 0.05 and ≥ 12/16 ⇒ **GUARDRAIL-HELPS**. |

**Outcomes.** S3-c ∧ S3-d ⇒ **STABILITY-WAS-THE-LEVER**. ¬S3-c ∧ S3-d ⇒ **HELPS-NOT-ENOUGH**,
with the residual decomposed by decile. ¬S3-d ⇒ **STABILITY-NOT-THE-LEVER**, closed in
`docs/hard-stops.md` with the exact guardrail, K, cell set and window so it cannot be revived by
forgetting.

## Carried limitations, stated before the data

1. **The scoping read is burned** (`gnn` 8 and 14, and reactive, on C1's trace-and-cell) and is
   excluded from every bar.
2. **One load rung.** x4000 drainable only. `drainable_regime_v1` showed conclusions at this
   cell do not survive a load change; nothing here may be quoted without its load factor.
3. **`S3_K = 3` is not tuned**, the same way `objective_pivot_v1` Phase 3 carried lr = 1e-4.
4. **Cells, not traces.** The rescaled traces are not independent draws, so replication is over
   topology/placement seed. A cell-specific result is a cell-specific result.
5. **No GNN-vs-pointwise claim may be founded here.** A queue guardrail is exactly the kind of
   per-platform-count-and-depth function the count theorem covers; `mpoff` is expected to gain
   as much, and that prediction is recorded now.

---

## 2026-09-15 — the reads: **S1 FIRES, S2 does not, S3 says STABILITY-NOT-THE-LEVER**

99 unguarded arms (job 768098) and 96 guarded (768219) across three cells; one arm absent in
each, the known `s7901 × gnn seed 3` livelock, declared before the runs.

### S1 — EARLY-ADVANTAGE-REAL. The registered expectation was UNCERTAIN and it fired at full strength

| cell | reactive early queue | `gnn` median delta | ahead | `mpoff` median delta | ahead |
|---|---|---|---|---|---|
| s7901 | 17.464 s | **−4.136 s** | **13/13** | −3.479 s | 16/16 |
| s9001 | 14.968 s | **−4.070 s** | **15/15** | −3.760 s | 16/16 |
| s9002 | 14.550 s | **−3.475 s** | **15/15** | −3.416 s | 16/16 |

**Every seed on every cell, both arms — 43/43 `gnn` and 48/48 `mpoff`.** The learned arms
really are faster than reactive Knative while queues are shallow, and it replicates on two
topologies that had never been looked at. This is the first thing in this program's record
that has a learned arm ahead of reactive and survives replication.

### S2 — STABILITY-NOT-SEPARATED (1 of 3 cells)

| cell | reactive ratio | `gnn` median | `mpoff` median | |
|---|---|---|---|---|
| s7901 | 1.79 | 10.67 | 7.38 | fires |
| s9001 | 2.01 | **2.17** | **2.29** | no — the arms are as stable as reactive |
| s9002 | **3.42** | 26.92 | 31.30 | no — reactive fails its own bar |

"The learned arms cannot control their queues" is **not** a general property.

### The mechanism story does not survive its own follow-up

Across the three cells, arm instability and the gap to reactive line up almost perfectly
(1.79 → −111 %, 2.01 → −6.8 %, 3.42 → −179 %). **That is three aggregates**, and this
program closed a lineage on exactly that shape the same day. Asked per seed **within** each
cell — exploratory, no bar — stability does not predict latency:

| cell | arm | ρ(stability, latency) | p |
|---|---|---|---|
| s7901 | `gnn` | **+0.825** | 0.0002 |
| s9001 | `gnn` | **−0.650** | 0.0064 |
| s9002 | `gnn` | +0.341 | 0.196 |
| **pooled-z, n = 95** | | **+0.088** | **0.395** |

Two cells significant in **opposite directions**, pooled zero. The correlational case for the
mechanism is not there, which is why S3 — an intervention — is the read that decides.

### S3 — the live gate

* **S3-a NO-DEADLOCK.** 0 hangs. Unlike `GNN_PREFIX_PLATFORM_CAP` (3/16 in this regime), the
  depth-relative guardrail is safe to serve. It did **not** rescue the `s7901 × seed 3`
  livelock either, so that hang is not queue runaway.
* **S3-b GUARDRAIL-BOUND.** The mask was active on **84–90 %** of decisions in every cell and
  arm, minimum 83.8 %, against a 5 % bar. It did what its name says.
* **S3-c REACTIVE-STILL-WINS, 0/3 cells.** Guarded `gnn` vs reactive: −35.2 % (s7901, 0/13),
  −5.1 % (s9001, 1/15), −216.3 % (s9002, 0/15).
* **S3-d GUARDRAIL-DOES-NOT-HELP** under the registered test, `gnn` 1/3 cells.

| cell | guarded | unguarded | change | seeds better | registered p (Mann-Whitney) |
|---|---|---|---|---|---|
| s7901 | 35.07 | 54.82 | **+36.0 %** | 13/13 | < 0.0001 → fires |
| s9001 | 22.02 | 22.35 | +1.5 % | 15/15 | 0.115 → no |
| s9002 | 74.27 | 65.40 | **−13.6 %** | 3/15 | 0.0009 → no, and worse |

**Outcome: STABILITY-NOT-THE-LEVER.**

### A mis-specified test in my own registration, disclosed and not switched

S3-d compares the **same seeds** with and without the guardrail, and the bar I signed names
**Mann-Whitney**, which is an *unpaired* test. The matched test is Wilcoxon signed-rank. Run
on all available seeds (the registered read additionally drops the two burned scoping seeds
on C1, hence its slightly different n and medians):

| cell | arm | better | registered MW p | paired Wilcoxon p |
|---|---|---|---|---|
| s7901 | `gnn` | 15/15 | < 0.0001 | 0.000061 |
| s9001 | `gnn` | **16/16** | **0.127** | **0.000031** |
| s9002 | `gnn` | 3/16 | 0.0006 | 0.025 (worse) |

**Under the correctly specified test S3-d would read GUARDRAIL-HELPS on 2 of 3 cells, and the
outcome would be HELPS-NOT-ENOUGH rather than STABILITY-NOT-THE-LEVER.** The registered
verdict stands as signed — swapping a test after seeing which way it moves the answer is how
a registration stops meaning anything — and the sensitivity is recorded here beside it so no
reader has to rediscover it. The mis-specification is filed in `docs/gates/gate-tools.md`.

Note what does **not** change under either test: **S3-c is 0/3 on both.** No configuration
here beats reactive Knative.

### What the guardrail actually does, and why the mechanism is refuted either way

It helps enormously where the arms were unstable (s7901, ratio 10.67 → +36 %), negligibly
where they were already stable (s9001, ratio 2.17 → +1.5 %), and **hurts** where they were
*most* unstable (s9002, ratio 26.92 → −13.6 %). "Stabilise the unstable arm and it recovers"
predicts the opposite of what s9002 did. The intervention's effect is cell-dependent in a way
the stability story does not explain.

`mpoff` gains as much as `gnn` throughout (S3-d fires on 2/3 cells for `mpoff` under the
registered test against `gnn`'s 1/3), which is the registered count-theorem prediction, and
**no GNN-vs-pointwise claim is founded here.**

## Outcome (2026-09-15): **EARLY-ADVANTAGE-REAL · STABILITY-NOT-THE-LEVER** — closed

| read | verdict |
|---|---|
| S0 controls | treatment is one field (`batch_timeout 0.02 → 16.0`); C2/C3 differ from C1 only by `network.topology.seed` |
| **S1** | **EARLY-ADVANTAGE-REAL** — 3/3 cells, 43/43 `gnn` and 48/48 `mpoff` seeds ahead |
| S2 | STABILITY-NOT-SEPARATED — 1/3 cells |
| S3-a | NO-DEADLOCK — 0 hangs, unlike the platform cap |
| S3-b | GUARDRAIL-BOUND — 84–90 % of decisions |
| S3-c | REACTIVE-STILL-WINS — 0/3 cells |
| S3-d | GUARDRAIL-DOES-NOT-HELP (registered test); HELPS on 2/3 under the paired test |

**Two findings, and they point in opposite directions about how hopeful to be.**

1. **The learned arms are genuinely better than reactive Knative early, everywhere.** Not a
   cell artifact, not a seed artifact: every one of 91 learned arms across three topologies
   is ahead over the first fifth of the trace. Whatever these models know, it is real and it
   is being destroyed later in the trace rather than never having existed.
2. **Keeping their queues bounded does not recover it.** The guardrail bound hard, never
   deadlocked, and still lost to reactive on every cell — while helping +36 % on one, +1.5 %
   on another and **hurting 13.6 %** on the third, the one where the arms were *least* stable.

**What is closed, precisely:** the depth-relative queue guardrail at **K = 3**, decode-time,
on the V = 1 checkpoints, at the x4000 drainable rung with a 16 s window, uncapped, Q = 100,
across cells s7901/s9001/s9002. Do not re-run this configuration. `K` was **not** tuned, and
a tuned `K` is a different experiment that must be registered as one.

**What is NOT closed:**

* **Why the early advantage is lost.** S1 says it exists; nothing here says what destroys it.
  That is the question the next lineage should ask, and it now has a measured phenomenon
  rather than a hunch to start from.
* **Any non-queue stabiliser.** This tested one mechanism — capping depth relative to the
  shallowest candidate. It says nothing about admission control, batch sizing by load, or
  giving the model a queue feature inside its trained range.
* **A tuned or adaptive K**, including one that is a no-op on cells like s9001 where the arms
  are already stable and the guardrail buys 1.5 %.

**Carried, unchanged:** one load rung; cells not traces; and no GNN-vs-pointwise claim — the
pointwise twin gained at least as much as the graph arm throughout, exactly as the count
theorem predicted before the gate ran.
