# offline_live_transfer_v1 — does the offline score have any resolving power over serving?

**Status:** `REGISTERED` (2026-09-15). Bars below are signed **before** any registered family
is read. Nothing in this node may be edited after its data exists except by a dated
amendment that says what changed and why.

**Parent context:** `peer_affinity_v1` (the offline/live reversal), `serving_gap_v1` and
`serving_gap_v2` (both CLOSED NO-GO hunting a mechanism for it), `drainable_objective_v1`
(the reversal reproduced on an unrelated label).

---

## The question

Every screen in this record uses an offline number — held-out plan regret on captured
single-batch states — to order work, select checkpoints, and in several cases to close a
direction. The program has also recorded, three times now, that the arm which wins offline
loses live.

The standing interpretation is that offline and live are **anti-correlated**, and two
lineages went looking for the mechanism behind that anti-correlation. Both failed.

This lineage tests a cheaper hypothesis that would explain the failures: **the offline score
may simply have no resolving power over live outcome at all.** An anti-correlation observed
across two arm means, or three corpus rungs, cannot be distinguished from zero signal plus an
offset. If that is what has been happening, there is no mechanism to find.

## The motivating measurement, disclosed, and the families it burns

Read on 2026-09-15 **before these bars were written**, on the two families that are therefore
**excluded from every bar below**: `peer_affinity_v1` T1b (unshaped label, 32 checkpoints) and
`drainable_objective_v1` V = 1 (shaped label, 31 checkpoints). Each checkpoint has both a
selected-checkpoint offline score (`final/val/regret_masked_topo`) and a live median latency
at the same cell.

| family | grouping | n | Spearman | p |
|---|---|---|---|---|
| T1b | pooled both arms | 32 | −0.212 | 0.245 |
| T1b | within `gnn` | 16 | −0.300 | 0.259 |
| T1b | within `mpoff` | 16 | +0.015 | 0.957 |
| V = 1 | pooled both arms | 31 | −0.341 | 0.060 |
| V = 1 | within `gnn` | 15 | −0.025 | 0.930 |
| V = 1 | within `mpoff` | 16 | +0.300 | 0.259 |

Within an arm the offline score carries no detectable signal in any of the four cases. The
spreads are the sharper fact: offline, T1b's 16 `gnn` seeds span 51.7–54.6 (a 5.6 % range);
**live, the same 16 checkpoints span 43.4–75.8 s — a 1.7× range.** The live outcome is
dominated by a seed-level term an order of magnitude larger than any architecture effect this
program has chased, and the offline metric does not see it.

Second motivating read, same date, same burned families: **live latency is queue time and
almost nothing else**, Spearman **+0.999** (p ≈ 4e-44) in both.

These two reads decide nothing. They are why this lineage exists, and both families are
disqualified from its bars.

## What is registered, and on what

**Registered families — none of them read for this statistic yet.** Each has 16 `gnn` +
16 `mpoff` per-seed live results at `lr 2e-3`, with matching per-seed W&B runs:

| family | live results | checkpoints |
|---|---|---|
| F1 | `results/x800p2_uncapped_gate` (32) | `peer-affinity-v1-x800p2-{gnn,mpoff}-lr2e3-seed{1..16}` |
| F2 | `results/x800p3_uncapped_gate` (32) | `peer-affinity-v1-x800p3-{gnn,mpoff}-lr2e3-seed{1..16}` |
| F3 | `results/warm_uncapped_gate` (32) | `peer-affinity-v1-warm-{gnn,mpoff}-lr2e3-seed{1..16}` |

**Offline statistic:** `final/val/regret_masked_topo` — the selected checkpoint's held-out
score, i.e. the number the selector actually uses. `final/test/regret_masked_topo` is read as
a secondary and reported, never substituted for the primary.
**Live statistic:** `averageElapsedTime`, the statistic every live reading in this program
quotes.

## Bars

### R0 — positive control on the read itself (runs first, gates everything)

The read must be able to detect a correlation that is known to be there. On each registered
family, correlate live latency against live `averageQueueTime`.

* **Bar:** Spearman ≥ **0.90** in **all three** families. Below it, the machinery is wrong and
  **every other read in this lineage is VOID** until it passes.

### R1 — is the offline score informative? (primary)

Two readings, both registered:

* **Within-arm:** Spearman between offline score and live latency, computed separately for
  `gnn` and `mpoff` in each family (six cells, n = 16 each).
  **Bar:** |ρ| ≥ **0.50** with p < 0.05 in ≥ **2 of 3 families**, in the same direction, for
  at least one arm ⇒ contributes to OFFLINE-INFORMATIVE.
* **Pooled-z (the powered reading):** z-score both variables **within each (family, arm)
  cell** — which removes the arm and family offsets, leaving only seed-level signal — then
  correlate all 96 points.
  **Bar:** |ρ| ≥ **0.30** with p < 0.05 ⇒ OFFLINE-INFORMATIVE.

**Verdict:** OFFLINE-INFORMATIVE if either bar fires; **OFFLINE-UNINFORMATIVE** otherwise.
*Registered expectation: OFFLINE-UNINFORMATIVE.* Recorded here, before the data, so the read
can refute it rather than be read backwards afterwards.

**Why pooled-across-arms is not the primary.** A pooled ρ is dominated by the two arm means,
which is the very thing under suspicion. It is reported for continuity with how the reversal
has been quoted, and it decides nothing.

**Power, stated in advance.** n = 16 within a cell detects ρ = 0.5 at ~47 % power, so a
within-arm null is weak evidence on its own. The pooled-z read at n = 96 detects ρ = 0.30 at
~80 %. A null on **both** is the finding; a null on the within-arm read alone is not.

### R2 — is there a statistic that does predict? (secondary; at most one promotion)

Four candidate surrogates, each computable offline from a checkpoint with no live run, scored
on the 151 captured live states from `drainable_debug_v1` D1:

1. **depth** — mean places the decoded plan sits above the shallowest legal replica;
2. **concentration** — max tasks the plan puts on one platform, ÷ batch size;
3. **one-step regret** — the decoded plan's regret against those states' sweep optimum;
4. **range sensitivity** — change in the plan when the queue column is scaled to its live
   range, which the corpus never contains.

* **Bar:** a candidate is a **SURROGATE** if its pooled-z Spearman against live latency is
  ≥ **0.50**, p < 0.05 after **Holm correction over the four candidates**, and it also clears
  ρ ≥ 0.40 within-family in ≥ 2 of 3 families.
* **At most one candidate is promoted**, the one with the higher pooled-z |ρ|; any other that
  clears is reported and not used. Registered so a surrogate cannot be picked after seeing
  which one would look best downstream.

### R3 — is a bad seed born bad, or does it become bad? (mechanism)

Take the best and worst seed by live latency in the `drainable_objective_v1` V = 1 `gnn` arm,
re-run both with raw per-task records retained, split the trace into deciles by task arrival,
and compare their per-decile mean queue time.

* **PRESENT-FROM-THE-START** if the decile-1 gap is ≥ **50 %** of the full-trace gap ⇒ the
  live states are outside the trained regime from the first decision (distribution shift).
* **COMPOUNDS** if the decile-1 gap is ≤ **20 %** of the full-trace gap ⇒ the model builds its
  own bad states (closed loop). This is what `drainable_debug_v1` D1/D2 point at: near-perfect
  one-step plans, excess queue that is 77–90 % cross-batch.
* **MIXED** in between, reported as such with the decile curve.

R3 explains; it does not decide, and it cannot make R1 fire.

### R4 — the live gate that closes the lineage (rule 6)

Offline reads order this work. They do not close it. The gate is the cost of the selector,
measured live.

Every training run in this program saves **two** checkpoints: the offline-selected one and the
last epoch. Gate them head to head at the drainable cell (`cell_s7901_f4000_pg16`, x4000 trace,
16 s window, Q = 100, uncapped, the configuration `drainable_objective_v1` used), 16 seeds per
arm, on the V = 1 `gnn` and `mpoff` families whose selected arms are already gated.

* **SELECTOR-WORKS** — selected beats last-epoch: median lower, ≥ **12/16** seeds better,
  Mann-Whitney p < 0.05, in **both** arms.
* **SELECTOR-IS-NOISE** — neither direction clears; the selector is choosing on something live
  cannot see.
* **SELECTOR-IS-HARMFUL** — last-epoch beats selected on both arms at p < 0.05.
* **If R2 promoted a surrogate**, a third arm is gated: the per-seed choice between the two
  saved checkpoints made by the surrogate, against the choice made by the offline score. No
  retraining — the surrogate picks between weights that already exist.

**Declared in advance:** `v1_gnn` seed 3 deterministically livelocks the simulator
(`drainable_objective_v1`, Phase C), so that arm is read at n = 15 against a ≥ 12 seed bar.

## Outcomes

* **R1 fires** ⇒ OFFLINE-INFORMATIVE. The reversal is a real cross-arm effect and
  `serving_gap_v3` is worth registering. This lineage closes having sharpened the target.
* **R1 nulls and R4 says SELECTOR-IS-NOISE** ⇒ **OFFLINE-SCREEN-HAS-NO-RESOLVING-POWER.**
  Every offline-only screen in the record is downgraded to "ordering evidence, never closing
  evidence", written into `docs/lessons.md`, and every future experiment stops paying for one
  that decides nothing.
* **R1 nulls and R2 promotes a surrogate that wins R4** ⇒ **SURROGATE-SELECTOR**, the
  deployable result: a cheap offline statistic that does track serving, and a checkpoint
  selector to replace the current one.
* **R1 nulls and no surrogate** ⇒ the honest null: nothing cheap predicts serving in this
  system, and a live gate is the only instrument. Expensive, and worth knowing.

## Carried limitations, stated before the data

1. **Two families are burned** (T1b, V = 1) by the motivating read and excluded from every bar.
2. **One system.** All families share this simulator, workload generator and cell family. This
   measures transfer inside this system, not a general property of offline evaluation.
3. **Live latency here is queue time** (ρ = 0.999 on the burned families). A surrogate that
   predicts queue time is predicting latency *at this cell*; that is a property of the drainable
   regime, not a law, and any promotion says so.
4. **The selector can only choose between two saved checkpoints per run.** R4 measures the cost
   of the current selector and the value of a surrogate over that same two-way choice, not over
   all 300 epochs.
5. **No architecture claim may be founded on this lineage.** It is a measurement about a
   metric, not about message passing.

---

## 2026-09-15 — R0 PASSES, **R1 nulls: OFFLINE-UNINFORMATIVE**

Job 767312, tool and bars committed at `6b2304f` before the read; the sbatch echoes the
bar constants into its own log so which values were in force is on the record.

### R0 — the positive control passes in all three families

| family | n | Spearman(live queue, live latency) | bar ≥ 0.90 |
|---|---|---|---|
| F1 x800p2 | 32 | **+1.0000** | pass |
| F2 x800p3 | 32 | **+1.0000** | pass |
| F3 warm | 32 | **+1.0000** | pass |

The read can detect a relationship that is there. It also restates, on three families that
were *not* burned, the fact first seen on the burned two: **live latency is queue time.**
Here it is not merely ρ = 0.999 but a perfectly monotone function of it, in every family.

### R1 — the primary read

| cell | n | ρ(offline, live) | p | offline range | live range |
|---|---|---|---|---|---|
| F1 x800p2 / `gnn` | 16 | +0.150 | 0.579 | 134.8–144.5 | 112,725–226,945 |
| F1 x800p2 / `mpoff` | 16 | +0.282 | 0.289 | 139.9–148.9 | 114,347–216,454 |
| F2 x800p3 / `gnn` | 16 | **−0.553** | **0.026** | 171.2–177.4 | 175,092–336,414 |
| F2 x800p3 / `mpoff` | 16 | −0.038 | 0.888 | 178.5–186.2 | 148,606–418,554 |
| F3 warm / `gnn` | 16 | +0.188 | 0.485 | 26,035–29,855 | 40,149–63,701 |
| F3 warm / `mpoff` | 16 | −0.400 | 0.125 | 26,407–28,332 | 37,750–42,635 |

* **within-arm bar: does not fire.** One cell clears |ρ| ≥ 0.50 at p < 0.05
  (F2/`gnn`, −0.553) and the bar requires the same arm and the same sign in ≥ 2 families —
  F2/`mpoff` reads −0.038 and F1/`gnn` reads **+0.150**, the opposite sign. Six cells were
  tested at α = 0.05, so ~0.3 false positives are expected and P(≥ 1) ≈ 26 %: one hit is
  what a pure null looks like, which is precisely why the bar was written to need two
  families in agreement.
* **pooled-z bar (the powered reading): does not fire.** ρ = **−0.030**, p = 0.772,
  n = 96, against a bar of 0.30 at ~80 % power. This is not "too few seeds to tell"; it is
  a measured zero.
* **pooled raw, which decides nothing: ρ = −0.525, p < 0.0001.**

### The last two lines are the finding

The raw pooled correlation — **−0.525, highly significant** — is the offline/live
"reversal" as this record has quoted it for three lineages. Strip the family and arm
offsets by z-scoring within each (family, arm) cell, and the same 96 points read
**−0.030**.

**So the reversal is not a relationship between the two measurements. It is the difference
between two arm averages, of a score that carries no per-checkpoint signal whatsoever.**
That was predicted in advance and is pinned as an executable test in
`tests/test_offline_live_transfer_reads.py::test_pooled_z_strips_an_arm_offset_that_the_raw_pooled_read_keeps`,
which builds a synthetic family with exactly this shape — better offline, worse live, pure
noise within arm — and reads ρ < −0.5 raw and ≈ 0 pooled-z.

It also explains why `serving_gap_v1` and `serving_gap_v2` both closed NO-GO: **they were
hunting a mechanism behind a pattern that needs no mechanism.** Neither was wrong; both were
looking for the cause of an offset that three or fewer points cannot distinguish from noise.

### What this does and does not decide

**Does not touch the live arm-level comparisons.** A 16-seed median test is still a valid
answer to "which arm is faster live", and every such reading in the record stands.

**Does undermine prediction and selection.** No offline number in this program has been
shown to rank two checkpoints by how they will serve. The cost of that is R4's question,
and R4 is the live gate that closes this lineage (rule 6) — offline reads order the work
and this one has ordered it.

**Carried limitation, restated:** three families, one simulator, one workload generator.
This is transfer inside this system, not a general claim about offline evaluation.

## 2026-09-15 — R4, the live gate: the offline score resolves **within** a run and **not across** runs

Job 767324, `--array=0-31%8`, `olt_r4_lastepoch_f4000_pg16`. Identical cell, trace, window,
physics and knobs to the selected arms' gate (`dobj_f4000_pg16`), so the two directories are
the same measurement twice with only the checkpoint differing. No retraining: both snapshots
were already on disk.

| arm | selected (best epoch) | last epoch | selected faster by | seeds | Mann-Whitney p | Wilcoxon paired p |
|---|---|---|---|---|---|---|
| `gnn` (n = 15) | **54.82 s** | 63.08 s | **+13.10 %** | 11/15 | 0.0279 | 0.0256 |
| `mpoff` (n = 16) | **51.40 s** | 61.50 s | **+16.42 %** | 14/16 | 0.0005 | 0.0010 |

*(Mann-Whitney is the registered test. The data is paired by seed, so the paired Wilcoxon is
reported beside it as a disclosed secondary; it agrees and changes nothing.)*

### The registered bar, scored honestly

SELECTOR-WORKS requires, in **both** arms: median lower, **≥ 12/16 seeds** better, and
p < 0.05.

* `mpoff` clears all three.
* `gnn` clears median and p (0.0279) and reads **11 of 15 seeds — one short of the bar.**

**So the composite bar does not fire.** It is recorded that way rather than rescored to fit:
the seed-count sub-bar was signed at ≥ 12 and 11 is 11. Neither SELECTOR-IS-NOISE (one arm
clears outright) nor SELECTOR-IS-HARMFUL (the direction is the opposite) applies either. The
honest statement is **SELECTOR-HELPS, COMPOSITE-BAR-SHORT-BY-ONE-SEED**, with both arms
agreeing in direction and significance.

### R1 and R4 are not in conflict — and together they are the finding

They ask different questions, and nothing in this record had separated them before:

| question | instrument | answer |
|---|---|---|
| **within a run** — which epoch's weights should be kept? | the offline score across epochs | **it resolves.** Worth 13–16 % of live latency. |
| **across runs** — is run A better than run B, or arm A than arm B? | the offline score across checkpoints | **it does not resolve.** ρ = −0.030, p = 0.772, n = 96. |

This is coherent rather than contradictory. Within a run the score tracks something real and
large: the models genuinely degrade after epoch ~27 (measured this same day —
`drainable_objective_v1` Phase B, 78–94 % of every run is post-selection memorisation), and
the score sees that degradation. Across runs, the residual differences in that score are
small relative to what actually moves live latency.

**The rule this produces: the offline score can say a model got worse than it was; it cannot
say one model is better than a different model.** Every model-class and arm comparison in
this program rests on the second reading, which is the one with no signal. Every checkpoint
selection rests on the first, which is earning 13–16 % and must be kept.

### Seed 3's second checkpoint livelocks too

`lastepoch_gnn_s3` froze exactly as its selected twin did and was killed by the job's
30-minute limit (TIMEOUT at 00:30:13). **Both snapshots from that training run are
unservable**, so the defect belongs to the run — the seed's whole optimisation trajectory
under this label — not to one epoch's weights. The `gnn` arm is therefore n = 15 here for
the same declared reason it was n = 15 in `drainable_objective_v1` Phase C.

The 30-minute `--time` on this gate, added that morning because of the first livelock, cost
30 minutes instead of the 12 hours the previous cap would have burned. That is now the
default shape for any gate serving freshly trained weights.

## 2026-09-15 — R2: **NO-SURROGATE.** All four candidates fail, and they fail R1's way

Job 767606, tool and bars committed before the correlation. Every input was complete: 96
checkpoints × 151 states, decoded twice (unscaled and at `R2_QUEUE_SCALE = 50.0`), all 96
array tasks COMPLETED, and **151 of 151 states scorable for every checkpoint** — no plan
landed outside its sweep, so the common-subset protocol never had to bind.

| candidate | pooled-z ρ | p | p (Holm/4) | families clearing | F1 x800p2 | F2 x800p3 | F3 warm | |
|---|---|---|---|---|---|---|---|---|
| depth | +0.025 | 0.811 | 1.000 | 2/3 | −0.151 | **−0.490** | **+0.710** | no |
| concentration | +0.053 | 0.609 | 1.000 | 1/3 | −0.237 | −0.261 | **−0.511** | no |
| one_step_regret | **+0.344** | **0.0006** | **0.0024** | **0/3** | +0.387 | −0.093 | −0.146 | no |
| range_sensitivity | +0.001 | 0.992 | 1.000 | 1/3 | −0.130 | −0.214 | **+0.749** | no |

### Every candidate that looks like something looks like something in ONE family

* **`one_step_regret` is the interesting failure.** It clears the significance bar outright —
  pooled-z ρ = +0.344, p = 0.0006, still 0.0024 after Holm over four — which is *more* signal
  than the selector's own score carries (R1: ρ = −0.030). The sign is even the sensible one:
  worse one-step plans, slower live. And it clears **0 of 3** families: +0.387, −0.093,
  −0.146. The pooled significance is one family's relationship diluted across three, not a
  property that reproduces. **This is exactly why the family sub-bar was registered**, and it
  is the only thing standing between "p = 0.0006" and a promoted surrogate.
* **`depth` and `range_sensitivity` both carry a strong `warm` reading** (+0.710, +0.749) that
  is absent or reversed in the two x800 families. A statistic that predicts in the family it
  was inspired by and nowhere else is the pattern this whole lineage exists to catch.

### A defect in the family sub-bar, disclosed and not fixed

As registered and implemented, the family sub-bar counts families with **|ρ| ≥ 0.40** and does
**not** require them to agree in sign — unlike R1's within-arm bar, which does. So `depth`
scores "2 of 3 families clearing" on **−0.490 and +0.710**, which are opposite relationships.

**It changed no verdict here** — `depth` failed the ρ and Holm bars anyway, and nothing was
promoted — but the bar is weaker than the one beside it and would matter if a future candidate
cleared the other two. It is recorded as signed, not retuned after seeing the data; a successor
lineage should copy R1's sign-checking form. Filed to `docs/gates/gate-tools.md`.

### What R2 does and does not rule out

**Rules out:** these four statistics, measured on one shared state set, as a way to rank
checkpoints *across* families.

**Does not rule out:** a within-family screen. `one_step_regret` at ρ = +0.387 in x800p2 and
`depth`/`range_sensitivity` at ~+0.7 in warm may well be real inside those corpora. Nothing
here says a screen calibrated and used *within one corpus* cannot work — only that none of
these four transfers, which is what a general screen would have to do.

**Verdict: NO-SURROGATE.** Combined with R1 (the selector's own score: no across-run signal)
and R4 (that same score: a real 13–16 % within-run signal), the position is now: **no cheap
offline number tested in this program ranks training runs by how they serve, and the live
gate — ~3 minutes an arm, against 41 minutes to train one — remains the only instrument that
does.**

## 2026-09-15 — R3: **COMPOUNDS**, and the curve is a single transient excursion, not a drift

Job 767666, on captures that each reproduced their gate latency exactly (43.9905 s and
74.7555 s, asserted to 0.01 s by the capture job). Best seed 14 against worst seed 8 of the
V = 1 `gnn` arm: same recipe, same corpus, same split, same lr, **only the training seed
differs**, and they serve the identical 50,000-task trace 1.7× apart.

| decile | n | best queue | worst queue | gap | share |
|---|---|---|---|---|---|
| 1 | 5000 | 16.285 | 16.914 | 0.630 | 0.002 |
| 2 | 5000 | 9.727 | 9.440 | −0.287 | −0.001 |
| 3 | 5000 | 27.216 | 34.071 | 6.855 | 0.022 |
| 4 | 5000 | 26.414 | 57.704 | 31.290 | 0.101 |
| 5 | 5000 | 44.937 | 124.828 | 79.891 | 0.258 |
| **6** | 5000 | 28.806 | **185.058** | **156.253** | **0.505** |
| 7 | 5000 | 41.684 | 68.377 | 26.693 | 0.086 |
| 8 | 5000 | 45.707 | 43.926 | −1.781 | −0.006 |
| 9 | 5000 | 28.792 | 34.003 | 5.211 | 0.017 |
| 10 | 5000 | 32.442 | 37.086 | 4.644 | 0.015 |
| **full** | 50000 | 30.201 | 61.141 | 30.940 | |

**Decile-1 share 0.002 ⇒ COMPOUNDS** (bar ≤ 0.20), and **PRESENT-FROM-THE-START is refuted
outright**: the two seeds are indistinguishable over the first three deciles — 0.2 % of the
total gap — so the worse seed is not handicapped by out-of-range live states from its first
decision. Whatever separates them, it is not that they start in different regimes.

### The verdict is right and too coarse; the curve is the finding

The registered bar cannot see shape, and the descriptive `shape` field this tool reports
(RISING, second half carries 0.617) is also too blunt for what is actually here. The gap is
**not a drift**. It is a single excursion:

* deciles 1–3: nothing (0.2 % of the gap)
* deciles 4–6: the divergence builds and then spikes — **decile 6 alone carries 50.5 %** of
  the whole gap, with the worst seed's mean queue at **185.1 s against the best seed's
  28.8 s, a 6.4× excursion over its own baseline**
* deciles 7–10: **it recovers.** Decile 8 is *negative* (the worst seed is briefly faster),
  and the last three deciles together carry 2.6 %.

**A closed loop that ran away would not come back.** This one does. So COMPOUNDS is the
correct reading of the registered bar, and the mechanism it is usually shorthand for —
progressive, self-reinforcing degradation — is **not** what the data shows. What the data
shows is a seed that fell into a deep-queue excursion partway through the trace and drained
out of it, and a 1.7× difference in headline latency that is one event's worth of queue
spread over 50,000 tasks.

### Why this matters to the rest of the lineage

It explains R1 and R2 at once. If a checkpoint's live latency is set by whether it happens to
hit one excursion in one trace, then **no property of the checkpoint measured on captured
states can predict it** — the excursion is a property of the interaction, not of the model.
That is exactly what R1 (ρ = −0.030 across 96 checkpoints) and R2 (no surrogate among four)
measured, and it is why both nulls are consistent rather than merely disappointing.

It also puts a number on how little the headline means: the difference between "our best
seed" and "our worst seed" — larger than any architecture effect in this program — is one
excursion. **A single-seed live number at this cell is not a measurement of a checkpoint.**

### A limitation of this read, stated

n = 2 checkpoints on n = 1 trace. The excursion's existence is measured; its *frequency* is
not. Whether most bad seeds are bad for this reason, whether a given seed excurses on other
traces, and what triggers the excursion are three separate questions this read does not
answer, and a successor should not quote it as though it did.

---

## Outcome (2026-09-15): **OFFLINE-SCORE-RESOLVES-WITHIN-RUN-ONLY** — closed

Five reads, all run, all on families and bars fixed before their data.

| read | verdict |
|---|---|
| R0 positive control | **CONTROL-PASSES** — ρ = 1.0000 in all three families |
| R1 primary | **OFFLINE-UNINFORMATIVE** — pooled-z ρ = −0.030, p = 0.772, n = 96 |
| R2 surrogates | **NO-SURROGATE** — all four fail; the best clears p and 0/3 families |
| R3 mechanism | **COMPOUNDS** — and the gap is one excursion, not a drift |
| R4 live gate | **SELECTOR-HELPS, COMPOSITE-BAR-SHORT-BY-ONE-SEED** — +13.1 % / +16.4 % |

### The registered outcome sentences do not map, and that is recorded rather than patched

The registration listed four outcomes, all of which assumed R1 and R4 would point the same
way — that the offline score either resolves serving or does not. **It does both, depending on
what is being ranked**, so none of the four sentences applies as written. The outcome is
named for what was measured:

> **The offline score resolves WITHIN a training run and not ACROSS runs.** It ranks epochs
> (worth 13–16 % of live latency, so keep it) and it cannot rank seeds, runs, arms or model
> classes (ρ = −0.030 at n = 96, ~80 % power at 0.30).

No bar was moved to reach that, and the two sub-bars that turned out to be defective — R2's
family bar not checking sign, R3's share normaliser — are recorded above and in
`docs/gates/gate-tools.md` rather than retuned.

### What this lineage changes about the record

1. **The offline/live reversal is not a phenomenon.** Pooled raw it reads ρ = −0.525,
   p < 0.0001; z-scored within (family, arm) the same 96 points read −0.030. It is the gap
   between two arm averages of a score with no per-checkpoint signal. `serving_gap_v1` and
   `serving_gap_v2` closed NO-GO hunting a mechanism for it; **they were looking for the cause
   of an offset.** Neither should be reopened on that basis.
2. **No offline-only comparison of two arms is evidence about serving** on these corpora. The
   structural offline findings are untouched — pointwise-separability, the count theorem, "the
   optimum never sees the mechanism" are facts about labels, not rankings of checkpoints.
3. **A single-seed live number is not a measurement of a checkpoint.** R3: two seeds of one
   recipe differ 1.7× live, and 50.5 % of that difference is **one decile** of one trace, with
   full recovery afterwards.
4. **The live gate is the cheap instrument, not the expensive one.** ~3 min per arm against
   41 min to train the checkpoint it judges. The offline screen costs more than what it
   filters and decides nothing across runs.

### What is NOT closed

* **A within-corpus screen.** `one_step_regret` reads ρ = +0.387 inside x800p2; `depth` and
  `range_sensitivity` read ~+0.7 inside warm. Those may be real *within* their corpus. R2 only
  rules out transfer across corpora, which is what a general screen would need.
* **A sequence-scored evaluator** — scoring k consecutive batches rather than one. Not tested
  here. It inherits a warning: `objective_pivot_v1` found rollout *rankings* unstable across
  horizon lengths (ρ(h2,h10) = −0.027), so any such evaluator owes a rank-stability control as
  a blocking bar before it is trusted.
* **The excursion itself.** R3 measured that one exists and that it recovers. Its frequency,
  its trigger, and whether the learned arms' loss to reactive Knative is made of the same thing
  are three open questions, on n = 2 checkpoints and n = 1 trace.
