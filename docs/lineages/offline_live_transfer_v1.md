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
