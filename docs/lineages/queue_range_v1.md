# queue_range_v1 — the served queue column leaves the contract it was trained under

**Status:** `REGISTERED` (2026-09-15). Every bar below is a module constant in
`scripts_cosim/queue_range_v1_read.py`, committed **before** the arms it reads exist. Amend by
dated amendment only.

**Parent:** `serving_stability_v1` (CLOSED, `EARLY-ADVANTAGE-REAL · STABILITY-NOT-THE-LEVER`).
That lineage measured, on 3/3 cells and all 91 learned arms, that the arms carry **3.4–4.1 s
less queue than reactive Knative over the first fifth of the trace** and lose it afterwards,
and that a decode-time queue guardrail (safe, 0 hangs, bound 84–90 % of decisions) does not
recover it. Its closing sentence — *"something destroys a real advantage, and no measurement
yet says what"* — is this lineage's question.

---

## The claim under test

> **The arms lose their early advantage because the queue column they rank candidates with
> leaves the contract it was trained under, and it leaves it as the trace gets busier.**

The mechanism is arithmetic, and it is measured, not assumed. Platform dim 7 is
`queue_depth / min(max(1, p90(depths over ALL platforms)), 100)` under `legacy_v0`
(`src/placement/queue_features.py`). Measured on the arms' own training corpus
`graphs_cache_drainable_objective_v1_v1` (516 datasets, 2026-09-15, this session):

| | all platform rows (n = 69,144) | candidate rows only (n = 2,544) |
|---|---|---|
| dim7 p50 | 0.00 | **12.00** |
| dim7 p90 | 0.00 | **25.00** |
| dim7 p99 | 18.00 | 34.00 |
| dim7 max | **42.00** | **42.00** |
| dim13 max | 0.44 | 0.44 |

Raw queue max is also 42, so **the divisor is 1.0 in every training dataset**: dim7 *is* raw
queue depth, on a scale whose largest value the model ever saw is 42. There are exactly two
ways to leave that contract live, and they are opposite:

* **OUT-OF-RANGE.** While ≥ 10 % of platforms are idle the p90 is 0, the divisor pins at 1.0,
  and dim7 is unbounded raw depth. A candidate 1,000 tasks deep reads 1000.0 against a trained
  maximum of 42.
* **COMPRESSED.** Once most platforms are busy the divisor inflates (capped at 100), and the
  same 20-task difference between two candidates reads as 0.2 of feature. The column stops
  resolving the pile it is standing in front of.

Both are out of contract, **both predict the early-good/late-bad shape** (early in the trace
queues are shallow and the column behaves as in training), and the intervention is the same
for both. Which one the live cell is actually in is **Q0's job and is not assumed here.**

`drainable_regime_v1`'s B5 control measured dim-7 p90 = 5 over 427 busy platform-queues at
this very cell — **inside** the trained range — and that is a p90 over a whole run. It does not
say what the column looks like in decile 9, which is the only place this claim lives. If Q0
finds the column in contract across the whole trace, the claim is refuted and Q2/Q3 are
reported as a null.

## Why no existing stop covers this

* **`herosim-queue-column-is-broken-live` (2026-09-12/13)** refuted the queue column as the
  cause of the **offline/live reversal** on the x200/x800 corpora, on a crossover-ordering
  argument. That is a different question, a different quantity (`gnn` − `mpoff`), a different
  load (940× overload) and a different cell. This lineage asks about `arm` − `reactive`
  **within** the trace, at f4000, and the refuted claim is not re-asserted.
* **It is not the guardrail** (`serving_stability_v1` S3, CLOSED). That masked candidates at
  decode time using the *raw* queue snapshot and left the feature alone. This changes the
  feature the model scores with and masks nothing.
* **It is not a retrain, a corpus or a label,** so the count theorem, the horizon-return stop,
  the closed-loop PG stop and the warm-corpus stop do not reach it. Nothing is trained here.
* **It is not `scale_invariant_v1`.** Serving that contract to a `legacy_v0` checkpoint is
  refused by `require_matching_queue_feature_contract`, correctly. The overrides below change
  magnitudes *within* `legacy_v0` and are reported in every result's `env` block.

## The arms

Three serving arms, **same checkpoints, same seeds, same trace, same cell, same commit**, run
as one array so no arm is compared against a number produced by an older tree:

| arm | knobs | what it is |
|---|---|---|
| **A** `plain` | none | the control — exactly what `serving_stability_v1` S1/S2 served |
| **B** `pinned` | `GNN_QUEUE_SERVE_DIVISOR=1.0` | training semantics restored: dim7 = raw depth |
| **C** `inrange` | `+ GNN_QUEUE_SERVE_DIM7_CLAMP=42`, `GNN_QUEUE_SERVE_DIM13_CLAMP=0.44` | the column can never leave the trained range |

B and C differ **only** above depth 42. C is order-preserving inside the range and ties
everything above it — which is what the model already does at magnitudes it never saw, made
explicit. A is re-run rather than reused.

3 cells × 16 seeds × {`gnn`, `mpoff`} × 3 arms = **288 runs**, ~3 min each, plus the reactive
arms already in `results/ss_s1s2`. `mpoff` is carried as the pointwise control: if the fix is
about the queue column and not about message passing, it must help both.

## Bars

**Q0 — where the column actually is (blocking, and it can kill the lineage).**
From arm A's own per-batch records, bucketed on the trace's arrival-order deciles:

* `Q0_OUT_OF_RANGE_FRAC = 0.20` — share of late-decile (9–10) batches whose maximum candidate
  dim7 exceeds `CORPUS_DIM7_CANDIDATE_MAX = 42`.
* `Q0_BLIND_FRAC = 0.20` — share of late-decile batches that are *blind*: candidate raw queues
  ≥ 10 apart while their dim7 values are < 0.05 apart.
* `Q0_EARLY_LATE_RATIO = 3.0` — late-decile value over early-decile (1–2) value.
* `Q0_MIN_CELLS = 2` of 3.

**`COLUMN-LEAVES-CONTRACT`** if, on ≥ `Q0_MIN_CELLS` cells, *either* the out-of-range share or
the blind share exceeds its threshold in deciles 9–10 **and** is ≥ `Q0_EARLY_LATE_RATIO` ×
its value in deciles 1–2. Otherwise **`COLUMN-IN-CONTRACT`**, the mechanism is refuted, and
Q2/Q3 are still run and still reported — they just cannot be attributed to this mechanism.

**Q1 — did the knob do what its name says (blocking).**
The registered general rule from `drainable_regime_v1`: a control bar that reads the arms'
own counters. For arms B and C:

* `Q1_MIN_PINNED_FRAC = 0.99` — share of B/C batches whose recorded divisor is exactly 1.0.
* `Q1_MAX_OUT_OF_RANGE_FRAC = 0.0` — arm C batches above the clamp. Blocking and exact: a
  single batch above 42 in the arm whose whole definition is "never above 42" is a bug.
* Arm A must show `qr_batches > 0`. A run with no records served no graph through the builder.

Failing Q1 makes that arm **VOID**, not negative.

**Q2 — paired latency against the control (primary).**
Per cell and policy arm, `A − B` and `A − C` in mean elapsed time, paired by seed.
**Wilcoxon signed-rank** (paired — the arms share seeds; `docs/gates/gate-tools.md`,
2026-09-15), `Q2_ALPHA = 0.05`, **Holm–Bonferroni over `Q2_HOLM_N = 6`** (2 interventions ×
3 cells), the registered family size, never shrunk to what was computed.

* `Q2_MIN_SEEDS = 12`, `Q2_MIN_CELLS = 2`.
* **`RANGE-FIX-HELPS`** if B or C improves on A with a positive median and a Holm-adjusted
  p < α on ≥ `Q2_MIN_CELLS` cells for the `gnn` arm. Otherwise **`RANGE-FIX-DOES-NOT-HELP`**.

**Q3 — the headline: does it beat reactive (primary).**
Best intervention arm vs `knative_network` on the same cell.

* **`BEATS-REACTIVE`** if its median mean-elapsed is below reactive's on ≥ `Q3_MIN_CELLS = 2`
  of 3 cells. Otherwise **`REACTIVE-STILL-WINS`**.
* **Registered expectation: NEGATIVE, and the arithmetic is stated before the run.** The
  measured early advantage is **3.4–4.1 s of queue** per task over deciles 1–2. Reactive's
  mean elapsed is 25.95 / 20.95 / 23.48 s against the unguarded `gnn` arm's 54.82 / 22.35 /
  65.40 s, so the full-trace deficits are **28.9 / 1.4 / 41.9 s**. Holding the early margin
  across all ten deciles is worth about **4 s of mean queue** — enough to flip **s9001 and
  nothing else**. Q3 firing on s7901 or s9002 would mean the fix does more than extend the
  early margin. **Q2 firing while Q3 fails on 2 of 3 cells is the outcome this registration
  expects**, and it is still a mechanism found: it would be the first measurement in this
  program that names something which destroys the arms' advantage and then removes it.

**Q4 — does the early advantage extend (the diagnostic that survives a Q3 failure).**
Number of deciles in which the arm's mean queue is below reactive's on the same cell.
Control value, measured: **2** (deciles 1–2, `serving_stability_v1` S1).

* **`ADVANTAGE-EXTENDS`** if the intervention arm's median count is ≥ `Q4_MIN_DECILES = 4` on
  ≥ `Q4_MIN_CELLS = 2` cells. Otherwise **`ADVANTAGE-DOES-NOT-EXTEND`**.

## Declared in advance

* **`gnn` seed 3's checkpoint deterministically livelocks the simulator** (`drainable_objective_v1`
  Phase C, reproduced byte-for-byte; cell-specific — it hangs on s7901 and completes on
  s9001/s9002). It is excluded by name from every bar, as in `serving_stability_v1`.
* **BURNED, and excluded:** the `jam_probe` scoping read used `gnn` seeds **8** and **14** and
  the reactive arm on `cell_s7901_f4000_pg16`'s trace-and-cell. Those stay excluded here, so
  C1's `gnn` arm reads at n ≤ 13.
* **No scoping read was taken before these bars.** The corpus table above is a property of the
  training cache, not of any arm's live behaviour, and no live number from the intervention
  arms existed when this file was committed.

## Not in scope

Retraining under `scale_invariant_v1` (that is the principled fix and it is a *training*
change — a separate registration, and only worth its 32 × 41 min if Q0 fires); the cluster-size
axis (`cluster_scale_v1`, registered separately); the batch window (fixed at 16 s); the
platform cap (closed — deadlocks 3/16); the load ladder.

## Record

*(dated entries appended below as the lineage runs)*

### 2026-09-15 — read, all bars. Outcome **QUEUE-RANGE-NOT-THE-LEVER**

285 of 288 arms. The 3 missing are `gnn` seed 3 on `cell_s7901`, the declared livelock — no
other failure, no `FAIL LOUD`, no timeout. `simulation_data/queue_range_v1/gate.json`.

**Parity, unasked for and worth recording:** the `plain` arms reproduce
`serving_stability_v1`'s unguarded medians **to three decimals** — 54.819 / 22.350 / 65.401
against S3-d's 54.82 / 22.35 / 65.40 — across a commit that changed the feature builder, the
scheduler and the orchestrator. The instrument is inert.

#### Q0 — `COLUMN-IN-CONTRACT` (1/3 cells) by the bar; **fires in shape on 2/3**

Per-decile share of batches whose maximum candidate dim7 exceeds the corpus maximum of 42,
median over seeds, `plain` `gnn`:

| cell | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | peak dim7 | vs reactive |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s9001 | .00 | .00 | .00 | .00 | .00 | .00 | .00 | .00 | .00 | .00 | **2** | −6.7 % |
| s7901 | .00 | .00 | .00 | **.21** | **.36** | .03 | .00 | .00 | .00 | .03 | **38** | −111.3 % |
| s9002 | .00 | .00 | .00 | .00 | .03 | .17 | **.83** | .24 | .33 | .47 | **113** | −182.9 % |

Three readings hold:

1. **Deciles 1–2 are in contract on all three cells** — exactly where the arms beat reactive.
2. **The cell that never leaves range is the cell that nearly ties reactive.** Severity of the
   range violation orders with severity of the loss across all three cells. n = 3 aggregates;
   this lineage's parent lost a mechanism claim to exactly that shape, so it is recorded as an
   ordering and nothing more.
3. **It is an excursion, not a drift.** s7901 blows out at deciles 4–5 and recovers; s9002
   peaks at decile 7 and stays bad.

**The bar as signed reads deciles 9–10 as "late" and therefore fires on s9002 only
(late share 0.418, early 0.000) — 1 of 3, `COLUMN-IN-CONTRACT`. It is not moved.** The bar was
mis-*shaped*, not mis-*tuned*: it assumed a monotone late drift and the mechanism is a
mid-trace excursion, which `docs/lessons.md` already distinguishes from
`offline_live_transfer_v1` R3. Filed in `docs/gates/gate-tools.md`. The `blind` sub-bar never
fires anywhere: the divisor is 1.0 in 100 % of batches on every cell, so the column never
compresses. **Of the two registered failure modes only OUT-OF-RANGE occurs here; COMPRESSED
does not occur at this load at all.**

#### Q1 — `KNOB-BOUND` (blocking bar, passed)

`inrange` drives the out-of-range share to **exactly 0.000** where `plain` carries 0.107
(s7901 `gnn`), 0.164 (s9002 `gnn`) and 0.197 (s9002 `mpoff`). `pinned` is an **exact no-op**,
as it must be: the divisor was already 1.0 in every batch of every arm, so pinning it changes
nothing. That makes `pinned` a free null control and it behaved as one — `+0.000 s` on all six
(cell, policy) combinations, 0 seeds moved. Two further provable no-ops also held: every arm
on s9001, where the column never approaches 42.

#### Q2 — `RANGE-FIX-DOES-NOT-HELP`

Paired Wilcoxon, `plain` − `inrange`, Holm over the registered family of 6:

| policy | cell | plain | inrange | gain | seeds | p |
|---|---|---|---|---|---|---|
| `gnn` | s7901 | 54.819 | 54.778 | **+0.041 s** | 9/13 | 0.075 |
| `gnn` | s9001 | 22.350 | 22.350 | +0.000 s | 0/15 | — |
| `gnn` | s9002 | 65.401 | 66.429 | **−0.367 s** | 3/15 | 0.011 |
| `mpoff` | s7901 | 51.403 | 50.210 | +0.344 s | 12/16 | 0.056 |
| `mpoff` | s9001 | 22.523 | 22.522 | +0.000 s | 6/16 | 0.028 |
| `mpoff` | s9002 | 73.998 | 73.135 | +0.525 s | 9/16 | 0.501 |

Nothing clears Holm. **The largest effect anywhere is 0.5 s on a 74 s arm — 0.7 %** — and on
the cell with the worst violation the fix is **significantly worse** (p = 0.011, 3/15). Putting
the queue column back inside the range it was trained on does not recover the early advantage
and does not move latency.

#### Q3 — `REACTIVE-STILL-WINS`, 0/3 cells, as registered

−111.3 % / −6.7 % / −182.9 %, 0 seeds below reactive on any cell. The registered expectation
was NEGATIVE on ≥ 2 of 3 and the registered arithmetic (the early margin is worth ~4 s against
deficits of 28.9 / 1.4 / 41.9 s) was right about the size and wrong only in being generous.

#### Q4 — `ADVANTAGE-EXTENDS` fires, and it is a **control artifact**

Deciles in which the arm's mean queue is below reactive's: s7901 **2**, s9001 **10**,
s9002 **4** — **identical for `plain`, `pinned` and `inrange`**. The intervention changed
nothing; the bar fired because the *control* already clears it on 2 of 3 cells. The registered
control value of 2 came from `serving_stability_v1` S1, which reported deciles 1–2 and was
never asked how many deciles the control wins. **A "does X extend Y" bar sized against a
control value measured on one cell, when the family has three.** Same error shape as
`drainable_objective_v1`'s C0, whose bar failed its own control. Not moved; filed.

#### What this measurement found instead — the decomposition

`plain` `gnn` against reactive, per task, medians over seeds:

| | s7901 | s9001 | s9002 |
|---|---|---|---|
| elapsed | **+28.87** | **+1.43** | **+42.02** |
| queue | +23.33 | **−2.67** | +37.19 |
| **batch wait** | **+6.89** | **+6.87** | **+6.72** |
| peer exchange | +0.07 | −0.85 | −0.13 |
| peer rendezvous | −1.48 | −1.85 | −1.84 |
| scale events | +5,838 | +5,930 | +3,455 |

**On s9001 the learned arm wins every term it was designed to win — queue by 2.67 s, peer
exchange by 0.85 s, rendezvous by 1.85 s — and loses the cell on batch wait alone.** Remove
the 6.87 s and it beats reactive by ~5.4 s (−26 %). Batch wait is a **flat ~6.8 s tax on every
cell**, independent of how badly placement fails; the queue blow-up is what separates the two
bad cells from the good one. Autoscaler churn is 2.2–6.6× reactive's everywhere and **does not
order with the loss** (+5,930 on the best cell, +3,455 on the worst), so it is not the
discriminator it looked like in `drainable_regime_v1`'s aggregate.

**Two distinct failures, not one.** A flat batching tax that costs the whole margin where
placement works, and a queue blow-up on two of three cells that is not the feature range.
`drainable_serving_config_v1` closed *removing* batching (−1731 %, the graph arm cannot decode
singletons) and swept the window at fixed load; **nobody has tested a policy that keeps
peer-group batching and caps the wait** — dispatch on group completion with a short deadline
rather than a fixed window. That is a new registration, not an amendment here.

#### Closed precisely

The **magnitude of the `legacy_v0` queue column at serve time** is not what destroys the early
advantage, at the x4000 rung, on these three cells, for these V = 1 checkpoints, with the
divisor measured at 1.0 in 100 % of batches. **Not closed:** the queue blow-up itself (real on
2/3 cells, cause unknown), batch wait, `scale_invariant_v1` as a *training* contract, and any
cell whose serving divisor is not 1.0 — none was observed here, so COMPRESSED remains untested
rather than refuted.
