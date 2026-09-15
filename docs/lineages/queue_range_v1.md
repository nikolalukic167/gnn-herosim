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
