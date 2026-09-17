# peer_only_v1 — message passing over the peer graph only, and the corpus it was starved of

**Status:** `CLOSED` (2026-09-17) — **PEERONLY-BEATS-POINTWISE at 80 servers, and loses to
the best pointwise arm anyway.** Registered 2026-09-16; every bar below is a module constant
in `scripts_cosim/peer_only_v1_read.py`, committed before any arm was trained.

**Outcome.** On the 1,654-dataset corpus at the 80-server rung, `peeronly` (PeerConv, no GIN)
beats its MP-OFF twin on elapsed by **−5.75 %** (16 checkpoints, 14/16, p = 0.0011; −7.10 % over
the 64 (cell, checkpoint) pairs) and beats reactive Knative by **−41.5 %** on **64/64** pairs
and **16/16** checkpoints — **the first measurement in this program where a message-passing arm
beats both its own pointwise twin and reactive on the same cells.** Read the clauses before
quoting it; none is optional. Both halves are read at **n = 16 checkpoints** after Amendments 1
and 2; the first write-up quoted −15.68 % from a 4-checkpoint subset that overstated the margin
by more than 2×.

1. **The original `n` was 4 checkpoints × 4 cells, not 16 seeds — and it mattered.** A2
   registered the pair as the unit and the bar was honoured as signed, but a claim about an
   *architecture* has the **checkpoint** as its independent unit. **B3 (Amendment 1)** re-read
   it on all 16: the margin fell from −15.68 % to **−5.75 %**, barely above the 5 % bar, with
   **2 of 16 checkpoints on the wrong side**. Every descriptive figure from the subset moved
   with it (the twin is −36.81 % vs reactive, not −28.00 %).
2. **The margin is queue, not peers.** Elapsed at R3 is ~99 % queue; the peer term — what
   `PeerConv` exists to improve — differs by **0.66 s of a 78 s gap, under 1 %**. The arm wins
   on queue and autoscaler churn (92 scale events vs 120). B2 is therefore **not** evidence
   that peer-graph reasoning is what pays.
3. **It loses to the best pointwise arm the program has.** `peeronly_1670` vs `mpoff_516` is
   **+11.17 %, 0/16 checkpoints, p = 0.0004** (**B4**, Amendment 2, 16 checkpoints — this
   clause got *stronger* with power). The win is corpus-matched, as registered; the best
   available scheduler at that rung is still pointwise, and **both** 516 arms beat **every**
   1,670 arm.
4. **More data made every arm worse live** — B1 reads CORPUS-DOES-NOT-HELP on 6/6 arm × rung,
   and the flip is mostly `mpoff` degrading **+35.49 %** (0/16), not `peeronly` improving. But
   the corpus **did** improve every arm *offline* (regret 516 → 1,670: `gnn` 38.40 → 35.85,
   `mpoff` 40.77 → 39.66, `peeronly` 41.92 → 41.28 %), and the two corpora are identical in
   tasks, candidates per task, peer edges and target scale — so B1 is this program's
   offline/live anti-correlation on a new axis, **not** a defective corpus.
5. **The offline ranking is exactly inverted.** Offline `gnn` < `mpoff` < `peeronly`; live at
   R3 `peeronly` < `mpoff` < `gnn`, same checkpoints.
6. **The winning rung is saturated** (reactive ~600 s vs 21–36 s) and the statistic is
   relative to reactive on the same cell. At the unsaturated 6-server rung **every arm loses
   to reactive** and B2 is a TIE.
7. **The arm that wins is not the GNN.** The full `gnn` is beaten by `peeronly` by 22.42 %
   (16/16 pairs, 4 checkpoints — `gnn` was not extended) at R3, and A3's registered mechanism is MECHANISM-NOT-CONFIRMED: `peeronly`'s
   queue improves against *both* twins, so "GIN is the over-reaction" does not isolate it.

One thing runs the *other* way and is carried with the rest: the advantage **grows across the
trace**, −6.2 % at the first decile to **−27.9 %** at the last. Every earlier positive in this
program was an early-trace effect that decayed (`serving_stability_v1`); this is the opposite
shape, so it is not that effect re-appearing.

Phase A (516 datasets) reads A1 ENCODING-COSTS offline, A2 TIE at both rungs, A4
PEER-TERM-KEPT: **PeerConv alone carries no live edge at the small corpus.** The edge appears
only at 1,654 datasets, and only because the pointwise twin falls further.

**Parents:** `partial_state_v3` (CLOSED — the size-free representation; its P3 harness, cells,
checkpoints and reactive arms are reused here), `peer_affinity_v1` (the offline MP edge, and
its serving asymmetry), `serving_gap_v1` (the graph arm is 3–7× *more* load-responsive than
its twin), `drainable_regime_v1` (the graph arm loses to its twin on **queue**, not peers),
`link_mp_v1` / `mp_ablation_v1` (message passing over the wrong graph is harmful; over the
right graph it ties or wins).

---

## The claim under test

Every live comparison in this program has the pointwise twin (`mpoff`) beating the graph arm
(`gnn`) or tying it, and the record says where the loss comes from:

* offline, message passing wins only on the **pair-indexed peer term** — `PeerConv` over
  task–task exchange edges — and that edge grew with corpus size (136 → 482 datasets:
  +2.08 → +5.14 pp);
* live, the graph arm loses to its twin on **queue**: it cuts peer rendezvous 5× and still
  loses, because it is 3–7× more load-responsive and over-reacts to platform queue state;
* the reason is structural — with MP off, a task embedding never touches a platform feature;
  with the bipartite GIN on, every platform's queue state is carried into every task
  embedding. That is why `gnn` degraded 16 pp against `mpoff`'s 7 pp when the queue column
  left its trained range (`peer_affinity_v1`, queue-scale probe).

> **Claim A.** Keep `PeerConv` over the task–task peer edges and **drop the bipartite GIN over
> task–platform edges**. Platform state then reaches the scorer exactly as in `mpoff`, while
> peer structure still flows between tasks. The arm (`peeronly`) keeps the offline MP edge
> (which came from `PeerConv`) and loses the queue over-reaction (which came from the GIN).
>
> **Claim B.** The MP edge is corpus-starved. Train all three arms on the **1,670-dataset**
> corpus (`train` 136 + the full `train2` 1,500 + `r2` 34, minus any alpha set-asides) instead
> of the 516 T1b parents, under the same recipe, and read the same live bars.

`mpoff` is carried as the pointwise control and — unlike every other lineage since 2026-09-04
— **a graph-vs-pointwise reading is the point here and is permitted**, because the arms share
corpus, label, recipe, split and training seeds and differ only in which message passing runs.
The selector caveat is carried: offline reads order the work and never close it.

## What `peeronly` is, exactly

`TaskPlacementGNN(mp_peer_edges=True, mp_platform_edges=False)`: encoders → `PeerConv` on
task embeddings → **no GIN** → `EdgeScorer` on `[task_emb, platform_emb, edge_attr,
partial_state_v3]`. `mp_platform_edges` is weight-invisible (the GIN is constructed and never
run, as in `mpoff`), so it is recorded in the sidecar, whitelisted in
`executesimulation.checkpoint_mp_config`, adopted-or-verified at serving
(`GNN_MP_PLATFORM_EDGES`), and a mismatch fails loud. `gnn` and `mpoff` must be **bit-identical**
to before the change (A0).

## Bars

### Phase A — the architecture, on the 516-dataset corpus

**A0 — instrument (blocking).** (i) Re-serving one `gnn` and one `mpoff` v3 checkpoint on one
P3 cell per rung reproduces `partial_state_v3` P3's mean elapsed to **three decimals**
(`A0_TOL = 0.0005`), so the model change is inert for the existing arms and their P3
summaries may be reused. (ii) The `peeronly` sidecar records `mp_platform_edges = false`,
`mp_peer_edges = true`, `disable_message_passing = false`, contract `partial_state_v3`;
serving refuses a sidecar/environment mismatch (unit test).

**A1 — offline, ordering only.** Held-out regret at the selected checkpoint (the P1 method),
`peeronly` vs `mpoff` and `peeronly` vs `gnn`, paired by training seed, 16 seeds
(`A1_MIN_SEEDS = 12`, `A1_TIE_PP = 1.0`, `A1_ALPHA = 0.05`). Registered expectation:
`peeronly` ≈ `gnn` (the edge came from `PeerConv`), both ahead of `mpoff`.

**A2 — the PRIMARY, live.** Rungs **R0 (6 servers, f4000)** and **R3 (80 servers, f300)** of
the `partial_state_v3` P3 harness — same 4 topology cells per rung, same checkpoint seeds
(1, 2, 4, 5), same reactive and `gnn`/`mpoff` arms (reused under A0). `peeronly` vs `mpoff`
**mean elapsed, paired by (cell, seed)**, n = 16 per rung (`A2_MIN_PAIRS = 12`), exact
Wilcoxon, `A2_TIE_PCT = 5.0`, `A2_ALPHA = 0.05`. Per rung: **`PEERONLY-BEATS-POINTWISE`**
(median < −5 % and p < 0.05), **`TIE`**, or **`POINTWISE-BETTER`**. The lineage headline is
`PEERONLY-BEATS-POINTWISE` only if it reads so on at least one rung and `POINTWISE-BETTER` on
none. **Registered expectation: TIE at R0, uncertain at R3.**

**A3 — the mechanism.** Mean queue time, paired by (cell, seed): `peeronly` vs `gnn` must be
lower by ≥ `A3_QUEUE_IMPROVE_PCT = 5.0` % (p < 0.05) on both rungs ⇒ **`GIN-IS-THE-OVERREACTION`**;
and `peeronly` vs `mpoff` queue within ± 5 % ⇒ the over-reaction is gone, not merely reduced.
**Registered expectation: POSITIVE.** If A3 does not fire, Claim A's mechanism is wrong even
if A2 ties.

**A4 — the peer term is retained.** `totalPeerExchangeTime + totalPeerRendezvousWait` per
task, `peeronly` vs `mpoff`, paired: `peeronly` ≤ `mpoff` × (1 + `A4_PEER_TOL = 0.05`) on both
rungs. An arm that lost the queue over-reaction *and* the peer advantage is `mpoff` with noise
and reads **`PEER-TERM-LOST`**.

### Phase B — the corpus, all three arms

**B0 — corpus instrument (blocking).** The 1,670 corpus needs the T1b preparation the 1,154
new `train2` datasets never had (`refresh_optimal_full_stats.py --rewrite-ssc`, then
`peer_affinity_alpha_prescan.py --alpha 2.0` with set-asides), then one cache under
`partial_state_v3` with the V = 1 label on the measured clock. Bars: contract, label, alpha
identical to the psv3 cache's; the **516 T1b parents' partial-state ingredients identical** to
the psv3 cache's, dataset for dataset (the P0 comparison restricted to those ids,
`B0_MAX_COLUMN_DIFF = 0.0`); the test split is the same 34 `r2` datasets; `B0_MIN_DATASETS =
1500`.

**B1 — the corpus lever, per arm.** Each arm's 1,670 checkpoints vs its 516 checkpoints,
paired by training seed: offline (A1 method, ordering) and **live** at R0 and R3 (A2 method).
`CORPUS-HELPS` per arm and rung if median < −5 % and p < 0.05. Registered expectation:
POSITIVE for `gnn` and `peeronly` (T1 → T1b), uncertain for `mpoff`.

**B2 — the Phase B headline.** `peeronly_1670` vs `mpoff_1670`, live at R0 and R3, the A2
bars and verdicts. Registered expectation: uncertain — this is the reading the question
"how can the graph arm win" turns on.

## Cells, seeds, cost — declared in advance

* Cells: `cs6s{9001,9002,9003,9005}` and `cs80s{9001,9002,9003,9005}` as minted for P3;
  workloads `drainable_f4000_n50000` / `drainable_f300_n50000`; window 16 s, poll 1 ms, batch
  10, `SIM_FORCE_FULL_STATS=1`; **the R3 rung is saturated** (reactive queue ~603 s) and every
  live number there is relative to reactive or paired between arms, never absolute.
* Checkpoint seed 3 excluded; topology 9004 excluded; 25 % attrition budgeted; every arm name
  carries cell, rung, arm, corpus and seed; `--patience 60 --min-epochs 100`; lr 2e-3 not
  re-tuned (carried limitation).
* Cost: Phase A — 16 training runs (~4 h wall) + 2 A0 arms + 32 live arms (~1 h).
  Phase B — corpus prep + cache on datalab (~4 h), 48 training runs (~6 h wall), 96 live arms
  (~2 h). About two days of cluster time.
* **Not in scope:** the window (closed), the platform cap (deadlocks), queue clamps (closed),
  horizon labels (chaos), closed-loop policy gradient (closed at n = 120), attention pooling
  over candidates (a different representation), load-matching the rungs (closed by
  `cluster_scale_v1` S0.a). A peer-dominated regime (peer term ≥ 50 % of the scored latency
  at a drainable load) is the follow-on registration if B2 reads `POINTWISE-BETTER`.

## Record

### 2026-09-17 — Phase A training and A1: **ENCODING-COSTS offline, against the expectation**

Job 781533: **16/16 COMPLETED**, every arm stopped on patience at 130–208 epochs (last
improvement at 69–147). `peeronly` selects later and more variably than either twin
(selected epoch 17–147, median 48; `gnn` 24–140, median 72; `mpoff` 22–56, median 38).

**A1** (`simulation_data/peer_only_v1/a1.json`, held-out regret at the selected checkpoint as
% of the 34 test datasets' mean optimal RTT, paired by training seed, exact Wilcoxon):

| comparison | `peeronly` | twin | paired median Δ | p | `peeronly` ahead | read |
|---|---|---|---|---|---|---|
| vs `gnn` | 41.92 % | 38.40 % | **+3.31 pp** | 0.0005 | 1/16 | ENCODING-COSTS |
| vs `mpoff` | 41.92 % | 40.77 % | **+1.17 pp** | 0.0131 | 4/16 | ENCODING-COSTS |

**The registered expectation (`peeronly` ≈ `gnn`, both ahead of `mpoff`) is refuted
offline.** On the supervised target, `PeerConv` without the GIN fits held-out *worse* than
either the full graph arm or the pointwise twin — the offline MP edge is not `PeerConv`
alone; it needs the platform-side message passing it was measured with. Ordering only
(rule 6); A2 is the live read and the arm whose offline score is worst has been the arm that
wins live before (`offline_live_transfer_v1`: the score ranks epochs, never arms). Recorded
before A2 was read.

### 2026-09-17 — Phase A live: **A0 bit-identical · A2 TIE at both rungs · A3 fires at 80 servers only · A4 kept**

Job 781985, **36/36 COMPLETED**, 0 hangs. `simulation_data/peer_only_v1/phase_a.json`,
summaries under `results/po_v1/`; reactive, `gnn` and `mpoff` arms reused from P3 under A0.

**A0 — INSTRUMENT-PASS, exactly.** The re-served v3 `gnn` and `mpoff` seed-1 arms reproduce
their P3 mean elapsed to **|Δ| = 0.00000 s** on both rungs (22.2818 / 21.4006 s at 6 servers,
465.1545 / 328.9311 s at 80): the model change is inert for every existing arm.

**A2 / A3 / A4**, `peeronly` vs `mpoff` paired by (cell, seed), n = 16 per rung:

| rung | A2 elapsed | A3 queue vs `gnn` | A3 queue vs `mpoff` | A4 peer cost vs `mpoff` |
|---|---|---|---|---|
| R0 (6) | +0.64 %, p = 0.57, ahead 7/16 → **TIE** | −9.1 %, p = 0.44 | +0.4 %, p = 0.53 → **not confirmed** | −0.3 %, p = 0.41 → KEPT |
| R3 (80) | +2.61 %, p = 0.0052, ahead 2/16 → **TIE** (inside the 5 % band) | **−22.3 %, p = 0.0004** | +2.6 %, p = 0.0052 → **GIN-IS-THE-OVERREACTION** | +2.8 %, p = 0.0006 → KEPT (inside 5 %) |

**Headline: TIE.** `peeronly` is neither better nor worse than the pointwise twin by the
registered bars, on either rung. Per cell, median over checkpoint seeds (mean elapsed, s):

| cell | reactive | `gnn` | `mpoff` | `peeronly` | queue `gnn` / `mpoff` / `peeronly` | scale events `gnn` / `mpoff` / `peeronly` |
|---|---|---|---|---|---|---|
| cs6s9001 | 20.95 | 22.50 | 21.55 | 21.27 | 9.5 / 8.6 / 8.3 | 7,574 / 6,769 / 6,864 |
| cs6s9002 | 23.48 | **54.07** | 83.24 | 78.20 | 41.9 / 70.5 / 65.6 | 5,701 / 6,456 / 6,430 |
| cs6s9003 | 20.90 | 22.48 | 22.54 | 22.91 | 9.7 / 9.9 / 10.3 | 7,874 / 8,094 / 8,268 |
| cs6s9005 | 35.91 | 90.48 | **53.74** | 63.28 | 77.4 / 40.9 / 50.5 | 7,168 / 6,846 / 6,848 |
| cs80s9001 | 596.0 | 428.8 | 321.8 | 330.1 | 423 / 317 / 325 | 196 / 95 / 93 |
| cs80s9002 | 626.8 | 443.1 | 325.9 | 328.3 | 437 / 321 / 323 | 129 / 88 / 96 |
| cs80s9003 | 611.8 | 441.3 | 327.4 | 331.3 | 435 / 323 / 326 | 148 / 102 / 103 |
| cs80s9005 | 609.3 | 420.2 | 321.6 | 340.8 | 414 / 317 / 336 | 126 / 86 / 89 |

**What the mechanism bar says, precisely.** At 80 servers the claim holds on every cell:
drop the bipartite GIN and the graph arm's queue falls from ~430 s to ~326 s — the pointwise
twin's level, within 2.6 % — and its autoscaler churn falls from 126–196 events to `mpoff`'s
86–103. **The GIN is the over-reaction at scale**, and removing it recovers all of `mpoff`'s
advantage over `gnn` (−28 % → −45 % vs reactive) but **nothing beyond it**. At 6 servers the
same block is a coin flip per topology: on cs6s9002 the GIN is what makes `gnn` the *best*
learned arm (54 s against 78–83 s), on cs6s9005 it is what makes it the worst (90 s against
54–63 s), so the paired read nulls (p = 0.44) and A3 does not fire there.

**Claim A, as read.** The over-reaction is real and the GIN is its source; taking the GIN out
turns the graph arm into the pointwise twin, live, on every statistic (elapsed, queue, churn,
peer cost) — and offline it fits worse than both (A1). `PeerConv` alone carries no live edge.
Whether the corpus is what it was starved of is Phase B's question; Claim A on its own does
not make the graph arm win.

### 2026-09-17 — Phase B corpus: **B0 INSTRUMENT-PASS**, 1,657 datasets

Job 781534, 69 min, 41.7 GB. `train2`'s 1,500 datasets received the T1b preparation for the
first time: SSC rewritten for **1,500/1,500** (from `optimal_result.json`, no simulation),
alpha pre-scan at 2.0 set aside **13** (1,487 remain) and **all 516 T1b parents stayed in
place** (asserted). Cache `graphs_cache_peer_only_v1_1670_psv3`: 136 + 1,487 + 34 =
**1,657 datasets**, 56,870,981 RTT rows, contract `partial_state_v3`, the same V = 1
measured-clock label and alpha 2.5 as the psv3 cache. **B0:** metadata agrees, the 516
parents' partial-state ingredients are identical to the psv3 cache's (max diff 0.0, none
missing), the 34-dataset test split is present in both ⇒ **INSTRUMENT-PASS**. Phase B
training submitted: job 782116, 48 arms (`gnn`, `mpoff`, `peeronly` × 16 seeds), which
refused to start without this artefact.

**Job 782116 FAILED in 9 s on every arm, correctly:** `assert_split_artifact_covers` refuses a
split artifact that does not enumerate exactly the cache's parents, and the T1b artifact knows
516. The registration's sentence "the split artifact is unchanged" was wrong as a file and
right as a fact: `experiments/peer_only_v1_1670_split.json` carries **T1b's test (34) and val
(96) verbatim** and only extends train (386 → 1,527 with the 1,141 new parents), so
checkpoint selection and the held-out set are identical between the 516- and 1,657-dataset
arms and B1 pairs like with like. Sha `0f1ee96edeb7…`. Resubmitted as job **782166**.

**Job 782166 FAILED at ~2.5 min on every arm, also correctly, on a corpus defect:**
`refresh_partial_state_edge_attr: task 1 has a candidate absent from the partial-state
context (missing key (22, 118))`. The demand table is built from the placement sweep's rows,
while the graph's candidate set lists every replica of the task's type; in **3 of the 1,141
new `train2` datasets** (ds_00427, ds_00493, ds_00925 — sweeps complete, 13,824 / 10,368 /
5,184 rows) a replica on node 22 / platform 118 is a candidate for tasks 1–2 that no sweep
row ever places on, and node 22 is absent from the ranked hosting nodes. **0 of the 516 T1b
parents** has the defect. Handled as a set-aside with its reason (`…_setaside_peer_only_v1/
candidate_not_in_sweep/REASON.md`), a cache-level guard that fails loud on the class
(`scripts_cosim/peer_only_v1_candidate_check.py`), the dobj recipe's sweep-completeness loop
restored in the corpus job, and the split artifact minted **from the cache** in the same job
(`scripts_cosim/peer_only_v1_split.py`) so cache and artifact cannot drift. The corpus is
rebuilt at **1,654** datasets; B0 and the artifact are re-read from the rebuild.

### 2026-09-17 — Phase B rebuild, training, and the offline read: **the graph arm wins offline at 1,654**

Corpus job **782258** (75 min) rebuilt the cache with the three set-asides in place:
`train2` 1,487 datasets, all sweeps complete on all three source dirs, **all 516 T1b parents
still present**, cache `graphs_cache_peer_only_v1_1670_psv3` at **1,654** datasets and
56,841,605 RTT rows under `partial_state_v3` with the same V = 1 measured-clock label.
`peer_only_v1_candidate_check.py` reads **offenders = 0** over all 1,654 graphs, so the class
that killed job 782166 is now proven absent rather than assumed. **B0 re-reads
INSTRUMENT-PASS** (big 1654, small 516, missing 0, metadata agrees, ingredients max diff
**0.0**, test ids same). The split artifact was minted from the cache in the same job:
train 1,524 · val 96 · test 34 · n 1,654 (1,138 new parents), sha `f9c79fd7bb77…`, committed
before training.

Training job **782603**: 48/48 arms COMPLETED, 48 checkpoints plus 48 final-epoch copies,
every one with a `.contract.json` sidecar, **16/16 patience stops on every arm** (no arm ran
the 300-epoch budget). The untrained eval logs at `task_acc = 38.3 %`, which is the chance
floor, so the curves now start where they should.

**Offline, held-out regret at the selected checkpoint** (% of the 34-dataset test split's
mean optimal RTT, 442.11 s; paired exact Wilcoxon over 16 training seeds, A1 method,
`simulation_data/peer_only_v1/b_offline.json`):

| pair | medians | paired median | p | ahead | verdict |
|---|---|---|---|---|---|
| `peeronly` vs `gnn` | 41.28 % vs 35.85 % | **+5.80 pp** | 0.0004 | 0/16 | ENCODING-COSTS |
| `peeronly` vs `mpoff` | 41.28 % vs 39.66 % | **+2.00 pp** | 0.0006 | 1/16 | ENCODING-COSTS |
| `gnn` vs `mpoff` | 35.85 % vs 39.66 % | **−4.20 pp** | 0.0006 | 15/16 | ENCODING-HELPS |

Two things, both ordering-only (rule 6 — B1 and B2 are the live bars):

1. **`peeronly` loses to both twins on the big corpus too**, more heavily than at 516
   (+5.80 pp vs gnn here, +3.31 pp there). PeerConv alone is not the carrier of the offline
   edge; A1's ENCODING-COSTS is not a small-corpus artefact.
2. **Full message passing beats its MP-OFF twin offline by 4.20 pp on 15/16 seeds** at 1,654
   datasets, reproducing `peer_affinity_v1` T1b's +5.14 pp at 482 on a 3.4× corpus, a
   size-free representation and patience-selected checkpoints. Selected epochs order the same
   way as the regret (gnn median 80, mpoff 140): the pointwise arm needs longer and still
   lands worse. **This is an offline number and this program's offline/live sign has reversed
   before** (`offline_live_transfer_v1`, ρ = −0.030 across 96 checkpoints); B2 is the reading
   that counts.

Gate submitted in two halves — `--array=36-131` is rejected with `AssocMaxSubmitJobLimit`
because an array counts every task against the account's `MaxSubmit = 50`, and 96 > 50.
Half 1 is job **782848** (`--array=36-83%12`); half 2 follows it.

### 2026-09-17 — Phase B live: **B1 CORPUS-DOES-NOT-HELP 6/6 · B2 PEERONLY-BEATS-POINTWISE at 80 servers**

Gate jobs **782848** (`--array=36-83%12`) and **782921** (`--array=84-131%12`), 96/96 arms
COMPLETED, 132 summaries in `results/po_v1`. **A0 re-read bit-identical** on all four
re-serves (`|d| = 0.00000`), so the `mp_platform_edges` flag remains inert on the arms that do
not set it, across a corpus rebuild and a second gate.

**B2 — the registered headline** (`peeronly_1670` vs `mpoff_1670`, elapsed, paired by
(cell, checkpoint seed), n = 16):

| rung | median | ahead | p | verdict |
|---|---|---|---|---|
| R0, 6 servers | −7.98 % | 12/16 | 0.0557 | TIE (inside the 5 % band on p) |
| **R3, 80 servers** | **−15.68 %** | **16/16** | **0.0004** | **PEERONLY-BEATS-POINTWISE** |

At R3 the same arm also **beats reactive Knative by 41.4 % on 16/16** pairs (per cell: −41.4,
−42.5, −42.1, −40.6 %), cuts queue **−15.77 %** against `mpoff` and **−22.50 %** against
`gnn`, and cuts the peer term **−11.59 %** on 15/16 (A4 PEER-TERM-KEPT). **This is the first
measurement in the program in which a message-passing arm beats both its own MP-OFF twin and
reactive Knative on the same cells and seeds.** Every clause below is part of that sentence.

**Clause 1 — it loses to the best pointwise arm the program has.** `peeronly_1670` vs
`mpoff_516` at R3 is **+10.40 %, 0/16, p = 0.0004**. The B2 contrast is corpus-matched by
registration, and corpus-matched is the comparison this program insists on
(`link_mp_v1`, `reliability_matched_v1`); against the *best available* scheduler the graph
arm still loses. Both statements are true and neither may be quoted without the other.

**Clause 2 — B1 says the corpus made every arm worse live, 6/6:** `gnn` −1.59 % (p = 0.72) /
+7.25 % (p = 0.098), `mpoff` **+14.61 %** (p = 0.026) / **+35.49 %** (0/16, p = 0.0004),
`peeronly` +6.62 % (p = 0.063) / +6.51 % (2/16, p = 0.0019), at R0 / R3. Registered
expectation was POSITIVE for `gnn` and `peeronly`; the read is **CORPUS-DOES-NOT-HELP
everywhere**. So the B2 flip (516: +2.61 % TIE → 1670: −15.68 %) is mostly **`mpoff`
degrading 35 %**, not `peeronly` improving: at 516 the arms read −45.26 % / −47.28 % vs
reactive, at 1670 −41.37 % / −28.00 %.

**Clause 3 — the offline ranking is exactly inverted.** Offline at 1,654 the order is
`gnn` (35.85 %) < `mpoff` (39.66 %) < `peeronly` (41.28 %); live at R3 it is `peeronly`
(−41.4 %) < `mpoff` (−28.0 %) < `gnn` (−23.6 %). The offline-best arm is the live-worst, on
the same checkpoints. This is `offline_live_transfer_v1`'s anti-correlation in its sharpest
form yet: a full reversal of a three-way ordering, not an offset.

**Clause 4 — R3 is saturated and the statistic is relative.** Reactive elapsed is ~600 s at
R3 against 21–36 s at R0, exactly as `partial_state_v3` P4 recorded. At the unsaturated R0
rung **every arm loses to reactive** (`peeronly` +21.6 %, `gnn` +33.9 %, `mpoff` +70.5 %,
1/16 pairs ahead).

**Clause 5 — the arm that wins is not the GNN.** `peeronly` is PeerConv only; the full
`gnn` (PeerConv + GIN) is beaten by `peeronly` at R3 by 22.42 % on 16/16. The registered A3
mechanism does not survive either: at 1670 `peeronly`'s queue improves against **both**
twins (−22.50 % vs `gnn`, −15.77 % vs `mpoff`), so "GIN is the over-reaction" no longer
isolates the cause ⇒ **MECHANISM-NOT-CONFIRMED** at both rungs.

Read: `scripts_cosim/peer_only_v1_gate_read.py --phase b`,
`simulation_data/peer_only_v1/phase_b.json`.

### 2026-09-17 — AMENDMENT 1: B3, the same contrast with the checkpoint as the unit

**Registered before any B3 arm was submitted.** Bars are module constants in
`scripts_cosim/peer_only_v1_read.py` (`B3_SEEDS`, `B3_MIN_SEEDS = 16`,
`B3_IMPROVE_PCT = 5.0`, `B3_ALPHA = 0.05`, `B3_RUNG = "R3"`), committed with this section.

**Why.** B2 pairs by (cell, checkpoint seed) and reports **n = 16** from `CKSEEDS = (1, 2, 4, 5)`
crossed with 4 topology cells. That is exactly what A2 registered (`A2_MIN_PAIRS = 12`) and the
bar was honoured as signed — this is not a violated bar. But the headline is a claim about an
**architecture**, and for that claim the independent unit is the **checkpoint**, of which there
are four. Their seed-level medians of the `peeronly`/`mpoff` ratio at R3 are −9.75 %, −27.16 %,
−20.15 %, −5.86 %: all four negative, and a two-sided sign test on four units cannot go below
**p = 0.125** however consistent they are. The reactive comparison is thinner still — one
reactive run per cell, so "16/16 vs reactive" re-uses 4 cells four times. `peer_affinity_v1`'s
"13/16 seeds" was 16 *training* seeds; this lineage's "16/16" is not the same quantity.

**B3.** All 16 trained checkpoints of `peeronly_1670` and `mpoff_1670`, R3 only, 4 cells each.
One value per checkpoint — the median over its four cells (`collapse_to_seed`, which fails loud
on a ragged row rather than averaging over what it has) — then the paired exact Wilcoxon over
16 checkpoints. `PEERONLY-BEATS-POINTWISE` if median ≤ −5 % and p < 0.05; otherwise
**`PEERONLY-BEATS-POINTWISE-UNDERPOWERED`**.

**The consequence is signed here, before the data.** If B3 does not clear, the node records the
headline as underpowered and CLAUDE.md's standing answer reverts to its pre-2026-09-17 wording.
If it does clear, the standing answer keeps its two halves and gains the seed-level number.

Cost: 96 arms at ~2 min each, submitted as two arrays of 48 (`MaxSubmit = 50`). The checkpoints
already exist; nothing is retrained. Tasks 132–227 are **appended** to the gate's table so
indices 0–131 keep their arms and their summaries are never re-read.

### 2026-09-17 — the B2 result re-checked against the raw summaries

Everything below is a read of the 274 gate summaries, not a re-run. **The direction survives
every check; two things in how B2 was first written up do not.**

**What held.**

| check | result |
|---|---|
| completion parity | **50,000 / 50,000 tasks in all 274 arms** — the margin is not a dropped-task artifact |
| served architecture | `(GNN_DISABLE_MESSAGE_PASSING, GNN_MP_PLATFORM_EDGES_OFF)` is `(0,0)` / `(1,0)` / `(0,1)` for `gnn` / `mpoff` / `peeronly`, 32 arms each, distinct `GNN_MODEL_PATH` per arm |
| code parity | `code.diff_sha256` is the empty-string hash in **every** summary, and `git diff 73632a6..b650fa4 -- src/` is empty, so A0's bit-identical re-serve carries to the Phase B gate |
| training recipe | the 516 and 1,670 yamls differ **only** in `cache_dir` and the split artifact — B1 is a clean corpus contrast |
| consistency | `peeronly` beats `mpoff` at R3 in **4/4 checkpoints and 4/4 cells** |

**The corpus is not off-distribution** — the "more data made it worse because the new data is
different" explanation is refuted, dataset for dataset:

| | datasets | tasks | candidates/task | nodes | platforms | peer edges | optimal RTT (mean) |
|---|---|---|---|---|---|---|---|
| T1b parents | 516 | 10.00 | 2.813 | 4.93 | 4.93 | 34.6 | 480.16 s |
| new parents | 1,138 | 10.00 | 2.795 | 4.93 | 4.93 | 34.6 | 473.13 s |

**Correction 1 — the margin is queue, not peers.** At R3 elapsed is ~99 % queue for every arm:

| arm | elapsed | queue | wait | peer/task | scale events |
|---|---|---|---|---|---|
| reactive | 610.56 | 603.31 | 0.00 | 7.18 | 150 |
| `1670_gnn` | 460.45 | 454.25 | 0.73 | 5.34 | 172 |
| `1670_mpoff` | 436.01 | 430.18 | 0.73 | 5.00 | 120 |
| `1670_peeronly` | 357.90 | 352.77 | 0.73 | 4.34 | 92 |
| `516_peeronly` | 332.22 | 327.32 | 0.73 | 4.13 | 93 |
| **`516_mpoff`** | **325.36** | 320.53 | 0.73 | 4.01 | 95 |

The **peer term differs by 0.66 s out of a 78 s gap — under 1 %**. The arm wins by building
less queue and triggering fewer autoscale events (92 vs 120), not by placing peers better.
That is the substance behind A3's `MECHANISM-NOT-CONFIRMED`, and it means B2 is **not**
evidence that peer-graph reasoning is what pays. Note also that `516_mpoff` is the fastest arm
at the rung and **both** 516 arms beat **every** 1,670 arm.

**Correction 2 — one genuinely new positive, previously unrecorded.** The advantage over
`mpoff` **grows monotonically across the trace**: −6.2, −8.5, −9.8, −9.2, −13.9, −17.6, −16.2,
−17.2, −23.4, **−27.9 %** by decile. Every previous positive in this program
(`serving_stability_v1`) was an early-trace effect that decayed; this is the opposite shape, so
it is not that effect re-appearing. `1670_gnn` meanwhile carries a mid-trace excursion (queue
567 → 702 s at deciles 2–4, recovering to ~320 s), which is the GIN over-reaction made visible.

**Correction 3 — B1 is the offline/live anti-correlation, not "the data is bad."** On the
identical 34-dataset test set and the identical denominator (442.11 s), held-out regret
516 → 1,670: `gnn` 38.40 → **35.85 %**, `mpoff` 40.77 → **39.66 %**, `peeronly` 41.92 →
**41.28 %**. **Every arm improved offline and worsened live.** The corpus did move the
objective it was optimising; the transfer is what failed, for the fourth time in this program.
Live seed variance also explodes with corpus size — across the four checkpoints `516_mpoff`
spans 318.6–329.9 s (~3 %) while `1670_mpoff` spans 368.3–483.2 s (~31 %) — which is the
mechanical reason four draws is too few here, and why B3 (Amendment 1) was registered.

### 2026-09-17 — B3: **PEERONLY-BEATS-POINTWISE holds on 16 checkpoints, at less than half the margin**

Jobs **783263** + **783325**, 96/96 arms COMPLETED, 144 R3 arms in `results/po_v1`. Read:
`peer_only_v1_gate_read.py --phase b3`, `simulation_data/peer_only_v1/phase_b3.json`.

**B3, one value per checkpoint (median over its 4 cells), all 16 trained seeds:**
median **−5.75 %**, **p = 0.0011**, **14/16 checkpoints ahead** ⇒ **PEERONLY-BEATS-POINTWISE**
against the registered bar (median ≤ −5 %, p < 0.05, all 16 seeds present). Per checkpoint:
−10.0, −27.3, −35.4, −19.0, −6.9, **+3.6**, −4.3, **+0.2**, −20.3, −0.4, −3.0, −9.1, −4.6,
−4.3, −32.0, −4.4 %.

**The claim stands and the effect size does not.** With the 64 (cell, checkpoint) pairs the
same A2 statistic reads **−7.10 %, 58/64, p < 0.0001**, against **−15.68 %** on the original
four checkpoints. **The 4-checkpoint read overstated the margin by more than 2×**, exactly the
failure the amendment was registered against, and the checkpoint-level median (−5.75 %) now
sits barely above the 5 % bar with **2 of 16 checkpoints on the wrong side**.

Every descriptive number from the 4-seed subset moves the same way and the node's earlier
figures are superseded by these:

| statistic at R3 | 4 checkpoints | **16 checkpoints** |
|---|---|---|
| `peeronly_1670` vs `mpoff_1670` (pairs) | −15.68 %, 16/16 | **−7.10 %, 58/64** |
| `peeronly_1670` vs `mpoff_1670` (checkpoints) | — | **−5.75 %, 14/16, p = 0.0011** |
| `peeronly_1670` vs reactive | −41.37 % | **−41.45 %, 64/64** (checkpoint-level −41.48 %, 16/16) |
| `mpoff_1670` vs reactive | −28.00 % | **−36.81 %, 64/64** |
| queue, `peeronly` vs `mpoff` | −15.77 % | **−7.15 %** |
| peer term, `peeronly` vs `mpoff` | −11.59 %, 15/16 | **−4.81 %, 56/64** |

`mpoff_1670` was **not** as bad as four checkpoints made it look (−36.81 %, not −28.00 %), so
B1's "the corpus cost `mpoff` +35.49 %" is itself a 4-checkpoint figure and is carried as such
until the 516 side is extended — which is what **B4 (Amendment 2)** now does.

**A0 re-read `INSTRUMENT-PASS`, `|d| = 0.00000` on all four re-serves**, across a third gate
submission. The bar, the arm table indices 0–131 and their summaries were untouched.

### 2026-09-17 — AMENDMENT 2: B4, clause 3 with the checkpoint as the unit

**Registered before any B4 arm was submitted.** Bars: `B4_MIN_SEEDS = 16`, `B4_TIE_PCT = 5.0`,
`B4_ALPHA = 0.05`, `B4_RUNG = "R3"` in `scripts_cosim/peer_only_v1_read.py`.

Clause 3 of the head — *"it loses to the best pointwise arm the program has,
`peeronly_1670` vs `mpoff_516` = +10.40 %"* — is also a 4-checkpoint read, and B3 has just
shown a 4-checkpoint read overstating an effect by more than 2×. `mpoff_516` **is**
`partial_state_v3`'s `mpoff`, all 16 of whose seeds exist and whose re-serve A0 proves
bit-identical, so the 12 unused ones extend the *same* arm (gate corpus tag `v3ext`, relabelled
to `516_mpoff` by the read; `tables()` now refuses two summaries for one arm name rather than
letting the last one win). Paired by training seed across the two corpora, as B1 pairs.

**Consequence signed before the data:** if B4 reads median ≥ +5 % with p < 0.05 the verdict is
`POINTWISE-STILL-BEST` and clause 3 stands as written. Otherwise it is
`BEST-ARM-NOT-ESTABLISHED`, clause 3 is **withdrawn**, and CLAUDE.md's second half — "the best
scheduler the program has is still a pointwise one" — is rewritten to match. 48 arms, ~10 min.

### 2026-09-17 — B4: **POINTWISE-STILL-BEST**, and clause 3 gets *stronger* with power

Job **783389**, 48/48 arms COMPLETED. `peeronly_1670` vs `mpoff_516` at R3, one value per
checkpoint, 16 seeds: median **+11.17 %**, **p = 0.0004**, **0/16 checkpoints ahead** ⇒
**POINTWISE-STILL-BEST**. Per checkpoint: +1.7, +8.0, +5.1, +13.3, +18.2, +15.5, +15.5, +31.2,
+31.1, +11.3, +11.0, +12.9, +7.9, +3.7, +9.5, +7.5 % — the pointwise arm ahead on **every
one**, and the 4-checkpoint estimate (+10.40 %) was, unusually, a slight *under*-statement.

**The two amendments move the headline in opposite directions, and both were signed before
their data.**

| claim | 4 checkpoints | 16 checkpoints | verdict |
|---|---|---|---|
| B3 — `peeronly` beats its MP-OFF twin | −15.68 % | **−5.75 %**, 14/16, p = 0.0011 | holds, **margin more than halved** |
| B4 — the best pointwise arm is still ahead | +10.40 % | **+11.17 %**, 0/16, p = 0.0004 | holds, **strengthened** |

So the lineage's two-part answer survives with its weaker half weakened and its stronger half
confirmed: **message passing over the peer graph does beat its own MP-OFF twin live, by about
6 % rather than 16 %, and the best scheduler the program has is still the pointwise arm trained
on the small corpus, by 11 % on every checkpoint.** Both halves are now read at n = 16
checkpoints; neither rests on four draws.

What this costs the "first arm to beat both" sentence: nothing in kind — `peeronly_1670` still
beats reactive on **64/64** pairs and **16/16** checkpoints (−41.5 %) while beating its twin —
and a great deal in degree. The arm that does it is PeerConv-only, its margin over the twin is
~6 %, its advantage is 99 % queue rather than peer placement, and a pointwise arm on a third of
the data is 11 % faster than it.
