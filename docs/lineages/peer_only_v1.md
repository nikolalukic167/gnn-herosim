# peer_only_v1 — message passing over the peer graph only, and the corpus it was starved of

**Status:** `REGISTERED` (2026-09-16). Every bar below is a module constant in
`scripts_cosim/peer_only_v1_read.py`, committed **before** any arm is trained. Amend by dated
amendment only.

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
