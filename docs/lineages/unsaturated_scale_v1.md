# unsaturated_scale_v1 — does a learned arm beat reactive on BIG infrastructure where the baseline is healthy?

**Status:** `REGISTERED` (2026-09-19). Every bar below is a module constant in
`scripts_cosim/unsaturated_scale_v1_read.py`, committed with its 13 tests before any arm ran.
Two live steps (rule 6): **S1** the baseline sweep, **S2** the learned arms at the rung S1
selects.

**Why this lineage exists.** The record has no unsaturated big-infrastructure measurement.
The server ladder scales arrivals with servers — R0 is 6 servers at `f4000` (0.460/s), R3 is 80
at `f300` (6.139/s), both ×13.3 — yet reactive's queue share goes **63 % → 94 % → 98 % → 99 %**
and its elapsed **22 s → 611 s** (`peer_only_v1`, B5's decomposition). Every 80-server win in
the record (`partial_state_v3` −28.7/−46.9 %, `peer_only_v1` B2 −41.5 %, `bipartite_edge_v1`
D1 −42.2 %) is therefore read against a drowning baseline: "less drowned", not "faster". The
three rungs where the standing wins were measured against a *healthy* baseline (R0, C40, C80;
63 / 77 / 80 % queue share) are all 6 servers. Whether anything learned works at scale where
Knative itself is fine has never been asked.

**Parents.** `peer_only_v1` (the cells, the ladder, the saturation bar), `bipartite_aggr_v1`
(the arm and the mechanism this rung tests at its predicted location), `corpus_matched_v1`
(the matched twin), `cluster_scale_v1` (the candidate-support finding: 47.92 candidates/task
at 80 servers against a corpus maximum of 5, **9.58× out of support at any load**).

**Entry points.** Bars and readers: `scripts_cosim/unsaturated_scale_v1_read.py`, tests in
`scripts_cosim/test_unsaturated_scale_v1_read.py`. S1 gate:
`scripts_cosim/datalab/unsaturated_scale_v1_s1.sbatch` → `results/us_v1/`. S2 will be
appended to `scripts_cosim/datalab/peer_only_v1_gate.sbatch` as a new rung index, so that it
shares the cells, checkpoints, sidecar pins and summary schema of every arm it is compared
against (the D/E/F/H precedent) → `results/po_v1/` with rung tag `U80`.

## Registration — 2026-09-19

### S1 — the baseline sweep (reactive only; 24 arms, ~25 s each)

Reactive `knative_network` on the four R3 cells (`cs80s9001/9002/9003/9005`; 9004 hangs on
every policy) across the arrival ladder `K_FACTORS = (300, 500, 700, 1000, 2000, 4000, 8000)`.
`f300` is reused from `results/psv3_p3` (it IS this cell at that factor); the six others are
minted. Every trace is a rescale of the gate trace cut at 50,000 events — the sbatch asserts
the trace's `arrival_rescale.factor` and event count, and a reactive arm that does not finish
the trace **fails loud rather than defining a rung** (the 5/10-client rungs of B7 were
excluded for exactly this).

**Statistic.** Reactive's queue share = `averageQueueTime / averageElapsedTime`, median over
the four cells, per factor.

**The selection rule (`select_factor`), signed now:** the **smallest** `f` (fastest arrivals,
hardest rung) whose median queue share is **≤ `K_TARGET_SHARE` = 0.80** — inside the
63–80 % band the standing unsaturated wins were measured in, not merely under the 90 % bar.
Smallest, not "closest to 80 %": choosing the load after seeing the ladder would be choosing
it in the arms' favour, since their known strength grows with load.

- some factor ≤ 0.80 ⇒ **`UNSATURATED-80-SERVER-RUNG-EXISTS`**, S2 is primary there.
- none ≤ 0.80 but some < 0.90 ⇒ **`UNSATURATED-ONLY-AT-THE-EDGE-OF-THE-BAR`**, S2 is
  primary at the smallest such `f`, and every quote must say it sits outside the band.
- none < 0.90 even at `f8000` (0.23/s, 5× slower per server than R0) ⇒
  **`REACTIVE-SATURATES-AT-80-SERVERS-AT-EVERY-LOAD`**. That is the primary finding — the
  baseline breaks at scale independent of load, and the standing suspect is the replica
  allocator's FCFS starvation (`docs/lessons.md`). S2 still runs, at the least-saturated
  factor, and is read as **SECONDARY** (`read_k4(..., primary=False)`).

Monotonicity of share in `f` is reported, never a bar.

### S2 — the learned arms at the selected rung (5 arms × 4 cells × 16 checkpoints = 320 arms, ~2 min each)

Arms, all existing checkpoints, no training: `be1670_gnnedge0` (the repaired graph arm, the
one that beats reactive at C40), `1670_mpoff` (its corpus-matched MP-OFF twin), `1670_peeronly`,
`1670_gnn` (the proper GNN: PeerConv + bipartite GIN with `sum`), `516_mpoff` (the best
pointwise arm, served as `v3ext`). Same env, window, poll and batch size as every other arm on
these cells. Unit: the checkpoint (median over its 4 cells), n = 16, paired two-sided Wilcoxon.
Bars: |median| ≥ 5 %, p < 0.05 — the chain's, unchanged.

- **K1** each arm vs reactive: `ARM-BEATS-REACTIVE-AT-UNSATURATED-SCALE` /
  `REACTIVE-FASTER-AT-UNSATURATED-SCALE` / `NOT-SEPARATED`.
- **K2** `gnnedge0` vs `mpoff`, both 1,670: the model-class contrast, matched, at scale.
- **K3** `gnn` vs `gnnedge0`: does the `sum` defect still cost where the baseline is healthy?
  `bipartite_aggr_v1`'s mechanism says the cost is a function of candidate-set size (~48/task
  here at any load), so it predicts **`SUM-COSTS-AT-UNSATURATED-SCALE`**. A null here would
  weaken the mechanism; a positive is its second confirmation at a new operating point.
- **K4** the composite: `LEARNED-ARM-BEATS-REACTIVE-AT-UNSATURATED-SCALE` if **any**
  registered arm fires K1; an unreadable arm is missing, never a loss; a not-separated ladder
  is not a win. The verdict names the winners.

**Registered expectation (signed, may be wrong):** S1 finds an in-band factor between `f1000`
and `f4000`. K4 fires through `gnnedge0` and `peeronly`; `gnn` does not beat reactive; K3 reads
`SUM-COSTS`. Reasoning: at C80 (80 % share, the same band) `gnnedge0` beat reactive by 13.5 %,
and the batching tax that sinks every arm at 20 clients (6.77 s of assembly wait) shrinks as
arrivals speed up (0.73 s at R3). The risk on the other side is `cluster_scale_v1`: 9.58× out
of candidate support, which a healthy baseline no longer hides.

**What a positive would and would not mean.** A K4 positive is the first defensible
big-infrastructure sentence in the record — and it is still a 6-server checkpoint served 9.6×
out of support with no big-cluster training data, so it is a transfer result, not a
trained-at-scale one. A K4 negative with S1 positive closes "a GNN on big infrastructure" for
this apparatus at the checkpoints that exist, and the route through is training data at scale,
which has no known generator.

**Cost.** S1: 24 arms × ~25 s. S2: 320 arms × ~2 min, in blocks of 48 (`MaxSubmit = 50`),
~1.5 h wall on the CPU partition.

**Explicitly not in scope.** Retraining; any new topology; the client axis; the 12/24-server
rungs (the same sweep there is a follow-up if S1 finds an unsaturated factor at 80).
