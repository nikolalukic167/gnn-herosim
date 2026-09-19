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

## 2026-09-19 — S1: **`UNSATURATED-80-SERVER-RUNG-EXISTS`, at f4000 — the same absolute arrival rate 6 servers runs at**

Job 789464, 24 reactive arms (6 factors × 4 cells) plus `f300` reused from `results/psv3_p3`.
Read: `scripts_cosim/unsaturated_scale_v1_gate_read.py s1`.

| factor | arrivals/s | reactive queue share (median of 4 cells) | saturated (≥ 0.90)? | in band (≤ 0.80)? |
|---|---|---|---|---|
| f300 (= R3) | 6.137 | 0.988 | yes | no |
| f500 | 3.682 | 0.987 | yes | no |
| f700 | 2.630 | 0.985 | yes | no |
| f1000 | 1.841 | 0.975–0.982 on 3 cells — **UNREADABLE** (below) | yes | no |
| f2000 | 0.920 | 0.955 (0.941–0.964) | yes | no |
| **f4000** | **0.460** | **0.660** (0.539 / 0.658 / 0.661 / 0.802) | **no** | **yes** |
| f8000 | 0.230 | 0.480 | no | yes |

Share is monotone in `f`. **Selected: `f4000`**, the smallest in-band factor, as the rule
signed above requires. Per cell at f4000, reactive's elapsed is 23.27 / 29.14 / 46.33 / 20.54 s
(median **26.2 s**; the 6-server R0 cells read 22.22 s at this same trace).

**The finding underneath the selection: reactive Knative's capacity does not grow with the
cluster.** At 6 servers `f4000` (0.460/s) is unsaturated at 63 %; at 80 servers — 13.3× the
servers — the *same absolute rate* is the fastest one that is unsaturated (66 %), and doubling
it to 0.92/s already reads 95.5 %. The server ladder's "arrivals per server held at R0's ratio"
therefore pushed reactive 13× past a throughput ceiling that never moved, and that is what the
99 % queue shares at R1/R2/R3 were measuring. This is a **baseline** property, and it is why no
80-server number in the record could be read as "faster than reactive": the mechanism (standing
suspect: the replica allocator, FCFS by task type — `docs/lessons.md`) is a follow-up, not this
lineage's question.

**`f1000` on `cs80s9001` hung** at simulated t ≈ 27,346 s with no further progress for 3 min
(other cells finish in ~40 s wall), the starved-client spin already in
`docs/lessons.md`; cancelled rather than left to time out. The rule drops an arm that did not
finish the trace, so `f1000` is UNREADABLE for selection; it could not have been selected
(monotone, between 0.985 and 0.955). Descriptive on 3 cells: 0.975–0.982.

Task 8 of 24 cancelled (789464_8); 23 completed, every one at `num_tasks = 50000`.

**S2 submitted** at `f4000`: `peer_only_v1_gate.sbatch` tasks 1364–1683, rung index 4 (`U80`),
in seven blocks of ≤ 48 via `PO_TASK_OFFSET`.
