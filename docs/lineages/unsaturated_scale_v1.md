# unsaturated_scale_v1 — does a learned arm beat reactive on BIG infrastructure where the baseline is healthy?

**Status:** `CLOSED` (2026-09-19) — **`NO-LEARNED-ARM-BEATS-REACTIVE-AT-UNSATURATED-SCALE`.**
Closed on a live gate (rule 6): 320 arms, 16 checkpoints × 4 cells × 5 arms. Registered
2026-09-19; every bar below is a module constant in `scripts_cosim/unsaturated_scale_v1_read.py`,
committed with its 13 tests before any arm ran. The registered expectation was **wrong**.

**Outcome — two findings, the first about the baseline.**

1. **Reactive Knative's capacity does not grow with the cluster** (S1). At 80 servers it is
   unsaturated only at the *same absolute arrival rate* 6 servers runs at — 0.460/s, queue share
   0.660 — and at 0.92/s it already reads 0.955. The server ladder held arrivals-per-server
   constant and thereby pushed the baseline 13× past a throughput ceiling that never moved;
   **that** is what the 99 % queue shares at R1/R2/R3 were, and why every 80-server "win" in the
   record (`partial_state_v3` −28.7/−46.9 %, `peer_only_v1` B2 −41.5 %, `bipartite_edge_v1` D1
   −42.2 %) was "less drowned", never "faster".
2. **At the one healthy 80-server rung, nothing learned is separated from reactive** (S2, K4).
   `gnnedge0` −6.46 % (p = 0.47, 9/16), `peeronly` −7.97 % (p = 0.12, 13/16), `mpoff_1670`
   −0.21 %, `mpoff_516` −0.20 %: all `NOT-SEPARATED`. The proper GNN **loses**, +16.98 %
   (p = 0.044). K2 (graph vs matched twin) −6.10 %, not separated. **K3 fired as predicted:**
   `sum` costs **+26.69 % (p = 0.039)** — `bipartite_aggr_v1`'s mechanism confirmed at a second
   operating point, one where the baseline is healthy.

**The cause is the batching tax, and it is a function of arrival rate, not cluster size.**
Every learned arm pays **6.82 s of batch-assembly wait** per task that reactive pays 0.00 for —
the same 6.77 s R0 pays, because the healthy 80-server rung runs at R0's arrival rate. The arms
win ~6 s of queue (11.2 vs 17.3 s) and give it all back on assembly; per checkpoint that nets
out as a coin flip (roughly half the checkpoints beat reactive by 6–13 %, the other half lose
by 14–130 %). A size-free representation and a mean aggregation got the model *to* 80 servers;
the serving design is what keeps it from winning there.

**Carry.** (a) These are 6-server checkpoints served 9.58× out of candidate support
(`cluster_scale_v1`); the result closes "the existing checkpoints on big infrastructure", not
"a GNN trained at scale", for which no data generator exists. (b) `peeronly` at 13/16 ahead
with a −8 % median is the closest thing to a signal and is **not** quotable as one: the three
losing checkpoints lose by +10 / +50 / +95 %. (c) `f1000` on `cs80s9001` hung (starved-client
spin) and was cancelled; it could not have been selected. (d) The reactive cells vary 20.5–46.3 s
at this rung, so the per-cell picture is uneven (`gnnedge0` wins 3 of 4 cells by 8–18 %, loses
`cs80s9001` by 6.7 %); the registered unit is the checkpoint and the read is what stands.

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
`scripts_cosim/datalab/unsaturated_scale_v1_s1.sbatch` → `results/us_v1/`. S2 is rung index 4
(`U80`) of `scripts_cosim/datalab/peer_only_v1_gate.sbatch`, tasks 1364–1683, so that it shares
the cells, checkpoints, sidecar pins and summary schema of every arm it is compared against
(the D/E/F/H precedent) → `results/po_v1/`. Gate read:
`scripts_cosim/unsaturated_scale_v1_gate_read.py s2 ... --factor 4000`.

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

## 2026-09-19 — S2: **`NO-LEARNED-ARM-BEATS-REACTIVE-AT-UNSATURATED-SCALE`** — the registered expectation was wrong

Jobs 789489 / 789539 / 789588 / 789639 / 789687 / 789743 / 789795, 320 arms, all COMPLETED,
every one at `num_tasks = 50000` on `drainable_f4000_n50000`, 0 failures. Wall ~3.5 min per arm.
Reactive at this rung (from S1): 23.27 / 29.14 / 46.33 / 20.54 s per cell, median **26.21 s**,
queue 17.30 s, batch wait 0.00 s, queue share 0.660.

**K1 — each arm vs reactive**, one value per checkpoint (median over 4 cells), n = 16, paired
two-sided Wilcoxon, bar |median| ≥ 5 % and p < 0.05:

| arm | median vs reactive | p | ahead | elapsed (median of ck) | verdict |
|---|---|---|---|---|---|
| `be1670_gnnedge0` | −6.46 % | 0.469 | 9/16 | 24.51 s | `NOT-SEPARATED` |
| `1670_peeronly` | −7.97 % | 0.121 | 13/16 | 24.12 s | `NOT-SEPARATED` |
| `1670_mpoff` | −0.21 % | 0.679 | 9/16 | 26.15 s | `NOT-SEPARATED` |
| `516_mpoff` | −0.20 % | 0.215 | 8/16 | 26.15 s | `NOT-SEPARATED` |
| `1670_gnn` | **+16.98 %** | **0.044** | 7/16 | 30.65 s | **`REACTIVE-FASTER-AT-UNSATURATED-SCALE`** |

**K2** `gnnedge0` vs `mpoff` (both 1,670): −6.10 %, p = 0.469, `NOT-SEPARATED`. The −31.65 %
(16/16) the same contrast reads at 40 clients does not appear here.
**K3** `gnn` vs `gnnedge0`: **+26.69 %, p = 0.039, `SUM-COSTS-AT-UNSATURATED-SCALE`** — as
registered. The mechanism (`sum` over ~48 candidates/task) predicted the cost would be there at
any load, and it is.
**K4:** `NO-LEARNED-ARM-BEATS-REACTIVE-AT-UNSATURATED-SCALE`, winners none, losers `1670_gnn`,
nothing missing, primary.

**Registered expectation was wrong.** It said K4 would fire through `gnnedge0` and `peeronly`,
reasoning from C80 (same queue-share band, −13.5 %) and from assembly wait shrinking with
arrival rate. The second premise was the error: assembly wait shrinks with arrival *rate*, and
the healthy 80-server rung runs at R0's rate, so the tax is R0's tax.

**Where the time goes (median s/task over checkpoints).**

| arm | elapsed | queue | batch wait |
|---|---|---|---|
| reactive | 26.21 | 17.30 | **0.00** |
| `gnnedge0` | 24.51 | 11.20 | 6.83 |
| `peeronly` | 24.12 | 10.90 | 6.82 |
| `mpoff` (1670) | 26.15 | 12.96 | 6.82 |
| `mpoff` (516) | 26.15 | 12.88 | 6.82 |
| `gnn` | 30.65 | 17.44 | 6.81 |

Every learned arm wins queue by 4–6 s and pays 6.8 s of assembly; `gnn` does not even win
queue. The 6.8 s is R0's 6.77 s (`peer_only_v1` B5 decomposition) to within 0.1 s.

**The distribution is bimodal, per checkpoint** (sorted relative elapsed vs reactive, %):
`gnnedge0` −13 −12 −11 −11 −10 −9 −9 −7 −6 | 0 +14 +23 +25 +30 +30 +73;
`peeronly` −13 −12 −12 −12 −12 −10 −10 −8 −7 −7 −7 −4 −1 | +10 +50 +95;
`gnn` −11 −9 −8 −8 −7 −7 0 | +11 +23 +24 +46 +46 +65 +74 +85 +132.
The losing tail is mostly `cs80s9003` (reactive 46.33 s there; `peeronly` 72.69 s and `mpoff`
84.36 s per-cell medians). This is the training-draw lottery the record already knows
(`docs/lessons.md`), at a rung where nothing masks it.

**Descriptive, per cell** (median over 16 checkpoints, vs reactive on the same cell):
`gnnedge0` +6.7 / −17.6 / −16.2 / −8.2 %; `peeronly` −1.0 / −17.2 / +56.9 / −9.2 %;
`mpoff_1670` +8.7 / −8.2 / +82.1 / −3.9 %; `mpoff_516` +4.5 / −13.8 / −10.7 / −7.4 %;
`gnn` +44.8 / −11.9 / +0.7 / −4.2 %. Cells are not the unit; this is context only.

**What closes and what does not.** Closed: "the existing checkpoints beat reactive on big
infrastructure" — at the one 80-server rung where reactive is healthy, they do not, and the
proper GNN loses. Not closed: a model trained at scale (no generator), and a serving design
that does not wait for a batch — the 6.8 s is the whole margin and it is a design choice, not
a model property (`queue_range_v1` already found the same term deciding a 6-server cell).
