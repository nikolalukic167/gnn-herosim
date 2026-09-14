# peer_affinity_warm_v1 — train on the cluster the model is served on

**Status:** `ACTIVE` — W0 read 2026-09-13 (W0.a PASS, **W0.b NO-GO** on both registered bars,
W0.c descriptive; attachment `peer_affinity_warm_v1/w0_read.json`). **Amendment 1 (2026-09-13,
signed before any W1 number exists): W1 runs regardless of the W0.b verdict**, by the user's
standing rule that a lineage ends with a live gate, never with an offline read (CLAUDE.md rule 6).
The W0.b NO-GO stands as a measurement and is not re-scored; the W1 design, arms, seeds, bars and
headline rule below are unchanged. What W1 can now add beyond W0.b: L1 measures whether the state
mismatch costs *latency live* even though its one-step label is pointwise (the corpus lever), and
L2/L3 are the live twin and baseline contrasts the headline rule requires.
**Parent:** [`peer_affinity_v1`](peer_affinity_v1.md) (the environment, the checkpoints
this compares against, the gate cell and trace). **Question it answers:** is the
train/serve *state* mismatch the H5 audit measured — a corpus captured from a cluster the
autoscaler has not built yet, served on one it has — the reason the offline message-passing
edge does not transfer live, and does closing it at the source produce a served graph
arm that beats both its pointwise twin and reactive Knative?

**Entry points:** `scripts_cosim/make_warm_corpus.py` (snapshot → dataset),
`scripts_cosim/split_warm_snapshots.py`, `scripts_cosim/datalab/peer_affinity_warm_v1_w0_generate.sbatch`,
`src/placement/live_snapshot_seed.py` (`candidate` flag, measured drain),
`src/placement/live_audit.py` (`platform_queue_drain_seconds`),
`src/policy/knative_network_batch/scheduler.py` (`KNATIVE_BATCH_BY_PEER_GROUP`),
`tests/test_warm_snapshot_corpus.py`.

## Why this lineage exists

Every `peer_affinity_v1` corpus captures its state from a cluster whose replica set is the
generator's deterministic plan — one replica per type on 90 % of the server hosts, first
suitable platform (so never `xavierGpu`), queues drawn from Poisson(2). The audit of
2026-09-13 measured what the same checkpoints meet when served on the production trace:
the autoscaler builds **14 replicas per type by t = 7 s**, `xavierGpu` among them (7.6 %
of live candidates, a type that is a candidate in 0 of 516 training datasets), and the
queues are **11k–19k deep** by the end of the 170 s arrival window because the cluster is
overloaded ~300× (2,650 arrivals/s against ~9 completions/s once every task pays its peer
transfers). The H5 read closed with: *"a corpus whose captured state is taken before the
autoscaler has built the cluster the model will be served on does not describe that
cluster's action space."* This lineage tests that sentence the only way it can be tested —
by building the corpus from the served cluster's own states.

**What is different from the closed P2 route.** `program_verdict_v1` ruled out
"live-snapshot labels" (P2) because on the option-1 corpora the one-step target is
pointwise-separable, so labelling live states one step at a time cannot teach anything a
pointwise scorer lacks. That ruling's premise is false on `peer_affinity_v1` — the cost is
indexed by pairs of task instances and the offline MP edge is measured (+5.14 pp, p =
0.001). Whether the premise is *also* false in the served regime — deep queues, where a
batch's cost may be dominated by queue drain rather than by the peer term — is exactly what
the W0 screen below measures before any corpus is generated.

## The mechanism (built and proven on a prototype, 2026-09-13)

1. **Capture.** A behaviour policy runs the production physics live
   (`HEROSIM_PEER_EXCHANGE=1`, `HEROSIM_SERVER_ONLY_REPLICAS=1`, `node_disk_v2`,
   `PYTHONHASHSEED=0`) with `LIVE_AUDIT_SNAPSHOT_PATH` set and batches that ARE peer groups
   (`GNN_BATCH_BY_PEER_GROUP=1` for a `gnn` arm; the new `KNATIVE_BATCH_BY_PEER_GROUP=1`
   for the reactive arm — measured on the 3k smoke, the window batcher's batches coincide
   with a peer group in only 14 of 387 cases, and a pair that straddles two batches cannot
   be priced by a batch co-sim). Every snapshot carries the full per-type replica set with
   `platform_type` and a **measured queue drain** (`platform_queue_drain_seconds`: per
   queued task its execution on that platform, storage I/O, source latency and the peer
   transfers it will actually pay to peers whose node is known). Without the measured
   drain the replay's exec+comm formula drains a 16k-deep live queue ~100× too fast.
2. **Replay.** `make_warm_corpus.py` turns a snapshot into a standard dataset directory:
   the cell's **live** topology (`prepare_infrastructure_for_real_simulation`, not the
   corpus generator's — the generator repairs reachability for its own replica plan and on
   the prototype that offered two tasks a candidate their live source could not reach: 1536
   plans against the 384 the live slate spans); the whole live cluster as a
   `live_snapshot_seed` (replicas, backlogs, drains); the snapshot's own peer group as the
   workload (trace events by task id, grouped by application in wsc order as the sweep
   requires, peer pairs re-indexed); and the ordinary brute-force sweep via
   `generate_single_dataset(..., infrastructure_override=)`. SSC rewrite, alpha pre-scan,
   `prepare_graphs_cache --dag-partial-state --platform-feature-dim 14` and the split
   tooling then run unchanged.
3. **Candidate subsampling.** A warm task has 6–10 reachable replicas (mean 7.2, max 14,
   measured on the H2 decode traces), so the full product is ~10^8 plans. Per task type a
   random subset of R live replicas is marked `candidate: true`; the rest are replayed as
   **occupied non-candidates** — they load the cluster exactly as live (backlog, drain,
   removed from the autoscaler's free pool) but are not offered to the sweep and so are not
   candidate edges in the cache: the label is the optimum over exactly the slate the model
   sees. Among draws whose per-task product fits `--target-combos` (20,000 — the
   generated corpora's median is 20.7k) the most balanced one is kept; draw seeded by
   (`--seed`, snapshot id) and recorded in `warm_snapshot.json` and the provenance.
4. **Prototype (mechanics only, not a registered number):** snapshot 5 of a
   `knative_network_batch` peer-group run on `cell_s7901` (t = 12.7 s): 384 plans, per-task
   candidates `[2,3,2,2,2,1,2,2,1,2]` equal to the live slate, sweep complete, SSC
   rewritten, alpha pre-scan 0 infeasible, cache built (`partial_state_v2`, alpha 2.5,
   `legacy_v0` queue contract — kept so the T1b arms can be evaluated on the same
   features). The cache carries **21 busy platforms** (the live count at that time), a
   dim-7 queue column reaching 25.5 (depth 2,545 over the live divisor's cap of 100 — the
   live column's own semantics, which the cold corpora never reach), and a **`xavierGpu`
   candidate** (`node1:116`) whose node cap at alpha 2.5 is 3.22 against 0.14–0.24 on the
   other nodes — the H5 inflation, now inside a training graph. Sweep RTT 15,584–18,928 s
   with a peer term of 52–97 s: the spread between plans is queue drain, not peer exchange.

## Stages, bars and costs — registered 2026-09-13 before any W0 dataset exists

### W0 — the screen (diagnostic only; nothing from it enters a corpus)

**Inputs.** Snapshots captured on the gate cell `cell_s7901` with the gate trace
`workload-150-150-peer_p2_x200.json`, two behaviour sources: (a) `knative_network_batch`
with `KNATIVE_BATCH_BY_PEER_GROUP=1`, batch 10 / 20 ms; (b) the T1b `gnn` lr2e3 seed 1
checkpoint under the deployable serving config (`masked_topo`, peer-group batching,
`GNN_PREFIX_PLATFORM_CAP=1`). Stride 2,000 arrivals; every 3rd aligned snapshot; **50
datasets per source** (`split_warm_snapshots.py --shards 10 --per 5 --every 3`), target
20,000 plans each, on `CPU-amd`. Because the topology is the gate cell, **no W0 dataset
ever enters a training corpus**; W0 only measures the regime.

**W0.a — support closure (descriptive, pass/fail).** Read from the cache built on the
100 datasets. PASS if (i) `xavierGpu` is a candidate in ≥ 20 % of datasets and ≥ 2 % of
candidate slots (live base rate 7.6 % of candidates), and (ii) the dim-7 queue column over
busy platforms reaches p90 ≥ 20 and max ≥ 150 across the datasets (live, on the H2 decode
traces: p50 52, p90 189, max 702). FAIL on either is a tooling defect to fix before W1, not
a finding.

**W0.b — joint structure in the served regime (GO / NO-GO for W1).** Statistic:
`scripts_cosim/score_route_b_contention.py --objective rtt --alphas 2.5` on the 100
datasets, `r_exact_band.mean_tied` at alpha 2.5 per dataset — the regret of the plan a
pointwise scorer with a perfect decoder would pick, the same statistic the
`peer_affinity_v1` R1/R2 reads used — plus its one-integer and k-integer count repairs.
**NO-GO** if the median `mean_tied` regret is ≤ 1.0 % **and** ≤ 20 % of datasets exceed
2.0 %: the warm one-step label is pointwise-recoverable, a corpus of such labels cannot give
message passing anything its twin lacks, W1 is not run, and the lineage closes with that
measurement (the served regime's batch target is queue-drain-dominated and separable).
**GO** otherwise. Both sources are read separately; the GO/NO-GO is on the pooled 100.

**W0.c — the cold checkpoints on warm states (descriptive).** All 16 T1b `gnn` and 16
`mpoff` lr2e3 checkpoints evaluated on the warm cache with the serving decoder (regret vs
the sweep optimum at alpha 2.5), paired by seed. Reports each arm's regret level and the
contrast; this is the offline analogue of the live gate on the states the gate visits. No
bar — it says how far from optimal the reversal-era checkpoints are on the states they were
served on.

**Cost.** 2 × ~15 min of local capture (done for the reactive source, running for the
`gnn` source); 100 sweeps of ≤ 20k plans on 20 CPU-amd nodes, ~1 h; one cache build.

### W1 — the warm corpus, training and gate (registered on a W0.b GO; **runs regardless per Amendment 1**)

**Cells and trace.** 24 training cells `cell_s9001`–`cell_s9024` and 4 held-out cells
`cell_s9101`–`cell_s9104`, minted by `make_peer_affinity_gate_cell.py`'s protocol (corpus
shape, parity check), seeds outside every corpus seed (7001–7034, 7100–8233) and the gate
cell 7901. Warm-up trace: `workload-150-100.json` (301k events, same per-client rate as
the gate trace, a different draw) with the corpus's peer structure put on it by
`make_peer_affinity_production_workload.py --seed 7301 --partners 2 --x-scale-bytes 200e6`
— the gate trace itself is never used to build training states.

**Sources.** On every cell both behaviour policies of W0 run the warm-up trace with
snapshot capture (stride 2,000, ≤ 150 aligned per run); `make_warm_corpus.py` cuts **12
datasets per (cell, source)** spread over the run (`--every`), 20k-plan target: **≈ 576
training datasets** (24 × 2 × 12) and **≈ 96 held-out** (4 × 2 × 12). Two sources because
the served state distribution is produced by the served policy's own placements (closed
loop) and the reactive baseline's is the policy-neutral reference; the source is recorded
per dataset and the read reports per-source regret as well.

**Cache and training.** SSC rewrite → alpha pre-scan at 2.0 (set-aside count recorded) →
`prepare_graphs_cache --dag-partial-state --platform-feature-dim 14` under
`DAG_PRIMARY_ALPHA_KEY=2.5`, `PARTIAL_STATE_CONTRACT=partial_state_v2`, queue contract
`legacy_v0` (the contract the T1b arms were trained and served under — the contrast
against them must not change the feature semantics; `scale_invariant_v1` is a separate
registration) → near-RTT sidecar → split artifact with the held-out cells as the test
block. Arms: `gnn` and `mpoff`, the T1b recipe verbatim (`experiments/peer_affinity_v1_t1b_*_lr2e3.yaml`
with the warm cache), **16 seeds each**, lr 2e-3 (the lr every prior rung selected), W&B
project `gnn-peer-affinity-v1`, run names `peer-affinity-warm-v1-{gnn,mpoff}-lr2e3`.

**Offline read** (`peer_affinity_t1_read.py` conventions, selected checkpoint AND last
epoch): `gnn` vs `mpoff` on the warm held-out block, 16 pairs, exact Wilcoxon, alpha 0.05:
GNN-NEEDED if median ≥ +1 pp, p < 0.05, `gnn` ahead on ≥ 11/16; POINTWISE-BETTER if the
mirror; TIE/INDETERMINATE otherwise.

**Live gate** — `cell_s7901`, `workload-150-150-peer_p2_x200.json` (450,729 tasks), the
registered stage-3 gate verbatim (`peer_affinity_v1_stage3_live_gate.sbatch`,
`HEROSIM_REPLICA_PLATFORM_TYPES` unset — the full cluster), statistic `total_rtt`,
16 seeds per learned arm, paired by training seed, exact Wilcoxon, alpha 0.05. **Primary
configuration: `GNN_PREFIX_PLATFORM_CAP=1`** (the deployable serving config in which the
+21.9 % over Knative was measured); the uncapped configuration is run and reported as
secondary. Arms: warm `gnn`, warm `mpoff`, the landed T1b `gnn` and `mpoff` (re-used
summaries where the code is unchanged, re-run otherwise), `knative_network`,
`knative_network_batch`. Three registered contrasts, all on the primary configuration:

| contrast | fires as | bar |
|---|---|---|
| **L1 — the lever:** warm `gnn` vs T1b `gnn` (same seed) | WARM-HELPS | median `total_rtt` improvement ≥ 3 % and p < 0.05; WARM-HURTS on the mirror; else NO-EFFECT |
| **L2 — the twin:** warm `gnn` vs warm `mpoff` | GNN-NEEDED-LIVE | median ≥ +1 %, p < 0.05, `gnn` ahead on ≥ 11/16; POINTWISE-BETTER on the mirror; else TIE |
| **L3 — the baseline:** warm `gnn` vs `knative_network` | BEATS-KNATIVE | warm `gnn` lower `total_rtt` on ≥ 12/16 seeds and median ≥ +3 % |

**The lineage's headline is fixed in advance:** "a winning GNN" is recorded **only** if L2
reads GNN-NEEDED-LIVE **and** L3 reads BEATS-KNATIVE in the same (primary) configuration
— the first measurement in the program where a graph arm beats both its twin and
reactive Knative. L1 alone answers the H5 sentence (is the state mismatch the lever); L2
alone repeats the T1b offline story live; neither is the headline. Also reported, not bars:
makespan, `effective_parallel_channels`, peer-exchange time, `frac_candidates_oos` (must
be ~0 for the warm arms — the intervention's own control: the training support now
contains the served types).

**Cost.** 56 warm-up runs (~30 min CPU each), ~670 sweeps of ≤ 20k plans (~7 h on 20
nodes), one cache build, 32 GPU training runs (~1 h each), 2 × 34 gate arms (~3 h each,
throttled). About two cluster-days end to end.

## What this can and cannot establish

- A W0.b **NO-GO** is a real answer: in the regime the gate measures, the one-step label
  is pointwise-recoverable, so *no* supervised corpus of that regime — warm or cold — can
  give message passing an edge; the reversal is then not a train/serve-state problem the
  supervised path can fix, and the remaining levers are the objective (closed loop) or the
  regime (an arrival rate the cluster can actually serve — a different gate, registered
  separately, not a re-scoring of this one).
- W1 WARM-HELPS without GNN-NEEDED-LIVE says the state mismatch cost latency but not
  through message passing — the corpus lever again, as in `link_mp_v1` and
  `reliability_matched_v1`.
- Nothing here re-scores or re-interprets any landed `peer_affinity_v1` number; the T1b
  arms are the fixed comparators.

## Record

### 2026-09-13 — measured before registration (mechanics, not bars)

- Local `knative_network_batch` on the gate cell and trace, window batching: replicas
  reach 14 + 14 by t = 7.01 s (28 busy platforms), the trace's 450,729 arrivals end at
  t = 169.5 s with **449,387 tasks still queued** (depth 11,448–19,480 per platform), and
  the run drains until t ≈ 1.5e5 s; `total_rtt` 2.013e10 (the landed Knative figure).
  Peer physics is a pure per-pair transfer charge plus a rendezvous wait for unplaced
  peers (`Platform._peer_exchange_time`, `_peer_rendezvous_events`); no coupling between
  queues beyond that, so a backlog's drain is a sum over its queued tasks and is
  capturable.
- Window-batched Knative batches coincide with a peer group in 14/387 (3k smoke);
  peer-group batching added to `knative_network_batch` behind `KNATIVE_BATCH_BY_PEER_GROUP`.
- Live H2 decode traces (x200 p2, `gnn_s1`): candidates per task mean 7.21, median 6,
  p90 10, max 14; log10(product) ≈ 8.2 per batch; dim-7 over busy platforms p50 52 /
  p90 189 / max 702.
- Prototype dataset as described above; the cache carries the live regime.

### 2026-09-13 — W0 attempt 1 VOID (truncated sweeps), engine fix, attempt 2 launched

Snapshots: 226 per source from local peer-group-batched runs on cell_s7901 (`knative_network_batch`
with `KNATIVE_BATCH_BY_PEER_GROUP=1`; `gnn` T1b seed 1, capped masked decode with
`GNN_BATCH_BY_PEER_GROUP=1`); 224 aligned each, 50 chosen per source (every 3rd, t = 5.9–109.1 s),
10 shards × 5. Datalab arrays 761329 (knb) and 761343 (gnn), CPU-amd, 32 CPUs / 64 GB, 20k-plan budget.

**What happened.** Every one of the 20 tasks reached the 64 GB cgroup limit (MaxRSS 67.1 GB) and
was OOM-killed at least once; only **14 of 95** datasets have `sweep_complete: true`. The other 81
carry 3–30 % of their plans (`worker_exception` 7k–18k rows: the worker pool broke and every pending
future raised) — yet each has a `best.json`, a manifest line and the array printed "done: 5 dataset(s)".
`generate_single_dataset` already refines that outcome to `truncated`; the driver recorded the status
and continued. **Nothing from attempt 1 is a number**; the two corpora are set aside on datalab as
`gnn_datasets_peer_affinity_warm_v1_w0_{knb,gnn}.truncated_<jobid>` (not deleted).

**Cause.** The label horizon on a warm snapshot is the queue drain (best RTT 590 s at t = 5.9 s,
5.9e5 s at t = 109 s — the regime the registration predicted). The autoscaler ticks every simulated
second (`reconcile_interval=1`), and each tick runs the determined autoscaler's per-type scaling pass
and appends one `systemEvents` row per task type: at a 1.4e5 s horizon one plan's result carried
15.5 MB of `systemEvents` and a plan cost ~20 s (0.8 sim/s on 31 workers vs 27.7 sim/s at t = 5.9 s).
Sweep rate and result size both track snapshot time monotonically across all 95 datasets.

**Fix (this commit).** `COSIM_AUTOSCALER_RECONCILE_INTERVAL` stretches the tick in
`execute_simulation` (unset → 1, unchanged; non-numeric or ≤ 0 raises). **Label-invariant, measured
locally**: early snapshot (t = 5.9 s, 384 plans) and late snapshot (t = 109 s, 288 plans), 1 s tick vs
1e12 — **0 of 672 rows differ** in `rtt`, `task_times` or `makespan`, best RTT bit-identical
(15,583.50 s and 409,425.36 s); sweep time 32 → 12 s and 327 → 20 s, result file 1.66 → 0.39 MB and
25.9 → 0.39 MB. `make_warm_corpus.py` now raises on any non-`success` status (rule 4), and the W0 read
sbatch counts a dataset complete only when `placement_metadata.json` says `sweep_complete` and
rows == plans. Attempt 2 uses the same shards, seeds and budget with the knob set to 1e12.

### 2026-09-13 — W0 attempt 2 VOID at the cache (46/100 cap-infeasible), draw fixed, attempt 3 launched

Arrays 761515/761516: all 20 tasks COMPLETED, 100/100 sweeps complete (`sweep_complete: true`,
rows == plans), 411 sim/s on the early snapshot vs 28 before the tick fix, no OOM. The read job
(761763) died in `prepare_graphs_cache`: **46 of 100 datasets have no sweep row feasible at
alpha = 2.0** (23 per source, spread evenly over snapshot time 11–109 s and draw size R 3–7), and a
dataset without a feasible row cannot carry a label set (training-contract 5.5). The prototype had
none. Cause: the candidate draw chose the most *balanced* subset of live replicas and never asked
whether a plan over it fits the per-node caps (α × max single candidate demand on the node) — with
10 tasks it packs a type's candidates onto too few nodes. **Not a regime finding:** with the full
live slate every one of the 224 aligned snapshots per source is cap-feasible at α = 2.0, and a
feasible draw inside the 20k budget exists for 224/224 of each with the same R distribution
(knb R 3/4/5/6/7 = 4/60/98/52/10; gnn 5/61/91/48/15/4 incl. R 8).

**Fix (this commit).** `make_warm_corpus.batch_demands` + `cap_feasible` apply the scorer's exact
rule (demand = demand_scale × memoryRequirements[type][platform]; caps `alpha_max`; a zero-demand
node uncapped) to a draw by exhaustive search with pruning; `choose_candidates` skips draws that
fail it and rejects the snapshot if none exists. Verified against `score_route_b_contention.Dataset`
on three real sweeps (caps equal, demand multisets equal, verdicts agree). Attempt 3 uses the same
shards, seeds and budget; a draw that was already feasible is unchanged (same key order), so the 54
feasible datasets regenerate bit-identically and only the 46 get a different slate. Attempt-2
corpora set aside as `..._w0_{knb,gnn}.capinfeasible_<jobid>`.

### 2026-09-13 — W0 read (attempt 3, arrays 761884/761885, read job 762208): **NO-GO, lineage CLOSED**

Corpus: 100/100 sweeps complete (`sweep_complete: true`, rows == plans), 0 rejected snapshots,
prescan **0/100** infeasible at alpha 2.0, cache built on all 100. Read against the bars registered
above, unchanged; result JSON attached as `peer_affinity_warm_v1/w0_read.json`.

**W0.a — support closure: PASS.** Pooled: `xavierGpu` a candidate in **67 %** of datasets (bar ≥ 20 %)
and **9.1 %** of candidate slots (bar ≥ 2 %; live base rate 7.6 %); dim-7 queue column over busy
platforms p50 48.2 / **p90 111.3** (bar ≥ 20) / **max 590** (bar ≥ 150); 27.5 busy platforms per
dataset. Per source: reactive 78 % / 11.3 %, p90 105, max 590; gnn 56 % / 6.9 %, p90 121, max 256.
The warm cache carries the served regime the 516 cold datasets never contained.

**W0.b — joint structure in the served regime: NO-GO (both bars, both sources).**
`score_route_b_contention --objective rtt --alphas 2.5 --allow-replica-reuse`, `r_exact_band.mean_tied`:
pooled median **0.0 %** (NO-GO bar ≤ 1.0 %), **0 of 100** datasets above 2 % (bar ≤ 20 %), 0 above
1 %; 88/100 exactly 0.0, the 12 non-zero values 0.0006–0.046 %. Reactive source: median 0.0, max
0.010 %, 4/50 non-zero; gnn source: median 0.0, max 0.046 %, 8/50 non-zero. A pointwise scorer with
a perfect decoder recovers the sweep optimum on the states the gate visits: the warm one-step label
is the platform's measured drain plus the batch's per-pair transfers, and both are sums over
(task, placement) terms. **W1 is not run.** The lever named by the H5 audit's closing sentence —
train on the cluster the model is served on — is closed *as a supervised route*: closing the state
mismatch at the source produces labels a pointwise model class fits exactly.

**W0.c — the cold checkpoints on warm states (descriptive, no bar).** T1b lr2e3, 16 seeds per arm,
serving decoder (reuse + relax, cap off), regret vs the alpha-2.5 sweep optimum, 0 infeasible:

| states | gnn median regret | mpoff median regret | mpoff − gnn (paired median) | gnn better | Wilcoxon p |
|---|---|---|---|---|---|
| pooled (100) | 42.2 % | 44.2 % | +3.54 pp | 12/16 | 0.021 |
| reactive-source (50) | 30.6 % | 37.2 % | +6.74 pp | 11/16 | 0.058 |
| gnn-source (50) | 48.3 % | 48.8 % | +1.93 pp | 11/16 | 0.21 |

Both reversal-era arms sit ~40 % above the optimum on the states they were served on (means 70–98 %:
heavy tails), and the graph arm is slightly less far off. This is the offline analogue of the live
gate and is not a claim: at that regret level the sweep optimum is a pointwise object (W0.b), so
whatever separates the arms here is fit, not graph reasoning.

**Cost as run.** 2 local captures; 3 generation attempts (attempts 1–2 VOID, records above), the
final one ~10 min per array on 20 CPU-amd nodes; one read job (~20 min incl. 32 checkpoint evals).

**What this closes and what it leaves.** Closed: a warm-state *supervised* corpus as the route to a
served graph arm that beats its twin and Knative (every option-1/route-B stop now has its warm-state
counterpart). Left open, unregistered: the served regime is ~300× overloaded, so the one-step label
is drain-dominated by construction — a *different* served regime (a trace the cluster can drain) would
be a new environment question, not a re-run of this screen; and the W0.c regrets say the serving
gap is mostly fit on out-of-support states, which is a serving/feature question the 2026-09-13 audit
already owns.

### 2026-09-13 — Amendment 1: W1 runs despite the W0.b NO-GO (signed before any W1 number)

User decision, same day as the W0 read: "I always want live gate not just offline as the end of the
discussion or lineage." Recorded as CLAUDE.md rule 6. Nothing in the W1 section changes — cells
s9001–s9024 / s9101–s9104, `workload-150-100` + peer seed 7301, two sources × 12 snapshots per cell,
T1b recipe × 16 seeds per arm, live gate on `cell_s7901` capped (primary) and uncapped (secondary),
contrasts L1/L2/L3 and the headline rule as written. The W0.b result is the prior: the corpus's
one-step label is pointwise-recoverable, so the pre-registered expectation is L2 = TIE; L1 and L3 are
the open readings.

### 2026-09-13 — W1 launched (Amendment 1), before any W1 number exists

Trace `workload-150-100-peer_p2_x200.json` (301,352 events, 536,124 peer pairs; peer seed 7301,
2 partners, 200 MB) built locally and rsynced (md5 equal). 28 cells minted by
`make_peer_affinity_gate_cell.py` (s9001–s9024 train, s9101–s9104 held-out), parity PASS on all 28.
Datalab chain, SLURM `afterok` dependencies, submitted 2026-09-13 ~21:30 UTC at f50e704:

| stage | job | what |
|---|---|---|
| capture | 762821 (56 tasks, %20) | `peer_affinity_warm_v1_w1_capture.sbatch`: each (cell, source) on the warm-up trace, stride 2,000, ≤ 150 snapshots; sources as W0 (`knative_network_batch` + peer-group batching; T1b `gnn` seed 1 capped) |
| generate | 762846 (56, %20) | `..._w1_generate.sbatch`: `every = aligned // 12`, 12 datasets per (cell, source), 20k plans, cap-feasible draw, tick knob 1e12; ids `12 × task` inside `_w1_train` / `_w1_heldout` |
| cache | 762847 | `..._w1_cache.sbatch`: SSC → prescan (set-aside) → cache `graphs_cache_peer_affinity_v1_warm` → near-RTT sidecar → `experiments/peer_affinity_v1_warm_split.json` (held-out cells = test block) |
| train | 762848 (32) | `..._w1_train.sbatch`: `experiments/peer_affinity_v1_warm_{gnn,mpoff}_lr2e3.yaml` (T1b verbatim), CPU-amd 16 threads, checkpoints `models/peer-affinity-v1-warm-{arm}-lr2e3-seed{1..16}.pt` |
| gate, primary | 762849 (32, %16) | stage-3 sbatch, `GNN_PREFIX_PLATFORM_CAP=1`, `CKPREFIX=models/peer-affinity-v1-warm`, `results/warm_capped_gate/` |
| gate, secondary | 762850 (32, %16) | same, cap unset, `results/warm_uncapped_gate/` |
| Knative check | 762851 (2) | `knative_network` + `knative_network_batch` re-run into `results/warm_knative_check/`; must reproduce the landed `total_rtt` 2.01309e10 (code has changed since) or the baselines are re-run in both configurations |

Naming note (not a bar): checkpoints and cache use the `peer_affinity_t1_read.py` tag convention
(`peer-affinity-v1-warm-…`, `graphs_cache_peer_affinity_v1_warm`) rather than the
`peer-affinity-warm-v1-…` spelling in the W1 text, so the T1 read tool runs unmodified with `--tag warm`.

### 2026-09-14 — W1 capture: 5 of 28 cells STALL (starved client), dropped; chain re-linked

Capture 762821 finished **46/56** tasks in 44–52 min each (150 snapshots per run). The other 10 —
cells **s9004, s9008, s9011, s9012** (train) and **s9102** (held-out), *both* sources — ran 1.7–2.6 h
at 100 % CPU with 1–11 snapshots written and the gateway's progress line stuck at event 1,353
(t ≈ 6 s). `py-spy` on `cell_s9008_knb`: 53 % of samples inside
`knative_network/autoscaler.py:create_first_replica` (46 % of that constructing DEBUG `LogRecord`s the
ERROR-level handler then drops), locals `task_type=dnn2, source_node_name=client_node19` on every
sample. Mechanism (`knative_network_batch/scheduler.py:_process_task_batch`): a task whose client can
reach no server holding a replica of its type is postponed, `create_first_replica` is asked for one,
under `HEROSIM_SERVER_ONLY_REPLICAS=1` every reachable server is already memory-full, the call returns
`StopIteration`, the task is re-queued and the next 0.02 s batch retries — forever. Nothing is logged
(the warning is below the handler level) and simulated time barely advances. These five topology
draws each contain one such client; the 23 others do not. The gate cell `cell_s7901` is one of the
23 kinds (its gates completed before).

Action (execution, not registration): the 10 stalled tasks and the pending generate array 762846
were cancelled; the partial snapshots are kept as `*.jsonl.stalled_client_starved`; generation
resubmitted as **763302** (`--array=0-5,8-13,16-19,24-49,52-55%20`, dataset ids keep their
`12 × task` slots, so the train corpus has gaps at ids 36–47, 84–95, 120–143 and the held-out block
at 24–35); cache 762847 and the Knative check 762851 re-chained `afterok:763302` (the check had been
chained on the capture array and went `DependencyNeverSatisfied`). Everything downstream is
unchanged. **The W1 corpus is therefore 20 train cells × 2 sources × 12 = 480 datasets (registered:
576) and 3 held-out cells × 24 = 72 (registered: 96).** 480 is the T1b corpus size (482) at which
the offline MP edge was measured, so the L1 comparison is not under-powered by the cut; the bars
are unchanged. The stall itself is a simulator liveness defect, recorded in
`docs/gates/gate-tools.md` (2026-09-14); it is not fixed here because a fix changes the served
physics and W1's sources must match the gate's.
