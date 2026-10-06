# residency_placement_v1

**Status:** `CLOSED` (2026-10-01) — **NO ESTABLISHED MESSAGE-PASSING WIN**.
Three-attempt protocol written before profile collection/training; a fourth
synthetic probe and an exact-control follow-up were separately registered after
the findings that motivated them. User requested multiple tries.

**Outcome.** Four attempts, 18 trained checkpoints, 96 held-out simulated batch
configurations and **128 real-container runs / 2,048 requests** complete and audit.
The original three configurations have no residual assignment headroom over hand
search on any of their 72 test cases; their physical batches contain one function
type. A separately trained six-function synthetic probe creates actual eviction
pressure, but provides no established GNN benefit over its matched MP-OFF twin.
In its paired exact-control physical gate, both learned arms lose on both cases.
The full 3^8 vectorized search is itself competitive with decoder latency, so
amortization is not established at this size. Only **two independent physical
application blocks**, reused across attempts: directions, not publishable effect
sizes. Eviction remains a fixed shared rule; joint learned eviction is untested.

## Question and scope

Can a bipartite request-host scorer improve placement when measured container
startup, execution and termination costs combine with binding resident-memory
admission? Three predeclared attempts cover equal CPU quotas/tight memory,
unequal quotas/tight memory, and more memory with distant trace windows.
All use six benchmark functions, three logical hosts and eight observed requests
per burst. This is a standalone measured-container co-simulator and Docker runtime,
not the existing HeROsim execution engine or historical peer-affinity checkpoints.

The user authorized small matched pilots across multiple attempts. The original
training uses one mixed corpus across three environments, three seeds each of a bipartite GNN,
its separately trained MP-OFF twin, and an MLP seeing the full fixed-size graph.
All arms receive the same 20-dimensional edge observations and hand summaries;
the full-input MLP is not restricted to seeing its own candidate edge.

Placement induces changes to residency; all arms use the same deterministic
frequency-times-startup-cost-per-byte eviction priority. This is inspired by
FaasCache, but lacks its Greedy-Dual aging and is not an implementation of that
paper. Joint optimization of eviction decisions is outside this pilot. The label
enumerates all 3^8 assignments at fixed request order and fixed eviction policy;
it is not an optimal online scheduling oracle. Only the current batch is labelled.

## Grounding and validity

- [Shahrad et al., ATC 2020](https://www.usenix.org/conference/atc20/presentation/shahrad)
  and the [Azure 2019 trace](https://github.com/Azure/AzurePublicDataset/blob/master/AzureFunctionsDataset2019.md)
  supply per-minute invocation counts. Select the first 648 distinct nonempty
  applications in day 1 (one function each); six-app blocks are disjoint by split.
  Counts determine composition after sampling eight requests. Benchmark mapping,
  downsampling and burst release are synthetic, not observed Azure timestamps.
- [FaasCache, ASPLOS 2021](https://homes.luddy.indiana.edu/prateeks/papers/faascache-asplos21.pdf)
  motivates memory-aware residency and a cost/size/frequency eviction control.
- [IceBreaker, ASPLOS 2022](https://par.nsf.gov/servlets/purl/10355252)
  motivates heterogeneous warming. Here heterogeneity is measured Docker CPU
  quota, not different physical CPU architectures or a replication of IceBreaker.
- [Gasse et al., NeurIPS 2019](https://arxiv.org/abs/1906.01629)
  establishes bipartite imitation as prior art. No architectural priority claim.

Microbenchmarks load generated records into resident indexes, then perform keyed
lookups, weighted aggregation or hashing, each at two dataset sizes. They are
reproducible application kernels, not an established benchmark suite or production
functions. Profile startup/stop from the controller and warm RPCs; admit memory at
1.5 times measured peak RSS, rounded to MiB, and enforce that reservation as each
container's cgroup limit. Host admission bounds the sum of these limits. Logical
hosts share one physical machine; the image is locally cached and containers have
no network access. Startup includes Docker CLI/runtime overhead.

## Frozen protocol and gates

Entry points: [protocol](../../experiments/residency_placement_v1.json),
[measurement/data runner](../../scripts_cosim/residency_study.py),
[physics](../../src/placement/residency.py),
[model](../../src/policy/residency/model.py),
[trainer](../../src/policy/residency/train.py).

Data and all per-plan scores live under `simulation_data/residency_placement_v1`;
models and mandatory sidecars under `models/residency_placement_v1`.
Every training run goes through `run_experiment.py` and offline W&B. Separate
whole application blocks: 64 training, 16 validation, 24 simulation-test and four
reserved physical blocks; only the first two physical blocks used per attempt.
Checkpoint selection uses mean normalized complete-batch cost on validation only.
Physical selection picks each architecture's best validation seed before opening
physical outcomes, preserving all nine simulation-test comparisons.

Controls: immediate cache-aware completion greedy, scarcity-aware greedy,
expanded-slot linear assignment, multi-start four-round coordinate search, and
the exhaustive fixed-policy assignment reference. Report planning time separately
and charge it to every actual request in the physical gate. No search-budget win
is inferred from a slow exhaustive reference.

Physical gate: two fresh six-app blocks per attempt, two bursts per run, two
repetitions reversing arm order; five arms (greedy, search, GNN, MP-OFF, full-input
MLP). Recompute decisions from actual surviving resident containers after the
first burst. No future-window input to policy features. Runs verify outputs,
memory admission, cold-start/eviction counts and predicted versus actual times.
This is a small closed-loop burst pilot, not a production-arrival replay. All
attempts run their physical gate irrespective of offline outcome. The same app
blocks recur across attempts: never count attempts/repeats/requests as independent
replicates. Two physical cases cannot establish a publishable architecture win.

## Record

### 2026-10-01 — four completed attempts and exact-control gate

All runs and audit artifacts are retained. Negative means faster; positive means
slower. Simulation entries below are medians of per-case ratios after averaging
the three independently trained seeds within each architecture. Do not subtract
two medians to infer their direct paired contrast.

| Attempt | Function mix | Hand search exactly matches assignment optimum | GNN vs hand search, simulated | Direct GNN vs MP-OFF median, simulated |
|---|---|---|---|---|
| Equal CPU, tight memory | Trace-derived | 24/24 | 0.00% | 0.00% |
| Unequal CPU, tight memory | Trace-derived | 24/24 | +0.33% | 0.00% |
| Distant trace windows, more memory | Trace-derived | 24/24 | +0.55% | 0.00% |
| Six-function mixed backlog | Explicit synthetic probe | 19/24 | +7.63% | 0.00% |

The mixed probe has individual reference headroom up to 23.42%, but its median
is zero; it does not provide a general residual optimization margin. The matched
MP-OFF is +0.30% versus hand search at its median. The full-input MLP is much
worse, which does not establish graph value when the separately trained MP-OFF
ties or does better. Parent training batches contain one function in 102/192
configurations; validation 42/48 and simulation test 42/72 do too. All original
evaluated physical input batches have one function. The unused reserved block
106 has a two-function second burst in two attempts; it was not substituted
after inspecting results.
The six-function probe is reported separately and never called a production mix.

**Physical gates.** The original five-arm gate completes 60 runs/960 requests;
the synthetic probe completes 20/320. A later paired exact/GNN/MP-OFF control
adds 36/576 and 12/192 respectively, for **128/2,048 total**. Each run has two
bursts; the second decision consumes the actual surviving container pool.
There are two application blocks per attempt, reused across attempts, not eight
independent samples. Two repetitions reverse policy order. Within each block
pair runs by repetition, take the median percentage, and only then aggregate
blocks. All physical percentages and per-block values are in the audit JSONs;
the small design supports directional description only.

- Original GNN-versus-coordinate-search directions are favorable, but the GNN
  does not beat the faster immediate greedy at the median in any original
  attempt. The GNN/MP-OFF ranking varies across attempts; 16/24 compared burst
  plans are identical. This is not a message-passing result.
- The mixed primary gate splits one case each against coordinate search.
  GNN/MP-OFF differences remain small. Real manipulation is present: the
  primary mixed gate records **140 cold starts and 134 evictions**; only 2/8
  compared GNN/MP-OFF burst plans are identical.
- In the separate mixed exact-control gate, **GNN and MP-OFF both lose to exact
  on both cases**. The original single-function exact comparisons sometimes
  favor learned arms, but they do not resolve the joint-placement question;
  the fixed-profile optimum is not a clairvoyant optimum for noisy wall time.

**Timing and calibration.** Median measured test sweep time is 7.88 ms in the
original corpus and 9.07 ms in the mixed one, versus 9.51/10.23 ms for GNN seed
202 and 17.96/21.81 ms for coordinate search. Thus this implementation cannot
claim that neural inference amortizes a necessarily slow search. These timing
reads motivated the explicitly post-primary exact-control amendment. In the
primary physical gates, median absolute prediction errors are 6.55%, 5.07%,
11.85% and 8.66%; p90 errors are 15.35%, 12.97%, 16.91% and 14.89%. These are
observed checks on this machine, not validation of arbitrary clusters or proof
that a small predicted ranking difference is real.

**Audit and reproducibility.** Independent scalar replay agrees with vectorized
costs, cold starts, evictions and final resident sets. Every physical decision
reproduces from its saved current-state snapshot. All function results, request
counts, memory admissions, observed peak RSS bounds, FIFO ordering and cross-burst
resident states pass. Exact test sweeps are reproduced and saved, all data hashes
match, and application identities do not cross train/validation/test/physical
splits. Checkpoint sidecars include architecture, seed, selected epoch, source
hashes and corpus hash. Eighteen offline W&B histories were read against the
uniform soft-cross-entropy reference ln(3), with selected epochs retained.

The original full sweeps contain 1,574,640 plans; the added training/validation
probe contains 524,880. Both collections retain `placements/placements.jsonl`.
All 96 test configurations retain exact sweep arrays as well. The original
corpus has 4,608 conditional-prefix training examples, the probe 1,536. No
models are reused between the two corpora, and all architecture quotes compare
arms trained on the same named corpus.

Artifacts:
- [Original audit](residency_placement_v1/results/audit.json) and
  [mixed-probe audit](residency_placement_v1/mixed_results/audit.json).
- [Training summary](residency_placement_v1/training_summary.json),
  [environment](residency_placement_v1/environment.json),
  [checkpoint/data manifest](residency_placement_v1/artifact_manifest.json).
- `results/` and `mixed_results/` each retain simulation scores, physical events,
  separate exact-control runs, frozen validation-only checkpoint selection,
  curve reads, exact test sweeps and source snapshots/manifests.
- Corpora: `simulation_data/residency_placement_v1` and
  `simulation_data/residency_mixed_v1`; weights: corresponding `models/` paths.

**Scope of closure.** Do not repeat these eight-request, fixed-eviction recipes
as evidence that memory pressure makes message passing necessary. They do not
establish an original GNN scheduling contribution. Larger instances, explicit
joint eviction actions and production-representative mixed arrivals remain
different questions; none is automatically qualified by this result.

### 2026-10-01 — exact serving control added after timing read

The complete 3^8 vectorized assignment sweep costs median 7.88 ms on the parent
test configurations and 9.07 ms on the mixed probe, versus GNN seed 202 at
9.51/10.23 ms. Calling the exhaustive reference intrinsically slow would therefore
be false. `residency_exact_control_v1.json` adds 48 paired physical runs of exact,
the frozen GNN and the frozen MP-OFF on the same two held-out blocks, reversing
order in repetition two. No new independent sample size or checkpoint selection.
The fixed host-tuple table is prepared before requests, like model weights;
state-dependent evaluation and all other planning work are charged. Original
five-arm gates remain separately reported. This amendment was registered after
their results and timings, before the additional exact-control runs.

### 2026-10-01 — fourth attempt registered after workload inspection

The original physical blocks all contain one function per eight-request batch;
the train/validation/test medians are also one. This is a workload manipulation
failure for the joint-placement question, regardless of model performance.
Keep every original attempt and finish all original physical runs.

A separate `residency_mixed_v1.json` protocol adds a synthetic mechanism probe:
one queued request for each of the six measured functions, plus two drawn using
the source minute's weights, shuffled deterministically. CPU quotas and tight
memory follow the heterogeneous parent. It is not an Azure replay or evidence
that such bursts occur at its prevalence in production. It reuses the measured
profiles but has a separate corpus, nine new checkpoints, fresh physical runs,
and held-out application blocks with the same train/validation/test boundaries.
Every architecture is retrained; no favorable parent checkpoint is reused.
Selection and physical sample size remain unchanged. The probe was registered
before its corpus, training or physical runs, after observing the parent
simulation's lack of median headroom and the single-function physical inputs.

### 2026-10-01 — registration and build

Protocol frozen before profile outcomes. New semantics deliberately avoid the
old co-sim warmup artifact in [objective_pivot_v1](objective_pivot_v1.md).
Earlier [workflow_amortized_v1](workflow_amortized_v1.md) and
[dag_memory_value_learning_v1](dag_memory_value_learning_v1.md) remain negative;
this study retains their requirement for strong learned and hand controls.
