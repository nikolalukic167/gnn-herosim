# shuffle_placement_s0_v1 — measured hash-aggregation shuffle placement screen

> Produced on local-record-2026-10-06 at afea04bd; code not merged; old physics.

**Status:** `CLOSED` (2026-10-01) — **NO-GO for GNN training on this shuffle/replay
recipe**. Static protocol preceded scoring; the follow-up live protocol preceded
its four fresh network configurations. No GNN training or HeROsim physics change.

**Current outcome:** static fluid-model headroom is 0.959%, with route search
reaching 30/32 optima. A real-TCP namespace gate completes 48 streams and 576 jobs;
adaptive search changes mean job latency by −1.38% / −0.52% at 0.25 / 0.10 s
arrival intervals, with only four independent configurations and no significance
claim. Calibration fails the 20% median-error limit on two configurations and
shows large per-pattern errors. The static headroom is **not a validated bound on
real-network performance**. These measurements do not qualify training and do not
rule out useful GNNs on larger or different application graphs.

## Question and scope

Does application-derived shuffle traffic leave meaningful placement headroom after
route-aware controls? This successor changes the workload and release semantics,
not the bandwidth of the closed peer-prefetch experiment. Its parents are
[core prefetch](g32_core_prefetch_s0_v1.md) and
[coordinated pairs](g32_coordinated_valley_s0_v1.md), whose strong controls recovered
most bounded search headroom.

The application is four producer processes sending hash-partitioned key/value records
over loopback TCP to four reducer processes. Each reducer performs a grouped sum after
all its input arrives. Inputs are generated, with producer-specific Zipf key skew;
this is **not a production trace**. Measured quantities are partition bytes, producer
data-ready times, reducer compute times and end-to-end local application duration.
Sender/receiver bytes, row counts, key sums and value sums must agree.

The standalone replay assumes directed access/core link capacities and progressive
max-min bandwidth sharing. It is **not validated multi-host TCP behavior**, and is
not the HeROsim execution engine. Local transfers consume an assumed memory resource;
each host serves one reducer at a time. There is no artificial co-location cap.
All sixteen network configurations share one two-rack graph shape; capacity, compute
speed and producer placement vary. Holding out eight configurations is not evidence
of generalization across graph families.

## Frozen protocol

`experiments/shuffle_placement_s0_v1.json` fixes four workload seeds, three real
measurement repetitions each, eight development and eight holdout configurations,
the 16-evaluation search budget and the regressors before scoring outcomes.
For each workload seed select the actual repetition with median observed local
duration, independently of placement performance. Preserve all twelve measurements.

Enumerate every one of the 256 static reducer assignments for each workload/configuration.
This is exhaustive only within that decision space. It is not an optimal online
scheduler or a global upper bound for an arrival stream.

Controls: route-load proxy; its top 16 plans exact-scored; Ridge and ExtraTrees
proposal selectors trained only on development configurations. Each learned selector
includes the proxy baseline and 15 proposed plans. All see raw traffic and placement
facts plus route-load/compute summaries. Screen regressors log to offline W&B and
persist joblib weights plus input/feature contracts; no GNN is trained.

The exploratory screen asks whether median configuration headroom reaches 8% and
whether strong 16-evaluation controls capture less than half. These are practical
screening thresholds, not scientific constants or sufficient conditions for GNN
training. A zero headroom denominator is reported as unavailable, not a pass.

The original MLP checkpoints and externality-aware CD cannot be directly transplanted:
this task places shuffle reducers and minimizes job completion, with new features
and network semantics. The screen therefore cannot claim to beat those policies.

## Execution and validation boundary

Entry point: `scripts_cosim/shuffle_placement_s0.py` with `measure` and `screen`
phases. `screen` requires `--save-checkpoints`. Results live in
`docs/lineages/shuffle_placement_s0_v1/run_2026-10-01/`.

The initial 32-job stream replay uses plans chosen in isolation and is not a live
policy gate. The subsequent `scripts_cosim/shuffle_network_gate.py` implements an
actual adaptive callback and real TCP execution in Linux network namespaces;
`scripts_cosim/audit_shuffle_network_gate.py` checks the observed snapshots,
decisions and completions. Artifacts are in `network_gate_2026-10-01/` beside this
node. Validation was attempted and failed for the fluid model's full timing
contract; it is not left pending or silently treated as a pass. Physical multi-host
validation, large graph families and model retraining are outside this closed recipe.

## Record

### 2026-10-01 — review correction and stop decision

The review correctly identifies this as an unsuitable GNN-training target:
only 4^4 = 256 assignments, one graph shape, and strong route-aware controls in
the larger parent studies. The 30/32 optimum recovery and failed physical
calibration leave no justification for extending this recipe. Any future proposal
first needs a paper analysis
of coupling at good placements, a timed strong-control comparison and a credible
decision budget; a task-count threshold alone is not sufficient.

All-to-all communication is not itself a proof of separability. For placement x,
link load is B_l(x) = sum_(p,r) b_pr * 1[l in route(p,x_r)], and shared capacity
introduces a bottleneck bound max_l B_l(x)/C_l. Changing one reducer's host can
therefore change other flows' completion times. This replay also serializes
reducers sharing a host. The measured lack of useful residual headroom is the
reason to stop, not a claimed general separability theorem for shuffles.

The obsolete sentence saying the adaptive gate was still pending has been
replaced below. Run `bvv5yux9` was deliberately configured with `mode="offline"`:
its local W&B transaction log records the regressors and metrics, while no
service upload was attempted. It remains unsynced to keep the experiment record
local; this is logged-but-unsynced, not an unlogged training run. No upload
rejection is claimed for this particular run. The repository also uses offline
W&B for the [workflow pilot](workflow_amortized_v1.md).

The scoped Git record includes protocols, scripts/tests, the index and stop
entries, measured traces, archived stream JSONs, calibration JSONs, audits, provenance,
source snapshots, compressed exact scores, regressor checkpoints/contracts and
the offline W&B transaction log. Generated payload/calibration `.bin` files,
the redundant uncompressed exact-score JSON and debug logs remain local and are
ignored. Decompress `run_2026-10-01/exact_scores.json.gz` to `exact_scores.json`
before running the static auditor on a fresh checkout. The network auditor
requires extraction of `network_gate_2026-10-01/streams.tar.gz` into that same
directory and the regenerated payload files; the archived payload manifest preserves
their checksums, and the runner's `materialize` function reconstructs them from
the frozen protocol. No new measurement is needed to regenerate those bytes.

### 2026-10-01 — real TCP adaptive gate and closure

All **48/48** runs finish: four fresh network configurations × two arrival
intervals × three policies × two repetitions. All **576 jobs**, **2,304 reducers**
and **9,216 payload transfers** audit, carrying **18,432,000,000 application bytes**.
Every payload passes its receiver SHA check; byte counts, key/value sums, row
counts, release ordering, reducer barriers and completions pass the independent
audit. All 576 decisions reproduce from recorded observations. Snapshot keys only
refer to prior jobs; no future job state enters the selector. Adaptive route
placement differs from static placement on **112/192 decisions**, so the callback
does consume state rather than replay fixed plans. Per-host compute is serialized
in the worker; all workers share physical hardware.

Per configuration, pair mean job durations within repetition and take the median
paired percent over the two repetitions; then report the median of four
configurations. Negative favors the adaptive policy. Repeats/jobs are not
independent units. With four configurations these are exploratory directions,
not powered magnitudes or a statistical win.

| Arrival interval | Adaptive route vs static | Adaptive search vs static | Search faster configurations |
|---|---:|---:|---:|
| 0.25 s (4 jobs/s) | +0.039% | −1.378% | 3/4 |
| 0.10 s (10 jobs/s) | +4.098% | −0.520% | 3/4 |

Median run-level median decision time, including state collection and scoring:
static **7.40 ms**, adaptive route **7.34 ms**, adaptive search **35.49 ms**.
Dispatch overhead and controller lag also contribute to measured response time.
The search forecast uses unfinished measured compute demand and outstanding
receiver bytes; it is not a perfect remaining-service measurement. Most mapper
generation is represented by recorded release offsets, with payloads materialized
before execution, not by concurrent real mapper CPU execution.

The calibration has **72 measurements** (six patterns × three repeats × four
configurations). Median absolute relative prediction errors by configuration are
**16.05%, 51.11%, 21.30%, 16.54%**: two exceed the frozen 20% tolerance. Local TCP
versus the assumed memory resource has median errors **76.1–86.6%**. Remote
patterns also fail on some configurations, including single-core **21.7–61.9%**.
An aggregate pass cannot validate every route or the local-transfer contract.
Do not quote the initial 0.959% as physical headroom, or these fluid scores as an
exact real-TCP oracle. No physics parameter was retuned after these outcomes.

The live and calibration results close this particular feasibility recipe without
qualifying GNN training. A successor needs a separately validated execution model
and an application/decision space with residual value beyond strong controls;
increasing payloads in this opened experiment is not a sealed confirmation.

Raw per-stream snapshots, plans, flow timing, completions, topology, calibrations,
protocol and source/provenance are retained in `network_gate_2026-10-01/`.
`audit.json` holds the independent recomputation. No physical host-network
interfaces or routes were modified; namespace workers and routers were terminated
by the runner. No models were trained during this live gate.

Run from the repository root in an environment supporting unprivileged user/network
namespaces, with a fresh output directory:

```bash
unshare --user --map-root-user --net env PIPENV_IGNORE_VIRTUALENVS=1 VIRTUAL_ENV= PYTHONPATH=/root/projects/my-herosim OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 pipenv run python3 scripts_cosim/shuffle_network_gate.py run --out /tmp/shuffle_network_reproduction
```

The auditor requires an executed-source snapshot `source_network_gate.py` whose
hash matches the run's `provenance.json`; preserve it before editing the runner.
Live wall-clock timings vary across repetitions and machines.

### 2026-10-01 — adaptive emulated-network gate registration

The user requested the next step. `experiments/shuffle_network_gate_v1.json`
freezes four fresh configurations (31201–31204), two stream intervals (0.25 and
0.10 s), twelve jobs per stream, two repetitions and three arms before evaluation:
static route scoring, route scoring with current traffic/unfinished compute, and
adaptive top-16 fluid-forecast search. Arm order reverses in the second repetition.
The live search objective sums remaining completion times of known unfinished jobs
and the arriving job, accounting for their externality within the forecast. It
does not see future arrivals. A CPU-time budget is not equated to an evaluation
count: actual decision and dispatch latency are included in job response time.

`scripts_cosim/shuffle_network_gate.py` creates four host and two router Linux
network namespaces inside an isolated user/network namespace. Real TCP travels
over directed TBF-shaped veth links, with netem propagation delay. Parent network
capacities and producer placements are reused, with all compute speed factors
set to one because the workers share actual host hardware. Local communication
is real local TCP, not the earlier assumed memory service. Payloads reconstruct
the original generated records byte-for-byte. Mapper readiness is replayed from
the measured trace; aggregation executes for real, with one lock per host.
Transfers have SHA-256 checks and aggregation totals must match per reducer.

At every arrival the controller collects receiver byte-progress and completed
reducers over Unix control sockets. Outstanding flow bytes, known data-ready
times and unfinished reducer demands feed placement. Snapshots, plans, queueing
of the controller, whole-decision time and dispatch time are recorded. The
snapshot is sampled, not an atomic network snapshot; unfinished computation is
estimated by its measured full duration, not a perfect remaining-service oracle.

Six calibration patterns (single/shared core, single/disjoint access, shared
receiver, and local TCP) each run three times per configuration with 8 MiB flows.
The predeclared median absolute relative error limit is 20%. Per-pattern errors
must also be disclosed: an aggregate pass cannot certify all network behavior.
Calibration failure does not cancel the registered live runs, but blocks treating
fluid scores as a validated oracle. No model fitting or GNN training is planned.

Development seed 31099 completed a three-job smoke for all arms, outside the
evaluation configurations. The initial smoke had three calibration patterns;
local, shared-receiver and single-access patterns were added for coverage before
fresh evaluation. The smoke was used for execution checks, not choosing seeds or
rates. Tests cover conservation, actual state-dependent decisions and exclusion
of finished jobs from forecasts. Namespace capability and shaping probes succeeded.
This is a single-machine network emulator, not a physical multi-machine testbed.

### 2026-10-01 — measured application, exact screen and audit

All twelve TCP measurements completed: 32 MB per job (four million 8-byte records),
with individual partitions 1.068–3.749 MB and local application duration
0.200–0.220 s. The observed partition volumes replace synthetic peer edges;
the key distribution remains generated. Instrumentation validates conservation,
not multi-host network fidelity.

The exact screen covers 64 instances, 256 assignments each: **16,384 simulations**.
Thirty-two instances belong to eight held-out capacity/source-placement configurations.
Median per-configuration oracle gain over the simple route proxy is **0.9586%**,
below the frozen 8% screen threshold. The route proxy's top-16 exact search recovers
the optimum in **30/32** held-out instances. The two misses are **1.366 ms** and
**0.425 ms**; the worst relative regret is **0.5767%**. Median gain capture is 100%
at each configuration, which must not be mistaken for 100% on every instance.
Ridge and ExtraTrees also have median-over-configuration capture of 100%; this
does not mean either reaches every optimum.

The event replay completed 40 fixed-plan streams (five arms × eight configurations),
32 jobs each: **1,280 completed jobs**. These runs exercise shared links and compute
queues but do not observe ongoing state when choosing plans. They cannot establish
an adaptive-policy result. Median configuration mean job duration is 0.24220 s for
the route proxy and 0.23670 s for its 16-evaluation search, excluding planning overhead.

An initial audit assertion incorrectly expected every instance to reach the optimum
from the configuration medians. Reading individual rows found the two misses above;
the audit now reports their actual number and magnitude rather than treating the
aggregate as a universal statement. No protocol, model or outcome was tuned after
that read. The independent auditor verifies full plan coverage, disjoint split,
baseline inclusion, proxy-search scores, a bit-identical event replay witness,
the executed source hash and a lossless compressed copy of all exact scores.

Artifacts: `run_2026-10-01/measurements.json`, `exact_scores.json.gz`, `results.json`,
`audit.json`, frozen `protocol.json`, phase provenance, executed `source_screen.py`,
and persisted control models/contracts in `checkpoints/`. W&B logged the CPU screen
controls locally in offline run `bvv5yux9`; it has not been synced to the service.

Reproduce from the repository root, using a fresh output directory for measurement:

```bash
PIPENV_IGNORE_VIRTUALENVS=1 VIRTUAL_ENV= PYTHONPATH=/root/projects/my-herosim OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 pipenv run python3 scripts_cosim/shuffle_placement_s0.py measure --out /tmp/shuffle_reproduction
PIPENV_IGNORE_VIRTUALENVS=1 VIRTUAL_ENV= PYTHONPATH=/root/projects/my-herosim OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 pipenv run python3 scripts_cosim/shuffle_placement_s0.py screen --out /tmp/shuffle_reproduction --save-checkpoints /tmp/shuffle_reproduction/checkpoints
PIPENV_IGNORE_VIRTUALENVS=1 VIRTUAL_ENV= PYTHONPATH=/root/projects/my-herosim pipenv run python3 scripts_cosim/audit_shuffle_placement_s0.py /tmp/shuffle_reproduction
```

Real timing varies across measurements. To reproduce deterministic placement scores,
reuse the archived measurements and frozen protocol instead of recollecting timings.
No GNN was trained. The subsequent adaptive real-TCP gate ran and the timing
validation failed, as recorded above. The lineage is CLOSED with no training
qualification.

### 2026-10-01 — implementation and measurement registration

User requested execution of the measured-shuffle/headroom feasibility study.
The protocol was written before measurements or placement scores. The local
measurement environment has NumPy 2.3.0 and scikit-learn 1.7.0. Six focused tests
cover max-min allocation, capacity conservation, release timing, reducer barriers,
host serialization, deterministic overlapping-job replay and route construction.
Hashes and environment versions are preserved per execution phase.
