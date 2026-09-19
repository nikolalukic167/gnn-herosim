# HeROsim

A discrete-event simulator for serverless task placement in heterogeneous clusters, and the
experimental apparatus around it. The research question is whether a graph-aware scheduler
beats a pointwise one; most of the vocabulary below exists to keep claims about that
question falsifiable.

## Language

### The simulated cluster

**Node**:
A machine in the cluster. Hosts platforms and local storage, and is the unit at which
network ingress and image caching are scoped.
_Avoid_: Host, machine, server

**Platform**:
An execution environment on a node — a set of CPU cores, a GPU — that accepts tasks into a
queue and runs them. The finest-grained place a task can be sent.
_Avoid_: Worker, executor, device

**Replica**:
A `(Node, Platform)` pair admitted to serve a given task type. A replica is an
**eligibility fact**, not an object: it is membership in a set keyed by task-type name, with
no identity, no lifecycle and no state of its own. "Creating a replica" adds a pair to that
set. There is deliberately no `Replica` class.
_Avoid_: Instance, container, pod

**Task**:
One unit of user work. Carries a task type, belongs to an application, and originates at a
source node — which is usually not the node it ends up executing on.
_Avoid_: Job, request, invocation

**Task type**:
The function a task runs. Determines which platforms are eligible to serve it and which
image must be resident before it can start.
_Avoid_: Function, kernel, workload type

**Application**:
A set of tasks submitted together with dependencies between them — a DAG, not a bag.
_Avoid_: Job, batch, workflow

**Storage**:
A disk attached to a node, local or remote, holding cached images and task data.
_Avoid_: Cache, disk, volume — "cache" is reserved (see below)

### Warmth

Three distinct predicates, kept separate because the physics differ in exactly whether two
of them are the same thing.

**Sandbox warmth**:
The platform's immediately previous task matched this task type, so its sandbox can be
reused.
_Avoid_: Warm, hot, warm replica

**Image locality**:
The task type's image is already resident on a node-local disk, so no pull is required.
_Avoid_: Cached, warm disk

**Warmth physics**:
Which predicate decides whether a task must pay for an image pull. Under
`platform_reuse_v1` sandbox warmth and image locality are **coupled** — a warm sandbox skips
the pull. Under `node_disk_v2` they are **decoupled** — only image locality skips it. Two
collections generated under different warmth physics can never be trained together.
_Avoid_: Warmth model, warm mode

**Cold start**:
The delay a task pays when neither the image nor the sandbox is ready for it.

### Placement

**Placement**:
The assignment of one task to one `(node, platform)`.

**Placement plan**:
An assignment for every task in a snapshot. A plan may leave a task to be auto-resolved by
the policy rather than forcing it.
_Avoid_: Mapping, assignment, schedule

**RTT**:
Round-trip time — a task's total wall time from dispatch to done, including queueing, image
pull, transfer and execution. `total_rtt` sums it over every task and is the primary
outcome measure everywhere.
_Avoid_: Latency, makespan, duration — **makespan** is a genuinely different composition
(`max(done) − min(dispatched)`) and must not be used loosely for RTT

**Policy**:
A scheduling strategy under test — GNN, MLP, Knative, and the reference implementations.
Each is an orchestrator, autoscaler and scheduler acting together; naming one names all
three.
_Avoid_: Algorithm, model, scheduler — "scheduler" is one component of a policy, not the
whole of it

**Contention**:
Any resource that co-placed tasks must share and thereby slow each other on. Distinguish the
level: **node contention** (compute on a node), **ingress contention** (a node's shared
inbound bandwidth), **link contention** (capacity on a backbone segment).

**Dispersal**:
How widely a policy spreads a batch across distinct nodes and platforms. A property of a
scheduler's behaviour over time, not of any single decision.
_Avoid_: Spread, load balancing

### Data the simulator produces

Four levels, deliberately not all called "dataset".

**Snapshot**:
One warmed system state captured at a decision point, together with the full sweep of
placement plans over it. The unit stored as `ds_XXXXX/`.
_Avoid_: Dataset, sample, episode

**Collection**:
A directory of snapshots generated from one grid preset under one set of physics.
_Avoid_: Dataset, corpus

**Corpus**:
The set of collections that are mutually compatible and are trained on together. Membership
is decided by physics, feature contract and task structure — never by convenience.
_Avoid_: Dataset, training set

**Cache**:
A corpus tensorized into the form a trainer consumes. A cache is derived, and can disagree
with what the same features look like at serving time — which is a defect, not a variant.
_Avoid_: Preprocessed data

**Placement sweep**:
Every valid placement plan for a snapshot paired with its resulting `total_rtt`. Mandatory:
a snapshot without one is incomplete, not merely unlabelled.
_Avoid_: Labels, ground truth

**Co-simulation**:
Generating a snapshot's sweep by brute-force simulating every plan. It produces a
**one-step** optimum: the best plan for this batch against a frozen state, not the best
policy over time.

### Whether the target has structure to learn

**Additivity**:
The degree to which a plan's RTT is the sum of independent per-task costs. Measured as
additive R². A fully additive target is one a pointwise model is correctly specified for,
and a graph-aware model cannot beat it by training.
_Avoid_: Separability (as a distinct concept — it is the same property; prefer additivity)

**Coupling**:
The residual: RTT that depends on which tasks were placed *together*. The quantity the whole
research programme is trying to manufacture.
_Avoid_: Interaction, non-linearity

**Collision**:
The specific coupling channel where a plan places more tasks on one platform or one node
than it has to. The dominant — and so far very nearly the only — source of coupling
measured in this simulator.

### Making a claim stick

**Lineage**:
A line of experimental work with a status and a recorded outcome. Not done until it has a
row in `LINEAGES.md`.

**Gate**:
A pre-registered comparison that a lineage must pass before its claim counts. Its criteria
are written down before it runs; a result read off an ungated run is exploratory, whatever
its size.

**Live gate**:
A gate run against a real trace through the actual simulator, as opposed to scoring against
a stored sweep. The only evidence that a checkpoint works, rather than that it fits.
_Avoid_: Evaluation, benchmark

**Sealed holdout**:
Evaluation data the model provably never saw, minted or withheld before training. Sealing is
per-cell: a cell is sealed when no snapshot with **this exact topology** was trained on. A
fresh topology seed is sufficient, even at a connection probability the corpus already
covered — sealed means unmemorized and in-distribution, not out-of-distribution. "Not in the
cache" is *not* sealed if it shares a topology with something that was.

**Cell**:
One evaluation instance — a topology (connection probability plus its topology seed) and the
workload run over it. Cells are the unit results are counted in ("wins 5/5 cells") and the
unit baselines are compared within (always against *same-cell* Knative).
_Avoid_: Run, config, seed

**Condition**:
The configuration held fixed while draws vary — cache and feature layout, typically. A
seed's effect is measured *within* a condition.

**Arm**:
One variant being compared in a gate: a policy plus its training configuration.
_Avoid_: Model, variant, treatment

**Draw**:
One training run's random initialization. A checkpoint is a sample from a draw
distribution, never a fixed property of its configuration — so reliability claims must
compare distributions, not single checkpoints.
_Avoid_: Seed, run

**Seed**:
Never use bare. Say **topology seed**, **training draw**, or **workload seed** — they are
three different things, and conflating them is what made every "multi-seed" live gate
secretly a multi-topology gate.

**Collapse**:
A detector firing: a run whose `total_rtt` exceeds same-cell Knative by the registered
threshold. It names a **signature, not a cause** — the occupation-collapse mechanism it was
named after has been falsified, and the detector is measured-invalid for candidate-relative
arms. Never infer a mechanism from the word.

**Pre-registration**:
Writing a gate's criteria, thresholds and decision rule down, committed, before the run that
tests them.

**Venue**:
The machine a result was produced on — local or datalab. Two numbers from different venues
are not comparable until the parity checks say so.
_Avoid_: Environment, platform — **platform** is taken

**Contract**:
A declared compatibility promise checked before a model is served. Never use bare: say
**queue-feature contract** (`legacy_v0` / `scale_invariant_v1`), **checkpoint contract** (the
`.contract.json` sidecar), **training-cache contract**, or **serving layout**. A checkpoint
without a checkpoint contract is not evidence.
