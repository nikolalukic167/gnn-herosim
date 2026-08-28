# CLAUDE.md

Guidance for Claude Code working in this repository.

## What this is

**HeROsim** — a SimPy discrete-event simulator for serverless task placement in
heterogeneous clusters, plus the experimental apparatus around it. Two modes: live
simulation of a workload trace, and **co-simulation**, which brute-forces every placement
of a task batch to produce brute-force-labelled GNN training data.

**The research question:** does a graph-aware scheduler (GNN) beat a pointwise one (MLP)
at task placement? Knative is the industry-standard reactive baseline. The MLP exists to
verify that a simple pointwise model *cannot* match the graph-aware one.

Three ways to get there, and **which one is live has changed again (2026-08-28)**:

1. Generate co-sim data good enough to train a GNN that beats Knative and MLP on latency.
   **Closed by measurement** — see `program_verdict_v1`. The co-sim target is
   pointwise-separable, so the MLP is the *correctly specified* model class and no amount
   of training data changes that. Do not restart this without reading that node.
2. **Change the environment** so exploitable joint structure exists
   (`route_b_env_pivot_v1`, chosen 2026-08-27 after route B stage 2 returned
   NO-GO-PREPROBE). **PARKED 2026-08-28** — the screen could not measure S0 on its
   overlap rungs, and even a pass would feed the objective option 1 closed. Resuming
   needs a signed amendment in its node.
3. **Change the training objective, not the environment.** **This is the live program**
   (`objective_pivot_v1`): establish the GNN's live reliability edge as a registered
   draw-distribution claim, then P3, then P1 closed-loop training against the live
   simulator — the one path `program_verdict_v1` left open to the latency claim. The
   simulator is now the training environment, not a label factory.

(Options 1/2 are cited as "CLAUDE.md option 1/2" from several lineage nodes — keep them.)

What a GNN needs in order to have anything to learn from a *supervised* target:
**multi-task placements under contention**. Route A proved coupling alone is not enough —
breaking separability is necessary but not sufficient; you need contention. And route B
proved even that is not sufficient for the supervised path — which is why option 3
changes the objective instead.

## Where knowledge lives — READ FIRST

**`LINEAGES.md` is the one entry point to the research record**, and an **index only**: a
status and a one-line outcome per lineage, each linking to the node with the full record.
`simulation_data/REGISTRY.json` does the same for datasets.

| Where | What |
|---|---|
| `docs/lineages/<name>.md` | One node per lineage — standing, entry points, datasets, full dated record. Attachments in `docs/lineages/<name>/`. |
| `docs/lessons.md` | Transferable rules — what generalises past any one lineage. |
| `docs/hard-stops.md` | Falsified directions + the measurement that closed each. **Check before proposing one.** |
| `docs/gates/gate-tools.md` | Corrections to the gates themselves, kept out of lineage narratives on purpose. |
| `docs/notes/` | Design notes on physics/features that outlive a lineage. |
| `docs/adr/` | Decisions with two live answers (warmth physics, queue contracts, mandatory sweep). |
| `CONTEXT.md` · `PARITY.md` · `CO_SIMULATION_GUIDE.md` | Vocabulary · cross-venue comparability · co-sim pipeline. |

Statuses: `ACTIVE` · `REGISTERED` (signed off, not run) · `CLOSED` (answered) ·
`SUPERSEDED` · `FAILED`/`FALSIFIED` · `SYNTHESIS` · `PAPER`.

**One fact, one home.** Before adding a paragraph, find the file that already owns that
fact and edit it. `LINEAGES.md` reached 4,995 lines because five files narrated the same
experiments and drifted apart. **Session handovers are ephemeral and never committed** —
write them to the scratchpad; promote anything still true a week later into a node,
`docs/lessons.md`, or `docs/gates/gate-tools.md`.

**`archive/` is retired code. Ignore it** unless the user names a lineage. Do not search
it, import from it, or treat it as current practice. Moved with `git mv` (so
`git log --follow` works); restore point is tag `pre-cleanup-2026-08`.

## The five rules that exist because they were broken

1. **Never import from `archive/`.** The live tree is verified closed against it;
   `LINEAGES.md` → Conventions carries the re-runnable gate.
2. **Never fork a training script per experiment.** That habit produced 40 near-identical
   `train_near_rtt_v2_*.py` differing only in cache dir and wandb name. New experiments get
   a config under `experiments/`, run via `run_experiment.py`.
3. **A lineage is not done until it has a `LINEAGES.md` row and a `docs/lineages/` node
   with an outcome.** A result never written down gets re-run months later.
4. **Fail loudly.** No silent failures, no skipping a failure for convenience. Fix the
   cause.
5. **Every training run logs to Weights & Biases.** No exceptions.

## Commands

All Python goes through pipenv. A stray local `.venv` hijacks `pipenv run` and surfaces as
a misleading `ModuleNotFoundError`, so when anything looks wrong, use the full form:

```bash
PIPENV_IGNORE_VIRTUALENVS=1 VIRTUAL_ENV= PYTHONPATH=/root/projects/my-herosim \
  pipenv run python3 <script> ...
```

```bash
# Live simulation / sweep runner used by the gates
pipenv run python3 src/executesimulation.py --policy <policy> <args>
pipenv run python3 scripts_cosim/run_simulation.py <args>

# Co-simulation: generate GNN training datasets
pipenv run python3 scripts_cosim/generate_gnn_datasets_fast.py --max-datasets 5 --quiet

# Recache, then train via an experiments/ config (never a new train_*.py)
pipenv run python3 src/notebooks/prepare_graphs_cache.py
pipenv run python3 run_experiment.py experiments/<config>.yaml

# Tests
pipenv run python3 -m pytest tests/ -q
```

`--policy` takes **registry names** (`knative_network_batch`), not `run_simulation.py`
strategy strings (`kn_network_kn_network`) — a wrong guess costs a 5 s startup round-trip.
Live-gate result JSONs are ~80 MB: read bounded prefixes
(`extract_gate_stats_summary.py`, `extract_platform_dispersal.py` are the patterns).

**Run this before training anything you intend to gate** (~12 s, no GPU). There is no CI;
running it is manual:

```bash
PIPENV_IGNORE_VIRTUALENVS=1 OMP_NUM_THREADS=1 pipenv run python3 -m pytest tests/test_trainer_determinism.py -q
```

Two runs of a trainer at one seed must give bit-identical weights. It covers **every**
trainer, not just the one that broke — see `docs/lineages/trainer_determinism_v1.md`.

## Datalab (TU Wien SLURM cluster)

**`ssh datalab` — that alias is configured and is the spelling to use.** It resolves to
`cluster.datalab.tuwien.ac.at` with the right user and key; writing the FQDN means
supplying `-i` and the user by hand for no benefit.

- Repo on the cluster: `/home/nikola.lukic/gnn-herosim`
- Environment: micromamba `gnn` (**not** pipenv) —
  `eval "$(micromamba shell hook --shell bash)" && micromamba activate gnn`
  (`--shell bash`, never the old `--bash`: it passes on login nodes and kills every
  compute-node job)
- Resources: GPU-a40, GPU-l40s, and CPU-only nodes
- Sync: **source by git push/pull, binaries by rsync** — then md5 both sides. `models/` is
  gitignored, so a checkpoint's `.contract.json` sidecar must travel with the `.pt`.

```bash
ssh datalab 'cd ~/gnn-herosim && sbatch scripts_cosim/datalab/<script>.sbatch'
ssh datalab 'squeue -u nikola.lukic'
```

**Never write `pipenv run python3` in anything that may run under `sbatch`.** `pipenv run`
resolves its own venv and shells straight past `micromamba activate gnn`; on the cluster
this silently created a third, undeclared environment that every gate actually used. Write
`${HEROSIM_PY:-pipenv run python3}` and `export HEROSIM_PY=python3` after activation. When
auditing, grep **both** spellings — the shell form and Python `["pipenv", "run", ...]` argv
lists — and read `run_provenance.python_env` from a result JSON rather than trusting an
sbatch banner.

Before writing an `.sbatch` or submitting, load the `datalab-pitfalls` skill. Before
comparing two numbers from different machines, read **`PARITY.md`** and run its checks in
order (`verify_code_identity.py` → `verify_live_infra_parity.py` →
`verify_venue_parity.py`). **Unknown is not a pass**, and a one-directional cross-venue gap
is a feature-code bug, not the venue — measured: library versions contribute exactly 0.0 to
GNN logits.

## Architecture — the parts you can't infer from the tree

Three extensible base classes; a policy implements all three:

- **`Orchestrator`** (`src/placement/orchestrator.py`) — system state, coordinates the other two
- **`Autoscaler`** (`src/placement/autoscaler.py`) — replica lifecycle, resource selection
- **`Scheduler`** (`src/placement/scheduler.py`) — picks a replica per incoming task

Runtime is `src/placement/simulation.py`. Infrastructure models (`Node`, `Platform`,
`Task`, `Application`, `Storage`) are in `src/placement/infrastructure.py`.

**There is no `Replica` class** — do not go looking for one. A *replica* is a
`(Node, Platform)` pair in `system_state.replicas[task_type_name]`: an eligibility fact
with no identity, lifecycle or state. The autoscaler "creates" one by adding a tuple to
that set; `_get_valid_replicas` is a filter, not a lookup. Full vocabulary in `CONTEXT.md`.

Policies live in `src/policy/` (18 of them). The ones that matter: `gnn/` (main approach;
`seq_decode.py` holds the sequential decode), `tabular/` (MLP baseline + `feature_builder.py`),
`knative*/` (baselines, several network/batch/ECT variants), `random/` (~20 lines — copy
this as a template for a new policy, then register it in `src/placement/simulation.py`).

**Feature contracts** are the thing that silently breaks a checkpoint:

- `src/placement/queue_features.py` — `legacy_v0` (existing caches/checkpoints) vs
  `scale_invariant_v1` (new training; invariant to uniform queue scaling). Selected by
  `QUEUE_FEATURE_CONTRACT`; enforce with `require_matching_queue_feature_contract()` in
  inference paths. See `docs/adr/0002-two-queue-feature-contracts.md`.
- `src/placement/warmth.py` — warmth/coldness physics. `node_disk_v2` vs
  `platform_reuse_v1` are **incompatible**; a mismatch changes live RTT ~100×, so it raises.
- A **checkpoint without a `.contract.json` sidecar is not evidence.**
  `_read_checkpoint_sidecar` returns `{}` and every check downstream silently adopts its
  default. `load_state_dict(strict=True)` is *not* a compatibility check — architecture
  flags invisible in weight shapes (e.g. `mp_residual`) must come from the contract.

### Co-simulation

`scripts_cosim/generate_gnn_datasets_fast.py` grid-searches the config space; the engine is
`src/executecosimulation.py` (capture state after warmup → enumerate valid placements →
simulate in parallel → write results). Topologies come from `src/generate_infrastructure.py`,
deterministic and seeded, with connectivity guaranteed by post-processing.

Every dataset **must** carry `placements/placements.jsonl` — the full `(placement_plan, rtt)`
sweep. Never treat it as optional; never `--resume` on `best.json` alone. See
`docs/notes/placements_jsonl_required.md` and `CO_SIMULATION_GUIDE.md`.

```
simulation_data/gnn_datasets/ds_XXXXX/
├── infrastructure.json    # topology, replicas, queues
├── workload.json          # task sequences
├── space_with_network.json
├── best.json              # optimal RTT + file reference
├── optimal_result.json    # full result for the best placement
└── placements/placements.jsonl   # MANDATORY: every plan with its RTT
```

Generation status codes: `SUCCESS` · `SKIPPED` (infeasible config) · `FAILED` (error).

## Running an experiment

1. Check `LINEAGES.md` — new lineage, or an extension of an ACTIVE one? Check
   `docs/hard-stops.md` before proposing a direction.
2. Add a grid preset to `generate_gnn_datasets_fast.py`, generate datasets.
3. Recache with `prepare_graphs_cache.py`.
4. Train via a config under `experiments/` — **this is what produces a checkpoint.**
5. Gate it with a live-gate / sealed-holdout comparison in `scripts_cosim/important/`.
6. Write the outcome into the node **and** the index row. Not done until you do.

**An ablation harness is not a substitute for step 5.** A comparison script that trains
in-process to compute an eval statistic has no reason to persist checkpoints and typically
doesn't — `topology_transfer_v1` ran a full pre-registered gate that way and ended with zero
deployable weights and no live-gate at all. If a result should ever face a real workload, its
training must go through step 4 at some point. If you run the ablation harness anyway, pass
`--save-checkpoints DIR` so each arm gets weights plus a `.contract.json`.

**Keep it simple, change small, test fast.** Small focused changes; quick test (5 datasets,
1–2 configs); verify; only then scale up.

## Dataset validation

Before training on a collection, check compatibility — collections mix only if they share
`warmth_physics`, `queue_feature_contract`, and task structure, and both are active.

```bash
pipenv run python3 scripts_cosim/extract_dataset_metadata.py --all      # METADATA.json + REGISTRY.json
pipenv run python3 scripts_cosim/validate_dataset_collection.py --active-only  # VALIDATION_REPORT.json
pipenv run python3 scripts_cosim/compute_compatibility_matrix.py        # COMPATIBILITY_MATRIX.json
```

Read `.results` / `.physics` from `METADATA.json`, `.status` from the validation report, and
`.training_groups` from the compatibility matrix. Structural completeness ≥97% is healthy —
some training subsets intentionally exclude datasets. The `dataset-validator` agent does this
end to end.

## Conventions

- **Answer analysis questions in chat.** Do not write a markdown document unless asked.
- **Simulation is deterministic when seeded properly.** Tie-breaks over sets of objects are
  the classic leak — `PYTHONHASHSEED` does not pin them (it randomizes str/bytes only).
- Dependencies: `Pipfile`. One env spec for cross-venue work: `envs/herosim-lock.txt`.
