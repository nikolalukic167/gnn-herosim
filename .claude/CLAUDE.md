# CLAUDE.md

Guidance for Claude Code working in this repository.

## What this is

**HeROsim** — a SimPy discrete-event simulator for serverless task placement in
heterogeneous clusters, plus the experimental apparatus around it. Two modes: live
simulation of a workload trace, and **co-simulation**, which brute-forces every placement
of a task batch to produce brute-force-labelled GNN training data.

**The research question:** does a graph-aware scheduler (GNN) beat a pointwise one (MLP)
at task placement? Knative is the industry-standard reactive baseline, and the MLP is the
pointwise control. **The MLP is a control, not a straw man** — the program's repeated
finding is that it ties, so treat "the MLP cannot match this" as a hypothesis to test, never
an assumption to write from.

Three routes to a positive answer. All three have been answered; route 2 was later reopened.

1. **Generate better co-sim data** so a GNN beats Knative and the MLP on latency.
   **Closed by measurement** (`program_verdict_v1`): the co-sim target is
   pointwise-separable, so the MLP is the *correctly specified* model class and more data
   cannot change that. Do not restart without reading that node.
2. **Change the environment** so exploitable joint structure exists. `route_b_env_pivot_v1`
   is `PARKED` (could not measure S0 on its overlap rungs). **Reopened** via
   `peer_affinity_v1`: a cost indexed by *pairs of task instances* under a binding cap is
   neither node-indexed nor routable-around. Before proposing any contention physics, read
   that node and `dag_fabric_contention_v1` — the one untried non-node-indexed lever, DAG
   output over the link fabric, is NO-GO before any code, and node-indexed CPU/memory
   contention is closed by the count theorem it cites.
3. **Change the training objective, not the environment** (`objective_pivot_v1`).
   **CLOSED 2026-09-03.** Phase 1 PASSED (reliability, scope-limited to severe collapse),
   Phase 2 CLOSED (horizon labels are deterministic chaos), Phase 3 MEASURED-NEGATIVE at
   n = 120 (−0.85 %, p = 0.928, powered). Do not restart without reading that node.

Options 1 and 2 are cited as "CLAUDE.md option 1/2" from several lineage nodes — keep the
numbering. What a GNN needs to have anything to learn from a *supervised* target is
**multi-task placements under contention**: route A proved coupling alone is not enough, and
route B proved contention alone is not enough either, which is why option 3 changed the
objective instead.

**Where the research question stands (rewritten 2026-09-21).**

**A two-line rule beats healthy reactive Knative by 13–16 % without waiting. One learned arm now
beats that rule in its own decoder seat** — trained on exactly the decision it is served and
served UNCAPPED (`joint_burst_v2`): `gnnedge0` beats the one-pass greedy −11.9 % (13/13), reactive
−34.7 %, random ~−48 %. **This is the first learned win over a rule with the model's own
information in the programme.** The `joint_burst_v1` loss to that greedy (+16.8 %) was a **serving
cap**, not the model or the corpus. **The remaining ceiling is a multi-pass coordinate-descent
greedy (+12.5 % ahead of the arm);** its lever `rollout_imitation_v1` is now **CLOSED
`RULE-FASTER-LIVE`** — a learned scorer beats it −12.7 % OFFLINE (held-out) but is +14–28 % SLOWER
live (R2 FAILS, 0/16 C40), an offline/live reversal. **No learned arm beats the rule live.** The
uncapped arm still only
ties its pointwise MLP twin — a win over the baselines, not yet over the pointwise control.

- **The rule** (`peer_greedy_live_v1`, 2026-09-20): Knative's candidate set scored in seconds as
  queue drain + cold + exec + latency + exchange to partners already placed, per arrival, no
  constant. On the 16 admissible environments per rung: **−12.96 % (C40) / −16.01 % (C80) vs
  reactive, 16/16 each**; −22.6 / −21.1 % vs random; the same rule with the exchange term off
  ties Knative, so the margin IS co-location (exchange 5.40 → 3.91 s per task and the queue
  *shorter*, 8.66 → 7.80 s). Served in `gnnedge0`'s seat (16 s peer-group batching, greedy in id
  order) it still beats `gnnedge0` −13.2 % (16/16, C80; −8.3 % disclosed at C40) and the
  no-wait flavour beats the batched one −8.3 %. The no-wait decoder's bar is now the rule.
- **The 6-server headline was never there** (`unsaturated_edge_v1`): "gnnedge0 −20.8 % (15/16)"
  was an unpaired ratio of medians over 4 cells, 2 saturated; paired per cell +4.6 / +75.8 /
  +9.7 / −36.8 %. On 16 environments with the paired statistic: C40 `gnnedge0` +6.68 % (0/14,
  disclosed), `peeronly` +8.36 %, `mpoff` +11.53 %, `gnn` +9.24 %; C80 `gnnedge0` +14.40 %
  (0/16). Every client-rung "vs reactive" number before 2026-09-20 flips sign when paired; the
  arm-vs-arm contrasts stand: `gnnedge0` beats its MP-OFF twin −6.82 % (14/14) at C40
  (`bipartite_aggr_v1`: use `mean`, never `sum`, over a variable-sized candidate set; name both
  corpora, 1670 and 516, in every model-class quote).
- **Batching relocates waiting** (`batch_window_edge_v1`): a 2 s window cuts the 7.1 s wait to
  1.7 s and the queue and rendezvous rise by the same amount (`WINDOW-NOT-THE-LEVER`). ECT loses
  +12.8 % to shortest-queue because its queue term is `len × exec` (0.1 s per queued task),
  not because physics awareness hurts.
- **Reactive's capacity does not grow with the cluster** (`unsaturated_scale_v1`): a cell
  enters a study only at reactive queue share ≤ 0.80, **unknown is not a pass**; 15 of 48
  6-server cells hang in the starved-client spin regardless of load; **w0** is the burstiest of
  the four arrival windows and every saturated cell is a w0 cell.
- **The environment levers** (`burst_groups_v1`, `payload_scale_v1`, `backbone_sparsity_v1`,
  2026-09-20). **Bursts** (groups dispatched together) remove the 7 s wait entirely (0.001 s) and
  Knative's rendezvous with it (Knative 17.9 → 8.3 s per task): `gnnedge0` then TIES Knative
  (−2.2 %, 11/16) while the rule reads **−26 %** (8/8) and beats `gnnedge0` +25 % (16/16) — 8
  environments, two signed amendments, disclosed; the wait was never the whole deficit. **Payload
  ×0.1** (20 MB): the rule ties Knative, every batching arm pays its wait (+15–16 %); **×3, ×10
  and a 250 Mbps backbone saturate reactive on every cell** — above the 200 MB scale every lever
  is a load lever at 0.46 arrivals/s on 6 servers. Regime: peer-aware placement pays from ×1 for
  the rule, at no measured scale for `gnnedge0`.
- **Trained on the served decision, `gnnedge0` beats the one-pass greedy in its own seat — the
  first learned win over a rule with the model's information** (`joint_burst_v1` → `joint_burst_v2`,
  2026-09-20/21): whole peer groups as bursts, from loaded states, labelled by the group optimum,
  16 seeds. v1 (capped) beat reactive −12.47 % (15/15) and its cold twin −7.82 % (J5) but lost the
  greedy's seat +16.83 %. v2 found that loss was the **serving cap** blocking the label's
  co-location move: served UNCAPPED, `gnnedge0` beats the one-pass greedy **−11.9 % (13/13)**,
  reactive −34.7 %, random ~−48 %; uncapping the v1 weights alone already clears the greedy (K6
  −8.8 %, 16/16) and the loaded-state corpus adds nothing beyond it (K5 CORPUS-NEUTRAL). It **loses
  to the coordinate-descent greedy +12.5 % (0/13)** — the honest ceiling — and only ties its MP-OFF
  twin (−4.4 %, under the bar). Its one-step-improvement follow-up `rollout_imitation_v1` CLOSED
  `RULE-FASTER-LIVE` (see the opening paragraph).
- **What survives from before:** the offline positive (`peer_affinity_v1`: MP beats its twin
  +5.14 pp offline, inverts live at a defensible load); a 150× trainability asymmetry
  (optimisation, never latency); `serving_stability_v1`'s early-trace positive; both model-class
  edges over the MLP fell to corpus matching (`link_mp_v1`, `reliability_matched_v1`);
  `partial_state_v3`'s size-free block is what lets 6-server checkpoints serve 80 at all.

**Pair on the environment before you take a median; quote a DIRECTION from a small design and a
MAGNITUDE only from a large one; a rule with the model's information is the bar, never Knative
alone.** Never a `peer_affinity` live number without its load factor, never a client-rung "vs
reactive" number from before 2026-09-20, never an arm comparison without both corpora. Start at
`docs/lineages/peer_greedy_live_v1.md`, `joint_burst_v2.md` and `throughline.md`.

## Where knowledge lives — READ FIRST

**`LINEAGES.md` is the one entry point to the research record**, and an **index only**: a
status and a one-line outcome per lineage, each linking to the node with the full record.
`simulation_data/REGISTRY.json` does the same for datasets.

| Where | What |
|---|---|
| `docs/lineages/<name>.md` | One node per lineage — standing, entry points, datasets, full dated record. Attachments in `docs/lineages/<name>/`. |
| `docs/lessons.md` | Transferable rules, one `##` section each — what generalises past any one lineage. |
| `docs/lessons-archive/` | Retired artifact inventories. Off the hot path; **not current practice.** |
| `docs/hard-stops.md` | Falsified directions + the measurement that closed each. **Check before proposing one.** |
| `docs/gates/gate-tools.md` | Corrections to the gates themselves, kept out of lineage narratives on purpose. |
| `docs/notes/` | Design notes on physics/features that outlive a lineage. |
| `docs/adr/` | Decisions with two live answers (warmth physics, queue contracts, mandatory sweep). |
| `CONTEXT.md` · `PARITY.md` · `CO_SIMULATION_GUIDE.md` | Vocabulary · cross-venue comparability · co-sim pipeline. |
| `tests/test_record_hygiene.py` | The checks that hold all of the above. ~2 s, no GPU. |

Statuses: `ACTIVE` · `REGISTERED` (signed off, not run) · `CLOSED` (answered) ·
`SUPERSEDED` · `FAILED`/`FALSIFIED` · `SYNTHESIS` · `PAPER`.

**One fact, one home.** Before adding a paragraph, find the file that already owns that
fact and edit it. This rule has been written down four times and broken four times, so it is
now a test: run `tests/test_record_hygiene.py`, and use the `close-a-lineage` skill when a
lineage closes, which is when the drift enters. **Session handovers are ephemeral and never
committed** — write them to the scratchpad; promote anything still true a week later into a
node, `docs/lessons.md`, or `docs/gates/gate-tools.md`.

**`archive/` is retired code. Ignore it** unless the user names a lineage. Do not search
it, import from it, or treat it as current practice. Moved with `git mv` (so
`git log --follow` works); restore point is tag `pre-cleanup-2026-08`.

## The rules that exist because they were broken

1. **Never import from `archive/`.** The live tree is verified closed against it;
   `tests/test_record_hygiene.py` carries the gate, along with the rest of the record's
   mechanical checks.
2. **Never fork a training script per experiment.** That habit produced 40 near-identical
   `train_near_rtt_v2_*.py` differing only in cache dir and wandb name. New experiments get
   a config under `experiments/`, run via `run_experiment.py`.
3. **A lineage is not done until it has a `LINEAGES.md` row and a `docs/lineages/` node
   with an outcome.** A result never written down gets re-run months later. The row is a
   status and *one line*; the record is the node. Run the `close-a-lineage` skill when a
   lineage closes — that is the moment the index, the node header and the stop all drift.
4. **Fail loudly.** No silent failures, no skipping a failure for convenience. Fix the
   cause.
5. **Every training run logs to Weights & Biases.** No exceptions.
6. **A lineage ends with a live gate, never with an offline read** (Nikola, 2026-09-13, after
   `peer_affinity_warm_v1` closed on its offline W0 screen). An offline screen may *order* the
   work; it does not *close* it. A NO-GO on an offline bar is recorded and the registered live
   gate still runs. Put the live gate in every plan and its cost.

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
order (`run_provenance.code` in both result JSONs → `verify_live_infra_parity.py` →
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
5. Read the curves before believing them:
   `pipenv run python3 scripts_cosim/read_training_curves.py wandb/run-<ts>-<id>`.
   It separates the metrics the objective can actually move from the dead and constant
   ones, prints every live curve against its **chance floor**, and names the selected
   epoch. A `task_acc` or `ce` quoted without that floor is not a result — the warm
   corpus averages 2.79 candidates per task, so 43% accuracy is chance. See
   `docs/lessons.md` → "Read a finished run's curves against their chance floor".
6. Gate it with a live-gate / sealed-holdout comparison in `scripts_cosim/important/`.
7. Write the outcome into the node **and** the index row. Not done until you do.

**An ablation harness is not a substitute for step 6.** A comparison script that trains
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

## Comment policy

When writing or editing code, don't add comments that just restate the code. Only comment
on non-obvious reasoning.
