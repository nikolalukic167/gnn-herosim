# AGENTS.md

Guidance for coding agents working in this repository — Claude Code, Codex, and others.
`.claude/CLAUDE.md` imports this file (`@../AGENTS.md`) for Claude Code sessions and adds
a short Claude-only appendix below the import; there is no second copy of anything below.

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

**Where the research question stands (rewritten 2026-10-07).**

**On the faithful workload, no learned arm beats coordinate-descent search (CD); the earlier "beats CD" headline was an
artifact of scattered request origins.** The setting is the Alibaba-grounded workload at ×2 / ×3 / ×5 on 19 unseen
topologies (`grounded_workload_v1` → `peak_load_v1` → `small_batch_confirm_v1`). The grounded mint gave each task of a peer
group its own client (97 % of groups span clients); a trace request has one caller (`client_local_v1`). With one origin
per group every rule runs 26–82 % faster, and:

- **The engineered GNN `sb1load` no longer beats CD**, zero-shot or retrained on a single-origin corpus
  (`small_batch_so_v1`, NO-WIN): +13.1 / +7.3 % (CD faster) and −3.5 % (not separated); retraining moves it 1–2 %. It
  still beats Knative −40 / −54 / −68 % and self-predict ~−27 %, ties locality-first and the one-pass greedy, and ties its
  MP-OFF twin. Under scattered origins it had beaten CD −10 / −35 / −17 % — **quote that only with the origin model named.**
- **Message passing is real but small, and never where it beats search.** The raw-plan GNN beats its same-input MLP
  (`local_features_v1` −11 / −24 / −20 %; retrained single-origin −10 / −7 % at ×2 / ×3, ×5 not separated) and its no-conv
  twin, but CD is faster at every rung (+13 to +19 %). Per-relation weights (`hetero_conv_v1`) add up to −7 % at ×5 and
  nothing else; edge physics in the convs (`raw_plan_v2`) and sum aggregation were worse or tied. Where a learned arm has
  beaten CD it was the engineered pointwise context, never the convolutions (`small_batch_confirm_v1`, `peak_controls_v1`).
- **Client execution does not pay** (`client_local_v1`): 2–7 of 20 clients can host the functions at all, and local
  execution is 0.1–2 % of calls under every policy.

**Earlier settings, still true:** per arrival the self-predict rule is the bar (`selfpredict_bar_v1`); in the 6-server burst
seat uncapped `gnnedge0` beats the one-pass greedy and loses to CD (`joint_burst_v2`); the co-sim single-batch target is
pointwise-separable (`program_verdict_v1`); rollout imitation and lookahead are closed (`rollout_imitation_v1`,
`lookahead_mp_v1`). Every client-rung "vs reactive" number before 2026-09-20 flips sign when paired.

**Pair on the environment before you take a median; quote a DIRECTION from a small design and a MAGNITUDE only from a
large one; a rule with the model's information is the bar, never Knative alone; name the origin model with any grounded
number.** Never a `peer_affinity` live number without its load factor, never an arm comparison without its corpus. Start at
`docs/lineages/small_batch_so_v1.md`, `client_local_v1.md`, `local_features_v1.md` and `throughline.md`.

## Current research entry points

For L2D job-shop architecture claims, read `docs/lineages/literature_reeval_v1.md`,
`docs/lineages/l2d_size_transfer_v1.md`, and
`docs/lineages/l2d_learning_efficiency_v1.md` before proposing transfer or
sample-efficiency experiments. For future-action rollout targets in L2D, also read
`docs/lineages/l2d_rollout_headroom_v1.md`.

For the compiled exact-search control required before workflow move-value training,
read `docs/lineages/workflow_move_value_v1.md`, `docs/lineages/workflow_basin_v1.md`,
`docs/lineages/radical_physics_v1.md`, `docs/lineages/mixed_physics_learning_v1.md`,
and `docs/lineages/mixed_pair_learning_v1.md`. For the active execution-order
control challenge, read `docs/lineages/mixed_dispatch_v1.md`.
Its learned children `docs/lineages/mixed_dispatch_learning_v1.md`,
`docs/lineages/mixed_dispatch_state_v1.md`, and
`docs/lineages/mixed_dispatch_value_v1.md` close total-rank imitation,
ready-set teacher imitation, and unguarded ready-set action values respectively.
All lose to the hand rule; the value model has only a small uncertain edge over
learned controls. Read all three before proposing another ordering learner.
For irregular DAG reservation proposals, also read
`docs/lineages/dag_reservation_learning_v1.md` before proposing another
complete-plan learner or training on the same physics.
For hidden-arrival online reservations, read
`docs/lineages/online_reservation_s0_v1.md` before repeating a one/two-decision
search or treating a materialized replay as a live policy callback.

For the follow-up complete-plan proposal-search experiment, read
`docs/lineages/workflow_proposal_v1.md`.

For DAG output-memory proposals, read `docs/lineages/dag_memory_lifetime_s0_v1.md`,
`docs/lineages/dag_memory_value_learning_v1.md`, and
`docs/lineages/dag_remat_s0_v1.md` before changing cache or recomputation physics.

For the complete-workflow environment search and its trained pilot, read
`docs/lineages/workflow_amortized_v1.md`. Its checkpoint, data and timing contracts
are separate from the historical peer-affinity arms.

Read `docs/lineages/joint_burst_v2.md`, `docs/lineages/rollout_imitation_v1.md`,
`docs/lineages/peer_affinity_v1.md`, `docs/lineages/peer_greedy_live_v1.md`, and
`docs/lineages/throughline.md` before proposing the next experiment.
For the fixed-replica burst result, also read
`docs/lineages/gnn_seeded_cd_fixed_v1.md`: the GNN seed beats matched hand
coordinate search on fresh topologies when eligible replicas are held fixed.
Its dynamic-autoscaler predecessor is negative, and
`docs/lineages/gnn_seeded_cd_mp_ablation_v1.md` shows its separately trained
MP-OFF twin beats the GNN seed in the fixed seat. Keep the learned-seed win
distinct from a message-passing win.
For the later fixed-replica proposal-portfolio result, read
`docs/lineages/gnn_proposal_selector_v1.md`; it tests the GNN checkpoint as one
candidate inside a deterministic selector, not as a stand-alone policy.
For its matched 32-task training and architecture test, read
`docs/lineages/g32_sampled_slate_v1.md` before attributing proposal value to
bipartite message passing. Its exact-validation-selected successor is
`docs/lineages/g32_exact_val_selection_v1.md`; read both before proposing
another checkpoint-selection or 16-plan sampled-slate training run.
For the four-type lower-payload fixed-replica move-pool screen, read
`docs/lineages/g32_fourtype_move_s0_v1.md` before proposing move-value training
from single-task reassignments or occupancy-preserving swaps.
For the contended 32-task peer-transfer fabric pilot, read
`docs/lineages/g32_peer_fabric_s0_v1.md` before changing peer-transfer physics
or attributing route-link contention to a GNN.
For group-commit transfer prefetch and its finite-pipe counterfactual, read
`docs/lineages/g32_peer_prefetch_s0_v1.md` before proposing training.
For the core-only capacity successor and route-aware controls, read
`docs/lineages/g32_core_prefetch_s0_v1.md` before changing the physics or training.
For two-task valley proposals under that physics, read
`docs/lineages/g32_coordinated_valley_s0_v1.md` before repeating same-node peer-pair search.
For 128-operation critical/lock/setup/block-swap proposals from a strong workflow start,
read `docs/lineages/dag_block_proposer_128_s0_v1.md` and its closed 384-operation parent
before training a move proposer or changing the block pool.

Keep the arms distinct: `jb2_mpoff` is a separately trained MP-OFF bipartite
scorer on group-optimum labels in the burst seat; `peer_greedy_learned_network`
is the five-feature rollout scorer served per arrival. A defect in one does not
establish a defect in the other. Pair on the environment before aggregation;
training draws and repeated environments are not independent replicates.
A rule with the model's information is the bar, never Knative alone.

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

## Codex and other tools

Codex-specific agent/skill ports, workflow corrections, and cloud-agent access notes
(Mitrix via Tailscale) live in `.codex/`; see `.codex/README.md`.

## Issue tracker

Issues live in GitHub Issues on `nikolalukic167/gnn-herosim`, via the `gh` CLI. See `docs/agents/issue-tracker.md`.

## Triage labels

The five canonical roles, each label string equal to its name. See `docs/agents/triage-labels.md`.

## Domain docs

Single-context: `CONTEXT.md` + `docs/adr/` at the repo root. See `docs/agents/domain.md`.
