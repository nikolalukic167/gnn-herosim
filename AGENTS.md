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

Routes 1–3 to a positive answer — better co-sim data, a changed environment, a changed objective — are all answered;
lineage nodes cite them as "CLAUDE.md option 1/2/3". Their text and status: `docs/lineages/throughline.md` → "Routes 1–3".

**Where the research question stands (rewritten 2026-10-12).**

**The bar is `cd_exactS_ib0` (`ib0_confirm_v1`, 2026-10-12) — exact search on CD's surrogate S over the top-5 slate with the
in-batch stacking term zeroed: −4.7 / −2.3 % vs `cd_exactS` (12/12 fresh topologies), which itself is −7 to −11 % vs CD across the
accel gates (24 topologies −8.6 / −6.8 %, `link_bound_s0_v1` −11.1 / −8.0 %, G8 −9.2 / −7.7 %). No learned arm beats
either, a full-coverage oracle beats `ib0` by only −2.8 / −2.0 %, and no learned win in this programme is a message-passing win.** Seat: R1.1 + WF1, replicas on the fastest
compatible platform, 160 clients × 24 servers, moderate and heavy rungs. Arm names: `CONTEXT.md` → Arms.
> ⚠ The GNN numbers here are on the cold-mispriced declared slate and not re-measured (`slate_fix_v1` running). Audit: gate-tools → "Numbers measured before a change".
- **Against plain `cd` a learned scorer wins at moderate, confirmed on fresh topologies:** `gnn_eng_nosplit` −6.9 / −7.7 %
  (`accel_nosplit_v1`), `gnn_selfsearch` −8.5 / −10.1 % (`selfsearch_confirm_v1`). Heavy: −1 to −4 %, under the −5 % bar.
  `gnn_selfsearch` ties its MP-OFF twin.
- **Against hand search** (`cd_exactS`, `cdxapply`): tie at moderate, 1.8–3.7 % slower at heavy, **but that deficit is a candidate-set
  handicap, not a scorer deficit** (`slate_bar_v1`, 2026-10-12): the GNN is trained and served on a declared top-5 slate whose cost charges
  no cold start on an initialized platform (the simulation does; gate-tools 2026-10-12), while `cd_exactS` may use any candidate. Restricting
  `cd_exactS` to that slate costs it +4.9 / +5.0 %; with matched information the GNN self-search beats it −4.85 % moderate (21/24, Holm .0003)
  / −1.94 % heavy (16/24, Holm .04). Still not an MP win (ties its twin). Whether the GNN reaches full-information `cd_exactS` on the uncapped
  candidate set is `slate_fix_v1` (training). The new hand bar is `cd_exactS_ib0` (`ib0_confirm_v1`, CLOSED: −4.7 / −2.3 % vs `cd_exactS`, 12/12); a full-coverage oracle beats it only −2.8 / −2.0 %, so no supervised re-ranker over the slate can clear it by more than that.
- **Why learning has little room:** median decision 3 tasks, 0 % over the enumeration cap; argmin-S optimal 79 %
  (regret 6 %); platforms 95–99 % idle; one-step lookahead headroom 3–5 %; more servers made it easier.
- **Closed levers:** post-batch value (`cost_to_go_v1`), bigger groups (`big_groups_s0_v1`), busy links
  (`link_bound_s0_v1`), a high-but-stable rung (`high_rho_s0_v1`), a longer horizon (`long_horizon_s0_v1`: 24 h, the GNN's deficit to cd_exactS widens, not narrows), supervised plan reranking (`plan_rerank_v1`), GNN self-search on over-cap batches only (`gnn_overcap_v1`: +0.2 / +2.5 % vs cd_exactS, HAND-FALLBACK-FASTER), GNN-ordered pruning of over-cap batches (`prune_g8_s0_v1`: pruning beats the expansion fallback −1.4 / −2.1 %, but any top-k order does; not learned, not MP). Numbers in
  their index rows.
- **Knative under R1:** quote only the concurrency-key arm (+48 / +68 % vs CD); the queue-keyed arm herds (gate-tools,
  2026-10-11).

**Earlier settings, still true:** on the grounded workload with one origin per group no learned arm beats CD
(`small_batch_so_v1`) — every grounded win before 2026-10-05 is scattered-origin; message passing helps raw inputs, not
engineered ones (`local_features_v1`, `r1_attribution_v1`); self-predict is the per-arrival bar (`selfpredict_bar_v1`);
the single-batch co-sim target is pointwise-separable (`program_verdict_v1`); every client-rung "vs reactive" number
before 2026-09-20 flips sign when paired.

**Pair on the environment before you take a median; quote a DIRECTION from a small design and a MAGNITUDE only from a
large one; a rule with the model's information is the bar, never Knative alone; name the origin model with any grounded
number.** Never a `peer_affinity` live number without its load factor, never an arm comparison without its corpus.

## Current research entry points

Read the node before proposing work in its area. Current seat: `docs/lineages/selfsearch_confirm_v1.md`, `docs/lineages/accel_nosplit_v1.md`,
`docs/lineages/big_groups_s0_v1.md`, `docs/lineages/cost_to_go_v1.md`, `docs/lineages/link_bound_s0_v1.md`, `docs/lineages/plan_rerank_v1.md`, `docs/lineages/throughline.md`.

| Area | Read first |
|---|---|
| L2D job-shop architecture, transfer, sample efficiency, rollout targets | `docs/lineages/literature_reeval_v1.md`, `docs/lineages/l2d_size_transfer_v1.md`, `docs/lineages/l2d_learning_efficiency_v1.md`, `docs/lineages/l2d_rollout_headroom_v1.md` |
| Workflow move values, exact-search control | `docs/lineages/workflow_move_value_v1.md`, `docs/lineages/workflow_basin_v1.md`, `docs/lineages/radical_physics_v1.md`, `docs/lineages/mixed_physics_learning_v1.md`, `docs/lineages/mixed_pair_learning_v1.md` |
| Execution order (all learners lose to the hand rule) | `docs/lineages/mixed_dispatch_v1.md`, `docs/lineages/mixed_dispatch_learning_v1.md`, `docs/lineages/mixed_dispatch_state_v1.md`, `docs/lineages/mixed_dispatch_value_v1.md` |
| DAG reservations, online reservations | `docs/lineages/dag_reservation_learning_v1.md`, `docs/lineages/online_reservation_s0_v1.md` |
| Complete-plan proposal search, amortised workflows | `docs/lineages/workflow_proposal_v1.md`, `docs/lineages/workflow_amortized_v1.md` |
| DAG output memory, cache, recomputation | `docs/lineages/dag_memory_lifetime_s0_v1.md`, `docs/lineages/dag_memory_value_learning_v1.md`, `docs/lineages/dag_remat_s0_v1.md` |
| Burst seat, peer greedy, rollout imitation | `docs/lineages/joint_burst_v2.md`, `docs/lineages/rollout_imitation_v1.md`, `docs/lineages/peer_affinity_v1.md`, `docs/lineages/peer_greedy_live_v1.md` |
| Learned seeds and proposals for search (learned-seed win ≠ MP win) | `docs/lineages/gnn_seeded_cd_fixed_v1.md`, `docs/lineages/gnn_seeded_cd_mp_ablation_v1.md`, `docs/lineages/gnn_proposal_selector_v1.md` |
| 32-task slates, selection, move pools | `docs/lineages/g32_sampled_slate_v1.md`, `docs/lineages/g32_exact_val_selection_v1.md`, `docs/lineages/g32_fourtype_move_s0_v1.md` |
| Peer-transfer fabric, prefetch, core contention, valleys | `docs/lineages/g32_peer_fabric_s0_v1.md`, `docs/lineages/g32_peer_prefetch_s0_v1.md`, `docs/lineages/g32_core_prefetch_s0_v1.md`, `docs/lineages/g32_coordinated_valley_s0_v1.md` |
| 128-operation block proposals | `docs/lineages/dag_block_proposer_128_s0_v1.md` |

Pair on the environment before aggregation; training draws and repeated environments are not independent replicates.

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
