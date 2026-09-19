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
   is `PARKED` (it could not measure S0 on its overlap rungs). **Currently reopened** via
   `peer_affinity_v1`: a cost indexed by *pairs of task instances* under a binding cap is
   neither node-indexed nor routable-around, so it clears the neighbouring stops. Before
   proposing any contention physics, read that node and `dag_fabric_contention_v1` — the
   one untried non-node-indexed lever, DAG output over the link fabric, is NO-GO before any
   code, and node-indexed CPU/memory contention is closed by the count theorem it cites.
3. **Change the training objective, not the environment** (`objective_pivot_v1`).
   **CLOSED 2026-09-03.** Phase 1 PASSED (reliability, scope-limited to severe collapse),
   Phase 2 CLOSED (horizon labels are deterministic chaos), Phase 3 MEASURED-NEGATIVE at
   n = 120 (closed-loop policy gradient does not beat the supervised checkpoint; −0.85%,
   p = 0.928, powered). Do not restart without reading that node.

Options 1 and 2 are cited as "CLAUDE.md option 1/2" from several lineage nodes — keep the
numbering. What a GNN needs to have anything to learn from a *supervised* target is
**multi-task placements under contention**: route A proved coupling alone is not enough, and
route B proved contention alone is not enough either, which is why option 3 changed the
objective instead.

**Where the research question stands (rewritten 2026-09-19).**

**A bipartite message-passing arm beats reactive Knative at an UNSATURATED rung by −20.8 %
(15/16), and beats its corpus-matched pointwise twin — but its own attribute-zeroed control
matches it, and the matched win does not reproduce on the smaller corpus.** All of that is the
standing answer; quoting one part alone misreports it. Everything below qualifies them.

- **The peer measurement** (`peer_only_v1`, 16 checkpoints throughout). `peeronly` (PeerConv
  kept, the bipartite GIN removed) beats its MP-OFF twin at **every** operating point tested
  (−4.6 to −21.9 % across 6/12/24/80 servers, −14.1 to −22.6 % across 20/40/80 clients) and
  beats reactive by −41.5 % at 80 servers and −9.3 % at 80 clients (13/16, p = 0.0097),
  **unsaturated**. Only message passing differs. **Four clauses, none optional:** the margin is
  **99 % queue** (the peer term moves 0.66 s of 78 s — *not* evidence peer reasoning pays, A3
  MECHANISM-NOT-CONFIRMED); **never quote a 4-checkpoint number** (−15.68 % more than halved on
  16); the client ladder is a **load sweep**, not the dispersion sweep intended; and the win is
  **not exclusive** — `mpoff_516` beats reactive there too (−7.2 %).
- **The bipartite penalty is SOLVED, and it was a `sum`** (`peer_only_v1` C1/C2 →
  `bipartite_edge_v1` D1–D4 → `bipartite_aggr_v1` E1–E3). `peeronly`
  beat `gnn` by **−18.94 %** at 80 servers and **not at all at 6** (p = 0.61) — an absence that
  survived five attempts at a mechanism. The cause: `GIN` aggregates with **`sum`**, a task
  aggregates over its **candidate platforms**, and that set is **3.55/task at 6 servers, 47.92
  at 80**. Replacing the `GIN` with a mean-aggregating `BipartiteEdgeConv` is worth
  **−15.72 %** (15/16) at 80 servers; that same conv given the `GIN`'s `sum` **collapses to
  −5.86 %, n.s.**; and `sum` vs `mean` alone is **+13.26 % (p = 0.0052) at 80 servers, not
  separated at 6** — the predicted scale-dependence, `SUM-COSTS-ONLY-WHERE-CANDIDATES-ARE-MANY`.
  **Edge-conditioning is NOT the reason** (D2: the zeroed-attribute control matches the
  treatment), nor is MLP shape. **Use `mean` for any bipartite stage over a variable-sized
  candidate set.** The repaired arm beats reactive by **−20.8 % (15/16) at the UNSATURATED
  40-client rung** where `gnn` loses by +8.44 %. Carry: the 6-server null is a non-separation,
  not proof of equality — but it **held at n = 32** and weakened (−6.22 %, p = 0.29).
- **The best-ARM clause stands; the best-MODEL-CLASS one does not** (`best_arm_v1` →
  `corpus_matched_v1`). The repaired graph arm vs `mpoff_516`, three
  unsaturated client rungs: **−14.76 % (16/16) at 40**, −11.32 % at 80, **+15.71 % (3/16,
  it LOSES) at 20** ⇒ `BEST-ARM-STILL-NOT-ESTABLISHED` — two wins and a loss where the signed
  rule needed two wins **and none**, so **the arm clause stands.** But **that single loss was a
  corpus effect.** Matched (same cache, split, seeds; only MP differs) at **1,670**: −31.65 %
  (16/16) at 40, −17.12 % at 80, **not behind at 20** (−0.73 %, p = 0.88) ⇒
  `MODEL-CLASS-EDGE-SURVIVES-MATCHING`; at **516**: one win (−10.78 % at 40), two ties ⇒
  `MODEL-CLASS-EDGE-IS-CORPUS-CONTINGENT`, so **name the corpus in every quote of either.** The
  confound is large: the same pointwise arm at 1,670 vs 516 reads **+18.71/+23.81/+8.64 %**,
  `CONFOUND-IS-MATERIAL` at all three rungs. Carry: at 20 clients **every** arm loses to
  reactive (+69 to +96 %, 0/16), a ranking among losers; `mpoff_516` is a GNN's MP-OFF twin,
  **not the MLP**; the contrast is the whole MP stack, not the bipartite conv; and
  `peeronly_1670` still loses to `mpoff_516` **+11.17 %, 0/16** at 80 servers / 20 clients,
  **B8 SCOPED to saturated rungs**. On the **server** ladder the arms beat reactive only at 24
  and 80; at 6 all lose.
- **The offline positive on the supervised target** is `peer_affinity_v1`: message passing
  beats its MP-OFF twin by **+5.14 pp** (p = 0.001, 13/16 seeds) at 482 datasets, reproduced
  at **−4.20 pp on 15/16** on the 1,654-dataset corpus under the size-free representation
  (`peer_only_v1`). Only `PeerConv` differs.
- **That offline edge does not transfer, and it inverts.** Offline and live are anti-correlated
  in peer-graph density; it is contingent on one platform type the corpus never contained; and
  at a defensible load (the gate ran at **940× overload**) it reads **−15.94 %** vs the twin,
  **−226 %** vs reactive.
- **Both model-class edges over the MLP fell to corpus matching** — latency (`link_mp_v1`) and
  reliability (`reliability_matched_v1`, p = 0.113). Corpus is the largest measured lever, and
  `corpus_matched_v1` shows a matched verdict can itself depend on the level matched at. Any
  GNN-vs-MLP number must name both arms' training cache.
- **What survives:** a trainability asymmetry (the closed loop moves the GNN 150× more —
  optimisation, never latency), a fit-ceiling split on the route B corpus, and one replicated
  live positive: over the first fifth of a trace the learned arms beat Knative on queue on 3/3
  cells and all 91 arms (`serving_stability_v1`).
- **The first whole-trace live win (`partial_state_v3`)** — a size-free rank block (dim 38 → 22,
  P0 bit-identical) lets 6-server checkpoints serve 12/24/80 and **beat reactive at 80 servers
  by 28.7 %/46.9 % on 4/4 seeds**, against a registered DEGRADES. Superseded as the headline by
  the unsaturated results above: those rungs are **saturated** and at 6 servers the arms lose.

**Before quoting any number from this program**, read the node and carry its caveats. Never
quote a `peer_affinity` live number without its load factor, nor `peer_only_v1`'s B2 without
its saturation and the `mpoff_516` comparison, nor any arm comparison without both corpora.
Start at `docs/lineages/peer_only_v1.md` and `docs/lineages/throughline.md` (last section).

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
