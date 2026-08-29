# HeROsim — a testbed for graph-based placement in the edge–cloud continuum

HeROsim is a **discrete-event simulator** (SimPy) for serverless task placement across a
heterogeneous edge–cloud cluster, plus the apparatus needed to turn it into **GNN training
data** and to **evaluate learned schedulers against real baselines** under a reproducibility
contract.

It exists to answer one question honestly: *does a graph-aware scheduler place tasks better
than a pointwise one?* The repository is built so that a negative answer is as publishable
as a positive one — every claim here is pre-registered, provenance-pinned, and scored by a
script that takes no tuning arguments.

## Two modes

| mode | what it does |
|---|---|
| **live simulation** | Replays a workload trace (~300k events) against a cluster, with a pluggable placement policy. Produces per-task records and a scalar `total_rtt`. |
| **co-simulation** | Captures cluster state after warmup, then **brute-forces every valid placement** of a small task batch and simulates each one. Yields exactly-labelled `(placement, rtt)` data — the supervised target for a GNN. |

## What is modelled

- **Compute heterogeneity** — Raspberry Pi class SBCs, Nvidia Jetson/Xavier edge AI devices,
  and servers, grouped into nodes with platforms, queues, warmth/cold-start physics and
  image caching.
- **Network** — an optional **backbone topology with finite link bandwidth**: core nodes,
  access links, per-link latency and bandwidth, and routing. This is the axis on which the
  graph model's advantage actually appears (see *Results*).
- **Applications** — serverless functions with DAG dependencies, data sizes and storage tiers.

## Quickstart

```bash
# 1. Generate co-simulation training data (brute-forced placement labels)
pipenv run python3 scripts_cosim/generate_gnn_datasets_fast.py --max-datasets 5 --quiet

# 2. Build the graph cache the trainer consumes
pipenv run python3 src/notebooks/prepare_graphs_cache.py

# 3. Train via a config (never a bespoke train_*.py — see Conventions)
pipenv run python3 run_experiment.py experiments/<config>.yaml

# 4. Evaluate live against baselines
pipenv run python3 src/executesimulation.py --policy <registry-name> <args>
```

Every dataset carries `placements/placements.jsonl` — the **full** placement sweep, not just
the optimum. Treat it as mandatory; a dataset without it cannot support most analyses here.

## Policies

Implement three base classes to add one: `Orchestrator` (system state), `Autoscaler` (replica
lifecycle), `Scheduler` (per-task replica choice). `src/policy/random/` is ~20 lines — copy it
as a template, then register in `src/placement/simulation.py`.

Shipped: `gnn/` (GIN over a bipartite task↔platform graph with a masked-softmax edge scorer),
`tabular/` (pointwise MLP), `knative*/` (reactive industry baselines, several network/batch
variants), plus assorted controls.

> **There is no `Replica` class.** A *replica* is a `(Node, Platform)` pair in
> `system_state.replicas[task_type]` — an eligibility fact with no identity or lifecycle.
> See `CONTEXT.md` for the full vocabulary.

## Reproducibility apparatus

This is the part worth reusing even if you do not care about placement:

- **Feature contracts.** A checkpoint without a `.contract.json` sidecar is not evidence.
  `load_state_dict(strict=True)` is *not* a compatibility check — architecture flags invisible
  in weight shapes must come from the contract.
- **Warmth physics** and **queue feature** contracts are explicit and incompatible variants
  raise rather than silently changing results by ~100×.
- **Determinism gate.** `tests/test_trainer_determinism.py` requires two runs of *every*
  trainer at one seed to give bit-identical weights.
- **Venue parity.** `PARITY.md` + `verify_venue_parity.py` establish when two numbers from
  different machines may be compared. Measured result: library versions contribute exactly
  0.0 to GNN logits, so a cross-venue gap is a code bug, not the venue.
- **Provenance.** Results record the commit, tree cleanliness and every registered env axis;
  scorers VOID rather than compute over unverified arms.

## Results established so far

Measured on 16 seeded training draws per model class × 30 live cells (~300–350k events each),
against Knative as the reactive baseline:

- **Reliability (registered, PASS).** The GNN's severe-collapse burden — cells at
  `total_rtt` ≥ +50% vs Knative — is stochastically smaller than the pointwise MLP's
  (rank-sum *p* = 0.00143; at +100% the GNN is clean in 16/16 draws, the MLP is not).
  **Scoped to severe collapse**: at +30% two GNN draws still carry real burden. The claim is
  "a tight distribution with a tail versus a lottery", never "the GNN does not collapse".
- **Latency is contention-dependent (exploratory).** The GNN averages **−25.1%** vs Knative on
  cells with a finite-bandwidth backbone, but **+2.5%** (slightly worse) on flat cells.
  Remove network contention and the advantage disappears.
- **A supervised negative.** On single-batch co-simulation targets the objective is
  pointwise-separable, so the MLP is the *correctly specified* model class and more data
  cannot change that. The diagnostic that establishes this is reusable and is, in our view,
  the more transferable contribution.

**Known open confound:** GNN and MLP differ in architecture, capacity and regularisation, not
only in graph-awareness. A message-passing ablation has not yet been run, so "graph-aware" is
not yet distinguished from "different architecture".

## Where knowledge lives

`LINEAGES.md` indexes the research record; `docs/lineages/<name>.md` holds each full dated
node. `docs/hard-stops.md` lists falsified directions and the measurement that closed each —
**read it before proposing a direction**. `CONTEXT.md` (vocabulary), `PARITY.md`
(cross-venue), `CO_SIMULATION_GUIDE.md` (pipeline) are the other entry points.

## Conventions

- New experiments get a config under `experiments/`, run via `run_experiment.py`. Never fork
  a training script per experiment.
- Simulation is deterministic when seeded. Tie-breaks over *sets* are the classic leak —
  `PYTHONHASHSEED` does not pin them.
- Dependencies in `Pipfile`; one pinned spec for cross-venue work in `envs/herosim-lock.txt`.
