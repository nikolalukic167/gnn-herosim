# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

**HeROsim** (Heterogeneous Resources Orchestration Simulator) is a discrete-event simulation environment for evaluating serverless resource allocation and task scheduling policies. The simulator supports both traditional simulation runs and co-simulation for generating GNN training datasets.

### Research Goals

**Primary Goal**: Demonstrate that Graph Neural Networks (GNNs) outperform simpler baselines for serverless task placement in heterogeneous networks.

- **GNN** is the main focus - we want it to be the best performing scheduler
- **MLP** exists as a verification baseline to ensure a simple pointwise model doesn't outperform the graph-aware approach
- **Knative** serves as the industry-standard reactive baseline

**Ultimate Goal**: Either:
1. Generate high-quality co-simulation data that trains the GNN to beat both Knative and MLP, OR
2. Fix the simulation environment to be more dynamic and better aligned for GNNs to leverage network topology advantages

**Key Requirement for GNN Advantage**: Multi-task placements are critical. GNNs need network heterogeneity and multi-task placement decisions to demonstrate advantages over pointwise MLPs. Single-task or overly homogeneous scenarios don't give GNNs enough structural signal to excel.

**Note**: Regime B is outdated. Focus is now on scenarios that create meaningful graph structure for the GNN to exploit.

## Commands

### Environment Setup

```bash
# Install Python 3.12 and dependencies (first time)
pipenv install

# Install development dependencies
pipenv install --dev

# Activate virtual environment
pipenv shell

# All Python commands must be run with pipenv
pipenv run python3 <script>
```

### Running Simulations

```bash
# Basic simulation scenario (from repository examples)
./scenario-ids.sh
./scenario-deepfake.sh
./scenario-proactive.sh

# Run simulation directly
pipenv run python -m src.placement <policy> <infrastructure> <workload> <output>

# Execute simulation with specific policy
pipenv run python src/executesimulation.py --policy <policy_name> <args>
```

### Co-Simulation (GNN Dataset Generation)

```bash
# Generate GNN training datasets (brute-force placement optimization)
pipenv run python scripts_cosim/generate_gnn_datasets_fast.py [--max-datasets N] [--quiet] [--workers N]

# Quick test (3-5 datasets)
pipenv run python scripts_cosim/generate_gnn_datasets_fast.py --max-datasets 5 --quiet

# Generate infrastructure only
pipenv run python -m src.generate_infrastructure --config <config> --sim-input <input> --output <output> --seed <seed>

# Run brute-force optimization (low-level)
pipenv run python src/executecosimulation.py --brute-force [--quiet] <config_file> <mapping_file> <output_dir>
```

### Testing

```bash
# Run pytest tests
pipenv run python3 -m pytest <test_file> -v
pipenv run python3 -m pytest <test_file> -q  # quiet mode

# Example: test queue features
pipenv run python3 -m pytest scripts_cosim/test_queue_features.py -q

# Many test files can be run directly
pipenv run python3 scripts_cosim/test_regime_b_metrics.py

# Run specific test function
pipenv run python3 -m pytest scripts_cosim/test_queue_features.py::test_function_name -v
```

### Training Models

**CRITICAL**: All training runs must be logged to Weights & Biases (wandb) for tracking and comparison.

```bash
# Train GNN models (ensure wandb is configured)
pipenv run python src/notebooks/train_near_rtt.py

# Train MLP models (verification baseline)
pipenv run python src/policy/tabular/train_mlp_dim22_from_batch.py

# Prepare graph cache for GNN training
pipenv run python src/notebooks/prepare_graphs_cache.py
```

### Datalab HPC Cluster

**Datalab** is the HPC cluster at TU Wien used for large-scale training and evaluation runs.

- **Cluster**: `cluster.datalab.tuwien.ac.at`
- **Repository**: `/home/nikola.lukic/gnn-herosim`
- **Environment**: micromamba with `gnn` environment (NOT pipenv)
- **Resources**: GPU nodes (GPU-a40, GPU-l40s) and CPU-only nodes
- **Workflow**: Code changes via git push/pull, large binaries (models, datasets) via rsync

**Activating environment on datalab**:
```bash
eval "$(micromamba shell hook --bash)" && micromamba activate gnn
```

**SLURM Batch Jobs** (submit on datalab):
```bash
# Submit batch job to SLURM cluster
sbatch <script>.sbatch

# Examples
sbatch scripts_cosim/datalab/live_cpu_amd.sbatch
sbatch scripts_cosim/datalab/contention_v4_deepq_cosim.sbatch

# Monitor jobs
squeue -u nikola.lukic
```

## Architecture

### Core Simulation Components

The simulator is built on **SimPy** (discrete-event simulation) with three extensible base classes:

1. **`Orchestrator`** (`src/placement/orchestrator.py`) - Entry point for policy implementation
   - Manages system state representation
   - Coordinates autoscaler and scheduler
   - Abstract methods for custom state structures

2. **`Autoscaler`** (`src/placement/autoscaler.py`) - Replica lifecycle management
   - Creates and removes function replicas
   - Resource selection for replica placement
   - Abstract methods: replica creation/removal, resource selection

3. **`Scheduler`** (`src/placement/scheduler.py`) - Task placement decisions
   - Selects replica for each incoming task
   - Abstract method: replica selection from pool

Key infrastructure models in `src/placement/infrastructure.py`:
- **`Node`** - Physical/virtual machines in the cluster
- **`Platform`** - Execution environments (CPU cores, GPUs, etc.)
- **`Replica`** - Function instances running on platforms
- **`Task`** - User requests to be executed
- **`Application`** - Collections of functions

Simulation runtime: `src/placement/simulation.py` (SimPy-based event loop)

### Policy Implementations

Policies are in `src/policy/` subdirectories, each implementing the Orchestrator/Autoscaler/Scheduler pattern:

- **`random/`** - Random placement (simplest reference implementation, ~20 lines)
- **`roundrobin/`** - Round-robin placement
- **`knative/`** - Knative-style autoscaling
- **`bpff/`** - Best Platform First Fit
- **`gnn/`** - Graph Neural Network scheduler (main ML approach)
  - `gnn/scheduler.py` - GNN-based task placement
  - `gnn/seq_decode.py` - Sequential decode with optional seqblend override
- **`tabular/`** - Tabular ML schedulers (MLP baseline)
  - `tabular/mlp_scheduler.py` - MLP batch scheduler (verification baseline)
  - `tabular/feature_builder.py` - Feature engineering for ML models

### Co-Simulation Pipeline

Located in `scripts_cosim/` - generates GNN training datasets via brute-force placement optimization.

**Main Script**: `scripts_cosim/generate_gnn_datasets_fast.py`
- Grid search across configuration space (network topology, replica configs, queue distributions, seeds)
- Single Python process (replaces old bash + jq pipeline)
- Parallel brute-force evaluation with worker initializer pattern

**Core Engine**: `src/executecosimulation.py`
- Phase 1: Capture system state after warmup
- Phase 2: Generate all valid placement combinations
- Phase 3: Execute simulations in parallel (ProcessPoolExecutor)
- Phase 4: Write results (`best.json`, `placements/placements.jsonl`)

**Infrastructure Generation**: `src/generate_infrastructure.py`
- Deterministic network topology with guaranteed connectivity
- Seeded randomness for reproducibility
- Replica placement across nodes
- Queue distributions for initial workload state

**Critical Requirement**: Every co-sim dataset **must** have `placements/placements.jsonl` - the full `(placement_plan, rtt)` sweep for RTT-hash / near-RTT training. Never treat the placement sweep as optional; never `--resume` on `best.json` alone without JSONL. See `memory/placements_jsonl_required.md`.

**Output Structure**:
```
simulation_data/gnn_datasets/ds_XXXXX/
├── infrastructure.json         # Network topology, replicas, queues
├── workload.json              # Task sequences
├── space_with_network.json    # Configuration
├── best.json                  # Optimal RTT and file reference
├── optimal_result.json        # Full simulation result for best placement
└── placements/
    └── placements.jsonl       # All placement combinations with RTTs (MANDATORY)
```

See `CO_SIMULATION_GUIDE.md` for comprehensive pipeline documentation.

### Data Flow

1. **Input**: Workload traces (`data/*/traces/`), infrastructure configs, task/platform metadata
2. **Simulation**: Event-driven execution via `src/placement/simulation.py`
3. **Policy Decision**: Orchestrator → Scheduler (placement) + Autoscaler (scaling)
4. **Execution**: Tasks run on platforms, SimPy advances time
5. **Output**: Results in `result/`, logs in `log/`, charts in `chart/`

### Feature Engineering

Queue-aware features are critical for ML schedulers:

- **`src/placement/queue_features.py`** - Queue normalization contracts
  - `legacy_v0`: Pre-split formulas (for backward compatibility with existing caches/checkpoints)
  - `scale_invariant_v1`: Invariant to uniform queue depth scaling (fixes training/live distribution mismatch)
  - Environment variable `QUEUE_FEATURE_CONTRACT` controls which contract is active

- **`src/policy/tabular/feature_builder.py`** - Feature extraction for tabular models
  - Builds inference feature bundles from system state
  - Handles task, platform, and edge features

- **`src/placement/warmth.py`** - Replica warmth/coldness modeling

### Model Training & Inference

**GNN Training**:
- Data preparation: `src/notebooks/prepare_graphs_cache.py`
- Training: `src/notebooks/train_near_rtt.py`
- Models saved in `models/` directory

**MLP Training** (verification baseline):
- Training script: `src/policy/tabular/train_mlp_dim22_from_batch.py`
- Model: `src/policy/tabular/mlp_model.py` (PointwiseEdgeMLP)
- Purpose: Verify that simple pointwise model doesn't outperform graph-aware GNN

**Inference**:
- GNN: `src/policy/gnn/scheduler.py` loads and runs GNN models
- MLP: `src/policy/tabular/mlp_scheduler.py` loads and runs MLP models
- Model loading uses `set_models()` method on scheduler instances

### Metrics & Analysis

**Comparison Scripts**: Various comparison tools in `scripts_cosim/important/`
- Focus on comparing GNN vs MLP vs Knative performance
- Analyze where GNN excels (network-aware decisions) vs where it struggles

## Key Patterns

### Creating a New Policy

1. Copy `src/policy/random/` as a template
2. Implement:
   - `orchestrator.py` - Subclass `Orchestrator`, define state structure
   - `autoscaler.py` - Subclass `Autoscaler`, implement scaling logic
   - `scheduler.py` - Subclass `Scheduler`, implement placement logic
3. Register in `src/placement/simulation.py` imports
4. Test with a simple scenario script

### Running Experiments

1. Define space configuration (JSON with infrastructure/workload parameters)
2. Generate samples: `src/generateall.py` + `src/sample.py` (LHS sampling)
3. Execute simulations: `src/executeinitial.py` for initial dataset
4. Optimize: `src/executeoptimization.py` for Bayesian optimization of underperforming samples
5. Analyze results: Results in `results/` folder, use `src/charts/` for visualization

### Working with Co-Simulation Datasets

**Generate datasets**:
```bash
pipenv run python scripts_cosim/generate_gnn_datasets_fast.py --max-datasets 20 --quiet
```

**Verify output**:
```bash
# Check dataset structure
ls -la simulation_data/gnn_datasets/ds_00000/

# Check best result
cat simulation_data/gnn_datasets/ds_00000/best.json

# CRITICAL: Verify placements.jsonl exists
ls -la simulation_data/gnn_datasets/ds_00000/placements/placements.jsonl

# Check progress log
cat logs/progress.txt
```

**Status codes**:
- `SUCCESS`: Dataset generated successfully
- `SKIPPED`: Configuration infeasible (e.g., no valid placements)
- `FAILED`: Error during generation

### Queue Feature Contracts

When working with queue-based features (dim 7 divisor, dim 13 usage ratio):

1. **Check active contract**: Look for `QUEUE_FEATURE_CONTRACT` environment variable
2. **Backward compatibility**: Use `legacy_v0` for existing caches/checkpoints (873/v5.5, regime B distill, ect_pull)
3. **New training**: Use `scale_invariant_v1` to prevent training/live distribution mismatch
4. **Contract validation**: Use `require_matching_queue_feature_contract()` in inference paths
5. **Testing**: See `scripts_cosim/test_queue_features.py` for contract property tests

## Dataset Metadata & Validation

All co-simulation dataset collections have comprehensive metadata tracking generation parameters, validation status, and compatibility for training. This system enables:
- Understanding what datasets exist and what problems they solve
- Validating data quality before training
- Intelligently combining compatible datasets for training

### Metadata System

**Global Registry**: `simulation_data/REGISTRY.json`
- Indexes all 17 collections (16 active, 1 deprecated)
- Tracks 3,566 total datasets (3,364 completed)
- Groups collections by status and problem category

**Per-Collection Metadata**: `simulation_data/<collection>/METADATA.json`
- Generation parameters (grid preset, connection probabilities, replica configs, queue distributions, seeds)
- Physics configuration (warmth_model, queue_feature_contract)
- Results (coupling rate, queue depth statistics, completion status)
- Training usage (which models used this data, associated graph caches)
- Compatibility information

**Schema**: `simulation_data/METADATA_SCHEMA.json` (v1.0.0)

### Validation Workflows

**Extract metadata for all collections**:
```bash
pipenv run python scripts_cosim/extract_dataset_metadata.py --all
```
Expected: 17 `METADATA.json` files + global `REGISTRY.json`

**Validate active collections**:
```bash
pipenv run python scripts_cosim/validate_dataset_collection.py --active-only
```
Checks:
- Structural completeness (required files: infrastructure.json, workload.json, best.json, placements/placements.jsonl)
- Physics consistency (same warmth_model across collection)
- Queue depth validation (match declared distributions within ±5%)
- Coupling rate estimation (RTT spread sampling)

Validation reports saved to: `simulation_data/<collection>/VALIDATION_REPORT.json`

**Generate compatibility matrix**:
```bash
pipenv run python scripts_cosim/compute_compatibility_matrix.py
```
Output: `simulation_data/COMPATIBILITY_MATRIX.json` with training group recommendations

### Compatibility Rules

Collections are **compatible** for training if:
- Same `warmth_physics` (node_disk_v2 vs platform_reuse_v1 are incompatible)
- Same `queue_feature_contract` (legacy_v0 vs scale_invariant_v1 are incompatible)
- Same task structure (4-task vs 1-task are incompatible)
- Both active (deprecated collections excluded)

**Training Groups** (from compatibility matrix):
1. **legacy_v0_node_disk_v2_4task**: 15 collections, 2,816 datasets
   - Contention series (v1-v5): scarce warm resources + varying queue depths
   - Warmth series: replica placement studies
   - Hetero baselines: Knative comparison
   - Production: highq_safe_20260606

2. **legacy_v0_platform_reuse_v1_4task**: 1 collection, 48 datasets
   - Regime B oracle_split_cosim only

### Checking Dataset Quality Before Training

Before training on a dataset collection:

1. **Check metadata**:
   ```bash
   cat simulation_data/<collection>/METADATA.json | jq '.results, .physics'
   ```

2. **Review validation report**:
   ```bash
   cat simulation_data/<collection>/VALIDATION_REPORT.json | jq '.status, .structural_completeness'
   ```

3. **Verify compatibility**:
   ```bash
   cat simulation_data/COMPATIBILITY_MATRIX.json | jq '.training_groups'
   ```

4. **Check for missing placements.jsonl** (CRITICAL requirement):
   ```bash
   # Count datasets with placements.jsonl
   ls -d simulation_data/<collection>/ds_*/placements/placements.jsonl | wc -l
   ```

**Key Quality Indicators**:
- Structural completeness ≥97% (some training subsets intentionally exclude datasets)
- Queue depths match declared distributions (within tolerance)
- Physics consistency across all datasets in collection
- Placements.jsonl present and non-empty (mandatory per CO_SIMULATION_GUIDE.md)

## Project Conventions

### Development Philosophy

**Keep it simple, change small, test fast**:
- Don't write a lot of code at once
- Make small, focused changes
- Test quickly before running big long runs
- Verify results immediately
- Iterate based on fast feedback

**Experimentation workflow**:
1. Make a small change
2. Run a quick test (5 datasets, 1-2 configs)
3. Verify the change works as expected
4. Only then scale up to full runs

### Error Handling

**Critical**: Make sure there are no silent failures. Fail loudly everywhere! Don't skip failures for convenience - solve the underlying issue.

### Documentation

When asked for explanations or analysis, **answer in chat directly** - do not write markdown or text documents.

### Training Visibility

**All training runs must be logged to Weights & Biases (wandb)**. No exceptions. This ensures:
- Progress tracking across experiments
- Hyperparameter comparison
- Reproducibility
- Team visibility

### Python Environment

Always use `pipenv run python3` or activate the pipenv environment first. Never run Python commands directly outside the virtual environment.

On datalab, use: `eval "$(micromamba shell hook --bash)" && micromamba activate gnn`

## Directory Structure

```
.
├── src/                          # Core simulation engine
│   ├── placement/                # Infrastructure, orchestrator, autoscaler, scheduler
│   ├── policy/                   # Policy implementations (random, gnn, tabular, etc.)
│   ├── generator/                # Workload/infrastructure generators
│   ├── notebooks/                # Training scripts (GNN, MLP)
│   └── executecosimulation.py   # Co-simulation brute-force engine
├── scripts_cosim/                # Co-simulation scripts and utilities
│   ├── generate_gnn_datasets_fast.py  # Main dataset generation script
│   ├── important/                # Key comparison and analysis scripts
│   ├── datalab/                  # SLURM batch job scripts
│   └── test_*.py                 # Test files
├── data/                         # Input data (traces, task/platform metadata)
├── simulation_data/              # Generated datasets and configurations
│   └── gnn_datasets/             # Co-simulation output datasets
├── models/                       # Trained ML models (GNN, MLP)
├── logs/                         # Simulation logs
├── result/                       # Simulation results
├── memory/                       # Design notes and decision records
├── paper/                        # Research paper content
└── scenario-*.sh                 # Example scenario scripts
```

## Important Files

- `CO_SIMULATION_GUIDE.md` - Comprehensive co-simulation pipeline documentation
- `memory/placements_jsonl_required.md` - Critical requirement for placement sweep
- `.cursor/rules/project-guidelines.mdc` - Project-specific guidelines
- `Pipfile` - Python dependencies (SimPy, PyTorch, torch-geometric, etc.)

## Notes

- The project uses **pipenv** for dependency management
- Simulation is **deterministic** when seeded properly (critical for reproducibility)
- Co-simulation supports both **cold start** (0% preinit) and **warm start** scenarios
- **Network connectivity** is guaranteed in generated infrastructures (post-processing ensures reachability)
- **orjson** is used for fast JSON serialization where possible, with fallback to stdlib json
