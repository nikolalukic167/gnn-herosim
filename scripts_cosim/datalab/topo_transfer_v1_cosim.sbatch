#!/bin/bash
#SBATCH --job-name=topo-transfer-v1
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/topo-transfer-v1-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/topo-transfer-v1-%A_%a.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=96G
#SBATCH --time=24:00:00

# topology_transfer_v1 Phase 4 corpus, at the registered `tier_launch` power tier
# (gate_statistics.PHASE4_TIERS, 2026-08-19): 900 datasets / server_node_count, 4,500
# total across the size ladder {20,28,40,60,80} (train {20,28,40}, held-out {60,80}).
# TOPO_TRANSFER_V1_GRID's seed range was bumped 30->75 to hit this (30 gave 360/size,
# tier_0.02 -- the ladder arithmetic showed that cannot resolve either observed
# shallow_v1 win_rate effect).
#
# Backbone is ON by default in the grid preset itself (non-binding 1000 MB/s, n_core=12
# fixed) -- no --link-bandwidth-mbps flag needed here. Verified locally on a 6-dataset
# smoke test before this job existed: 52 core links, n_core=12, placements.jsonl
# populated on every dataset.
#
# NOT --allow-non-unique-replicas: the local smoke test (and the full local run this
# job replaces) succeeded at 100% without it, unlike the netc_* grids this shard runner
# was written for. Set explicitly below to keep this job matched to what was verified,
# not to the shard runner's netc-oriented default of 1.
#
# 450 datasets per (conn_prob, server_count) topology variant (10 variants: 2 conn
# probs x 5 sizes), so NUM_SHARDS=50 divides every variant into exactly 5 equal shards
# (90 datasets each) -- shards stay cost-homogeneous instead of straddling a variant
# boundary, and the expensive srv=80 variant (450 datasets, ~117s/dataset at 8 local
# workers) gets 5-way parallelism instead of running as one long shard.
#
# Submit:
#   sbatch --array=0-49 scripts_cosim/datalab/topo_transfer_v1_cosim.sbatch
# Monitor:
#   squeue -u nikola.lukic
#   tail -f logs/topo-transfer-v1-<jobid>_*.out
# After every array task finishes, certify before trusting the corpus (placements.jsonl
# is mandatory, not optional -- see docs/notes/placements_jsonl_required.md):
#   for d in simulation_data/gnn_datasets_4tasks_topo_transfer_v1/ds_*; do
#     [[ -s "$d/placements/placements.jsonl" ]] || echo "MISSING: $d"
#   done

set -euo pipefail
PROJECT_ROOT="/home/nikola.lukic/gnn-herosim"
cd "$PROJECT_ROOT"

export OUTPUT_SUBDIR="${OUTPUT_SUBDIR:-gnn_datasets_4tasks_topo_transfer_v1}"
export GRID=topo_transfer_v1
export WARMTH_PHYSICS=node_disk_v2
export REMAINING_START="${REMAINING_START:-0}"
export TOTAL_DATASETS="${TOTAL_DATASETS:-4500}"
export NUM_SHARDS="${NUM_SHARDS:-50}"
export ONLY_MISSING_JSONL="${ONLY_MISSING_JSONL:-0}"
export ALLOW_NON_UNIQUE=0
export INGRESS_BANDWIDTH_MBPS=
export WORKLOAD_SEED=
export GNN_CAPTURE_DATASET_STATE=0
export COSIM_SUPPRESS_SIM_PRINTS=1

bash scripts_cosim/datalab/run_contention_regen_shard.sh
