#!/bin/bash
#SBATCH --job-name=netc-v1
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/netc-v1-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/netc-v1-%A_%a.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=96G
#SBATCH --time=24:00:00

# network_contention_v1 co-sim: the shallow_v1 grid under a shared per-node ingress pipe.
#
# Why this lineage exists: the co-sim target is additive to 98.8-100%, so a pointwise MLP
# is the correctly specified model class. Every term of the RTT sum is a function of
# (task, platform) alone except added_in_batch. Network latency -- the LARGEST term in the
# shallow regime, ~40% of per-task RTT -- is a static table lookup that never congests, so
# routing N tasks over a link costs exactly what routing 1 costs.
#
# This arm splits that term: propagation latency stays additive, while input transmission
# (stateSize / bandwidth) is served through the destination node's single shared pipe. Two
# batch-mates placed on the same node then serialize, so the cost of a placement depends on
# where the OTHER tasks went -- which is what a pointwise scorer cannot express.
#
# Measured on a 2-dataset matched A/B before writing this (identical workload and topology,
# only the flag differs). Mean RTT delta vs baseline, by extra co-located tasks in the plan:
#   0 extra -> 3.20 transfers   (pure additive transmission)
#   1 extra -> 4.63 transfers   (+1.4 transfers of WAITING)
#   2 extra -> 6.50 transfers   (+3.3 transfers of WAITING)
# Monotone in co-location: the coupling is real. node_contention_v3 by contrast measured
# nodeContentionTime == 0.0 on every placed task, because tasks held its resource for only
# ~0.024 s and never overlapped. A transfer at 1.5 MB/s is ~0.098 s, 4x the exec time.
#
# Bandwidth sweep (153,600 B nofs-* input): 5.0 MB/s -> 0.031 s, 1.5 -> 0.098 s (target,
# ~= the 0.095 s median propagation latency), 0.5 -> 0.31 s.
#
# WORKLOAD_SEED is pinned so every arm draws the SAME task source nodes. Before 2026-08-18
# that draw came from the unseeded global RNG, which made matched arms impossible.
#
# No #SBATCH --array here on purpose: pass --array at submit time so pilot and full runs
# share one file. The shard runner fails loud without SLURM_ARRAY_TASK_ID.
#
# Pilot (n=40, one arm per submission -- run all four, then gate):
#   for a in "base:" "bw5p0:5.0" "bw1p5:1.5" "bw0p5:0.5"; do
#     tag="${a%%:*}"; bw="${a#*:}"
#     sbatch --array=0-3 --export=ALL,TOTAL_DATASETS=40,NUM_SHARDS=4,\
# OUTPUT_SUBDIR=gnn_datasets_4tasks_netc_v1_${tag},INGRESS_BANDWIDTH_MBPS=${bw} \
#       scripts_cosim/datalab/netc_v1_cosim.sbatch
#   done
#
# Then, per arm, certify the corpus is FINISHED before scoring it -- the retracted 31%
# coupled fraction came from reading an in-flight corpus:
#   python3 scripts_cosim/audit_placements_jsonl_integrity.py --corpus <subdir>
#   python3 scripts_cosim/separability_diagnostic.py simulation_data/<subdir> \
#     --integrity-manifest simulation_data/placement_integrity_manifest.json \
#     --gate-coupled-fraction 0.20 --output simulation_data/separability_<subdir>.json
#
# PRE-REGISTERED GATE: coupled fraction (M1 greedy regret >1%) >= 20%, against shallow_v1's
# 4.5% and contention_v2's 7.1%. Secondary: additive R^2 < 0.95, and node-collision R^2 gain
# exceeding platform-collision gain. The coupled fraction is primary because the target is
# bimodal (shallow_v1 median R^2 0.9999 vs mean 0.956) and the mean hides the coupled
# minority that is the whole point.

set -euo pipefail
PROJECT_ROOT="/home/nikola.lukic/gnn-herosim"
cd "$PROJECT_ROOT"

export OUTPUT_SUBDIR="${OUTPUT_SUBDIR:-gnn_datasets_4tasks_netc_v1_bw1p5}"
export GRID=shallow_v1
export WARMTH_PHYSICS=node_disk_v2
export REMAINING_START="${REMAINING_START:-0}"
export TOTAL_DATASETS="${TOTAL_DATASETS:-40}"
export NUM_SHARDS="${NUM_SHARDS:-4}"
export ONLY_MISSING_JSONL="${ONLY_MISSING_JSONL:-0}"
export ALLOW_NON_UNIQUE=1
# Empty = baseline arm: no ingress pipe, node_disk_v2 physics.
export INGRESS_BANDWIDTH_MBPS="${INGRESS_BANDWIDTH_MBPS:-}"
export WORKLOAD_SEED="${WORKLOAD_SEED:-42}"
export GNN_CAPTURE_DATASET_STATE=0
export COSIM_SUPPRESS_SIM_PRINTS=1

bash scripts_cosim/datalab/run_contention_regen_shard.sh
