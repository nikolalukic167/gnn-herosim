#!/bin/bash
#SBATCH --job-name=wt-cont-gate
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/wt-cont-gate-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/wt-cont-gate-%A_%a.err
#SBATCH --partition=GPU-l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=05:00:00
#SBATCH --array=0-11

# Weighted-merge contention live gate (12 jobs):
#   0-2  mlp × 3 configs
#   3-5  gnn argmax_uniq × 3
#   6-8  gnn argmax × 3
#   9-11 knative × 3 (CPU-fast; still on GPU partition for queue simplicity — knative skips GPU)
# Submit: sbatch scripts_cosim/datalab/merged_contention_weighted_live_gate_gpu.sbatch

set -euo pipefail

PROJECT_ROOT="/home/nikola.lukic/gnn-herosim"
cd "$PROJECT_ROOT"

export SWEEP_DIR="${SWEEP_DIR:-simulation_data/normal_sim_sweeps/merged_contention_weighted_live_gate_20260616_105500}"
export WORKLOAD="${WORKLOAD:-data/nofs-ids/traces/workload-125-225.json}"
export GNN_MODEL="${GNN_MODEL:-models/near-rtt-v2-warmth-sparse-contention-weighted-dim14-ce-only.pt}"
export MLP_MODEL="${MLP_MODEL:-models/tabular/batch_edge_mlp_warmth_sparse_contention_v2_weighted_dim22_batchcache.pt}"
export GNN_BATCH_SIZE=4
export GNN_BATCH_TIMEOUT=0.002
export TIMEOUT="${TIMEOUT:-18000}"

CONFIGS=(
  "sparse_p25|simulation_data/normal_sim_sweeps/knative_network_20260606_192413/configs/05_sparse_40_40_p25.json"
  "sparse_p35|simulation_data/normal_sim_sweeps/knative_network_20260606_192413/configs/00_balanced_30_30_p35.json"
  "sparse_p25_skew|simulation_data/normal_sim_sweeps/atomic21_skew_configs/05_sparse_40_40_p25_degree_skew.json"
)

cfg_idx=$(( SLURM_ARRAY_TASK_ID % 3 ))
entry="${CONFIGS[$cfg_idx]}"
name="${entry%%|*}"
path="${entry#*|}"

if (( SLURM_ARRAY_TASK_ID < 3 )); then
  bash scripts_cosim/datalab/run_merged_contention_weighted_live_gate_one.sh mlp "$name" "$path"
elif (( SLURM_ARRAY_TASK_ID < 6 )); then
  export GNN_DECODE_MODE=argmax_uniq
  bash scripts_cosim/datalab/run_merged_contention_weighted_live_gate_one.sh gnn "$name" "$path" gnn_uniq
elif (( SLURM_ARRAY_TASK_ID < 9 )); then
  export GNN_DECODE_MODE=argmax
  bash scripts_cosim/datalab/run_merged_contention_weighted_live_gate_one.sh gnn "$name" "$path" gnn_argmax
else
  bash scripts_cosim/datalab/run_merged_contention_weighted_live_gate_one.sh knative "$name" "$path"
fi
