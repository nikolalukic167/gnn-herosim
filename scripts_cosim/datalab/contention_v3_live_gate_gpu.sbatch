#!/bin/bash
#SBATCH --job-name=cv3-gate
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/cv3-gate-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/cv3-gate-%A_%a.err
#SBATCH --partition=GPU-a40,GPU-a100s,GPU-l40s,GPU-a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --array=0-8

# contention_v3 live gate: knative/mlp/gnn × 3 sparse configs (9 jobs).

set -euo pipefail

PROJECT_ROOT="/home/nikola.lukic/gnn-herosim"
cd "$PROJECT_ROOT"

export SWEEP_DIR="${SWEEP_DIR:-simulation_data/normal_sim_sweeps/contention_v3_live_gate_20260620}"
export WORKLOAD="${WORKLOAD:-data/nofs-ids/traces/workload-125-225.json}"
export GNN_MODEL="${GNN_MODEL:-models/near-rtt-v2-contention-v3-dim14-ce-only.pt}"
export MLP_MODEL="${MLP_MODEL:-models/tabular/batch_edge_mlp_contention_v3_dim22_batchcache.pt}"
export GNN_BATCH_SIZE=4
export GNN_BATCH_TIMEOUT=0.002
export TIMEOUT="${TIMEOUT:-18000}"

CONFIGS=(
  "sparse_p25|simulation_data/normal_sim_sweeps/knative_network_20260606_192413/configs/05_sparse_40_40_p25.json"
  "sparse_p35|simulation_data/normal_sim_sweeps/knative_network_20260606_192413/configs/00_balanced_30_30_p35.json"
  "sparse_p25_skew|simulation_data/normal_sim_sweeps/atomic21_skew_configs/05_sparse_40_40_p25_degree_skew.json"
)

policies=(knative mlp gnn)
policy="${policies[$(( SLURM_ARRAY_TASK_ID / 3 ))]}"
cfg_idx=$(( SLURM_ARRAY_TASK_ID % 3 ))
entry="${CONFIGS[$cfg_idx]}"
name="${entry%%|*}"
path="${entry#*|}"

bash scripts_cosim/datalab/run_contention_v3_live_gate_one.sh "$policy" "$name" "$path"
