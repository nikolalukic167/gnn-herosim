#!/bin/bash
#SBATCH --job-name=sm-wssm-gate
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/sm-wssm-gate-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/sm-wssm-gate-%A_%a.err
#SBATCH --partition=GPU-l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=05:00:00
#SBATCH --array=0-8

# Strategic-merge WSSM live gate (9 jobs: knative/mlp/gnn × hub k4/k6/k8).
# Skip-if-valid-json in run_*_one.sh — safe to submit full array after partial local runs.
# Submit: sbatch scripts_cosim/datalab/strategic_merge_wss_live_gate_gpu.sbatch

set -euo pipefail

PROJECT_ROOT="/home/nikola.lukic/gnn-herosim"
cd "$PROJECT_ROOT"

export SWEEP_DIR="${SWEEP_DIR:-simulation_data/normal_sim_sweeps/strategic_merge_wss_live_gate_20260616}"
export WORKLOAD="${WORKLOAD:-data/nofs-ids/traces/workload-125-225.json}"
export GNN_MODEL="${GNN_MODEL:-models/near-rtt-v2-strategic-merge-wss-cont-v2-dim14-ce-only.pt}"
export MLP_MODEL="${MLP_MODEL:-models/tabular/batch_edge_mlp_strategic_merge_wss_cont_v2_dim22_batchcache.pt}"
export GNN_BATCH_SIZE=4
export GNN_BATCH_TIMEOUT=0.002
export TIMEOUT="${TIMEOUT:-18000}"
export QUEUE_LENGTH=100

CONFIGS=(
  "hub_k4_seek50|simulation_data/normal_sim_sweeps/sweep_bipartite_coordination_v1/configs/hub_k4_seek50.json"
  "hub_k6_seek50|simulation_data/normal_sim_sweeps/sweep_bipartite_coordination_v1/configs/hub_k6_seek50.json"
  "hub_k8_seek50|simulation_data/normal_sim_sweeps/sweep_bipartite_coordination_v1/configs/hub_k8_seek50.json"
)

policies=(knative mlp gnn)
policy="${policies[$(( SLURM_ARRAY_TASK_ID / 3 ))]}"
cfg_idx=$(( SLURM_ARRAY_TASK_ID % 3 ))
entry="${CONFIGS[$cfg_idx]}"
name="${entry%%|*}"
path="${entry#*|}"

bash scripts_cosim/datalab/run_strategic_merge_wss_live_gate_one.sh "$policy" "$name" "$path"
