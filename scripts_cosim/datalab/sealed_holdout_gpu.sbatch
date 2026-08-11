#!/bin/bash
#SBATCH --job-name=cv2-sealed
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/cv2-sealed-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/cv2-sealed-%A_%a.err
#SBATCH --partition=GPU-a40,GPU-a100s,GPU-l40s,GPU-a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --array=0-19

# Sealed holdout GNN-only: 4 unused topologies × 5 seeds = 20 jobs.
# Knative/MLP already completed locally; submit those separately if needed
# (POLICY=all array 0-59 via submit_sealed_holdout.sh).
#
# Index: cfg = task // 5, seed = task % 5
# Override POLICY via --export when submitting.

set -euo pipefail

PROJECT_ROOT="/home/nikola.lukic/gnn-herosim"
cd "$PROJECT_ROOT"

export SWEEP_DIR="${SWEEP_DIR:-simulation_data/normal_sim_sweeps/contention_v2_873_v5.5_sealed_holdout_20260806}"
export WORKLOAD="${WORKLOAD:-data/nofs-ids/traces/workload-125-225.json}"
export GNN_MODEL="${GNN_MODEL:-models/near-rtt-v2-contention-v2-873-v5.5-dim14-ce-only.pt}"
export MLP_MODEL="${MLP_MODEL:-models/tabular/batch_edge_mlp_contention_v2_873_v5.5_dim22_batchcache.pt}"
export GNN_BATCH_SIZE=4
export GNN_BATCH_TIMEOUT=0.002
export TIMEOUT="${TIMEOUT:-18000}"
POLICY="${POLICY:-gnn}"

CONFIGS=(
  "balanced_p50|simulation_data/normal_sim_sweeps/knative_network_20260606_192413/configs/01_balanced_40_40_p50.json"
  "balanced_p60|simulation_data/normal_sim_sweeps/knative_network_20260606_192413/configs/02_balanced_50_50_p60.json"
  "client_heavy_p50|simulation_data/normal_sim_sweeps/knative_network_20260606_192413/configs/03_client_heavy_50_35_p50.json"
  "server_heavy_p50|simulation_data/normal_sim_sweeps/knative_network_20260606_192413/configs/04_server_heavy_35_50_p50.json"
)
SEEDS=(42 43 44 45 46)

tid="${SLURM_ARRAY_TASK_ID:?}"
cfg_idx=$(( tid / 5 ))
seed_idx=$(( tid % 5 ))
entry="${CONFIGS[$cfg_idx]}"
name="${entry%%|*}"
path="${entry#*|}"
seed="${SEEDS[$seed_idx]}"

bash scripts_cosim/datalab/run_sealed_holdout_one.sh "$POLICY" "$name" "$path" "$seed"
