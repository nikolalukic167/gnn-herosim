#!/bin/bash
#SBATCH --job-name=mega-all7
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/mega-all7-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/mega-all7-%A_%a.err
#SBATCH --partition=GPU-l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-41

# Experiment Group 1: mega compare all7 — 6 new model configs x 7 topology configs = 42 jobs.
#
# Group layout (array index -> policy x config):
#   0-6   : gnn_warmth   x 7 configs  (near-rtt-v2-warmth-dim14-ce-only.pt)
#   7-13  : gnn_wsm      x 7 configs  (near-rtt-v2-warmth-sparse-merged-dim14-ce-only.pt)
#   14-20 : gnn_wssm     x 7 configs  (near-rtt-v2-warmth-sparse-skew-merged-dim14-ce-only.pt)
#   21-27 : mlp_warmth   x 7 configs  (batch_edge_mlp_warmth_v2.pt)
#   28-34 : mlp_wsm      x 7 configs  (batch_edge_mlp_warmth_sparse_merged.pt)
#   35-41 : mlp_wssm     x 7 configs  (batch_edge_mlp_warmth_sparse_skew_merged_ce_reduced.pt)
#
# Knative/HRC/GNN-dim14-ce baselines are already done; referenced by compare script.
# Submit: sbatch --array=0-41 scripts_cosim/datalab/mega_compare_all7.sbatch
# Partial rerun: sbatch --array=<indices> scripts_cosim/datalab/mega_compare_all7.sbatch

set -euo pipefail

PROJECT_ROOT="/home/nikola.lukic/gnn-herosim"
cd "$PROJECT_ROOT"

export SWEEP_DIR="${SWEEP_DIR:-simulation_data/normal_sim_sweeps/mega_compare_all7_20260614}"
export WORKLOAD="${WORKLOAD:-data/nofs-ids/traces/workload-100-100.json}"

POLICIES=("gnn_warmth" "gnn_wsm" "gnn_wssm" "mlp_warmth" "mlp_wsm" "mlp_wssm")

CONFIGS=(
  "default_20_20_p50|simulation_data/space_with_network.json"
  "00_balanced_30_30_p35|simulation_data/normal_sim_sweeps/knative_network_20260606_192413/configs/00_balanced_30_30_p35.json"
  "01_balanced_40_40_p50|simulation_data/normal_sim_sweeps/knative_network_20260606_192413/configs/01_balanced_40_40_p50.json"
  "02_balanced_50_50_p60|simulation_data/normal_sim_sweeps/knative_network_20260606_192413/configs/02_balanced_50_50_p60.json"
  "03_client_heavy_50_35_p50|simulation_data/normal_sim_sweeps/knative_network_20260606_192413/configs/03_client_heavy_50_35_p50.json"
  "04_server_heavy_35_50_p50|simulation_data/normal_sim_sweeps/knative_network_20260606_192413/configs/04_server_heavy_35_50_p50.json"
  "05_sparse_40_40_p25|simulation_data/normal_sim_sweeps/knative_network_20260606_192413/configs/05_sparse_40_40_p25.json"
)

group=$(( SLURM_ARRAY_TASK_ID / 7 ))
cfg_idx=$(( SLURM_ARRAY_TASK_ID % 7 ))

policy="${POLICIES[$group]}"
entry="${CONFIGS[$cfg_idx]}"
name="${entry%%|*}"
path="${entry#*|}"

bash scripts_cosim/datalab/run_mega_compare_all7_one.sh "$policy" "$name" "$path"
