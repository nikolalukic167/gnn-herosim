#!/usr/bin/env bash
# Submit strategic-merge + weighted-merge live gate SLURM arrays on datalab.
# Usage:
#   bash scripts_cosim/datalab/submit_strategic_merge_live_gates.sh
#   PARTITION=GPU-a40 GRES=gpu:a40:1 bash scripts_cosim/datalab/submit_strategic_merge_live_gates.sh
set -euo pipefail

PROJECT_ROOT="/home/nikola.lukic/gnn-herosim"
cd "$PROJECT_ROOT"

PARTITION="${PARTITION:-GPU-l40s}"
GRES="${GRES:-gpu:l40s:1}"
export TIMEOUT="${TIMEOUT:-18000}"
export WORKLOAD="${WORKLOAD:-data/nofs-ids/traces/workload-125-225.json}"

export SWEEP_DIR="simulation_data/normal_sim_sweeps/strategic_merge_wss_live_gate_20260616"
wssm_job=$(sbatch \
  --partition="$PARTITION" --gres="$GRES" --export=ALL \
  scripts_cosim/datalab/strategic_merge_wss_live_gate_gpu.sbatch | awk '{print $NF}')
echo "WSSM gate:        ${wssm_job} (array 0-8, skip-if-exists)"

export SWEEP_DIR="simulation_data/normal_sim_sweeps/strategic_merge_contention_live_gate_20260616"
export GNN_MODEL="models/near-rtt-v2-strategic-merge-wss-cont-v2-dim14-ce-only.pt"
export MLP_MODEL="models/tabular/batch_edge_mlp_strategic_merge_wss_cont_v2_dim22_batchcache.pt"
cont_job=$(sbatch \
  --partition="$PARTITION" --gres="$GRES" --export=ALL \
  scripts_cosim/datalab/strategic_merge_contention_live_gate_gpu.sbatch | awk '{print $NF}')
echo "Strategic cont:   ${cont_job} (array 0-8, skip-if-exists)"

export SWEEP_DIR="simulation_data/normal_sim_sweeps/merged_contention_weighted_live_gate_20260616_105500"
export GNN_MODEL="models/near-rtt-v2-warmth-sparse-contention-weighted-dim14-ce-only.pt"
export MLP_MODEL="models/tabular/batch_edge_mlp_warmth_sparse_contention_v2_weighted_dim22_batchcache.pt"
wt_job=$(sbatch \
  --partition="$PARTITION" --gres="$GRES" --export=ALL \
  scripts_cosim/datalab/merged_contention_weighted_live_gate_gpu.sbatch | awk '{print $NF}')
echo "Weighted cont:    ${wt_job} (array 0-11, skip-if-exists)"

echo ""
echo "Monitor: squeue -u nikola.lukic"
echo "Logs:    logs/sm-wssm-gate-*  logs/sm-cont-gate-*  logs/wt-cont-gate-*"
echo "Compare when done:"
echo "  python3 scripts_cosim/important/compare_wssm_expanded_live_gate.py --sweep-dir simulation_data/normal_sim_sweeps/strategic_merge_wss_live_gate_20260616"
echo "  python3 scripts_cosim/important/compare_contention_v2_live_gate.py --sweep-dir simulation_data/normal_sim_sweeps/strategic_merge_contention_live_gate_20260616"
echo "  python3 scripts_cosim/important/compare_merged_contention_live_gate.py --sweep-dir simulation_data/normal_sim_sweeps/merged_contention_weighted_live_gate_20260616_105500"
