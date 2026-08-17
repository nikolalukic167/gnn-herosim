#!/usr/bin/env bash
# Submit 6-run sparse-merged ce-reduced skew3 sweep (3 configs × MLP + GNN).
set -euo pipefail

PROJECT_ROOT="/home/nikola.lukic/gnn-herosim"
cd "$PROJECT_ROOT"

export SWEEP_DIR="${SWEEP_DIR:-simulation_data/normal_sim_sweeps/warmth_sparse_merged_ce_reduced_skew3_20260611}"
export WORKLOAD="${WORKLOAD:-data/nofs-ids/traces/workload-100-100.json}"
export HEROSIM_WARMTH_PHYSICS=node_disk_v2
export GNN_MODEL="${GNN_MODEL:-models/near-rtt-v2-warmth-sparse-merged-ce-reduced.pt}"
export MLP_MODEL="${MLP_MODEL:-models/tabular/batch_edge_mlp_warmth_sparse_merged_ce_reduced.pt}"

PARTITION="${1:-GPU-l40s}"
GRES="${2:-gpu:l40s:1}"

for f in "$GNN_MODEL" "$MLP_MODEL" "$WORKLOAD" \
  simulation_data/space_with_network.json \
  simulation_data/normal_sim_sweeps/atomic21_skew_configs/default_20_20_degree_skew.json \
  simulation_data/normal_sim_sweeps/atomic21_skew_configs/05_sparse_40_40_p25_degree_skew.json; do
  if [[ ! -f "$f" ]]; then
    echo "ERROR: missing $f" >&2
    exit 1
  fi
done

mkdir -p logs "${SWEEP_DIR}/results"

echo "SWEEP_DIR=${SWEEP_DIR}"
echo "GNN_MODEL=${GNN_MODEL}"
echo "MLP_MODEL=${MLP_MODEL}"
echo "Partition: ${PARTITION} (${GRES})"
echo "User queue before submit:"
squeue -u "$(whoami)" -o "%.10i %.12P %.16j %.2t %R" || true
echo "Existing results:"
ls -1 "${SWEEP_DIR}/results/" 2>/dev/null || true

echo ""
echo "Test-only GPU array (6 jobs):"
sbatch --test-only --partition="$PARTITION" --gres="$GRES" --array=0-5 \
  --export=ALL,SWEEP_DIR,WORKLOAD,HEROSIM_WARMTH_PHYSICS,GNN_MODEL,MLP_MODEL \
  scripts_cosim/datalab/warmth_sparse_merged_ce_reduced_skew3_gpu.sbatch

gpu_job=$(sbatch \
  --job-name=wsmcr-gpu \
  --output="/home/nikola.lukic/gnn-herosim/logs/wsmcr-gpu-%A_%a.out" \
  --error="/home/nikola.lukic/gnn-herosim/logs/wsmcr-gpu-%A_%a.err" \
  --partition="$PARTITION" \
  --gres="$GRES" \
  --array=0-5 \
  --export=ALL,SWEEP_DIR,WORKLOAD,HEROSIM_WARMTH_PHYSICS,GNN_MODEL,MLP_MODEL \
  scripts_cosim/datalab/warmth_sparse_merged_ce_reduced_skew3_gpu.sbatch | awk '{print $NF}')

echo ""
echo "Submitted MLP/GNN array: ${gpu_job} (tasks 0-5)"
echo "Logs: logs/wsmcr-gpu-${gpu_job}_*.out"
sleep 3
echo "Queue after submit:"
squeue -u "$(whoami)" -n wsmcr-gpu -o "%.10i %.2t %.10M %R"
