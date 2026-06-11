#!/usr/bin/env bash
# Submit 9-run warmth-v2 skew3 sweep: 3 Knative (no GPU) + 6 MLP/GNN (1 GPU each).
# Uses GPU-a40 to avoid GPU-a100s warmth-regen CPU jobs and GPU-l40s saturation.
set -euo pipefail

PROJECT_ROOT="/home/nikola.lukic/gnn-herosim"
cd "$PROJECT_ROOT"

export SWEEP_DIR="${SWEEP_DIR:-simulation_data/normal_sim_sweeps/warmth_v2_physics_skew3_20260611}"
export WORKLOAD="${WORKLOAD:-data/nofs-ids/traces/workload-100-100.json}"
export HEROSIM_WARMTH_PHYSICS=node_disk_v2

PARTITION="${1:-GPU-a40}"
GRES="${2:-gpu:a40:1}"

mkdir -p logs "${SWEEP_DIR}/results"

echo "SWEEP_DIR=${SWEEP_DIR}"
echo "Partition: ${PARTITION} (Knative=no GPU, MLP/GNN=${GRES})"
echo "Existing results:"
ls -1 "${SWEEP_DIR}/results/" 2>/dev/null || true

echo ""
echo "Test-only Knative array (3 jobs, 4 CPU, no GPU):"
sbatch --test-only --partition="$PARTITION" --array=0-2 \
  --export=ALL,SWEEP_DIR,WORKLOAD,HEROSIM_WARMTH_PHYSICS \
  scripts_cosim/datalab/warmth_v2_physics_skew3_knative.sbatch

echo ""
echo "Test-only GPU array (6 jobs, 4 CPU, 1 GPU):"
sbatch --test-only --partition="$PARTITION" --gres="$GRES" --array=0-5 \
  --export=ALL,SWEEP_DIR,WORKLOAD,HEROSIM_WARMTH_PHYSICS \
  scripts_cosim/datalab/warmth_v2_physics_skew3_gpu.sbatch

kn_job=$(sbatch \
  --job-name=wv2-kn \
  --output="/home/nikola.lukic/gnn-herosim/logs/wv2-kn-%A_%a.out" \
  --error="/home/nikola.lukic/gnn-herosim/logs/wv2-kn-%A_%a.err" \
  --partition="$PARTITION" \
  --array=0-2 \
  --export=ALL,SWEEP_DIR,WORKLOAD,HEROSIM_WARMTH_PHYSICS \
  scripts_cosim/datalab/warmth_v2_physics_skew3_knative.sbatch | awk '{print $NF}')

gpu_job=$(sbatch \
  --job-name=wv2-gpu \
  --output="/home/nikola.lukic/gnn-herosim/logs/wv2-gpu-%A_%a.out" \
  --error="/home/nikola.lukic/gnn-herosim/logs/wv2-gpu-%A_%a.err" \
  --partition="$PARTITION" \
  --gres="$GRES" \
  --array=0-5 \
  --export=ALL,SWEEP_DIR,WORKLOAD,HEROSIM_WARMTH_PHYSICS \
  scripts_cosim/datalab/warmth_v2_physics_skew3_gpu.sbatch | awk '{print $NF}')

echo ""
echo "Submitted Knative array: ${kn_job} (tasks 0-2)"
echo "Submitted MLP/GNN array: ${gpu_job} (tasks 0-5)"
echo "Logs: logs/wv2-kn-${kn_job}_*.out · logs/wv2-gpu-${gpu_job}_*.out"
echo "Monitor: squeue -u \$USER -n wv2-kn,wv2-gpu"
