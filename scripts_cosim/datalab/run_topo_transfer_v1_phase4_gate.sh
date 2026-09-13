#!/usr/bin/env bash
# topology_transfer_v1 Phase 4 pre-registered gate run (see LINEAGES.md).
# One SLURM array task per seed (42..46). Trains pointwise/gnn_base/gnn_node,
# split-mode topology_size (train sizes 20/28/40, held-out 60/80), tier_launch.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/nikola.lukic/gnn-herosim}"
cd "$PROJECT_ROOT"

SEEDS=(42 43 44 45 46)
IDX="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID not set — this script must run under sbatch --array}"
SEED="${SEEDS[$IDX]}"

CACHE_DIR="simulation_data/graphs_cache_topo_transfer_v1"
OUTPUT="simulation_data/topo_transfer_v1_phase4_seed${SEED}.json"
ENV_NAME="${HEROSIM_ENV_NAME:-gnn}"

mkdir -p logs

if ! command -v micromamba >/dev/null 2>&1; then
  source "${HOME}/.bashrc"
fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate "${ENV_NAME}"

[[ -d "$CACHE_DIR" ]] || { echo "ERROR: missing cache dir ${CACHE_DIR}" >&2; exit 1; }

echo "=== topo_transfer_v1 phase4 gate seed=${SEED} (array idx=${IDX}) ==="

export PYTHONPATH="${PROJECT_ROOT}"

python3 scripts_cosim/gnn_necessity_ablation.py \
  --cache "$CACHE_DIR" \
  --corpus-root simulation_data \
  --split-mode topology_size --train-sizes 20 28 40 --held-out-sizes 60 80 \
  --epochs 120 --seed "$SEED" \
  --models pointwise gnn_base gnn_node \
  --power-tier tier_launch \
  --output "$OUTPUT"

[[ -f "$OUTPUT" ]] || { echo "ERROR: expected output missing: ${OUTPUT}" >&2; exit 1; }
echo "=== seed=${SEED} complete -> ${OUTPUT} ==="
