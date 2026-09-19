#!/usr/bin/env bash
# topology_transfer_v1 Phase 4 gate, gnn_topo arm (network-entity-aware GNN; see
# LINEAGES.md and /root/.claude/plans/fair-enough-a-cheerful-raven.md).
# The original gate (run_topo_transfer_v1_phase4_gate.sh) never exercised
# use_network_entities=True on any config -- gnn_base/gnn_node are topology-blind
# bipartite GINs. This reruns pointwise (fresh comparison baseline) + gnn_topo only,
# same split/tier/epochs, so the result is directly comparable to the original 5 JSONs.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/nikola.lukic/gnn-herosim}"
cd "$PROJECT_ROOT"

SEEDS=(42 43 44 45 46)
IDX="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID not set — this script must run under sbatch --array}"
SEED="${SEEDS[$IDX]}"

CACHE_DIR="simulation_data/graphs_cache_topo_transfer_v1"
OUTPUT="simulation_data/topo_transfer_v1_phase4_topo_seed${SEED}.json"
ENV_NAME="${HEROSIM_ENV_NAME:-gnn}"

mkdir -p logs

if ! command -v micromamba >/dev/null 2>&1; then
  source "${HOME}/.bashrc"
fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate "${ENV_NAME}"

[[ -d "$CACHE_DIR" ]] || { echo "ERROR: missing cache dir ${CACHE_DIR}" >&2; exit 1; }

echo "=== topo_transfer_v1 phase4 gate (gnn_topo) seed=${SEED} (array idx=${IDX}) ==="

export PYTHONPATH="${PROJECT_ROOT}"

python3 scripts_cosim/gnn_necessity_ablation.py \
  --cache "$CACHE_DIR" \
  --corpus-root simulation_data \
  --split-mode topology_size --train-sizes 20 28 40 --held-out-sizes 60 80 \
  --epochs 120 --seed "$SEED" \
  --models pointwise gnn_topo \
  --power-tier tier_launch \
  --output "$OUTPUT"

[[ -f "$OUTPUT" ]] || { echo "ERROR: expected output missing: ${OUTPUT}" >&2; exit 1; }
echo "=== seed=${SEED} complete -> ${OUTPUT} ==="
