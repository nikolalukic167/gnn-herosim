#!/usr/bin/env bash
# Full 1060 warmth regen + graph recache + dim14-ce retrain pipeline.
# Run only after pilot_warmth_regen_audit.py passes.
#
# Each ds_* MUST retain placements/placements.jsonl (placement–RTT sweep). Not optional.
# docs/notes/placements_jsonl_required.md
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== Pilot audit ==="
pipenv run python3 scripts_cosim/pilot_warmth_regen_audit.py

echo "=== Full co-sim regen (1060 datasets, node_disk_v2) ==="
pipenv run python3 scripts_cosim/generate_gnn_datasets_fast.py \
  --max-datasets 1060 \
  --warmth-physics node_disk_v2 \
  --resume \
  --output-subdir gnn_datasets_4tasks_1060_warmth_v2

echo "=== Graph recache (dim14) ==="
PYTHONPATH="$(pwd)" pipenv run python3 src/notebooks/prepare_graphs_cache.py \
  --datasets-dir simulation_data/gnn_datasets_4tasks_1060_warmth_v2 \
  --cache-dir simulation_data/graphs_cache_gnn_datasets_4tasks_1060_warmth_v2

echo "=== Retrain dim14-ce from scratch ==="
export NEAR_RTT_CACHE_DIR="$(pwd)/simulation_data/graphs_cache_gnn_datasets_4tasks_1060_warmth_v2"
bash scripts_cosim/important/run_dim14_ce_only_train_and_sweep_nohup.sh

echo "Done. Update model path to new checkpoint before live sweeps."
