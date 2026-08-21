#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

AUDIT_DIR="$ROOT/logs/live_oracle_audit"
mkdir -p "$AUDIT_DIR"

echo "=== Cleaning prior audit artifacts ==="
rm -f "$AUDIT_DIR"/*.jsonl "$AUDIT_DIR"/*.csv "$AUDIT_DIR"/capture_run.log "$AUDIT_DIR"/cosim_audit_run.log "$AUDIT_DIR"/capture_sim_results.json 2>/dev/null || true

echo "=== Phase 1: Capture live snapshots (normal 150-150, sparse network) ==="
echo "  Filters: batch size >= 4, each task >= 4 candidates"
export LIVE_AUDIT_SNAPSHOT_PATH="$AUDIT_DIR/knative_batch_4candidate_150_150.jsonl"
export LIVE_AUDIT_MAX_SNAPSHOTS=500
export LIVE_AUDIT_MIN_BATCH_SIZE=4
export LIVE_AUDIT_MIN_CANDIDATES=4
export KNATIVE_BATCH_SIZE=4
export GNN_CAPTURE_DATASET_STATE=0

${HEROSIM_PY:-pipenv run python3} -m src.executesimulation \
  --config simulation_data/space_with_network.json \
  --workload data/nofs-ids/traces/workload-150-150.json \
  --policy knative_network_batch \
  --seed 101 \
  --output "$AUDIT_DIR/capture_sim_results.json" \
  2>&1 | tee "$AUDIT_DIR/capture_run.log"

SNAP_COUNT="$(wc -l < "$LIVE_AUDIT_SNAPSHOT_PATH" 2>/dev/null || echo 0)"
echo "Captured ${SNAP_COUNT} qualifying snapshots"
if [ "${SNAP_COUNT}" -eq 0 ]; then
  echo "ERROR: no qualifying snapshots captured on normal 150-150 sim"
  exit 1
fi

echo "=== Phase 2+3: Co-sim brute-force oracle + Knative/HRC/GNN comparison ==="
${HEROSIM_PY:-pipenv run python3} scripts_cosim/live_snapshot_oracle_audit.py \
  --snapshots "$LIVE_AUDIT_SNAPSHOT_PATH" \
  --output "$AUDIT_DIR/knative_batch_4candidate_150_150_cosim.csv" \
  --config simulation_data/space_with_network.json \
  --sim-input data/nofs-ids \
  --seed 101 \
  --max-combos 300000 \
  --max-runtime-s 7200 \
  --gnn-model src/notebooks/models/brisk-cosmos-41.pt \
  --progress 5 \
  2>&1 | tee "$AUDIT_DIR/cosim_audit_run.log"

echo "=== Pipeline complete ==="
