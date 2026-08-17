#!/usr/bin/env bash
# GNN live sim with seqblend+1 decode (override min-queue only if chosen_queue > min_queue + 1).
# Classic seqblend (any queue above min): GNN_SEQBLEND_QUEUE_MARGIN=0
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODEL="${GNN_MODEL_PATH:-models/good-plasma-43.pt}"
TAG="${1:-good-plasma-43}"
OUT="simulation_data/results/150-150/simulation_result_gnn_seqblend_p1_${TAG}_150-150.json"
LOG="logs/gnn_seqblend_p1_${TAG}_150-150.nohup.log"
STATS="${OUT%.json}.decode_stats.json"

mkdir -p logs simulation_data/results/150-150

GNN_CAPTURE_DATASET_STATE=0 \
GNN_DECODE_MODE=seqblend \
GNN_SEQBLEND_QUEUE_MARGIN=1 \
GNN_MODEL_PATH="$MODEL" \
nohup pipenv run python3 -u -m src.executesimulation \
  --config simulation_data/space_with_network.json \
  --workload data/nofs-ids/traces/workload-150-150.json \
  --policy gnn \
  --seed 101 \
  --output "$OUT" \
  > "$LOG" 2>&1 &

echo $! > "${LOG%.log}.pid"
echo "Started seqblend+1 GNN 150-150 (pid $(cat "${LOG%.log}.pid"))"
echo "  model : $MODEL"
echo "  output: $OUT"
echo "  stats : $STATS"
echo "  log   : $LOG"
echo "  rule  : override to min-queue iff live_queue(gnn_pick) > min_queue(candidates) + 1"
