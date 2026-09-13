#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

LIVE_DS="$ROOT/simulation_data/artifacts/run_queue_big/gnn_datasets_live_150_150"
CACHE="$ROOT/simulation_data/artifacts/run_queue_big/graphs_cache_gnn_datasets_live_150_150_seq"
LOG="$ROOT/logs/live_finetune_pipeline.log"

mkdir -p "$(dirname "$LOG")"

{
  echo "=== Phase 1: Label live snapshots (2h budget, smallest combos first) ==="
  pipenv run python3 -u scripts_cosim/label_live_snapshots_for_training.py \
    --snapshots logs/live_oracle_audit/knative_batch_4candidate_150_150.jsonl \
    --output-dir "$LIVE_DS" \
    --config simulation_data/space_with_network.json \
    --sim-input data/nofs-ids \
    --seed 101 \
    --max-combos 200000 \
    --max-runtime-s 7200 \
    --sort-by-combos

  echo "=== Phase 2: Build sequential graph cache ==="
  pipenv run python3 -u src/notebooks/prepare_graphs_cache_seq.py \
    --base-dirs "$LIVE_DS" \
    --cache-dir "$CACHE" \
    --allow-missing-queue-data

  echo "=== Phase 3: Fine-tune from brisk-cosmos-41 ==="
  pipenv run python3 -u src/notebooks/train_live_finetune.py \
    --epochs 15 \
    --num-dataloader-workers 0

  echo "=== Live fine-tune pipeline complete ==="
} 2>&1 | tee "$LOG"
