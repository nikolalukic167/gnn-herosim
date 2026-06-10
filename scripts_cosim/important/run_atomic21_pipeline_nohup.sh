#!/usr/bin/env bash
# Orchestrate atomic-21 experiment: cache -> parquet -> train GNN/MLP -> sweeps -> compare.
set -euo pipefail

ROOT="/root/projects/my-herosim"
cd "$ROOT"
export PYTHONPATH="${ROOT}:${ROOT}/src/notebooks"

TS="$(date +%Y%m%d)"
LOG="logs/atomic21_pipeline_${TS}.log"
CACHE_DIR="simulation_data/artifacts/run_queue_big/graphs_cache_gnn_datasets_4tasks_atomic21_seq"
BASE_DIRS="simulation_data/artifacts/run_queue_big/gnn_datasets_4tasks"
PARQUET="simulation_data/artifacts/tabular/batch_edges_atomic21.parquet"
GNN_MODEL="models/near-rtt-v2-atomic21-ce-only.pt"
MLP_MODEL="models/tabular/batch_edge_mlp_atomic21.pt"
SWEEP_OUT="simulation_data/normal_sim_sweeps/gnn_near_rtt_v2_atomic21_ce_only_${TS}"

log() {
  echo "[$(date -Is)] $*" | tee -a "$LOG"
}

mkdir -p logs simulation_data/artifacts/tabular models models/tabular

log "=== Atomic-21 pipeline start ==="

log "Step 1: rebuild graph cache"
pipenv run python3 src/notebooks/prepare_graphs_cache_seq.py \
  --cache-dir "$CACHE_DIR" \
  --base-dirs "$BASE_DIRS" \
  2>&1 | tee -a "$LOG"

log "Step 2: tabular parquet"
pipenv run python3 src/notebooks/prepare_tabular_dataset.py \
  --cache-dir "$CACHE_DIR" \
  --output "$PARQUET" \
  --regime batch \
  2>&1 | tee -a "$LOG"

log "Step 3: train GNN"
cd src/notebooks
pipenv run python3 train_near_rtt_v2_atomic21_ce_only.py \
  2>&1 | tee -a "../../${LOG}"
cd "$ROOT"
cp "src/notebooks/models/near-rtt-v2-atomic21-ce-only.pt" "$GNN_MODEL" 2>/dev/null || true

log "Step 4: train MLP"
pipenv run python3 -m src.policy.tabular.train_mlp \
  --input "$PARQUET" \
  --output "$MLP_MODEL" \
  --epochs 100 --patience 10 --hidden-dim 64 \
  2>&1 | tee -a "$LOG"

log "Step 5: GNN sweep"
bash scripts_cosim/important/run_gnn_near_rtt_v2_atomic21_ce_only_skew_sweep_nohup.sh "$SWEEP_OUT" \
  2>&1 | tee -a "$LOG"

log "Step 6: MLP sweep"
bash scripts_cosim/important/run_mlp_atomic21_skew_sweep_nohup.sh "$SWEEP_OUT" \
  2>&1 | tee -a "$LOG"

log "Step 7: compare"
pipenv run python3 scripts_cosim/important/compare_atomic21_gnn_mlp_sweep.py \
  --sweep-dir "$SWEEP_OUT" \
  2>&1 | tee -a "$LOG"

log "=== Atomic-21 pipeline complete ==="
