#!/usr/bin/env bash
# Full pipeline: pull → audit → weighted merged cache → train → live gate (argmax_uniq GNN).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}:${ROOT}/src/notebooks"

TS="$(date +%Y%m%d_%H%M%S)"
LOG="logs/merged_contention_pipeline_${TS}.log"
WARMTH_DIR="simulation_data/gnn_datasets_4tasks_1060_warmth_v2"
SPARSE_DIR="simulation_data/gnn_datasets_4tasks_sparse_warmth_v2"
CONT_DIR="simulation_data/gnn_datasets_4tasks_contention_v2"
MANIFEST="simulation_data/oversample_warmth_sparse_contention_v2.json"
CACHE_DIR="simulation_data/graphs_cache_warmth_sparse_contention_v2_weighted"
GNN_CKPT="models/near-rtt-v2-warmth-sparse-contention-weighted-dim14-ce-only.pt"
MLP_CKPT="models/tabular/batch_edge_mlp_warmth_sparse_contention_v2_weighted_dim22_batchcache.pt"
SWEEP_DIR="simulation_data/normal_sim_sweeps/merged_contention_weighted_live_gate_${TS}"

log() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

mkdir -p logs models models/tabular

log "=== merged contention weighted pipeline ==="

log "Phase 0: pull warmth jsonl from datalab (if behind)"
warmth_jsonl=$(pipenv run python3 -c "from pathlib import Path; b=Path('$WARMTH_DIR'); print(sum(1 for d in b.glob('ds_*') if (d/'placements/placements.jsonl').stat().st_size>0 if (d/'placements/placements.jsonl').exists()))" 2>/dev/null || echo 0)
if [[ "${warmth_jsonl}" -lt 498 ]]; then
  rsync -az --include='*/' --include='placements/placements.jsonl' --include='optimal_result.json' --include='system_state_captured_unique.json' --exclude='*' \
    -e "ssh -o BatchMode=yes" datalab:/home/nikola.lukic/gnn-herosim/simulation_data/gnn_datasets_4tasks_1060_warmth_v2/ "$WARMTH_DIR/" 2>&1 | tee -a "$LOG"
fi

log "Phase 1: separability audit"
for corp in "$WARMTH_DIR" "$SPARSE_DIR" "$CONT_DIR"; do
  pipenv run python3 scripts_cosim/separability_diagnostic.py "$corp" 2>&1 | tee -a "$LOG"
done

log "Phase 2: oversample manifest"
pipenv run python3 scripts_cosim/build_coupled_oversample_manifest.py \
  --corpus "$WARMTH_DIR" --corpus "$SPARSE_DIR" --corpus "$CONT_DIR" \
  --coupled-threshold 0.01 --coupled-weight 8 --base-weight 1 \
  --output "$MANIFEST" 2>&1 | tee -a "$LOG"

log "Phase 3: refresh SSC (warmth only if needed) + build weighted cache"
pipenv run python3 scripts_cosim/refresh_optimal_full_stats.py --base-dir "$WARMTH_DIR" --rewrite-ssc 2>&1 | tail -3 | tee -a "$LOG" || true
rm -rf "$CACHE_DIR"
PYTHONPATH="$(pwd)" pipenv run python3 src/notebooks/prepare_graphs_cache.py \
  --base-dirs "$WARMTH_DIR" "$SPARSE_DIR" "$CONT_DIR" \
  --cache-dir "$CACHE_DIR" \
  --oversample-manifest "$MANIFEST" 2>&1 | tee -a "$LOG"

n_graphs=$(pipenv run python3 -c "import pickle; print(len(pickle.load(open('${CACHE_DIR}/graphs.pkl','rb'))))")
log "Cache built: ${n_graphs} graphs"

log "Phase 4: train GNN"
export NEAR_RTT_CACHE_DIR="$(pwd)/${CACHE_DIR}"
export WANDB_MODE=offline
cd src/notebooks
pipenv run python3 train_near_rtt_v2_warmth_sparse_contention_weighted_dim14_ce_only.py >> "${ROOT}/${LOG}" 2>&1
cd "$ROOT"
cp -f src/notebooks/models/near-rtt-v2-warmth-sparse-contention-weighted-dim14-ce-only.pt "$GNN_CKPT" 2>/dev/null || \
  cp -f models/near-rtt-v2-warmth-sparse-contention-weighted-dim14-ce-only.pt "$GNN_CKPT"
log "GNN -> ${GNN_CKPT}"

log "Phase 5: train MLP"
pipenv run python3 src/notebooks/train_mlp_warmth_sparse_contention_weighted_dim22_batchcache.py >> "$LOG" 2>&1
log "MLP -> ${MLP_CKPT}"

log "Phase 6: live gate (GNN argmax_uniq + baselines)"
export GNN_MODEL="$GNN_CKPT"
export MLP_MODEL="$MLP_CKPT"
export SWEEP_DIR
export WORKLOAD="data/nofs-ids/traces/workload-125-225.json"
export GNN_DECODE_MODE=argmax_uniq

CONFIGS=(
  "sparse_p25|simulation_data/normal_sim_sweeps/knative_network_20260606_192413/configs/05_sparse_40_40_p25.json"
  "sparse_p35|simulation_data/normal_sim_sweeps/knative_network_20260606_192413/configs/00_balanced_30_30_p35.json"
  "sparse_p25_skew|simulation_data/normal_sim_sweeps/atomic21_skew_configs/05_sparse_40_40_p25_degree_skew.json"
)

failed=0
for policy in knative mlp gnn; do
  for entry in "${CONFIGS[@]}"; do
    name="${entry%%|*}"; path="${entry#*|}"
    if [[ "$policy" == "gnn" ]]; then
      export GNN_DECODE_MODE=argmax_uniq
      suffix="gnn_uniq"
    else
      unset GNN_DECODE_MODE 2>/dev/null || true
      suffix=""
    fi
    log "  live ${policy} ${name}"
    if ! SWEEP_DIR="$SWEEP_DIR" bash scripts_cosim/important/run_merged_contention_live_gate_one.sh "$policy" "$name" "$path" "$suffix" >> "$LOG" 2>&1; then
      log "ERROR: live ${policy} ${name} (${suffix:-default}) failed"
      failed=$((failed + 1))
    fi
  done
done

# Also GNN argmax baseline for comparison
export GNN_DECODE_MODE=argmax
for entry in "${CONFIGS[@]}"; do
  name="${entry%%|*}"; path="${entry#*|}"
  log "  live gnn_argmax ${name}"
  if ! SWEEP_DIR="$SWEEP_DIR" bash scripts_cosim/important/run_merged_contention_live_gate_one.sh gnn "$name" "$path" "gnn_argmax" >> "$LOG" 2>&1; then
    log "ERROR: live gnn_argmax ${name} failed"
    failed=$((failed + 1))
  fi
done

n_ok=$(find "${SWEEP_DIR}/results" -maxdepth 1 -name '*.json' | wc -l)
if [[ "$failed" -gt 0 || "$n_ok" -lt 12 ]]; then
  log "ERROR: merged live gate incomplete (${n_ok}/12 results, ${failed} failures)"
  exit 1
fi

pipenv run python3 scripts_cosim/important/compare_merged_contention_live_gate.py --sweep-dir "$SWEEP_DIR" >> "$LOG" 2>&1

log "=== pipeline complete === sweep=${SWEEP_DIR} log=${LOG}"
