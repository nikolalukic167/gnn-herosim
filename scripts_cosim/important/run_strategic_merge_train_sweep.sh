#!/usr/bin/env bash
# Strategic merge: warmth + sparse + contention_v2 (coupled oversample, no v3/skew)
# → recache → GNN + MLP (wandb online) → deployment live gates (wssm + contention).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}:${ROOT}/src/notebooks"

PHASE_DIR="${ROOT}/logs/strategic_merge_pipeline"
mkdir -p "$PHASE_DIR" logs models models/tabular src/notebooks/models

TS="$(date +%Y%m%d)"
LOG="${PHASE_DIR}/pipeline_${TS}.log"
MANIFEST="simulation_data/strategic_merge_weights.json"
CACHE="simulation_data/graphs_cache_strategic_merge_wss_cont_v2"
GNN_CKPT="models/near-rtt-v2-strategic-merge-wss-cont-v2-dim14-ce-only.pt"
MLP_CKPT="models/tabular/batch_edge_mlp_strategic_merge_wss_cont_v2_dim22_batchcache.pt"
WSSM_SWEEP="simulation_data/normal_sim_sweeps/strategic_merge_wss_live_gate_${TS}"
CONT_SWEEP="simulation_data/normal_sim_sweeps/strategic_merge_contention_live_gate_${TS}"

WARMTH_DIR="simulation_data/gnn_datasets_4tasks_1060_warmth_v2"
SPARSE_DIR="simulation_data/gnn_datasets_4tasks_sparse_warmth_v2"
CONT_DIR="simulation_data/gnn_datasets_4tasks_contention_v2"

log() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }
phase_done() { touch "${PHASE_DIR}/phase_${1}.done"; }

export WANDB_MODE=online
export WANDB_PROJECT="${WANDB_PROJECT:-gnn-near-rtt-strategic-merge-jun2026}"
unset WANDB_DISABLED 2>/dev/null || true

log "=== strategic merge train+sweep (wandb=${WANDB_MODE} project=${WANDB_PROJECT}) ==="

if [[ ! -f "${PHASE_DIR}/phase_manifest.done" ]]; then
  log "Phase 0: build strategic merge manifest"
  pipenv run python3 scripts_cosim/build_strategic_merge_manifest.py \
    --out "$MANIFEST" >> "$LOG" 2>&1
  phase_done manifest
fi

if [[ ! -f "${PHASE_DIR}/phase_recache.done" ]]; then
  log "Phase 1: prepare_graphs_cache (strategic oversample)"
  PYTHONPATH="$(pwd)" pipenv run python3 src/notebooks/prepare_graphs_cache.py \
    --base-dirs "$WARMTH_DIR" "$SPARSE_DIR" "$CONT_DIR" \
    --cache-dir "$CACHE" \
    --oversample-manifest "$MANIFEST" >> "$LOG" 2>&1
  n_graphs=$(pipenv run python3 -c "import pickle; print(len(pickle.load(open('${CACHE}/graphs.pkl','rb'))))")
  log "  cache graphs: ${n_graphs}"
  [[ "${n_graphs}" -ge 1500 ]] || { log "ERROR: cache too small (${n_graphs})"; exit 1; }
  phase_done recache
fi

if [[ ! -f "${PHASE_DIR}/phase_train.done" ]]; then
  export NEAR_RTT_CACHE_DIR="$(pwd)/${CACHE}"
  export NEAR_RTT_TRAIN_EPOCHS="${NEAR_RTT_TRAIN_EPOCHS:-100}"

  if [[ ! -f "${PHASE_DIR}/phase_train_gnn.done" ]]; then
  log "Phase 2a: GNN dim14 CE-only"
  cd src/notebooks
  pipenv run python3 train_near_rtt_v2_strategic_merge_dim14_ce_only.py >> "${LOG}" 2>&1
  NB_GNN="models/near-rtt-v2-strategic-merge-wss-cont-v2-dim14-ce-only.pt"
  cp -f "$NB_GNN" "../../${GNN_CKPT}"
  cd "$ROOT"
  phase_done train_gnn
  fi

  log "Phase 2b: MLP dim22 batchcache"
  export WANDB_RUN_NAME="mlp-dim22-strategic-merge-wss-cont-v2-batchcache"
  export WANDB_TAGS="mlp,dim22,strategic-merge,warmth-v2,sparse-v2,contention-v2,deploy"
  pipenv run python3 src/notebooks/train_mlp_strategic_merge_dim22_batchcache.py >> "$LOG" 2>&1
  [[ -f "$MLP_CKPT" ]] || { log "ERROR: MLP missing"; exit 1; }
  phase_done train
fi

if [[ ! -f "${PHASE_DIR}/phase_sweep_wssm.done" ]]; then
  log "Phase 3: wssm deployment live gate"
  mkdir -p "${WSSM_SWEEP}/results"
  export SWEEP_DIR="$WSSM_SWEEP"
  export GNN_MODEL="$GNN_CKPT"
  export MLP_MODEL="$MLP_CKPT"
  export WORKLOAD="data/nofs-ids/traces/workload-125-225.json"
  CONFIGS=(
    "hub_k4_seek50|simulation_data/normal_sim_sweeps/sweep_bipartite_coordination_v1/configs/hub_k4_seek50.json"
    "hub_k6_seek50|simulation_data/normal_sim_sweeps/sweep_bipartite_coordination_v1/configs/hub_k6_seek50.json"
    "hub_k8_seek50|simulation_data/normal_sim_sweeps/sweep_bipartite_coordination_v1/configs/hub_k8_seek50.json"
  )
  failed=0
  for policy in knative mlp gnn; do
    for entry in "${CONFIGS[@]}"; do
      name="${entry%%|*}"; path="${entry#*|}"
      log "  wssm ${policy} ${name}"
      if ! bash scripts_cosim/important/run_wssm_expanded_live_gate_one.sh "$policy" "$name" "$path" >> "$LOG" 2>&1; then
        log "ERROR: wssm ${policy} ${name} failed"
        failed=$((failed + 1))
      fi
    done
  done
  n_ok=$(find "${WSSM_SWEEP}/results" -maxdepth 1 -name '*.json' ! -name '*.decode_stats.json' | wc -l)
  if [[ "$failed" -gt 0 || "$n_ok" -lt 9 ]]; then
    log "ERROR: wssm sweep incomplete (${n_ok}/9 results, ${failed} failures)"
    exit 1
  fi
  pipenv run python3 scripts_cosim/important/compare_wssm_expanded_live_gate.py \
    --sweep-dir "$WSSM_SWEEP" >> "$LOG" 2>&1
  phase_done sweep_wssm
fi

if [[ ! -f "${PHASE_DIR}/phase_sweep_contention.done" ]]; then
  log "Phase 4: contention deployment live gate"
  mkdir -p "${CONT_SWEEP}/results"
  export SWEEP_DIR="$CONT_SWEEP"
  export GNN_MODEL="$GNN_CKPT"
  export MLP_MODEL="$MLP_CKPT"
  export WORKLOAD="data/nofs-ids/traces/workload-125-225.json"
  CONFIGS=(
    "sparse_p25|simulation_data/normal_sim_sweeps/knative_network_20260606_192413/configs/05_sparse_40_40_p25.json"
    "sparse_p35|simulation_data/normal_sim_sweeps/knative_network_20260606_192413/configs/00_balanced_30_30_p35.json"
    "sparse_p25_skew|simulation_data/normal_sim_sweeps/atomic21_skew_configs/05_sparse_40_40_p25_degree_skew.json"
  )
  failed=0
  for policy in knative mlp gnn; do
    for entry in "${CONFIGS[@]}"; do
      name="${entry%%|*}"; path="${entry#*|}"
      log "  contention ${policy} ${name}"
      if ! bash scripts_cosim/important/run_contention_v2_live_gate_one.sh "$policy" "$name" "$path" >> "$LOG" 2>&1; then
        log "ERROR: contention ${policy} ${name} failed"
        failed=$((failed + 1))
      fi
    done
  done
  n_ok=$(find "${CONT_SWEEP}/results" -maxdepth 1 -name '*.json' | wc -l)
  if [[ "$failed" -gt 0 || "$n_ok" -lt 9 ]]; then
    log "ERROR: contention sweep incomplete (${n_ok}/9 results, ${failed} failures)"
    exit 1
  fi
  pipenv run python3 scripts_cosim/important/compare_contention_v2_live_gate.py \
    --sweep-dir "$CONT_SWEEP" >> "$LOG" 2>&1
  phase_done sweep_contention
fi

phase_done all
log "=== ALL COMPLETE ==="
log "Models: ${GNN_CKPT}, ${MLP_CKPT}"
log "Sweeps: ${WSSM_SWEEP}, ${CONT_SWEEP}"
log "Manifest: ${MANIFEST}"
log "Cache: ${CACHE}"
