#!/usr/bin/env bash
# contention_v2: train GNN + MLP from cache, then 3-config live gate (GNN vs MLP vs Knative).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}:${ROOT}/src/notebooks"

TS="$(date +%Y%m%d_%H%M%S)"
LOG="logs/contention_v2_train_live_gate_${TS}.log"
CACHE="simulation_data/graphs_cache_contention_v2"
GNN_CKPT="models/near-rtt-v2-contention-v2-dim14-ce-only.pt"
MLP_CKPT="models/tabular/batch_edge_mlp_contention_v2_dim22_batchcache.pt"
SWEEP_DIR="simulation_data/normal_sim_sweeps/contention_v2_live_gate_20260615"

log() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

mkdir -p logs models models/tabular src/notebooks/models

log "=== contention_v2 train + live gate pipeline ==="
log "Cache: ${CACHE}"
log "Sweep: ${SWEEP_DIR}"

n_graphs=$(${HEROSIM_PY:-pipenv run python3} -c "import pickle; print(len(pickle.load(open('${CACHE}/graphs.pkl','rb'))))")
log "Graphs in cache: ${n_graphs}"
if [[ "${n_graphs}" -lt 500 ]]; then
  log "ERROR: cache too small (${n_graphs} graphs)" >&2
  exit 1
fi

export NEAR_RTT_CACHE_DIR="$(pwd)/${CACHE}"
export WANDB_MODE=offline
export NEAR_RTT_TRAIN_EPOCHS="${NEAR_RTT_TRAIN_EPOCHS:-100}"

log "Phase 1: GNN dim14 CE-only train"
cd src/notebooks
if ! ${HEROSIM_PY:-pipenv run python3} train_near_rtt_v2_contention_v2_dim14_ce_only.py >> "${ROOT}/${LOG}" 2>&1; then
  log "ERROR: GNN training failed"
  exit 1
fi
cd "$ROOT"

NB_GNN="src/notebooks/models/near-rtt-v2-contention-v2-dim14-ce-only.pt"
if [[ -f "$NB_GNN" ]]; then
  cp -f "$NB_GNN" "$GNN_CKPT"
elif [[ -f "models/near-rtt-v2-contention-v2-dim14-ce-only.pt" ]]; then
  cp -f "models/near-rtt-v2-contention-v2-dim14-ce-only.pt" "$GNN_CKPT"
else
  log "ERROR: GNN checkpoint missing after training"
  exit 1
fi
log "GNN checkpoint: ${GNN_CKPT}"

log "Phase 2: MLP dim22 batchcache train"
if ! ${HEROSIM_PY:-pipenv run python3} src/notebooks/train_mlp_contention_v2_dim22_batchcache.py >> "$LOG" 2>&1; then
  log "ERROR: MLP training failed"
  exit 1
fi
[[ -f "$MLP_CKPT" ]] || { log "ERROR: MLP checkpoint missing: ${MLP_CKPT}"; exit 1; }
log "MLP checkpoint: ${MLP_CKPT}"

log "Phase 3: live gate (3 sparse configs × 3 policies)"
export GNN_MODEL="$GNN_CKPT"
export MLP_MODEL="$MLP_CKPT"
export SWEEP_DIR
export WORKLOAD="data/nofs-ids/traces/workload-125-225.json"

CONFIGS=(
  "sparse_p25|simulation_data/normal_sim_sweeps/knative_network_20260606_192413/configs/05_sparse_40_40_p25.json"
  "sparse_p35|simulation_data/normal_sim_sweeps/knative_network_20260606_192413/configs/00_balanced_30_30_p35.json"
  "sparse_p25_skew|simulation_data/normal_sim_sweeps/atomic21_skew_configs/05_sparse_40_40_p25_degree_skew.json"
)

for policy in knative mlp gnn; do
  for entry in "${CONFIGS[@]}"; do
    name="${entry%%|*}"
    path="${entry#*|}"
    log "  -> ${policy} ${name}"
    bash scripts_cosim/important/run_contention_v2_live_gate_one.sh "$policy" "$name" "$path" >> "$LOG" 2>&1
  done
done

log "Phase 4: compare"
${HEROSIM_PY:-pipenv run python3} scripts_cosim/important/compare_contention_v2_live_gate.py \
  --sweep-dir "$SWEEP_DIR" >> "$LOG" 2>&1

log "=== pipeline complete ==="
log "Log: ${LOG}"
