#!/usr/bin/env bash
# contention_v3: recache → train GNN+MLP → 3-config live gate → compare (local pipenv).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}:${ROOT}/src/notebooks"

TS="$(date +%Y%m%d_%H%M%S)"
LOG="logs/contention_v3_train_live_gate_${TS}.log"
CORPUS="simulation_data/gnn_datasets_4tasks_contention_v3"
CACHE="simulation_data/graphs_cache_contention_v3"
GNN_CKPT="models/near-rtt-v2-contention-v3-dim14-ce-only.pt"
MLP_CKPT="models/tabular/batch_edge_mlp_contention_v3_dim22_batchcache.pt"
SWEEP_DIR="simulation_data/normal_sim_sweeps/contention_v3_live_gate_${TS}"

log() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

mkdir -p logs models models/tabular src/notebooks/models

log "=== contention_v3 train + live gate (local) ==="
log "Corpus: ${CORPUS}  Cache: ${CACHE}  Sweep: ${SWEEP_DIR}"

jsonl=$(find "$CORPUS" -path '*/placements/placements.jsonl' -size +0 2>/dev/null | wc -l)
if [[ "$jsonl" -lt 850 ]]; then
  log "ERROR: corpus incomplete (${jsonl} jsonl)" >&2
  exit 1
fi

if [[ ! -f "${CACHE}/graphs.pkl" || "${FORCE_RECACHE:-0}" == "1" ]]; then
  log "Phase 1: refresh SSC + prepare_graphs_cache"
  rm -rf "$CACHE"
  PYTHONPATH="$(pwd)" ${HEROSIM_PY:-pipenv run python3} scripts_cosim/refresh_optimal_full_stats.py \
    --base-dir "$CORPUS" --rewrite-ssc >> "$LOG" 2>&1
  PYTHONPATH="$(pwd)" ${HEROSIM_PY:-pipenv run python3} src/notebooks/prepare_graphs_cache.py \
    --base-dirs "$CORPUS" --cache-dir "$CACHE" >> "$LOG" 2>&1
fi

n_graphs=$(${HEROSIM_PY:-pipenv run python3} -c "import pickle; print(len(pickle.load(open('${CACHE}/graphs.pkl','rb'))))")
log "Graphs: ${n_graphs}"
[[ "$n_graphs" -ge 850 ]] || { log "ERROR: cache too small"; exit 1; }

export NEAR_RTT_CACHE_DIR="$(pwd)/${CACHE}"
export WANDB_MODE=offline
export NEAR_RTT_TRAIN_EPOCHS="${NEAR_RTT_TRAIN_EPOCHS:-100}"

log "Phase 2: GNN train"
cd src/notebooks
${HEROSIM_PY:-pipenv run python3} train_near_rtt_v2_contention_v3_dim14_ce_only.py >> "${ROOT}/${LOG}" 2>&1
cd "$ROOT"
cp -f src/notebooks/models/near-rtt-v2-contention-v3-dim14-ce-only.pt "$GNN_CKPT" 2>/dev/null || true
[[ -f "$GNN_CKPT" ]] || { log "ERROR: GNN ckpt missing"; exit 1; }

log "Phase 3: MLP train"
${HEROSIM_PY:-pipenv run python3} src/notebooks/train_mlp_contention_v3_dim22_batchcache.py >> "$LOG" 2>&1
[[ -f "$MLP_CKPT" ]] || { log "ERROR: MLP ckpt missing"; exit 1; }

log "Phase 4: live gate"
export GNN_MODEL="$GNN_CKPT"
export MLP_MODEL="$MLP_CKPT"
export SWEEP_DIR
export WORKLOAD="data/nofs-ids/traces/workload-125-225.json"
export TIMEOUT="${TIMEOUT:-18000}"

CONFIGS=(
  "sparse_p25|simulation_data/normal_sim_sweeps/knative_network_20260606_192413/configs/05_sparse_40_40_p25.json"
  "sparse_p35|simulation_data/normal_sim_sweeps/knative_network_20260606_192413/configs/00_balanced_30_30_p35.json"
  "sparse_p25_skew|simulation_data/normal_sim_sweeps/atomic21_skew_configs/05_sparse_40_40_p25_degree_skew.json"
)

for policy in knative mlp gnn; do
  for entry in "${CONFIGS[@]}"; do
    name="${entry%%|*}"; path="${entry#*|}"
    log "  -> ${policy} ${name}"
    bash scripts_cosim/important/run_contention_v2_live_gate_one.sh "$policy" "$name" "$path" >> "$LOG" 2>&1
  done
done

log "Phase 5: compare"
${HEROSIM_PY:-pipenv run python3} scripts_cosim/important/compare_contention_v2_live_gate.py \
  --sweep-dir "$SWEEP_DIR" >> "$LOG" 2>&1

log "=== complete === sweep=${SWEEP_DIR} log=${LOG}"
