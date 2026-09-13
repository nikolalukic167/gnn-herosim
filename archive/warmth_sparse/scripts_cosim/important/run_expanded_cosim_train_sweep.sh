#!/usr/bin/env bash
# Expanded co-sim: recache → train GNN+MLP → live gates (warmth/sparse/wssm + contention_v2).
# Idempotent; safe to re-run. Writes phase markers under logs/expanded_cosim_pipeline/.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}:${ROOT}/src/notebooks"

PHASE_DIR="${ROOT}/logs/expanded_cosim_pipeline"
mkdir -p "$PHASE_DIR" logs models models/tabular src/notebooks/models

TS="$(date +%Y%m%d)"
LOG="${PHASE_DIR}/pipeline_${TS}.log"
STATE="${PHASE_DIR}/state.env"

WARMTH_DIR="simulation_data/gnn_datasets_4tasks_1060_warmth_v2"
SPARSE_DIR="simulation_data/gnn_datasets_4tasks_sparse_warmth_v2"
SKEW_DIR="simulation_data/gnn_datasets_4tasks_skew_warmth_v2"
CONT_DIR="simulation_data/gnn_datasets_4tasks_contention_v2"

WSSM_CACHE="simulation_data/graphs_cache_warmth_v2_sparse_skew_merged"
CONT_CACHE="simulation_data/graphs_cache_contention_v2"

GNN_WSSM="models/near-rtt-v2-warmth-sparse-skew-merged-dim14-ce-only.pt"
MLP_WSSM="models/tabular/batch_edge_mlp_warmth_sparse_skew_merged_dim22_batchcache.pt"
GNN_CONT="models/near-rtt-v2-contention-v2-dim14-ce-only.pt"
MLP_CONT="models/tabular/batch_edge_mlp_contention_v2_dim22_batchcache.pt"

WSSM_SWEEP="simulation_data/normal_sim_sweeps/wssm_expanded_nu_live_gate_${TS}"
CONT_SWEEP="simulation_data/normal_sim_sweeps/contention_v2_expanded_live_gate_${TS}"

log() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }
phase_done() { touch "${PHASE_DIR}/phase_${1}.done"; echo "PHASE_${1}_DONE" >> "$STATE"; }
phase_ok() { [[ -f "${PHASE_DIR}/phase_${1}.done" ]]; }

log "=== expanded co-sim train+sweep pipeline ==="
log "Log: ${LOG}"

# --- Phase 0: verify rsync / non-unique labels ---
if ! phase_ok rsync; then
  log "Phase 0: verify expanded corpora on disk"
  warmth_jsonl=$(find "$WARMTH_DIR" -name placements.jsonl 2>/dev/null | wc -l)
  sparse_jsonl=$(find "$SPARSE_DIR" -name placements.jsonl 2>/dev/null | wc -l)
  cont_jsonl=$(find "$CONT_DIR" -name placements.jsonl 2>/dev/null | wc -l)
  log "  warmth jsonl=${warmth_jsonl} sparse=${sparse_jsonl} contention=${cont_jsonl}"

  nu_check=$(pipenv run python3 -c "
import json
from pathlib import Path
p = Path('${WARMTH_DIR}/ds_00000/placements/placements.jsonl')
if not p.exists():
    print('MISSING')
else:
    n = sum(1 for _ in open(p))
    m = json.loads(Path('${WARMTH_DIR}/ds_00000/placement_metadata.json').read_text())
    print(f'{n}|{m.get(\"non_unique_placements\",0)}')
" 2>/dev/null || echo "MISSING")

  if [[ "$nu_check" == "MISSING" ]] || [[ "${nu_check%%|*}" -lt 2000 ]]; then
    log "ERROR: warmth ds_00000 not cartesian-expanded (got ${nu_check}). Run transfer_expanded_cosim_from_datalab.sh first." >&2
    exit 2
  fi
  if [[ "$cont_jsonl" -lt 850 ]]; then
    log "ERROR: contention_v2 jsonl=${cont_jsonl} < 850. Pull from datalab first." >&2
    exit 2
  fi
  phase_done rsync
fi

# --- Phase 1: SSC refresh ---
if ! phase_ok ssc; then
  log "Phase 1: refresh_optimal_full_stats --rewrite-ssc"
  for dir in "$WARMTH_DIR" "$SPARSE_DIR" "$SKEW_DIR" "$CONT_DIR"; do
    log "  SSC ${dir##*/}"
    pipenv run python3 scripts_cosim/refresh_optimal_full_stats.py \
      --base-dir "$dir" --rewrite-ssc >> "$LOG" 2>&1
  done
  phase_done ssc
fi

# --- Phase 2: recache ---
if ! phase_ok recache; then
  log "Phase 2a: prepare_graphs_cache wssm merged"
  PYTHONPATH="$(pwd)" pipenv run python3 src/notebooks/prepare_graphs_cache.py \
    --base-dirs "$WARMTH_DIR" "$SPARSE_DIR" "$SKEW_DIR" \
    --cache-dir "$WSSM_CACHE" >> "$LOG" 2>&1

  log "Phase 2b: prepare_graphs_cache contention_v2"
  PYTHONPATH="$(pwd)" pipenv run python3 src/notebooks/prepare_graphs_cache.py \
    --base-dirs "$CONT_DIR" \
    --cache-dir "$CONT_CACHE" >> "$LOG" 2>&1

  wssm_n=$(pipenv run python3 -c "import pickle; print(len(pickle.load(open('${WSSM_CACHE}/graphs.pkl','rb'))))")
  cont_n=$(pipenv run python3 -c "import pickle; print(len(pickle.load(open('${CONT_CACHE}/graphs.pkl','rb'))))")
  log "  wssm cache: ${wssm_n} graphs; contention cache: ${cont_n} graphs"
  if [[ "$wssm_n" -lt 800 || "$cont_n" -lt 850 ]]; then
    log "ERROR: recache too small (wssm=${wssm_n} cont=${cont_n})" >&2
    exit 1
  fi
  phase_done recache
fi

# --- Phase 3: train GNN + MLP ---
if ! phase_ok train; then
  export NEAR_RTT_CACHE_DIR="$(pwd)/${WSSM_CACHE}"
  export WANDB_MODE=offline
  export NEAR_RTT_TRAIN_EPOCHS="${NEAR_RTT_TRAIN_EPOCHS:-100}"

  log "Phase 3a: GNN wssm dim14 CE-only"
  cd src/notebooks
  pipenv run python3 train_near_rtt_v2_warmth_sparse_skew_merged_dim14_ce_only.py >> "${LOG}" 2>&1
  cp -f models/near-rtt-v2-warmth-sparse-skew-merged-dim14-ce-only.pt "../../${GNN_WSSM}"
  cd "$ROOT"

  log "Phase 3b: MLP wssm dim22 batchcache"
  export NEAR_RTT_CACHE_DIR="$(pwd)/${WSSM_CACHE}"
  pipenv run python3 src/notebooks/train_mlp_v2_warmth_sparse_skew_merged_dim22_batchcache.py >> "$LOG" 2>&1
  [[ -f "$MLP_WSSM" ]] || { log "ERROR: MLP wssm missing"; exit 1; }

  log "Phase 3c: GNN contention_v2 dim14 CE-only"
  export NEAR_RTT_CACHE_DIR="$(pwd)/${CONT_CACHE}"
  cd src/notebooks
  pipenv run python3 train_near_rtt_v2_contention_v2_dim14_ce_only.py >> "${LOG}" 2>&1
  cp -f models/near-rtt-v2-contention-v2-dim14-ce-only.pt "../../${GNN_CONT}"
  cd "$ROOT"

  log "Phase 3d: MLP contention_v2 dim22 batchcache"
  export NEAR_RTT_CACHE_DIR="$(pwd)/${CONT_CACHE}"
  pipenv run python3 src/notebooks/train_mlp_contention_v2_dim22_batchcache.py >> "$LOG" 2>&1
  [[ -f "$MLP_CONT" ]] || { log "ERROR: MLP contention missing"; exit 1; }

  phase_done train
fi

# --- Phase 4: wssm live gate (3 bipartite hubs × 3 policies) ---
if ! phase_ok sweep_wssm; then
  log "Phase 4: wssm expanded live gate → ${WSSM_SWEEP}"
  mkdir -p "${WSSM_SWEEP}/results" "${WSSM_SWEEP}/logs"
  export SWEEP_DIR="$WSSM_SWEEP"
  export GNN_MODEL="$GNN_WSSM"
  export MLP_MODEL="$MLP_WSSM"
  export WORKLOAD="data/nofs-ids/traces/workload-125-225.json"
  export QUEUE_LENGTH="${QUEUE_LENGTH:-100}"

  CONFIGS=(
    "hub_k4_seek50|simulation_data/normal_sim_sweeps/sweep_bipartite_coordination_v1/configs/hub_k4_seek50.json"
    "hub_k6_seek50|simulation_data/normal_sim_sweeps/sweep_bipartite_coordination_v1/configs/hub_k6_seek50.json"
    "hub_k8_seek50|simulation_data/normal_sim_sweeps/sweep_bipartite_coordination_v1/configs/hub_k8_seek50.json"
  )

  for policy in knative mlp gnn; do
    for entry in "${CONFIGS[@]}"; do
      name="${entry%%|*}"
      path="${entry#*|}"
      log "  wssm ${policy} ${name}"
      bash scripts_cosim/important/run_wssm_expanded_live_gate_one.sh "$policy" "$name" "$path" >> "$LOG" 2>&1
    done
  done

  pipenv run python3 scripts_cosim/important/compare_wssm_expanded_live_gate.py \
    --sweep-dir "$WSSM_SWEEP" >> "$LOG" 2>&1 || true

  phase_done sweep_wssm
fi

# --- Phase 5: contention live gate ---
if ! phase_ok sweep_contention; then
  log "Phase 5: contention_v2 expanded live gate → ${CONT_SWEEP}"
  mkdir -p "${CONT_SWEEP}/results"
  export SWEEP_DIR="$CONT_SWEEP"
  export GNN_MODEL="$GNN_CONT"
  export MLP_MODEL="$MLP_CONT"
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
      log "  contention ${policy} ${name}"
      bash scripts_cosim/important/run_contention_v2_live_gate_one.sh "$policy" "$name" "$path" >> "$LOG" 2>&1
    done
  done

  pipenv run python3 scripts_cosim/important/compare_contention_v2_live_gate.py \
    --sweep-dir "$CONT_SWEEP" >> "$LOG" 2>&1

  phase_done sweep_contention
fi

log "=== ALL PHASES COMPLETE ==="
log "Models: ${GNN_WSSM}, ${MLP_WSSM}, ${GNN_CONT}, ${MLP_CONT}"
log "Sweeps: ${WSSM_SWEEP}, ${CONT_SWEEP}"
phase_done all
