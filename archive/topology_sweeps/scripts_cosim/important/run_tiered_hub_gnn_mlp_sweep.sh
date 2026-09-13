#!/usr/bin/env bash
# Tiered-hub matrix: 9 configs × 5 policies = 45 runs (local sequential).
set -euo pipefail

ROOT="/root/projects/my-herosim"
cd "$ROOT"

SWEEP_DIR="${1:-simulation_data/normal_sim_sweeps/tiered_hub_gnn_mlp_20260610}"
CFG_DIR="${SWEEP_DIR}/configs"
RES_DIR="${SWEEP_DIR}/results"
JOBS_TSV="${CFG_DIR}/jobs.tsv"
TS="$(basename "$SWEEP_DIR" | grep -oE '[0-9]{8}(_[0-9]{6})?' | tail -1 || date +%Y%m%d_%H%M%S)"
PROGRESS_LOG="logs/tiered_hub_gnn_mlp_sweep_${TS}.log"

WORKLOAD="data/nofs-ids/traces/workload-100-100.json"
GNN_MODEL="models/near-rtt-v2-dim14-ce-only.pt"
MLP_DIM22="models/tabular/batch_edge_mlp.pt"
MLP_ATOMIC21="models/tabular/batch_edge_mlp_atomic21.pt"
TIMEOUT=3600
SEED=42

export GNN_DECODE_MODE=argmax
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

mkdir -p "$CFG_DIR" "$RES_DIR" logs

log() { echo "[$(date -Is)] $*" | tee -a "$PROGRESS_LOG"; }

log "=== tiered hub sweep (45 jobs) ==="
pipenv run python3 scripts_cosim/important/generate_tiered_hub_configs.py --out-dir "$CFG_DIR"

run_job() {
  local policy="$1" name="$2" config="$3"
  unset INFERENCE_FEATURE_LAYOUT
  local output args=()
  case "$policy" in
    gnn_dim22)
      export INFERENCE_FEATURE_LAYOUT=dim22 GNN_MODEL_PATH="$GNN_MODEL"
      output="${RES_DIR}/${name}_gnn_dim22.json"
      args=(--gnn)
      ;;
    gnn_atomic21)
      export INFERENCE_FEATURE_LAYOUT=atomic21 GNN_MODEL_PATH="$GNN_MODEL"
      output="${RES_DIR}/${name}_gnn_atomic21.json"
      args=(--gnn)
      ;;
    mlp_dim22)
      export INFERENCE_FEATURE_LAYOUT=dim22 MLP_MODEL_PATH="$MLP_DIM22"
      output="${RES_DIR}/${name}_mlp_dim22.json"
      args=(--mlp_batch --mlp-model "$MLP_DIM22")
      ;;
    mlp_atomic21)
      export INFERENCE_FEATURE_LAYOUT=atomic21 MLP_MODEL_PATH="$MLP_ATOMIC21"
      output="${RES_DIR}/${name}_mlp_atomic21.json"
      args=(--mlp_batch --mlp-model "$MLP_ATOMIC21")
      ;;
    knative)
      output="${RES_DIR}/${name}_knative.json"
      args=(--knative_network)
      ;;
    *) log "ERROR unknown policy $policy"; return 1 ;;
  esac
  if [[ -f "$output" ]]; then
    log "${policy} ${name} SKIP"
    return 0
  fi
  log "${policy} Running ${name}"
  pipenv run python3 scripts_cosim/run_simulation.py \
    "${args[@]}" --config "$config" --workload "$WORKLOAD" \
    --output "$output" --timeout "$TIMEOUT" --seed "$SEED" \
    || { log "${policy} ${name} FAILED"; return 1; }
}

while IFS=$'\t' read -r policy name path _partition; do
  [[ "$policy" == "policy" ]] && continue
  run_job "$policy" "$name" "$path"
done < "$JOBS_TSV"

pipenv run python3 scripts_cosim/important/compare_tiered_hub_gnn_mlp_sweep.py \
  --sweep-dir "$SWEEP_DIR" | tee -a "$PROGRESS_LOG"
