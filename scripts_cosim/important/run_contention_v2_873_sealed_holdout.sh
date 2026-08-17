#!/usr/bin/env bash
# Sealed multi-seed live holdout for contention_v2 873/v5.5 deploy ckpts.
#
# Uses previously unused topology cells (NOT sparse_p25/p35/skew development trio).
# Paired GNN / MLP / Knative over SEEDS (default 42..46 = 5 seeds).
#
# Fail-loud: missing models/configs abort; nonzero sim exit aborts the job.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
LOG="${LOG:-logs/contention_v2_873_v5.5_sealed_holdout_${TS}.log}"
SWEEP_DIR="${SWEEP_DIR:-simulation_data/normal_sim_sweeps/contention_v2_873_v5.5_sealed_holdout_20260806}"
GNN_MODEL="${GNN_MODEL:-models/near-rtt-v2-contention-v2-873-v5.5-dim14-ce-only.pt}"
MLP_MODEL="${MLP_MODEL:-models/tabular/batch_edge_mlp_contention_v2_873_v5.5_dim22_batchcache.pt}"
WORKLOAD="${WORKLOAD:-data/nofs-ids/traces/workload-125-225.json}"
TIMEOUT="${TIMEOUT:-18000}"
CPU_PARALLEL="${CPU_PARALLEL:-4}"
# Learnable policies: 2 concurrent sims fit in ~8GB RSS on 32GB host; T4 can share.
GPU_PARALLEL="${GPU_PARALLEL:-2}"
# Seeds: at least five simulation seeds (data-leakage audit gate)
SEEDS_CSV="${SEEDS_CSV:-42,43,44,45,46}"

export HEROSIM_WARMTH_PHYSICS="${HEROSIM_WARMTH_PHYSICS:-node_disk_v2}"
export GNN_DECODE_MODE=argmax
export GNN_BATCH_SIZE="${GNN_BATCH_SIZE:-4}"
export GNN_BATCH_TIMEOUT="${GNN_BATCH_TIMEOUT:-0.002}"
export INFERENCE_FEATURE_LAYOUT=dim22
export PYTHONUNBUFFERED=1
export GNN_MODEL_PATH="$GNN_MODEL"
export MLP_MODEL_PATH="$MLP_MODEL"

CFG_DIR="${SWEEP_DIR}/configs"
OUT_DIR="${SWEEP_DIR}/results"
mkdir -p "$OUT_DIR" logs "$CFG_DIR"

log() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

IFS=',' read -r -a SEEDS <<< "$SEEDS_CSV"

CONFIGS=(
  "balanced_p50|${CFG_DIR}/01_balanced_40_40_p50.json"
  "balanced_p60|${CFG_DIR}/02_balanced_50_50_p60.json"
  "client_heavy_p50|${CFG_DIR}/03_client_heavy_50_35_p50.json"
  "server_heavy_p50|${CFG_DIR}/04_server_heavy_35_50_p50.json"
)

[[ -f "$WORKLOAD" ]] || { echo "ERROR: workload missing: $WORKLOAD" >&2; exit 1; }
[[ -f "$GNN_MODEL" ]] || { echo "ERROR: GNN missing: $GNN_MODEL" >&2; exit 1; }
[[ -f "$MLP_MODEL" ]] || { echo "ERROR: MLP missing: $MLP_MODEL" >&2; exit 1; }
for entry in "${CONFIGS[@]}"; do
  path="${entry#*|}"
  [[ -f "$path" ]] || { echo "ERROR: config missing: $path" >&2; exit 1; }
done

# Provenance manifest (fail if write fails).
# Callers that re-run this sweep under different physics/code override the
# descriptive fields so the artifact never misrepresents itself as the original.
MANIFEST_KIND="${MANIFEST_KIND:-sealed_multi_seed_live_holdout}"
MANIFEST_NOTE="${MANIFEST_NOTE:-Unused topology cells vs contention_v2_live_gate_20260615 development trio (sparse_p25/p35/skew).}"
MANIFEST_EXTRA_JSON="${MANIFEST_EXTRA_JSON:-{\"development_trio_excluded\": [\"sparse_p25\", \"sparse_p35\", \"sparse_p25_skew\"]\}}"
manifest_names=()
for entry in "${CONFIGS[@]}"; do manifest_names+=("${entry%%|*}"); done
MANIFEST_CONFIG_NAMES="$(IFS=,; echo "${manifest_names[*]}")"

pipenv run python3 scripts_cosim/important/write_sweep_manifest.py \
  --sweep-dir "$SWEEP_DIR" \
  --kind "$MANIFEST_KIND" \
  --note "$MANIFEST_NOTE" \
  --physics "$HEROSIM_WARMTH_PHYSICS" \
  --workload "$WORKLOAD" \
  --seeds "$SEEDS_CSV" \
  --configs "$MANIFEST_CONFIG_NAMES" \
  --gnn-model "$GNN_MODEL" \
  --mlp-model "$MLP_MODEL" \
  --extra-json "$MANIFEST_EXTRA_JSON" \
  --force | tee -a "$LOG"

# Fast total_rtt peek — result JSONs are 100MB+; never full-parse.
peek_rtt() {
  local f="$1"
  python3 - "$f" <<'PY'
import re, sys
from pathlib import Path
p = Path(sys.argv[1])
if not p.is_file():
    print(0); raise SystemExit(0)
size = p.stat().st_size
with open(p, "rb") as fh:
    head = fh.read(65536)
    if size > 131072:
        fh.seek(max(0, size - 65536))
        tail = fh.read()
    else:
        tail = b""
blob = head.decode("utf-8", "ignore") + "\n" + tail.decode("utf-8", "ignore")
m = None
for m in re.finditer(r'"total_rtt"\s*:\s*([0-9.eE+-]+)', blob):
    pass
print(m.group(1) if m else 0)
PY
}

run_one() {
  local policy="$1" name="$2" path="$3" seed="$4"
  local output tag
  case "$policy" in
    gnn) tag="gnn"; export GNN_MODEL_PATH="$GNN_MODEL" ;;
    mlp) tag="mlp_dim22"; export MLP_MODEL_PATH="$MLP_MODEL" ;;
    knative) tag="knative" ;;
    *) echo "ERROR: bad policy $policy" >&2; return 1 ;;
  esac
  output="${OUT_DIR}/${name}_s${seed}_${tag}.json"

  if [[ -f "$output" && "${FORCE_RERUN:-0}" != "1" ]]; then
    local rtt
    rtt=$(peek_rtt "$output")
    if [[ "$rtt" != "0" && "$rtt" != "0.0" ]]; then
      log "SKIP exists: $output total_rtt=${rtt}"
      return 0
    fi
    log "WARN stale: $output (total_rtt=${rtt}); re-running"
  fi

  local args=()
  case "$policy" in
    gnn) args=(--gnn) ;;
    mlp) args=(--mlp_batch --mlp-model "$MLP_MODEL") ;;
    knative) args=(--knative_network) ;;
  esac

  log "RUN ${policy} ${name} seed=${seed}"
  local start elapsed rtt
  start=$(date +%s)
  pipenv run python3 scripts_cosim/run_simulation.py \
    --config "$path" \
    --workload "$WORKLOAD" \
    --output "$output" \
    --seed "$seed" \
    --timeout "$TIMEOUT" \
    "${args[@]}"
  elapsed=$(( $(date +%s) - start ))
  rtt=$(peek_rtt "$output")
  if [[ "$rtt" == "0" || "$rtt" == "0.0" || -z "$rtt" ]]; then
    log "ERROR: empty/zero RTT for $output"
    return 1
  fi
  log "DONE ${policy} ${name} seed=${seed} elapsed=${elapsed}s total_rtt=${rtt}"
}

log "=== sealed holdout start ==="
log "SWEEP_DIR=${SWEEP_DIR}"
log "GNN=${GNN_MODEL}"
log "MLP=${MLP_MODEL}"
log "SEEDS=${SEEDS_CSV}"
log "CPU_PARALLEL=${CPU_PARALLEL}"

# Phase 1: Knative (CPU) — parallel pool
log "Phase 1: Knative (parallel=${CPU_PARALLEL})"
pids=()
fail=0
for entry in "${CONFIGS[@]}"; do
  name="${entry%%|*}"
  path="${entry#*|}"
  for seed in "${SEEDS[@]}"; do
    while (( ${#pids[@]} >= CPU_PARALLEL )); do
      if ! wait -n; then fail=1; fi
      # prune finished
      new_pids=()
      for pid in "${pids[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then new_pids+=("$pid"); fi
      done
      pids=("${new_pids[@]}")
    done
    (
      run_one knative "$name" "$path" "$seed"
    ) >>"$LOG" 2>&1 &
    pids+=($!)
  done
done
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then fail=1; fi
done
if (( fail )); then
  log "ERROR: Knative phase had failures"
  exit 1
fi

# Phase 2+3: MLP then GNN.
# Prefer serial for learnable policies (GPU_PARALLEL>1 thrash ~4× slower on this host).
# Avoid editing this file while a holdout is running — bash re-reads the script stream.
run_policy_serial() {
  local policy="$1"
  log "Phase: ${policy} (serial)"
  local entry name path seed
  for entry in "${CONFIGS[@]}"; do
    name="${entry%%|*}"
    path="${entry#*|}"
    for seed in "${SEEDS[@]}"; do
      if ! run_one "$policy" "$name" "$path" "$seed" >>"$LOG" 2>&1; then
        log "ERROR: ${policy} ${name} seed=${seed} failed"
        return 1
      fi
    done
  done
  return 0
}

if [[ "${SKIP_MLP:-0}" != "1" ]]; then
  run_policy_serial mlp
fi
if [[ "${SKIP_GNN:-0}" != "1" ]]; then
  run_policy_serial gnn
fi


log "Phase 4: compare"
pipenv run python3 scripts_cosim/important/compare_sealed_live_holdout.py \
  --sweep-dir "$SWEEP_DIR" \
  --report "${SWEEP_DIR}/compare.json" | tee -a "$LOG"

log "=== sealed holdout complete ==="
log "Log: ${LOG}"
log "Sweep: ${SWEEP_DIR}"
