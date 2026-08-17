#!/usr/bin/env bash
# Multi-seed live gate for contention_v2 873/v5.5 deploy ckpts on the COUPLED trio.
#
# Why: the sealed holdout (20260806) only covered uncoupled cells (balanced /
# client_heavy / server_heavy) where MLP wins total_rtt 13/20 and p99 4/4. The
# coupled development trio (sparse_p25 / sparse_p35 / sparse_p25_skew) was only
# ever run at seed 42 with the pre-repair 711-era checkpoints. On that single
# seed the GNN won sparse_p35 on total_rtt (11.08M vs Kn 12.77M vs MLP 16.79M)
# AND on p99 (110.2s vs Kn 164.7s vs MLP 783.5s) — an MLP collision cliff.
#
# Pre-registered question: does that cliff reproduce across 5 seeds with the
# repaired 873/v5.5 checkpoints?
#
# Success criterion (declared before running):
#   PRIMARY  GNN beats MLP on total_rtt in >=2 of 3 configs (seed-averaged)
#   TAIL     GNN beats MLP on p99 in >=2 of 3 configs (seed-averaged)
# Reaching TAIL without PRIMARY still supports a bounded collision-robustness
# claim. Reaching neither closes the coupled-cell question against the GNN.
#
# Fail-loud: missing models/configs abort; undeclared warmth physics aborts;
# zero/empty RTT aborts.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
LOG="${LOG:-logs/contention_v2_873_v5.5_coupled_trio_${TS}.log}"
SWEEP_DIR="${SWEEP_DIR:-simulation_data/normal_sim_sweeps/contention_v2_873_v5.5_coupled_trio_20260813}"
GNN_MODEL="${GNN_MODEL:-models/near-rtt-v2-contention-v2-873-v5.5-dim14-ce-only.pt}"
MLP_MODEL="${MLP_MODEL:-models/tabular/batch_edge_mlp_contention_v2_873_v5.5_dim22_batchcache.pt}"
WORKLOAD="${WORKLOAD:-data/nofs-ids/traces/workload-125-225.json}"
TIMEOUT="${TIMEOUT:-18000}"
SEEDS_CSV="${SEEDS_CSV:-42,43,44,45,46}"
# 32-core host, no usable GPU: cap per-sim threads so pools do not thrash.
CPU_PARALLEL="${CPU_PARALLEL:-4}"
ML_PARALLEL="${ML_PARALLEL:-2}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-6}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-6}"

# Physics must be declared: the implicit default (platform_reuse_v1) inflates
# live total RTT ~100x and is not the Regime A regime.
export HEROSIM_WARMTH_PHYSICS="${HEROSIM_WARMTH_PHYSICS:-node_disk_v2}"
export HEROSIM_REQUIRE_EXPLICIT_PHYSICS=1
export GNN_DECODE_MODE=argmax
export GNN_BATCH_SIZE="${GNN_BATCH_SIZE:-4}"
export GNN_BATCH_TIMEOUT="${GNN_BATCH_TIMEOUT:-0.002}"
export INFERENCE_FEATURE_LAYOUT=dim22
export PYTHONUNBUFFERED=1

OUT_DIR="${SWEEP_DIR}/results"
mkdir -p "$OUT_DIR" logs

log() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

IFS=',' read -r -a SEEDS <<< "$SEEDS_CSV"

KN_CFG="simulation_data/normal_sim_sweeps/knative_network_20260606_192413/configs"
CONFIGS=(
  "sparse_p25|${KN_CFG}/05_sparse_40_40_p25.json"
  "sparse_p35|${KN_CFG}/00_balanced_30_30_p35.json"
  "sparse_p25_skew|simulation_data/normal_sim_sweeps/atomic21_skew_configs/05_sparse_40_40_p25_degree_skew.json"
)

[[ -f "$WORKLOAD" ]] || { echo "ERROR: workload missing: $WORKLOAD" >&2; exit 1; }
[[ -f "$GNN_MODEL" ]] || { echo "ERROR: GNN missing: $GNN_MODEL" >&2; exit 1; }
[[ -f "$MLP_MODEL" ]] || { echo "ERROR: MLP missing: $MLP_MODEL" >&2; exit 1; }
for entry in "${CONFIGS[@]}"; do
  path="${entry#*|}"
  [[ -f "$path" ]] || { echo "ERROR: config missing: $path" >&2; exit 1; }
done

pipenv run python3 - <<PY | tee -a "$LOG"
import hashlib, json
from pathlib import Path
from datetime import datetime, timezone

def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

sweep = Path("${SWEEP_DIR}")
manifest = {
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "kind": "multi_seed_live_gate_coupled_trio",
    "note": (
        "Coupled development trio with repaired 873/v5.5 ckpts. Complement to "
        "contention_v2_873_v5.5_sealed_holdout_20260806 (uncoupled cells only)."
    ),
    "prior_single_seed_observation": {
        "sweep": "contention_v2_live_gate_20260615",
        "models": "711-era contention_v2 ckpts",
        "sparse_p35": {
            "total_rtt": {"knative": 12772421, "gnn": 11084659, "mlp": 16794833},
            "p99_s": {"knative": 164.7, "gnn": 110.2, "mlp": 783.5},
        },
    },
    "success_criterion": {
        "primary": "GNN < MLP on seed-averaged total_rtt in >=2 of 3 configs",
        "tail": "GNN < MLP on seed-averaged p99 in >=2 of 3 configs",
    },
    "warmth_physics": "${HEROSIM_WARMTH_PHYSICS}",
    "decode_mode": "argmax",
    "inference_feature_layout": "dim22",
    "gnn_batch_size": "${GNN_BATCH_SIZE}",
    "gnn_batch_timeout": "${GNN_BATCH_TIMEOUT}",
    "workload": "${WORKLOAD}",
    "seeds": [int(s) for s in "${SEEDS_CSV}".split(",")],
    "configs": ["sparse_p25", "sparse_p35", "sparse_p25_skew"],
    "gnn_model": {"path": "${GNN_MODEL}", "md5": md5("${GNN_MODEL}")},
    "mlp_model": {"path": "${MLP_MODEL}", "md5": md5("${MLP_MODEL}")},
    "policies": ["knative", "mlp", "gnn"],
}
(sweep / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print("Wrote", sweep / "manifest.json")
PY

peek_rtt() {
  local f="$1"
  pipenv run python3 - "$f" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, ".")
from scripts_cosim.sweep_metrics import peek_scalar
p = Path(sys.argv[1])
if not p.is_file():
    print(0)
    raise SystemExit(0)
print(peek_scalar(p, "total_rtt") or 0)
PY
}

run_one() {
  local policy="$1" name="$2" path="$3" seed="$4"
  local output tag args
  case "$policy" in
    gnn) tag="gnn"; export GNN_MODEL_PATH="$GNN_MODEL"; args=(--gnn) ;;
    mlp) tag="mlp_dim22"; export MLP_MODEL_PATH="$MLP_MODEL"; args=(--mlp_batch --mlp-model "$MLP_MODEL") ;;
    knative) tag="knative"; args=(--knative_network) ;;
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

run_policy_pool() {
  local policy="$1" width="$2"
  log "Phase: ${policy} (parallel=${width})"
  local pids=() fail=0 entry name path seed
  for entry in "${CONFIGS[@]}"; do
    name="${entry%%|*}"
    path="${entry#*|}"
    for seed in "${SEEDS[@]}"; do
      while (( ${#pids[@]} >= width )); do
        if ! wait -n; then fail=1; fi
        local new_pids=()
        for pid in "${pids[@]}"; do
          if kill -0 "$pid" 2>/dev/null; then new_pids+=("$pid"); fi
        done
        pids=("${new_pids[@]}")
      done
      ( run_one "$policy" "$name" "$path" "$seed" ) >>"$LOG" 2>&1 &
      pids+=($!)
    done
  done
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then fail=1; fi
  done
  if (( fail )); then
    log "ERROR: ${policy} phase had failures"
    return 1
  fi
  return 0
}

log "=== coupled trio gate start ==="
log "SWEEP_DIR=${SWEEP_DIR}"
log "physics=${HEROSIM_WARMTH_PHYSICS} (strict) GNN=${GNN_MODEL} MLP=${MLP_MODEL}"
log "SEEDS=${SEEDS_CSV} CPU_PARALLEL=${CPU_PARALLEL} ML_PARALLEL=${ML_PARALLEL}"

if [[ "${SKIP_KNATIVE:-0}" != "1" ]]; then
  run_policy_pool knative "$CPU_PARALLEL"
fi
if [[ "${SKIP_MLP:-0}" != "1" ]]; then
  run_policy_pool mlp "$ML_PARALLEL"
fi
if [[ "${SKIP_GNN:-0}" != "1" ]]; then
  run_policy_pool gnn "$ML_PARALLEL"
fi

log "Phase: compare"
pipenv run python3 scripts_cosim/important/compare_sealed_live_holdout.py \
  --sweep-dir "$SWEEP_DIR" \
  --report "${SWEEP_DIR}/compare.json" | tee -a "$LOG"

log "=== coupled trio gate complete ==="
log "Log: ${LOG}"
