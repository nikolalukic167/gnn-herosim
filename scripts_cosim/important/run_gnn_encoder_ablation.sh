#!/usr/bin/env bash
# Stage 1: ECT ceiling + GNN encoder/decode ablation ladder on the coupled trio.
#
# Why: the 45-cell full_corpus_siv1 gate put the GNN at 12.40x Knative (0/15) with
# GNN_DECODE_MODE=argmax and full GIN. An ad-hoc probe (siv1_hardcell_probe_20260814,
# n=1, no manifest) showed GNN_DISABLE_MESSAGE_PASSING=1 takes sparse_p35/s42 from
# 287.7M -> 23.2M total_rtt. This script separates three claims that are currently
# conflated:
#   (1) message passing is harmful          -> rung C
#   (2) same-node edges specifically are    -> rung B
#   (3) the decode is unguarded             -> rungs D/E/F
# and measures the ceiling nobody has ever measured on these cells (ECT arms), which
# decides whether there is headroom above the MLP at all.
#
# The GNN control (rung A) is NOT re-run: it already exists as
# full_corpus_siv1_coupled_trio_20260815/results/*_gnn.json (same ckpt, same env).
#
# Fail-loud: missing models/configs abort; undeclared warmth physics aborts;
# zero/empty RTT aborts.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
LOG="${LOG:-logs/gnn_encoder_ablation_${TS}.log}"
SWEEP_DIR="${SWEEP_DIR:-simulation_data/normal_sim_sweeps/gnn_encoder_ablation_20260816}"
GNN_MODEL="${GNN_MODEL:-models/near-rtt-v2-full-corpus-siv1-dim14-ce-only.pt}"
MLP_MODEL="${MLP_MODEL:-models/tabular/batch_edge_mlp_full_corpus_siv1_dim22_batchcache.pt}"
WORKLOAD="${WORKLOAD:-data/nofs-ids/traces/workload-125-225.json}"
BASELINE_SWEEP="${BASELINE_SWEEP:-simulation_data/normal_sim_sweeps/full_corpus_siv1_coupled_trio_20260815}"
TIMEOUT="${TIMEOUT:-18000}"
SEED="${SEED:-42}"
ECT_PARALLEL="${ECT_PARALLEL:-3}"
GNN_PARALLEL="${GNN_PARALLEL:-3}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

# Physics must be declared: the implicit default (platform_reuse_v1) inflates live
# total RTT ~100x and is not the Regime A regime.
export HEROSIM_WARMTH_PHYSICS="${HEROSIM_WARMTH_PHYSICS:-node_disk_v2}"
export HEROSIM_REQUIRE_EXPLICIT_PHYSICS=1
export QUEUE_FEATURE_CONTRACT="${QUEUE_FEATURE_CONTRACT:-scale_invariant_v1}"
export INFERENCE_FEATURE_LAYOUT=dim22
export GNN_BATCH_SIZE="${GNN_BATCH_SIZE:-4}"
export GNN_BATCH_TIMEOUT="${GNN_BATCH_TIMEOUT:-0.002}"
export PYTHONUNBUFFERED=1

OUT_DIR="${SWEEP_DIR}/results"
DISTILL_DIR="${SWEEP_DIR}/ect_pull_distill"
mkdir -p "$OUT_DIR" logs

log() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

KN_CFG="simulation_data/normal_sim_sweeps/knative_network_20260606_192413/configs"
CONFIGS=(
  "sparse_p25|${KN_CFG}/05_sparse_40_40_p25.json"
  "sparse_p35|${KN_CFG}/00_balanced_30_30_p35.json"
  "sparse_p25_skew|simulation_data/normal_sim_sweeps/atomic21_skew_configs/05_sparse_40_40_p25_degree_skew.json"
)
# The tightest / most diagnostic cell: GNN 13.06x Kn seed-mean, p99 3684s vs Kn 254s.
HARD_CELL="sparse_p35"

# Ablation ladder: tag|env assignments (space-separated KEY=VAL), GNN arm only.
LADDER=(
  "gnn_dropnodeedges|GNN_DROP_NODE_EDGES=1"
  "gnn_nomp|GNN_DISABLE_MESSAGE_PASSING=1"
  "gnn_uniq|GNN_DECODE_MODE=argmax_uniq"
  "gnn_lqb15|GNN_LQB_LAMBDA=1.5"
  "gnn_qfilter8|GNN_QUEUE_FILTER_MAX_DELTA=8"
)

[[ -f "$WORKLOAD" ]] || { echo "ERROR: workload missing: $WORKLOAD" >&2; exit 1; }
[[ -f "$GNN_MODEL" ]] || { echo "ERROR: GNN missing: $GNN_MODEL" >&2; exit 1; }
[[ -d "$BASELINE_SWEEP/results" ]] || { echo "ERROR: baseline sweep missing: $BASELINE_SWEEP" >&2; exit 1; }
for entry in "${CONFIGS[@]}"; do
  path="${entry#*|}"
  [[ -f "$path" ]] || { echo "ERROR: config missing: $path" >&2; exit 1; }
done

peek_rtt() {
  pipenv run python3 - "$1" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, ".")
from scripts_cosim.sweep_metrics import peek_scalar
p = Path(sys.argv[1])
print(peek_scalar(p, "total_rtt") or 0 if p.is_file() else 0)
PY
}

# run_one <tag> <cell_name> <config_path> <extra_env...>
run_one() {
  local tag="$1" name="$2" path="$3"; shift 3
  local output="${OUT_DIR}/${name}_s${SEED}_${tag}.json"
  local args=() kv

  if [[ -f "$output" && "${FORCE_RERUN:-0}" != "1" ]]; then
    local rtt; rtt=$(peek_rtt "$output")
    if [[ "$rtt" != "0" && "$rtt" != "0.0" ]]; then
      log "SKIP exists: $output total_rtt=${rtt}"
      return 0
    fi
    log "WARN stale: $output (total_rtt=${rtt}); re-running"
  fi

  case "$tag" in
    ect)      args=(--knative_network_ect) ;;
    ect_pull) args=(--knative_network_ect_pull)
              # Harvesting is OPT-IN: this workload is 561,848 tasks and ect_pull dumps a
              # PyG frame per decision (~100KB/frame => ~50GB per cell). Only enable it
              # when the teacher is actually worth distilling.
              if [[ "${HARVEST_DISTILL:-0}" == "1" ]]; then
                export ECT_PULL_DISTILL_DIR="${DISTILL_DIR}"
                mkdir -p "$ECT_PULL_DISTILL_DIR"
              fi ;;
    gnn_*)    args=(--gnn); export GNN_MODEL_PATH="$GNN_MODEL" ;;
    *) echo "ERROR: bad tag $tag" >&2; return 1 ;;
  esac

  for kv in "$@"; do export "${kv?}"; done

  log "RUN ${tag} ${name} seed=${SEED} env=[$*]"
  local start elapsed rtt
  start=$(date +%s)
  pipenv run python3 scripts_cosim/run_simulation.py \
    --config "$path" \
    --workload "$WORKLOAD" \
    --output "$output" \
    --seed "$SEED" \
    --timeout "$TIMEOUT" \
    "${args[@]}"
  elapsed=$(( $(date +%s) - start ))
  rtt=$(peek_rtt "$output")
  if [[ "$rtt" == "0" || "$rtt" == "0.0" || -z "$rtt" ]]; then
    log "ERROR: empty/zero RTT for $output"
    return 1
  fi
  log "DONE ${tag} ${name} seed=${SEED} elapsed=${elapsed}s total_rtt=${rtt}"
}

# Bounded-width job pool over a list of "tag|cell|cfg|env..." specs.
run_pool() {
  local width="$1"; shift
  local pids=() fail=0 spec
  for spec in "$@"; do
    while (( ${#pids[@]} >= width )); do
      if ! wait -n; then fail=1; fi
      local new=()
      for pid in "${pids[@]}"; do kill -0 "$pid" 2>/dev/null && new+=("$pid"); done
      pids=("${new[@]}")
    done
    IFS='|' read -r tag cell cfg envs <<< "$spec"
    # shellcheck disable=SC2086
    ( run_one "$tag" "$cell" "$cfg" $envs ) >>"$LOG" 2>&1 &
    pids+=($!)
  done
  for pid in "${pids[@]}"; do wait "$pid" || fail=1; done
  return $fail
}

log "=== stage 1: ceiling + encoder ablation ==="
log "SWEEP_DIR=${SWEEP_DIR} GNN=${GNN_MODEL}"
log "physics=${HEROSIM_WARMTH_PHYSICS} contract=${QUEUE_FEATURE_CONTRACT} seed=${SEED}"

# --- 1a. Ceiling: ECT + ECT_pull on all three cells --------------------------
if [[ "${SKIP_ECT:-0}" != "1" ]]; then
  specs=()
  for entry in "${CONFIGS[@]}"; do
    name="${entry%%|*}"; path="${entry#*|}"
    specs+=("ect|${name}|${path}|")
    specs+=("ect_pull|${name}|${path}|")
  done
  log "Phase: ECT ceiling (${#specs[@]} cells, parallel=${ECT_PARALLEL})"
  run_pool "$ECT_PARALLEL" "${specs[@]}" || { log "ERROR: ECT phase failed"; exit 1; }
fi

# --- 1b. Ablation ladder on the hard cell ------------------------------------
if [[ "${SKIP_LADDER:-0}" != "1" ]]; then
  hard_cfg=""
  for entry in "${CONFIGS[@]}"; do
    [[ "${entry%%|*}" == "$HARD_CELL" ]] && hard_cfg="${entry#*|}"
  done
  [[ -n "$hard_cfg" ]] || { echo "ERROR: hard cell $HARD_CELL not in CONFIGS" >&2; exit 1; }

  specs=()
  for rung in "${LADDER[@]}"; do
    specs+=("${rung%%|*}|${HARD_CELL}|${hard_cfg}|${rung#*|}")
  done
  log "Phase: ablation ladder on ${HARD_CELL} (${#specs[@]} cells, parallel=${GNN_PARALLEL})"
  run_pool "$GNN_PARALLEL" "${specs[@]}" || { log "ERROR: ladder phase failed"; exit 1; }
fi

log "=== stage 1 complete ==="
log "Log: ${LOG}"
