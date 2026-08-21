#!/usr/bin/env bash
# First real live-gate for the siv1_full_corpus lineage: knative | mlp | gnn on a real
# workload trace, over topology cells verified to match the training corpus.
#
# Differences from run_contention_v2_873_sealed_holdout.sh, all deliberate:
#
#   * NO --seed is passed to the simulator. `executesimulation.py:842-854` lets --seed
#     OVERRIDE the config's topology seed, so a seed sweep is a *topology* sweep: at
#     seed 42 against ds_00000's config, 144/210 edges differ and every shared edge
#     disagrees on latency. Replication comes from distinct verified cells instead.
#   * Cells are minted and parity-checked by make_full_corpus_siv1_gate_cells.py, and
#     re-verified here before any simulation starts. The older gate ran 40/40 p50 configs
#     against a 20/20 p25 corpus with nothing to say so.
#   * INFERENCE_FEATURE_LAYOUT is NOT exported. The checkpoint's .contract.json declares
#     it and load_gnn_model now enforces it; exporting a conflicting value is an error.
#
# Usage: run_full_corpus_siv1_live_gate.sh [knative|mlp|gnn|all] [cell_name]
#   cell_name restricts the run to one cell, which is how the SLURM array fans out.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

WHICH="${1:-all}"
ONLY_CELL="${2:-}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
SWEEP_DIR="${SWEEP_DIR:-simulation_data/normal_sim_sweeps/full_corpus_siv1_live_gate_20260820}"
LOG="${LOG:-logs/full_corpus_siv1_live_gate_${TS}.log}"
GNN_MODEL="${GNN_MODEL:-models/near-rtt-v2-full-corpus-siv1-dim14-ce-only.pt}"
MLP_MODEL="${MLP_MODEL:-models/tabular/batch_edge_mlp_full_corpus_siv1_dim22_batchcache.pt}"
WORKLOAD="${WORKLOAD:-data/nofs-ids/traces/workload-125-225.json}"
TIMEOUT="${TIMEOUT:-18000}"

# Placement tie-breaks are now sorted deterministically by (node.id, platform.id), so this
# is defense-in-depth against any unaudited set/dict-hash dependency, not the primary fix.
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"

# Physics comes from the checkpoint sidecar, which now records it; setting a conflicting
# value here would raise rather than silently serve a different cost model.
export HEROSIM_WARMTH_PHYSICS="${HEROSIM_WARMTH_PHYSICS:-node_disk_v2}"
export GNN_DECODE_MODE=argmax
export GNN_BATCH_SIZE="${GNN_BATCH_SIZE:-4}"
export GNN_BATCH_TIMEOUT="${GNN_BATCH_TIMEOUT:-0.002}"
export PYTHONUNBUFFERED=1
export PIPENV_IGNORE_VIRTUALENVS=1
export VIRTUAL_ENV=
export PYTHONPATH="$ROOT"

CFG_DIR="${SWEEP_DIR}/configs"
INFRA_DIR="${SWEEP_DIR}/cell_infrastructure"
OUT_DIR="${SWEEP_DIR}/results"
mkdir -p "$OUT_DIR" logs

log() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

[[ -f "$WORKLOAD" ]] || { echo "ERROR: workload missing: $WORKLOAD" >&2; exit 1; }
[[ -f "$GNN_MODEL" ]] || { echo "ERROR: GNN model missing: $GNN_MODEL" >&2; exit 1; }
[[ -f "$MLP_MODEL" ]] || { echo "ERROR: MLP model missing: $MLP_MODEL" >&2; exit 1; }

if [[ ! -d "$CFG_DIR" ]]; then
  log "Cells missing; minting them"
  pipenv run python3 scripts_cosim/important/make_full_corpus_siv1_gate_cells.py \
    --sweep-dir "$SWEEP_DIR" | tee -a "$LOG"
fi

# --- preflight: every cell must still match the corpus before a single sim runs --------
log "=== Infra parity preflight ==="
if [[ -n "$ONLY_CELL" ]]; then
  CELLS=("${INFRA_DIR}/${ONLY_CELL}")
  [[ -d "${CELLS[0]}" ]] || { echo "ERROR: no such cell: ${CELLS[0]}" >&2; exit 1; }
else
  mapfile -t CELLS < <(find "$INFRA_DIR" -mindepth 1 -maxdepth 1 -type d | sort)
fi
[[ ${#CELLS[@]} -gt 0 ]] || { echo "ERROR: no cells under $INFRA_DIR" >&2; exit 1; }
PARITY_ARGS=()
for cell in "${CELLS[@]}"; do PARITY_ARGS+=(--dataset "$cell"); done
# PARITY_EXTRA_ARGS lets a caller add a narrowly-scoped relaxation without editing this
# script -- currently only --allow-backbone-latency-divergence, for the link_contention_v1
# live A/B (see that flag's help). It is deliberately NOT defaulted to anything.
read -r -a PARITY_EXTRA <<< "${PARITY_EXTRA_ARGS:-}"
pipenv run python3 scripts_cosim/verify_live_infra_parity.py \
  "${PARITY_ARGS[@]}" "${PARITY_EXTRA[@]}" -v --json-out "${SWEEP_DIR}/infra_parity.json" | tee -a "$LOG"
log "Infra parity preflight PASSED for ${#CELLS[@]} cells"

run_one() {
  local policy="$1" cell_name="$2" config_path="$3" output run_args=()
  case "$policy" in
    gnn)     output="${OUT_DIR}/${cell_name}_s0_gnn.json";       export GNN_MODEL_PATH="$GNN_MODEL"; run_args=(--gnn) ;;
    mlp)     output="${OUT_DIR}/${cell_name}_s0_mlp_dim22.json"; export MLP_MODEL_PATH="$MLP_MODEL"; run_args=(--mlp_batch --mlp-model "$MLP_MODEL") ;;
    knative) output="${OUT_DIR}/${cell_name}_s0_knative.json";   run_args=(--knative_network) ;;
    *) echo "ERROR: unknown policy $policy" >&2; return 1 ;;
  esac

  if [[ -f "$output" && "${FORCE_RERUN:-0}" != "1" ]]; then
    local rtt
    rtt=$(pipenv run python3 -c "import json;print(json.load(open('${output}')).get('total_rtt',0))" 2>/dev/null || echo 0)
    if [[ "$rtt" != "0" && "$rtt" != "0.0" ]]; then
      log "SKIP (exists): $output total_rtt=${rtt}"; return 0
    fi
  fi

  log "--- ${policy} / ${cell_name} ---"
  local start; start=$(date +%s)
  # NOTE: no --seed. See the header.
  pipenv run python3 scripts_cosim/run_simulation.py \
    --config "$config_path" \
    --workload "$WORKLOAD" \
    --output "$output" \
    --timeout "$TIMEOUT" \
    "${run_args[@]}" 2>&1 | tee -a "$LOG"
  local rtt; rtt=$(pipenv run python3 -c "import json;print(json.load(open('${output}')).get('total_rtt','?'))" 2>/dev/null || echo "?")
  log "DONE ${policy}/${cell_name} elapsed=$(( $(date +%s) - start ))s total_rtt=${rtt}"
}

case "$WHICH" in
  all) POLICIES=(knative mlp gnn) ;;
  *)   POLICIES=("$WHICH") ;;
esac

for policy in "${POLICIES[@]}"; do
  for cell in "${CELLS[@]}"; do
    name="$(basename "$cell")"
    run_one "$policy" "$name" "${CFG_DIR}/${name}.json"
  done
done

log "=== Gate complete. Score with: ==="
log "  pipenv run python3 scripts_cosim/important/compare_sealed_live_holdout.py --sweep-dir $SWEEP_DIR"
