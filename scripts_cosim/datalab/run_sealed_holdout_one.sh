#!/usr/bin/env bash
# Sealed multi-seed live holdout — one SLURM task.
# Usage: run_sealed_holdout_one.sh <knative|mlp|gnn> <config_name> <config_json> <seed>
#
# Output: {config}_s{seed}_{knative|gnn|mlp_dim22}.json
# Models: 873/v5.5 contention_v2 deploy ckpts.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/nikola.lukic/gnn-herosim}"
cd "$PROJECT_ROOT"

POLICY="${1:?knative, mlp, or gnn}"
CONFIG_NAME="${2:?config name}"
CONFIG_PATH="${3:?config json path}"
SEED="${4:?seed}"
ENV_NAME="${HEROSIM_ENV_NAME:-gnn}"
WORKLOAD="${WORKLOAD:-data/nofs-ids/traces/workload-125-225.json}"
GNN_MODEL="${GNN_MODEL:-models/near-rtt-v2-contention-v2-873-v5.5-dim14-ce-only.pt}"
MLP_MODEL="${MLP_MODEL:-models/tabular/batch_edge_mlp_contention_v2_873_v5.5_dim22_batchcache.pt}"
OUT_DIR="${SWEEP_DIR:?SWEEP_DIR required}/results"
TIMEOUT="${TIMEOUT:-18000}"

export HEROSIM_WARMTH_PHYSICS="${HEROSIM_WARMTH_PHYSICS:-node_disk_v2}"
export GNN_DECODE_MODE=argmax
export GNN_BATCH_SIZE="${GNN_BATCH_SIZE:-4}"
export GNN_BATCH_TIMEOUT="${GNN_BATCH_TIMEOUT:-0.002}"
export PYTHONUNBUFFERED=1

mkdir -p "$OUT_DIR" logs

MODEL_CHECK=""
case "$POLICY" in
  gnn)
    tag="gnn"
    export GNN_MODEL_PATH="${GNN_MODEL}"
    export INFERENCE_FEATURE_LAYOUT=dim22
    MODEL_CHECK="$GNN_MODEL"
    RUN_ARGS=(--gnn)
    ;;
  mlp)
    tag="mlp_dim22"
    export MLP_MODEL_PATH="${MLP_MODEL}"
    export INFERENCE_FEATURE_LAYOUT=dim22
    MODEL_CHECK="$MLP_MODEL"
    RUN_ARGS=(--mlp_batch --mlp-model "$MLP_MODEL")
    ;;
  knative)
    tag="knative"
    RUN_ARGS=(--knative_network)
    ;;
  *)
    echo "ERROR: policy must be knative, mlp, or gnn; got: ${POLICY}" >&2
    exit 1
    ;;
esac

OUTPUT="${OUT_DIR}/${CONFIG_NAME}_s${SEED}_${tag}.json"

[[ -f "$WORKLOAD" ]] || { echo "ERROR: workload missing: $WORKLOAD" >&2; exit 1; }
[[ -f "$CONFIG_PATH" ]] || { echo "ERROR: config missing: $CONFIG_PATH" >&2; exit 1; }
[[ -z "$MODEL_CHECK" || -f "$MODEL_CHECK" ]] || { echo "ERROR: model missing: $MODEL_CHECK" >&2; exit 1; }

peek_rtt() {
  python3 - "$1" <<'PY'
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

if [[ -f "$OUTPUT" && "${FORCE_RERUN:-0}" != "1" ]]; then
  rtt=$(peek_rtt "$OUTPUT")
  if [[ "$rtt" != "0" && "$rtt" != "0.0" ]]; then
    echo "SKIP (exists): $OUTPUT  total_rtt=${rtt}"
    exit 0
  fi
  echo "WARN: stale output ${OUTPUT} (total_rtt=${rtt}); re-running"
fi

if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate "${ENV_NAME}"

echo "=== sealed holdout: ${POLICY} / ${CONFIG_NAME} / seed=${SEED} ==="
[[ -n "$MODEL_CHECK" ]] && echo "  MODEL=${MODEL_CHECK}"

start=$(date +%s)
python3 scripts_cosim/run_simulation.py \
  --config "$CONFIG_PATH" \
  --workload "$WORKLOAD" \
  --output "$OUTPUT" \
  --seed "$SEED" \
  --timeout "$TIMEOUT" \
  "${RUN_ARGS[@]}"
elapsed=$(( $(date +%s) - start ))

rtt=$(peek_rtt "$OUTPUT")
if [[ "$rtt" == "0" || "$rtt" == "0.0" || -z "$rtt" ]]; then
  echo "ERROR: invalid total_rtt in ${OUTPUT}" >&2
  exit 1
fi
echo "DONE: $OUTPUT  elapsed=${elapsed}s  total_rtt=${rtt}"
