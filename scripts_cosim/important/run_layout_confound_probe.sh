#!/usr/bin/env bash
# Does INFERENCE_FEATURE_LAYOUT change live outcomes for a siv1 dim14 checkpoint?
#
# Why this exists. The deployed checkpoint's .contract.json declares
# inference_feature_layout=dim22, so load_gnn_model adopts it. The prefixctl (variance
# control) and tempfix (corrected-cache) checkpoints declare **null**, and
# feature_builder._inference_feature_layout defaults an unset env var to **atomic21**.
# Their live gates therefore ran with layout=None recorded in run_provenance.env while
# every deployed-checkpoint gate ran under dim22. dim22 vs atomic21 changes whether
# platform queue features are normalized (`use_norm_queue`), i.e. the same tensor shapes
# carry different meanings -- the exact failure class LINEAGES.md keeps recording.
#
# If the two layouts move total_rtt materially, then the "training-draw lottery" table
# (LINEAGES.md 2026-08-22) compared deployed@dim22 against prefixctl@atomic21 and is
# confounded by serving configuration, not just training nondeterminism.
#
# Usage: run_layout_confound_probe.sh [cell_name]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

CELL="${1:-cell01_p25_s9001}"
SRC_SWEEP="simulation_data/normal_sim_sweeps/a4_wl150100"
OUT_DIR="${OUT_DIR:-simulation_data/normal_sim_sweeps/layout_confound_probe}/results"
WORKLOAD="${WORKLOAD:-data/nofs-ids/traces/workload-150-100.json}"
MODEL="${MODEL:-models/near-rtt-v2-full-corpus-siv1-dim14-ce-only-prefixctl.pt}"
TIMEOUT="${TIMEOUT:-18000}"

export PYTHONHASHSEED=0
export HEROSIM_WARMTH_PHYSICS=node_disk_v2
export GNN_DECODE_MODE=argmax
export GNN_BATCH_SIZE=4
export GNN_BATCH_TIMEOUT=0.002
export PYTHONUNBUFFERED=1
export PIPENV_IGNORE_VIRTUALENVS=1
export VIRTUAL_ENV=
export PYTHONPATH="$ROOT"
export GNN_MODEL_PATH="$MODEL"

mkdir -p "$OUT_DIR" logs
CFG="${SRC_SWEEP}/configs/${CELL}.json"
[[ -f "$CFG" ]] || { echo "ERROR: missing config $CFG" >&2; exit 1; }
[[ -f "$MODEL" ]] || { echo "ERROR: missing model $MODEL" >&2; exit 1; }

TAG="$(basename "$MODEL" .pt)"
for layout in unset dim22; do
  OUT="${OUT_DIR}/${CELL}_${TAG}_layout-${layout}.json"
  if [[ -f "$OUT" ]]; then echo "SKIP (exists): $OUT"; continue; fi
  echo "=== ${CELL} / ${TAG} / layout=${layout} ==="
  if [[ "$layout" == "unset" ]]; then
    unset INFERENCE_FEATURE_LAYOUT
  else
    export INFERENCE_FEATURE_LAYOUT="$layout"
  fi
  ${HEROSIM_PY:-pipenv run python3} scripts_cosim/run_simulation.py \
    --config "$CFG" --workload "$WORKLOAD" --output "$OUT" \
    --timeout "$TIMEOUT" --gnn
done

echo "=== probe complete ==="
${HEROSIM_PY:-pipenv run python3} - <<'PY'
import glob, json, os
out = os.environ.get("OUT_DIR", "simulation_data/normal_sim_sweeps/layout_confound_probe") + "/results"
rows = []
for f in sorted(glob.glob(out + "/*.json")):
    if f.endswith(".decode_stats.json"):
        continue
    d = json.load(open(f))
    rows.append((os.path.basename(f), d.get("total_rtt"),
                 d["run_provenance"]["env"].get("INFERENCE_FEATURE_LAYOUT")))
for name, rtt, layout in rows:
    print(f"{name:70s} layout={str(layout):8s} total_rtt={rtt:,.1f}")
if len(rows) >= 2:
    a, b = rows[0][1], rows[1][1]
    print(f"\ndelta = {100.0*(b-a)/a:+.3f}%  (0.1-0.4% is the measured simulation noise floor)")
PY
