#!/usr/bin/env bash
# Compare contention_v3 live gate results; fail if incomplete.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/nikola.lukic/gnn-herosim}"
cd "$PROJECT_ROOT"

SWEEP_DIR="${SWEEP_DIR:-simulation_data/normal_sim_sweeps/contention_v3_live_gate_20260620}"
PHASE_DIR="logs/contention_v3_pipeline"
EXPECTED="${EXPECTED:-9}"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="logs/contention_v3_compare_${TS}.log"

mkdir -p logs "$PHASE_DIR"

exec > >(tee -a "$LOG") 2>&1
echo "=== contention_v3 compare ${TS} ==="
echo "Sweep: ${SWEEP_DIR}"

if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate "${HEROSIM_ENV_NAME:-gnn}"

results="${SWEEP_DIR}/results"
[[ -d "$results" ]] || { echo "ERROR: missing ${results}" >&2; exit 1; }

n=0
while IFS= read -r f; do
  [[ "$f" == *decode_stats* ]] && continue
  rtt=$(python3 -c "import json; d=json.load(open('${f}')); print(d.get('total_rtt', 0))")
  if [[ "$rtt" != "0" && "$rtt" != "0.0" ]]; then
    n=$((n + 1))
  else
    echo "ERROR: invalid total_rtt in ${f}" >&2
    exit 1
  fi
done < <(find "$results" -maxdepth 1 -name '*.json' -type f | sort)

if [[ "$n" -lt "$EXPECTED" ]]; then
  echo "ERROR: only ${n}/${EXPECTED} valid sim JSONs" >&2
  exit 1
fi

python3 scripts_cosim/important/compare_contention_v2_live_gate.py --sweep-dir "$SWEEP_DIR" || {
  rc=$?
  if [[ "$rc" -eq 2 ]]; then
    echo "NOTE: live gate metric FAIL (GNN vs MLP sum) — results above are valid"
  else
    exit "$rc"
  fi
}

touch "${PHASE_DIR}/phase_live_gate.done"
touch "${PHASE_DIR}/phase_all.done"
echo "=== compare complete (${n}/${EXPECTED}) log=${LOG} ==="
