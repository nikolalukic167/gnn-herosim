#!/usr/bin/env bash
# Regime B benchmark sweep — mixed cold bursts on hub9 configs.
#
# Prepares hub configs + mixed burst workload, then runs:
#   knative_network | knative_network_ect | mlp_batch (seqblend1, ce_reduced)
# across hub_k{4,6,8}_seek{35,50,65}.
#
# Usage:
#   bash scripts_cosim/datalab/run_regime_b_benchmark.sh
#   SWEEP_DIR=... FORCE_RERUN=1 bash scripts_cosim/datalab/run_regime_b_benchmark.sh
set -euo pipefail

ROOT="${PROJECT_ROOT:-/root/projects/my-herosim}"
cd "$ROOT"

SWEEP_DIR="${SWEEP_DIR:-simulation_data/normal_sim_sweeps/regime_b_hub9_20260612}"
K_CORE_VALUES="${K_CORE_VALUES:-4,6,8}"
SEEK_FRACTIONS="${SEEK_FRACTIONS:-0.35,0.5,0.65}"
SEED="${SEED:-42}"
WORKLOAD="${WORKLOAD:-data/nofs-ids/traces/workload-cold-burst-mixed.json}"
HUB_CONFIG_FOR_WORKLOAD="${HUB_CONFIG_FOR_WORKLOAD:-${SWEEP_DIR}/configs/hub_k6_seek50.json}"

export HEROSIM_WARMTH_PHYSICS="${HEROSIM_WARMTH_PHYSICS:-node_disk_v2}"
export HEROSIM_DEFER_COLD_REPLICA_INIT="${HEROSIM_DEFER_COLD_REPLICA_INIT:-1}"
export HEROSIM_FAST_FORWARD_WARMUP="${HEROSIM_FAST_FORWARD_WARMUP:-1}"
export SIM_FORCE_FULL_STATS="${SIM_FORCE_FULL_STATS:-1}"
export PYTHONUNBUFFERED=1

echo "=== Regime B benchmark sweep ==="
echo "SWEEP_DIR=${SWEEP_DIR}"
echo "HEROSIM_WARMTH_PHYSICS=${HEROSIM_WARMTH_PHYSICS}"
echo "HEROSIM_DEFER_COLD_REPLICA_INIT=${HEROSIM_DEFER_COLD_REPLICA_INIT}"
echo "HEROSIM_FAST_FORWARD_WARMUP=${HEROSIM_FAST_FORWARD_WARMUP}"
echo "SIM_FORCE_FULL_STATS=${SIM_FORCE_FULL_STATS}"

mkdir -p "${SWEEP_DIR}/configs" "${SWEEP_DIR}/results"

if ! compgen -G "${SWEEP_DIR}/configs/hub_k*.json" > /dev/null; then
  echo "[1/3] Generating hub configs..."
  pipenv run python3 scripts_cosim/important/generate_tiered_hub_configs.py \
    --out-dir "${SWEEP_DIR}/configs" \
    --policies dim22 \
    --k-core-values "${K_CORE_VALUES}" \
    --seek-fractions "${SEEK_FRACTIONS}" \
    --latency-core-ms 5 \
    --latency-periphery-ms 30 \
    --seed "${SEED}"
else
  echo "[1/3] Hub configs already present under ${SWEEP_DIR}/configs"
fi

if [[ ! -f "${WORKLOAD}" || "${REGENERATE_WORKLOAD:-0}" == "1" ]]; then
  echo "[2/3] Generating mixed cold-burst workload -> ${WORKLOAD}"
  pipenv run python3 scripts_cosim/important/generate_cold_burst_workload.py \
    --output "${WORKLOAD}" \
    --hub-config "${HUB_CONFIG_FOR_WORKLOAD}" \
    --num-bursts 3 \
    --burst-sizes 4,8,16 \
    --burst-interval 180 \
    --task-types dnn1,dnn2 \
    --seed "${SEED}"
else
  echo "[2/3] Workload exists: ${WORKLOAD}"
fi

CONFIGS=(hub_k4_seek35 hub_k4_seek50 hub_k4_seek65 hub_k6_seek35 hub_k6_seek50 hub_k6_seek65 hub_k8_seek35 hub_k8_seek50 hub_k8_seek65)
POLICIES=(knative knative_ect mlp_seqblend1)

echo "[3/3] Running ${#CONFIGS[@]} configs × ${#POLICIES[@]} policies = $(( ${#CONFIGS[@]} * ${#POLICIES[@]} )) jobs"

failures=0
for cfg in "${CONFIGS[@]}"; do
  cfg_path="${SWEEP_DIR}/configs/${cfg}.json"
  if [[ ! -f "$cfg_path" ]]; then
    echo "ERROR: missing config ${cfg_path}" >&2
    failures=$((failures + 1))
    continue
  fi
  for pol in "${POLICIES[@]}"; do
    if ! bash scripts_cosim/datalab/run_regime_b_benchmark_one.sh "$pol" "$cfg" "$cfg_path"; then
      echo "FAIL: ${cfg} ${pol}" >&2
      failures=$((failures + 1))
    fi
  done
done

summary_path="${SWEEP_DIR}/results/regime_b_summary.json"
pipenv run python3 - <<PY
import json
from pathlib import Path

sweep = Path("${SWEEP_DIR}") / "results"
rows = []
for path in sorted(sweep.glob("*_regime_b_*.json")):
    try:
        d = json.loads(path.read_text())
    except Exception as exc:
        rows.append({"file": path.name, "status": "invalid", "error": str(exc)})
        continue
    rb = d.get("regime_b") or {}
    rows.append({
        "file": path.name,
        "policy": d.get("policy"),
        "config_file": d.get("config_file"),
        "total_rtt": d.get("total_rtt"),
        "regime_b_primary_score_s": d.get("regime_b_primary_score_s") or rb.get("regime_b_primary_score_s"),
        "burst_summaries": rb.get("burst_summaries"),
        "total_rtt_trap": (rb.get("total_rtt_trap") or {}),
    })

out = {"jobs": rows, "failures": int("${failures}")}
(sweep / "regime_b_summary.json").write_text(json.dumps(out, indent=2) + "\\n")
print(f"Wrote {sweep / 'regime_b_summary.json'} ({len(rows)} result files)")
PY

if [[ "$failures" -gt 0 ]]; then
  echo "=== DONE with ${failures} failure(s) ===" >&2
  exit 1
fi
echo "=== DONE — all Regime B benchmark jobs succeeded ==="
echo "Summary: ${summary_path}"
