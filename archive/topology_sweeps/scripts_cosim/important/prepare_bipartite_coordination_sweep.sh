#!/usr/bin/env bash
# Generate configs + job manifest for sweep_bipartite_coordination_v1.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

SWEEP_DIR="${SWEEP_DIR:-simulation_data/normal_sim_sweeps/sweep_bipartite_coordination_v1}"
K_CORE_VALUES="${K_CORE_VALUES:-4,6,8}"
SEEK_FRACTIONS="${SEEK_FRACTIONS:-0.35,0.5,0.65}"
LATENCY_CORE_MS="${LATENCY_CORE_MS:-5}"
LATENCY_PERIPHERY_MS="${LATENCY_PERIPHERY_MS:-30}"
SEED="${SEED:-42}"

pipenv run python3 scripts_cosim/important/generate_tiered_hub_configs.py \
  --out-dir "${SWEEP_DIR}/configs" \
  --policies dim22 \
  --k-core-values "${K_CORE_VALUES}" \
  --seek-fractions "${SEEK_FRACTIONS}" \
  --latency-core-ms "${LATENCY_CORE_MS}" \
  --latency-periphery-ms "${LATENCY_PERIPHERY_MS}" \
  --seed "${SEED}"

job_count=$(($(wc -l < "${SWEEP_DIR}/configs/jobs_dim22.tsv") - 1))
hub_count=$(python3 -c "k='${K_CORE_VALUES}'; s='${SEEK_FRACTIONS}'; print(len(k.split(','))*len(s.split(',')))")

if [[ "$job_count" -ne $((hub_count * 2)) ]]; then
  echo "ERROR: expected $((hub_count * 2)) jobs, got ${job_count}" >&2
  exit 1
fi

mkdir -p "${SWEEP_DIR}/results"
pipenv run python3 - <<PY
import json
from pathlib import Path

meta = {
    "sweep_id": "sweep_bipartite_coordination_v1",
    "workload": "data/nofs-ids/traces/workload-125-225.json",
    "gnn_batch_size": 4,
    "gnn_batch_timeout_s": 0.002,
    "k_core_values": [int(x) for x in "${K_CORE_VALUES}".split(",")],
    "seek_fractions": [float(x) for x in "${SEEK_FRACTIONS}".split(",")],
    "latency_core_ms": float("${LATENCY_CORE_MS}"),
    "latency_periphery_ms": float("${LATENCY_PERIPHERY_MS}"),
    "seed": int("${SEED}"),
    "policies": ["gnn_dim22", "mlp_dim22"],
    "hub_configs": int("${hub_count}"),
    "gpu_jobs": int("${job_count}"),
}
path = Path("${SWEEP_DIR}/configs/sweep_meta.json")
path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
PY

echo "[+] Prepared ${SWEEP_DIR}"
echo "    Hub configs: ${hub_count} · GPU jobs: ${job_count} (gnn_dim22 + mlp_dim22)"
echo "    k_core=[${K_CORE_VALUES}] seek=[${SEEK_FRACTIONS}] latency=${LATENCY_CORE_MS}/${LATENCY_PERIPHERY_MS}ms"
echo "    GNN batch size locked at 4 (training regime)"
