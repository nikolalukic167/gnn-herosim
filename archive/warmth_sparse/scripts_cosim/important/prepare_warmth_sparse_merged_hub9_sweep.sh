#!/usr/bin/env bash
# Generate 9 hub configs + job manifest for sparse-merged ce-reduced hub9 sweep (125-225).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

SWEEP_DIR="${SWEEP_DIR:-simulation_data/normal_sim_sweeps/warmth_sparse_merged_ce_reduced_hub9_20260611}"
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

hub_count=$(python3 -c "k='${K_CORE_VALUES}'; s='${SEEK_FRACTIONS}'; print(len(k.split(','))*len(s.split(',')))")

pipenv run python3 - <<PY
import json
from pathlib import Path

sweep = Path("${SWEEP_DIR}")
configs_dir = sweep / "configs"
hub_jsons = sorted(configs_dir.glob("hub_k*.json"))
if len(hub_jsons) != int("${hub_count}"):
    raise SystemExit(f"expected ${hub_count} hub configs, got {len(hub_jsons)}")

lines = ["policy\tconfig_name\tconfig_path"]
for path in hub_jsons:
    name = path.stem
    rel = path.as_posix()
    for policy in ("gnn", "mlp", "knative"):
        lines.append(f"{policy}\t{name}\t{rel}")

manifest = configs_dir / "jobs_smcr_hub9.tsv"
manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
job_count = len(lines) - 1
if job_count != int("${hub_count}") * 3:
    raise SystemExit(f"expected {int('${hub_count}') * 3} jobs, got {job_count}")

meta = {
    "sweep_id": "warmth_sparse_merged_ce_reduced_hub9",
    "workload": "data/nofs-ids/traces/workload-125-225.json",
    "gnn_model": "models/near-rtt-v2-warmth-sparse-merged-ce-reduced.pt",
    "mlp_model": "models/tabular/batch_edge_mlp_warmth_sparse_merged_ce_reduced.pt",
    "inference_feature_layout": "ce_reduced",
    "warmth_physics": "node_disk_v2",
    "gnn_batch_size": 4,
    "gnn_batch_timeout_s": 0.002,
    "k_core_values": [int(x) for x in "${K_CORE_VALUES}".split(",")],
    "seek_fractions": [float(x) for x in "${SEEK_FRACTIONS}".split(",")],
    "latency_core_ms": float("${LATENCY_CORE_MS}"),
    "latency_periphery_ms": float("${LATENCY_PERIPHERY_MS}"),
    "seed": int("${SEED}"),
    "policies": ["gnn", "mlp", "knative"],
    "hub_configs": int("${hub_count}"),
    "total_jobs": job_count,
}
(sweep / "configs" / "sweep_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
print(f"Wrote {manifest} ({job_count} jobs)")
PY

mkdir -p "${SWEEP_DIR}/results"
echo "[+] Prepared ${SWEEP_DIR}"
echo "    Hub configs: ${hub_count} · jobs: $((hub_count * 3)) (gnn + mlp + knative per hub)"
echo "    k_core=[${K_CORE_VALUES}] seek=[${SEEK_FRACTIONS}] latency=${LATENCY_CORE_MS}/${LATENCY_PERIPHERY_MS}ms"
