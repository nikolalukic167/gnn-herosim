#!/usr/bin/env bash
# Generate 18-job manifest for hub9 MLP ablation (9 seqblend1 + 9 dim22).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

SWEEP_DIR="${SWEEP_DIR:-simulation_data/normal_sim_sweeps/warmth_sparse_merged_ce_reduced_hub9_20260611}"

if [[ ! -d "${SWEEP_DIR}/configs" ]]; then
  bash scripts_cosim/important/prepare_warmth_sparse_merged_hub9_sweep.sh
fi

pipenv run python3 - <<PY
import json
from pathlib import Path

sweep = Path("${SWEEP_DIR}")
configs_dir = sweep / "configs"
hub_jsons = sorted(configs_dir.glob("hub_k*.json"))
if len(hub_jsons) != 9:
    raise SystemExit(f"expected 9 hub configs, got {len(hub_jsons)}")

lines = ["policy\tconfig_name\tconfig_path"]
for path in hub_jsons:
    name = path.stem
    rel = path.as_posix()
    lines.append(f"mlp_seqblend1\t{name}\t{rel}")
for path in hub_jsons:
    name = path.stem
    rel = path.as_posix()
    lines.append(f"mlp_dim22\t{name}\t{rel}")

manifest = configs_dir / "jobs_smcr_hub9_ablation.tsv"
manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
job_count = len(lines) - 1
if job_count != 18:
    raise SystemExit(f"expected 18 jobs, got {job_count}")

meta = {
    "sweep_id": "warmth_sparse_merged_ce_reduced_hub9_ablation",
    "parent_sweep": "warmth_sparse_merged_ce_reduced_hub9",
    "workload": "data/nofs-ids/traces/workload-125-225.json",
    "warmth_physics": "node_disk_v2",
    "policies": {
        "mlp_seqblend1": {
            "mlp_model": "models/tabular/batch_edge_mlp_warmth_sparse_merged_ce_reduced.pt",
            "inference_feature_layout": "ce_reduced",
            "gnn_decode_mode": "seqblend",
            "gnn_seqblend_queue_margin": 1,
            "output_suffix": "_mlp_sparse_merged_ce_reduced_seqblend1.json",
        },
        "mlp_dim22": {
            "mlp_model": "models/tabular/batch_edge_mlp.pt",
            "inference_feature_layout": "dim22",
            "gnn_decode_mode": "argmax",
            "output_suffix": "_mlp_dim22.json",
        },
    },
    "hub_configs": 9,
    "total_jobs": job_count,
}
(sweep / "configs" / "sweep_ablation_meta.json").write_text(
    json.dumps(meta, indent=2) + "\n", encoding="utf-8"
)
print(f"Wrote {manifest} ({job_count} jobs)")
PY

mkdir -p "${SWEEP_DIR}/results"
echo "[+] Prepared ablation manifest under ${SWEEP_DIR}"
