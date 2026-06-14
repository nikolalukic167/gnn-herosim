#!/usr/bin/env bash
# Generate 9 hub configs + 27-job manifest for warmth+sparse+skew merged ce-reduced hub9 sweep.
set -euo pipefail

export SWEEP_DIR="${SWEEP_DIR:-simulation_data/normal_sim_sweeps/warmth_sparse_skew_merged_ce_reduced_hub9_20260612}"
export GNN_MODEL="${GNN_MODEL:-models/near-rtt-v2-warmth-sparse-skew-merged-ce-reduced.pt}"
export MLP_MODEL="${MLP_MODEL:-models/tabular/batch_edge_mlp_warmth_sparse_merged_ce_reduced.pt}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

bash scripts_cosim/important/prepare_warmth_sparse_merged_hub9_sweep.sh

# Patch sweep meta for skew-merged GNN checkpoint.
pipenv run python3 - <<PY
import json
from pathlib import Path

sweep = Path("${SWEEP_DIR}")
meta_path = sweep / "configs" / "sweep_meta.json"
meta = json.loads(meta_path.read_text(encoding="utf-8"))
meta["sweep_id"] = "warmth_sparse_skew_merged_ce_reduced_hub9"
meta["gnn_model"] = "${GNN_MODEL}"
meta["mlp_model"] = "${MLP_MODEL}"
meta["parent_sweep"] = "warmth_sparse_merged_ce_reduced_hub9"
meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
print(f"Updated {meta_path}")
PY

echo "[+] Skew-merged hub9 sweep ready: ${SWEEP_DIR}"
echo "    GNN: ${GNN_MODEL}"
