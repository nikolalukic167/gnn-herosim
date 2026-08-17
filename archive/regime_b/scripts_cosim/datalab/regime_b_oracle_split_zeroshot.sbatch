#!/bin/bash
#SBATCH --job-name=rb-zs-rb
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/regime-b-oracle-split-zs-%j.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/regime-b-oracle-split-zs-%j.err
#SBATCH --partition=GPU-l40s,GPU-a40,GPU-a100s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00

# Regime B oracle_split_v1 zero-shot with Regime B–trained ckpts (post-retrain).

set -euo pipefail
PROJECT_ROOT="/home/nikola.lukic/gnn-herosim"
cd "$PROJECT_ROOT"

ENV_NAME="${HEROSIM_ENV_NAME:-gnn}"
if ! command -v micromamba >/dev/null 2>&1; then
  source "${HOME}/.bashrc"
fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate "${ENV_NAME}"

export PYTHONUNBUFFERED=1
export SIM_FORCE_FULL_STATS=1
export GNN_CAPTURE_DATASET_STATE=0
export HEROSIM_WARMTH_PHYSICS=platform_reuse_v1
export GNN_BATCH_SIZE=4
export GNN_BATCH_TIMEOUT=0.002
export INFERENCE_FEATURE_LAYOUT=dim24
export GNN_DECODE_MODE=argmax
export GNN_MODEL="${GNN_MODEL:-models/near-rtt-v2-regime-b-cold-burst-v1-dim16-ce-only.pt}"
export MLP_MODEL="${MLP_MODEL:-models/tabular/batch_edge_mlp_regime_b_cold_burst_v1_dim24_batchcache.pt}"
export EXPECTED_GNN_MD5_PREFIX="${EXPECTED_GNN_MD5_PREFIX:-any}"
export EXPECTED_MLP_MD5_PREFIX="${EXPECTED_MLP_MD5_PREFIX:-any}"

echo "=== Regime B oracle_split_v1 zero-shot (Regime B dim16/dim24 pull-obs ckpts) ==="
echo "host=$(hostname)  cuda=${CUDA_VISIBLE_DEVICES:-none}"
md5sum "$GNN_MODEL" "$MLP_MODEL"

python3 scripts_cosim/build_regime_b_live_stub.py --variant oracle_split_v1
python3 scripts_cosim/run_regime_b_live_stub_baselines.py \
  --variant oracle_split_v1 \
  --output-dir simulation_data/normal_sim_sweeps/regime_b_cold_burst_v1_oracle_split_v1_pullobs_zeroshot

python3 - <<'PY'
import json
from pathlib import Path
p = Path("simulation_data/normal_sim_sweeps/regime_b_cold_burst_v1_oracle_split_v1_pullobs_zeroshot/compare.json")
d = json.loads(p.read_text())
print("winner=", d.get("winner"))
print("physics_ceiling_pass=", d.get("physics_ceiling_pass"))
print("physics_ceiling_s=", d.get("physics_ceiling_s"))
print("intel_margin_s=", d.get("intelligence_margin_s"))
print("intel_margin_pass=", d.get("intelligence_margin_pass"))
for row in d.get("ranking_primary_asc") or []:
    print(f"  {row['policy']:12s}  {row['regime_b_primary_score_s']:.3f}s")
for name in ("oracle_parallel", "ect_pull", "knative", "mlp", "gnn"):
    pol = (d.get("policies") or {}).get(name) or {}
    primary = pol.get("regime_b_primary_score_s")
    if primary is not None:
        print(f"detail {name}: {primary:.3f}s ratio={pol.get('oracle_regret_ratio')}")
PY
