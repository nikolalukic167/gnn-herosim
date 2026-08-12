#!/bin/bash
#SBATCH --job-name=rb-zs
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/regime-b-zeroshot-%j.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/regime-b-zeroshot-%j.err
#SBATCH --partition=GPU-l40s,GPU-a40,GPU-a100s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00

# Regime B live-stub zero-shot: Kn / MLP / GNN on N=12 platform_reuse_v1 cell.
# Requires identical 873/v5.5 ckpts (md5 checked in runner).

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
export INFERENCE_FEATURE_LAYOUT=dim22
export GNN_DECODE_MODE=argmax
export GNN_MODEL="${GNN_MODEL:-models/near-rtt-v2-contention-v2-873-v5.5-dim14-ce-only.pt}"
export MLP_MODEL="${MLP_MODEL:-models/tabular/batch_edge_mlp_contention_v2_873_v5.5_dim22_batchcache.pt}"

echo "=== Regime B zero-shot ==="
echo "host=$(hostname)  cuda=${CUDA_VISIBLE_DEVICES:-none}"
md5sum "$GNN_MODEL" "$MLP_MODEL"

python3 scripts_cosim/build_regime_b_live_stub.py
python3 scripts_cosim/calibrate_regime_b.py --mode target
python3 scripts_cosim/run_regime_b_live_stub_baselines.py \
  --stub-dir simulation_data/regime_b_cold_burst_v1/live_stub \
  --output-dir simulation_data/normal_sim_sweeps/regime_b_cold_burst_v1_zeroshot \
  --policies knative,mlp,gnn

echo "=== compare.json ==="
python3 - <<'PY'
import json
from pathlib import Path
p = Path("simulation_data/normal_sim_sweeps/regime_b_cold_burst_v1_zeroshot/compare.json")
d = json.loads(p.read_text())
print("winner:", d.get("winner"))
for row in d.get("ranking_primary_asc") or []:
    print(f"  {row['policy']:12s}  {row['regime_b_primary_score_s']:.3f}s")
PY
