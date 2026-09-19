#!/bin/bash
# peer_affinity_v1 T1 preflight (datalab-pitfalls #9): train the 5-epoch smoke GNN config on
# the R2 smoke cache under the cluster env, then run the held-out evaluator on it. Proves the
# lazy imports, the env, CUDA and the whole train->sidecar->eval chain BEFORE the 96-task arrays.
#SBATCH --job-name=pa-t1-preflight
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/pa-t1-preflight-%j.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/pa-t1-preflight-%j.err
#SBATCH --partition=GPU-a40,GPU-a100s,GPU-l40s,GPU-a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:40:00

set -euo pipefail
ulimit -n 65536 2>/dev/null || true
cd /home/nikola.lukic/gnn-herosim
if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3 PYTHONPATH="$PWD" PYTHONUNBUFFERED=1 WANDB_ENTITY=nikolalukic167-tu-wien
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
python3 -c "import sys,torch;print('[INTERP]',sys.executable,torch.__version__,'cuda',torch.cuda.is_available())"
echo "[CODE] $(git describe --dirty --always) on $(git rev-parse --abbrev-ref HEAD)"
mkdir -p logs models
rm -f models/peer-affinity-v1-smoke-gnn-seed1.pt models/peer-affinity-v1-smoke-gnn-seed1.contract.json
python3 -u run_experiment.py experiments/peer_affinity_v1_smoke_gnn.yaml --seed 1
[[ -f models/peer-affinity-v1-smoke-gnn-seed1.pt ]] || { echo "FAIL LOUD: no checkpoint" >&2; exit 1; }
export PARTIAL_STATE_CONTRACT=partial_state_v2 PARTIAL_STATE_PEER_MASS=1 EVAL_DECODE_REPLICA_REUSE=1 EVAL_DECODE_RELAX=1
python3 -u scripts_cosim/eval_route_b_stage2_arm.py --checkpoint models/peer-affinity-v1-smoke-gnn-seed1.pt \
  --cache-dir simulation_data/graphs_cache_peer_affinity_v1_r2_smoke --split-artifact experiments/peer_affinity_v1_smoke_split.json \
  --alpha-key 2.5 --report logs/pa-t1-preflight-eval.json
python3 -c "import json; r=json.load(open('logs/pa-t1-preflight-eval.json')); print('[EVAL] datasets', r['num_datasets'], 'infeasible', r['num_infeasible'], 'median regret all', r['median_regret_pct_all'])"
echo "=== preflight ok ==="
