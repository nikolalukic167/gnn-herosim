#!/bin/bash
# peer_affinity_v1 -- denser-graph rung: score every trained checkpoint against the held-out
# block, one array task per (arm, lr, seed), then aggregate with the registered read.
# The read script (peer_affinity_t1_read.py) runs these evals itself but serially; here each
# one is a task, and the read then finds every report already on disk and only aggregates.
# GNN arms carry a `-final` twin (NEAR_RTT_SAVE_FINAL=1) and both are scored, because the
# registration reports the val-selected AND the last-epoch checkpoint for every GNN arm.
#   T = 48*arm + 16*lr + (seed-1);  arms gnn, mpoff, mlp_t1, mlp_t1x;  lrs 5e4, 1e3, 2e3
#   sbatch --export=ALL,RUNG=p3 scripts_cosim/datalab/peer_affinity_v1_x800_score.sbatch
#
#SBATCH --job-name=pa-x800-score
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/pa-x800-score-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/pa-x800-score-%A_%a.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --array=0-191

set -euo pipefail
cd /home/nikola.lukic/gnn-herosim
if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3
export PYTHONPATH="$PWD"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=""

RUNG="${RUNG:?set RUNG=p3 or RUNG=p2}"
TAG="x800${RUNG}"
CACHE=simulation_data/graphs_cache_peer_affinity_v1_x800_${RUNG}
SPLIT=experiments/peer_affinity_v1_x800_${RUNG}_split.json
REPORTS=simulation_data/peer_affinity_${TAG}_reports
mkdir -p "$REPORTS" logs
[[ -f "$CACHE/metadata.json" ]] || { echo "FAIL LOUD: no cache at $CACHE" >&2; exit 1; }
[[ -f "$SPLIT" ]] || { echo "FAIL LOUD: no split at $SPLIT" >&2; exit 1; }

T=${SLURM_ARRAY_TASK_ID}
ARMS=(gnn mpoff mlp_t1 mlp_t1x); LRS=(lr5e4 lr1e3 lr2e3)
ARM=${ARMS[$(( T / 48 ))]}; LR=${LRS[$(( (T % 48) / 16 ))]}; SEED=$(( T % 16 + 1 ))
echo "[TASK $T] rung=$RUNG arm=$ARM lr=$LR seed=$SEED"

# the arm environment the registration fixes (same as peer_affinity_t1_read.arm_env)
export PARTIAL_STATE_CONTRACT=partial_state_v2 EVAL_DECODE_REPLICA_REUSE=1 EVAL_DECODE_RELAX=1
if [[ $ARM == mlp_t1 ]]; then export PARTIAL_STATE_PEER_MASS=0; else export PARTIAL_STATE_PEER_MASS=1; fi
if [[ $ARM == mpoff ]]; then export GNN_DISABLE_MESSAGE_PASSING=1; else unset GNN_DISABLE_MESSAGE_PASSING; fi

score () {  # $1 = checkpoint, $2 = report
  [[ -f "$2" ]] && { echo "[skip] report exists: $2"; return 0; }
  [[ -f "$1" ]] || { echo "FAIL LOUD: checkpoint missing: $1" >&2; exit 1; }
  python3 scripts_cosim/eval_route_b_stage2_arm.py --checkpoint "$1" --cache-dir "$CACHE" \
    --split-artifact "$SPLIT" --alpha-key 2.5 --report "$2.partial"
  mv "$2.partial" "$2"
  echo "[done] $2"
}

if [[ $ARM == mlp_t1 || $ARM == mlp_t1x ]]; then
  score "models/tabular/peer-affinity-v1-${TAG}-${ARM}-${LR}_seed${SEED}.pt" \
        "$REPORTS/peer-affinity-v1-${TAG}-${ARM}-${LR}_seed${SEED}.json"
else
  score "models/peer-affinity-v1-${TAG}-${ARM}-${LR}-seed${SEED}.pt" \
        "$REPORTS/peer-affinity-v1-${TAG}-${ARM}-${LR}-seed${SEED}.json"
  score "models/peer-affinity-v1-${TAG}-${ARM}-${LR}-seed${SEED}-final.pt" \
        "$REPORTS/peer-affinity-v1-${TAG}-${ARM}-${LR}-seed${SEED}-final.json"
fi
echo "=== task $T done ==="
