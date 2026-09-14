#!/bin/bash
# peer_affinity_warm_v1 -- W1 offline read, part 1: score every warm checkpoint (val-selected
# AND -final) against the warm held-out block, one array task per (arm, seed). Adapted from
# peer_affinity_v1_x800_score.sbatch: W1 trains one lr (2e-3) and two arms (gnn, mpoff).
#   T = 16*arm + (seed-1);  arms gnn, mpoff
#   sbatch scripts_cosim/datalab/peer_affinity_warm_v1_w1_score.sbatch
#
#SBATCH --job-name=pa-warm-w1-score
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/pa-warm-w1-score-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/pa-warm-w1-score-%A_%a.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --array=0-31

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

TAG=warm
CACHE=simulation_data/graphs_cache_peer_affinity_v1_warm
SPLIT=experiments/peer_affinity_v1_warm_split.json
REPORTS=simulation_data/peer_affinity_${TAG}_reports
mkdir -p "$REPORTS" logs
[[ -f "$CACHE/metadata.json" ]] || { echo "FAIL LOUD: no cache at $CACHE" >&2; exit 1; }
[[ -f "$SPLIT" ]] || { echo "FAIL LOUD: no split at $SPLIT" >&2; exit 1; }

T=${SLURM_ARRAY_TASK_ID}
ARMS=(gnn mpoff); LR=lr2e3
ARM=${ARMS[$(( T / 16 ))]}; SEED=$(( T % 16 + 1 ))
echo "[TASK $T] arm=$ARM lr=$LR seed=$SEED"
echo "[CODE] $(git describe --dirty --always) on $(git rev-parse --abbrev-ref HEAD)"

# the arm environment the registration fixes (same as peer_affinity_t1_read.arm_env)
export PARTIAL_STATE_CONTRACT=partial_state_v2 EVAL_DECODE_REPLICA_REUSE=1 EVAL_DECODE_RELAX=1
export PARTIAL_STATE_PEER_MASS=1
if [[ $ARM == mpoff ]]; then export GNN_DISABLE_MESSAGE_PASSING=1; else unset GNN_DISABLE_MESSAGE_PASSING; fi

score () {  # $1 = checkpoint, $2 = report
  [[ -f "$2" ]] && { echo "[skip] report exists: $2"; return 0; }
  [[ -f "$1" ]] || { echo "FAIL LOUD: checkpoint missing: $1" >&2; exit 1; }
  [[ -f "${1%.pt}.contract.json" ]] || { echo "FAIL LOUD: sidecar missing for $1" >&2; exit 1; }
  python3 scripts_cosim/eval_route_b_stage2_arm.py --checkpoint "$1" --cache-dir "$CACHE" \
    --split-artifact "$SPLIT" --alpha-key 2.5 --report "$2.partial"
  mv "$2.partial" "$2"
  echo "[done] $2"
}

score "models/peer-affinity-v1-${TAG}-${ARM}-${LR}-seed${SEED}.pt" \
      "$REPORTS/peer-affinity-v1-${TAG}-${ARM}-${LR}-seed${SEED}.json"
score "models/peer-affinity-v1-${TAG}-${ARM}-${LR}-seed${SEED}-final.pt" \
      "$REPORTS/peer-affinity-v1-${TAG}-${ARM}-${LR}-seed${SEED}-final.json"
echo "=== task $T done ==="
