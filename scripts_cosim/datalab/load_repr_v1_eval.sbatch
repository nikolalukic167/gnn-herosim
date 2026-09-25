#!/bin/bash
# load_repr_v1 O1 (docs/lineages/load_repr_v1.md) -- the 8 checkpoints offline on the v4 cache, exactly as
# cd_gap_v1 A1/D0 (eval_route_b_stage2_arm.py, alpha key inf, the yaml env block + EVAL_DECODE_*).
# Tasks 0-3 v4load seeds 1-4, 4-7 v4twin seeds 1-4. Runs in the worktree ~/gnn-herosim-wt/load_repr.
#   sbatch --array=0-7 load_repr_v1_eval.sbatch
#SBATCH --job-name=lr1-eval
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/lr1-eval-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/lr1-eval-%A_%a.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=08:00:00

set -euo pipefail
WT=/home/nikola.lukic/gnn-herosim-wt/load_repr
cd "$WT"
if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3
export PYTHONPATH="$WT"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=""
echo "[CODE] $(git describe --dirty --always) at $WT"

T=${SLURM_ARRAY_TASK_ID:?FAIL LOUD: SLURM_ARRAY_TASK_ID unset}
if (( T < 4 )); then ARM=v4load; SEED=$(( T + 1 )); LS=1; else ARM=v4twin; SEED=$(( T - 3 )); LS=0; fi
echo "[TASK $T] arm=$ARM seed=$SEED"

export NEAR_RTT_SAVE_FINAL=1 NEAR_RTT_SPLIT_ARTIFACT=experiments/joint_burst_v2_split.json
export NEAR_RTT_TASK_TYPE_ONEHOT=1 NEAR_RTT_PARTIAL_STATE_EDGES=1 NEAR_RTT_DAG_ALPHA_KEY=inf
export NEAR_RTT_TIED_MAX_PLANS=0 NEAR_RTT_TRAIN_OBJECTIVE=ce_only NEAR_RTT_VAL_EXACT_REGRET=1
export NEAR_RTT_DECODE_REPLICA_REUSE=1 NEAR_RTT_DECODE_RELAX=1
export PARTIAL_STATE_CONTRACT=partial_state_v4 PARTIAL_STATE_LOAD_SECONDS=$LS PARTIAL_STATE_PEER_MASS=1
export NEAR_RTT_MP_PEER_EDGES=1 NEAR_RTT_MP_BIPARTITE_EDGE_CONV=1 NEAR_RTT_MP_BIPARTITE_EDGE_ATTR_ZERO=1
export EVAL_DECODE_REPLICA_REUSE=1 EVAL_DECODE_RELAX=1
unset GNN_DISABLE_MESSAGE_PASSING TRAIN_INIT_CHECKPOINT NEAR_RTT_LABEL_OVERRIDE_JSON
env | grep -E '^(NEAR_RTT|PARTIAL_STATE|EVAL_|GNN_)' | sort

CKPT=models/load-repr-v1-${ARM}-gnnedge0-lr2e3-seed${SEED}.pt
mkdir -p simulation_data/load_repr_v1
REPORT=simulation_data/load_repr_v1/eval_${ARM}_s${SEED}.json
[[ -f "$CKPT" ]] || { echo "FAIL LOUD: checkpoint missing: $CKPT" >&2; exit 1; }
[[ -f "${CKPT%.pt}.contract.json" ]] || { echo "FAIL LOUD: sidecar missing for $CKPT" >&2; exit 1; }
[[ -f "$REPORT" ]] && { echo "FAIL LOUD: $REPORT exists; refusing to overwrite" >&2; exit 1; }
python3 scripts_cosim/eval_route_b_stage2_arm.py --checkpoint "$CKPT" \
  --cache-dir simulation_data/graphs_cache_joint_burst_v2_psv4_inf \
  --split-artifact experiments/joint_burst_v2_split.json \
  --alpha-key inf --report "$REPORT.partial"
mv "$REPORT.partial" "$REPORT"
echo "[done] $REPORT"
