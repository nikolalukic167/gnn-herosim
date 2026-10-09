#!/bin/bash
#SBATCH --job-name=rank
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/rank-%j.out
#SBATCH --partition=CPU-amd
#SBATCH --cpus-per-task=48
#SBATCH --mem=190G
#SBATCH --time=12:00:00
# Label-horizon ranking test (forced plans, isolated vs admitted-arrival evaluation). Needs the forced-batch hook (HEROSIM_FORCED_PLACEMENTS).
# usage: WT=<pinned worktree> EXPECT_HEAD=<sha> RANK_ROOTS=<dataset dirs, comma separated> RANK_OUT=<out dir> [RANK_ARGS="--select FILE --max N --gate-only"] sbatch --export=ALL ...
set -euo pipefail
WT=${WT:?FAIL LOUD: set WT (pinned detached worktree)}; EXPECT_HEAD=${EXPECT_HEAD:?FAIL LOUD: set EXPECT_HEAD}; cd $WT
eval "$(micromamba shell hook --shell bash)"; micromamba activate gnn
[[ $(git rev-parse HEAD) == $EXPECT_HEAD* ]] || { echo "FAIL LOUD: HEAD is not $EXPECT_HEAD"; exit 1; }; [[ -z "$(git status --porcelain -- src scripts_cosim)" ]] || { echo "FAIL LOUD: worktree dirty"; exit 1; }
: "${RANK_ROOTS:?}" "${RANK_OUT:?}"
export HEROSIM_TRANSFER_MODEL=pipelined HEROSIM_REPLICA_RELEASE=1 HEROSIM_SCALEOUT=kpa GATE_FIXED_POLICY_TIME_SCALE=1.0 HEROSIM_SHARED_AUTOSCALER=0
export HEROSIM_PEER_EXCHANGE=1 HEROSIM_SERVER_ONLY_REPLICAS=1 HEROSIM_WARMTH_PHYSICS=node_disk_v2 PYTHONHASHSEED=0 HEROSIM_GNN_DEVICE=cpu SIM_FORCE_FULL_STATS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= HEROSIM_POLICY_TIME_SCALE=1.0 PYTHONPATH=$WT
export GNN_DECODE_MODE=masked_topo GNN_BATCH_BY_PEER_GROUP=1
export HEROSIM_RAW_DIR=/tmp/rank_$SLURM_JOB_ID; mkdir -p $HEROSIM_RAW_DIR
python3 scripts_cosim/ranking_label_horizon.py "$RANK_OUT" "$RANK_ROOTS" ${RANK_ARGS:-} 2>&1 | tee ${RANK_OUT}.txt
