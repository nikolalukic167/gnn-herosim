#!/bin/bash
#SBATCH --job-name=rdiag-decode
#SBATCH --output=/share/nikola.lukic/uncut_groups/replaydiag/decode-%A_%a.out
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
set -euo pipefail
W=${WT:?worktree}
D=/share/nikola.lukic/uncut_groups/replaydiag
eval "$(micromamba shell hook --shell bash)"; micromamba activate gnn
cd $W
for v in $(compgen -e | grep -E "^(HEROSIM_|GNN_|COSIM_|LIVE_AUDIT_|GATE_|SIM_|QUEUE_FEATURE|MAX_PLACEMENT|NEAR_RTT_|PARTIAL_STATE|R1A_)"); do unset "$v"; done
export HEROSIM_PY=python3 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=${THREADS:-1} MKL_NUM_THREADS=${THREADS:-1} CUDA_VISIBLE_DEVICES= PYTHONPATH="$W:$W/src/notebooks"
export DAG_PRIMARY_ALPHA_KEY=inf
mkdir -p $D/${DEC:-decode}
echo "node $(hostname) job $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID HEAD $(git rev-parse HEAD) cpu $(lscpu | grep -m1 'Model name')"
NS=$(ls $D/shards/g_*.pkl | wc -l); STRIDE=${STRIDE:-8}
for K in $(seq "$SLURM_ARRAY_TASK_ID" "$STRIDE" $((NS - 1))); do
  T=$(printf "%03d" "$K")
  for ARM in gnn_eng twin_eng; do
    python3 scripts_cosim/uncut_replay_diag.py decode --arm $ARM --shard $D/shards/g_$T.pkl --out $D/${DEC:-decode}/d_${T}_$ARM.json
  done
done
