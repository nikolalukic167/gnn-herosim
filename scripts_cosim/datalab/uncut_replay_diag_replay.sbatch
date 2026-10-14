#!/bin/bash
#SBATCH --job-name=rdiag-replay
#SBATCH --output=/share/nikola.lukic/uncut_groups/replaydiag/replay-%A_%a.out
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=04:00:00
set -euo pipefail
W=${WT:?worktree}
D=/share/nikola.lukic/uncut_groups/replaydiag
eval "$(micromamba shell hook --shell bash)"; micromamba activate gnn
cd $W
for v in $(compgen -e | grep -E "^(HEROSIM_|GNN_|COSIM_|LIVE_AUDIT_|GATE_|SIM_|QUEUE_FEATURE|MAX_PLACEMENT|NEAR_RTT_|PARTIAL_STATE|R1A_)"); do unset "$v"; done
export HEROSIM_PY=python3 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONPATH="$W:$W/src/notebooks"
export DAG_PRIMARY_ALPHA_KEY=inf
mkdir -p $D/replay
NS=$(ls $D/shards/g_*.pkl | wc -l)
echo "node $(hostname) job $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID HEAD $(git rev-parse HEAD) cpu $(lscpu | grep -m1 'Model name')"
python3 scripts_cosim/uncut_replay_diag.py replay --shards $D/shards --n-shards $NS --task $SLURM_ARRAY_TASK_ID --stride 8 --decode-dir $D/decode --draw docs/lineages/uncut_groups_s0_v1/replay_diag_draw.json --out $D/replay/r_$(printf "%02d" $SLURM_ARRAY_TASK_ID).json
