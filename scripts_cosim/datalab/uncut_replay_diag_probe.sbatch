#!/bin/bash
#SBATCH --job-name=rdiag-probe
#SBATCH --output=/share/nikola.lukic/uncut_groups/replaydiag/probe-%j.out
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
set -euo pipefail
W=${WT:?worktree}
eval "$(micromamba shell hook --shell bash)"; micromamba activate gnn
cd $W
for v in $(compgen -e | grep -E "^(HEROSIM_|GNN_|COSIM_|LIVE_AUDIT_|GATE_|SIM_|QUEUE_FEATURE|MAX_PLACEMENT|NEAR_RTT_|PARTIAL_STATE|R1A_)"); do unset "$v"; done
export HEROSIM_PY=python3 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONPATH="$W:$W/src/notebooks"
echo "node $(hostname) job $SLURM_JOB_ID HEAD $(git rev-parse HEAD) cpu $(lscpu | grep -m1 'Model name')"
python3 scripts_cosim/uncut_replay_diag.py probe --groups "${GROUPS_CSV:?}"
