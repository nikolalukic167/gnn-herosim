#!/bin/bash
#SBATCH --job-name=rdiag-prep
#SBATCH --output=/share/nikola.lukic/uncut_groups/replaydiag/prep-%j.out
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
set -euo pipefail
ulimit -n 65536 2>/dev/null || true
W=${WT:?worktree}
D=/share/nikola.lukic/uncut_groups/replaydiag
eval "$(micromamba shell hook --shell bash)"; micromamba activate gnn
cd $W
for v in $(compgen -e | grep -E "^(HEROSIM_|GNN_|COSIM_|LIVE_AUDIT_|GATE_|SIM_|QUEUE_FEATURE|MAX_PLACEMENT|NEAR_RTT_|PARTIAL_STATE|R1A_)"); do unset "$v"; done
export HEROSIM_PY=python3 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONPATH="$W:$W/src/notebooks"
echo "node $(hostname) job $SLURM_JOB_ID HEAD $(git rev-parse HEAD)"
python3 scripts_cosim/uncut_replay_diag.py prep --draw docs/lineages/uncut_groups_s0_v1/replay_diag_draw.json --out $D/shards --per-shard 25
