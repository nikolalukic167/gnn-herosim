#!/bin/bash
#SBATCH --job-name=lever-simmerge
#SBATCH --output=/share/nikola.lukic/uncut_groups/lever_sim/lever-simmerge-%A_%a.out
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
set -euo pipefail
ulimit -n 65536 2>/dev/null || true
W=${WT:?worktree}
D=/share/nikola.lukic/uncut_groups/lever_sim
eval "$(micromamba shell hook --shell bash)"; micromamba activate gnn
cd $W
for v in $(compgen -e | grep -E "^(HEROSIM_|GNN_|COSIM_|LIVE_AUDIT_|GATE_|SIM_|QUEUE_FEATURE|MAX_PLACEMENT|NEAR_RTT_|PARTIAL_STATE|R1A_|SERVEFAST_)"); do unset "$v"; done
export HEROSIM_PY=python3 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONPATH="$W:$W/src/notebooks"
export DAG_PRIMARY_ALPHA_KEY=inf
CACHE=/share/nikola.lukic/uncut_arm1/cache_run/cache/graphs.pkl
mkdir -p $D
echo "node $(hostname) job $SLURM_JOB_ID task ${SLURM_ARRAY_TASK_ID:-0} HEAD $(git rev-parse HEAD) cpu $(lscpu | grep -m1 'Model name')"
WHICH=${WHICH:-train}
OUT=$D/move_regret_simlabels.pkl; [ "$WHICH" = val ] && OUT=$D/unscored_mask_val.pkl
python3 scripts_cosim/uncut_move_regret_label.py merge --which $WHICH --parts $D/parts_$WHICH --labels ${LABELS:?FAIL LOUD: set LABELS} --out $OUT
md5sum $OUT; sha256sum $OUT
