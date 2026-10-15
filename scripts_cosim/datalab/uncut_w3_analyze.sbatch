#!/bin/bash
#SBATCH --job-name=w3-analyze
#SBATCH --output=/share/nikola.lukic/uncut_groups/w3/w3-analyze-%A_%a.out
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
set -euo pipefail
ulimit -n 65536 2>/dev/null || true
W=${WT:?worktree}
D=/share/nikola.lukic/uncut_groups/w3
eval "$(micromamba shell hook --shell bash)"; micromamba activate gnn
cd $W
for v in $(compgen -e | grep -E "^(HEROSIM_|GNN_|COSIM_|LIVE_AUDIT_|GATE_|SIM_|QUEUE_FEATURE|MAX_PLACEMENT|NEAR_RTT_|PARTIAL_STATE|R1A_|SERVEFAST_)"); do unset "$v"; done
export HEROSIM_PY=python3 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONPATH="$W:$W/src/notebooks"
export DAG_PRIMARY_ALPHA_KEY=inf
mkdir -p $D
echo "node $(hostname) job $SLURM_JOB_ID task ${SLURM_ARRAY_TASK_ID:-0} HEAD $(git rev-parse HEAD) cpu $(lscpu | grep -m1 'Model name')"
python3 scripts_cosim/uncut_w3_prefix.py analyze --rows-dir $D/rows --out $D/w3_analysis.json --md $D/w3_analysis.md
