#!/bin/bash
#SBATCH --job-name=lever-tests
#SBATCH --output=/share/nikola.lukic/uncut_groups/lever/tests-%j.out
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=02:00:00
set -euo pipefail
ulimit -n 65536 2>/dev/null || true
W=${WT:?worktree}
eval "$(micromamba shell hook --shell bash)"; micromamba activate gnn
cd $W
for v in $(compgen -e | grep -E "^(HEROSIM_|GNN_|COSIM_|LIVE_AUDIT_|GATE_|SIM_|QUEUE_FEATURE|MAX_PLACEMENT|NEAR_RTT_|PARTIAL_STATE|R1A_|SERVEFAST_)"); do unset "$v"; done
export HEROSIM_PY=python3 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONPATH="$W:$W/src/notebooks"
echo "node $(hostname) job $SLURM_JOB_ID HEAD $(git rev-parse HEAD) cpu $(lscpu | grep -m1 'Model name')"
python3 -m pytest tests/test_batched_tf.py tests/test_trainer_determinism.py ${EXTRA_TESTS:-} -q -p no:cacheprovider 2>&1 | tail -${TAIL:-40}
