#!/bin/bash
#SBATCH --job-name=w3-prep
#SBATCH --output=/share/nikola.lukic/uncut_groups/w3/w3-prep-%A_%a.out
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
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
M=/home/nikola.lukic/gnn-herosim/models
[ "$(md5sum $M/uncut-arm1b-gnn-eng-seed1-final.pt | cut -c1-32)" = f8b8be77761a020b1179f0853ae2f0a2 ] || { echo "FAIL LOUD: s1 md5"; exit 1; }
[ "$(md5sum $M/uncut-arm1b-gnn-eng-seed2-final.pt | cut -c1-32)" = 95f391d1a1638f456cd3e091f01f10c6 ] || { echo "FAIL LOUD: s2 md5"; exit 1; }
DRAW=docs/lineages/uncut_groups_s0_v1/w3_draw.json
python3 scripts_cosim/uncut_replay_diag.py prep --draw $DRAW --out $D/shards --per-shard 25
mkdir -p $D/decode
for S in $D/shards/g_*.pkl; do
  T=$(basename $S .pkl); T=${T#g_}
  python3 scripts_cosim/uncut_replay_diag.py decode --arm gnn_eng --runs gnn_eng_s1_final,gnn_eng_s2_final --shard $S --out $D/decode/d_${T}_gnn_eng.json
done
python3 scripts_cosim/uncut_w3_prefix.py points --shards $D/shards --decode-dir $D/decode --draw $DRAW --out $D/points.json
