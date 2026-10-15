#!/bin/bash
#SBATCH --job-name=d0-regret
#SBATCH --output=/share/nikola.lukic/uncut_groups/d0/d0-%j.out
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
set -euo pipefail
ulimit -n 65536 2>/dev/null || true
W=${WT:?worktree}
D=/share/nikola.lukic/uncut_groups/d0
eval "$(micromamba shell hook --shell bash)"; micromamba activate gnn
cd $W
for v in $(compgen -e | grep -E "^(HEROSIM_|GNN_|COSIM_|LIVE_AUDIT_|GATE_|SIM_|QUEUE_FEATURE|MAX_PLACEMENT|NEAR_RTT_|PARTIAL_STATE|R1A_|SERVEFAST_)"); do unset "$v"; done
export HEROSIM_PY=python3 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONPATH="$W:$W/src/notebooks"
export DAG_PRIMARY_ALPHA_KEY=inf
mkdir -p $D
echo "node $(hostname) job $SLURM_JOB_ID HEAD $(git rev-parse HEAD) cpu $(lscpu | grep -m1 'Model name')"
M=/home/nikola.lukic/gnn-herosim/models
chk() { [ "$(md5sum $M/$1 | cut -c1-32)" = "$2" ] || { echo "FAIL LOUD: md5 of $1"; exit 1; }; }
chk uncut-arm1b-gnn-eng-seed1-final.pt f8b8be77761a020b1179f0853ae2f0a2
chk uncut-arm1b-gnn-eng-seed2-final.pt 95f391d1a1638f456cd3e091f01f10c6
chk uncut-arm1b-twin-eng-seed1-final.pt 2c22db5270c05c3f6a115b6a47fdf903
chk uncut-arm1b-twin-eng-seed2-final.pt 9e337dd6a9c8659e082e9292102a8fe1
DRAW=docs/lineages/uncut_groups_s0_v1/d0_draw.json
python3 scripts_cosim/uncut_replay_diag.py prep --draw $DRAW --out $D/shards --per-shard 25
python3 scripts_cosim/uncut_d0_regret.py regret --shards $D/shards --draw $DRAW --out $D/regret.json
python3 scripts_cosim/uncut_d0_regret.py tf --shards $D/shards --arm gnn_eng --out $D/tf_gnn_eng.json
python3 scripts_cosim/uncut_d0_regret.py tf --shards $D/shards --arm twin_eng --out $D/tf_twin_eng.json
python3 scripts_cosim/uncut_d0_regret.py combine --regret $D/regret.json --tf $D/tf_gnn_eng.json --tf-twin $D/tf_twin_eng.json --out $D/d0.json --md $D/d0.md
