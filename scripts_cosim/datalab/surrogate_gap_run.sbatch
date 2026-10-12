#!/bin/bash
# surrogate_gap_screen on the uncapped accel corpus (full candidate set, <= 1e5 plans): select, run (S, ib0 S, GNN plan score), walls, read.
#   sbatch --export=ALL,WT=<pinned worktree>,EXPECT_HEAD=<sha>,O=<out dir>,PER_N=60,GNN_MAX=20000,WALLS=3 surrogate_gap_run.sbatch
# PER_N=2 WALLS=1 is the smoke. The corpus is read only.
#SBATCH --job-name=sgap
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/sgap-%j.out
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=24:00:00
set -euo pipefail
WT=${WT:?FAIL LOUD: set WT}; EXPECT_HEAD=${EXPECT_HEAD:?FAIL LOUD: set EXPECT_HEAD}; O=${O:?FAIL LOUD: set O}
PER_N=${PER_N:-60}; GNN_MAX=${GNN_MAX:-20000}; WALLS=${WALLS:-3}; WORKERS=${WORKERS:-30}
cd "$WT"
for v in $(compgen -e | grep -E '^(HEROSIM_|GNN_|COSIM_|LIVE_AUDIT_|GATE_|SIM_|QUEUE_FEATURE|MAX_PLACEMENT|NEAR_RTT_|PARTIAL_STATE|WF1_)'); do unset "$v"; done
if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"; micromamba activate gnn
HEAD=$(git rev-parse HEAD); echo "HEAD $HEAD node $(hostname) job=${SLURM_JOB_ID:-}"
[[ "$HEAD" == "$EXPECT_HEAD"* ]] || { echo "FAIL LOUD: worktree HEAD $HEAD is not $EXPECT_HEAD"; exit 1; }
export PYTHONPATH="$WT" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
export COSIM_SUPPRESS_SIM_PRINTS=1 GNN_CAPTURE_DATASET_STATE=0 HEROSIM_PEER_EXCHANGE=1 HEROSIM_PY=python3 HEROSIM_REPLICA_RELEASE=1 HEROSIM_RETAIN_PEER_STATS=1 \
  HEROSIM_RETAIN_TASK_TIMES=1 HEROSIM_SCALEOUT=kpa HEROSIM_SERVER_ONLY_REPLICAS=1 HEROSIM_SHARED_AUTOSCALER=0 HEROSIM_SNAPSHOT_FIDELITY=1 \
  HEROSIM_TRANSFER_MODEL=pipelined HEROSIM_WARMTH_PHYSICS=node_disk_v2 MAX_PLACEMENT_COMBINATIONS_SKIP=2000000 PYTHONHASHSEED=0
U=/share/nikola.lukic/accel_uncap/4f41b771
CK=/share/nikola.lukic/accel_prod/gate_inputs/models/accel-replica-v1-gnn-eng-seed1.pt
mkdir -p "$O"
python3 -m pytest tests/test_surrogate_gap_screen.py -q 2>&1 | tail -3
python3 scripts_cosim/surrogate_gap_screen.py select $O/states.tsv $U/heldout:heldout $U/train:train --per-n $PER_N
python3 scripts_cosim/surrogate_gap_screen.py walls $O/walls.jsonl $O/states.tsv --per-cell $WALLS --plans 5 --workers 16
python3 scripts_cosim/surrogate_gap_screen.py run $O/rows.jsonl $O/states.tsv --ckpt $CK --gnn-max-plans $GNN_MAX --workers $WORKERS
python3 scripts_cosim/surrogate_gap_screen.py read $O/rows.jsonl --walls $O/walls.jsonl --json $O/read.json | tee $O/read.txt
