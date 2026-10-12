#!/bin/bash
# extend_corpus_slate production run, one split, array shards. The source corpus is read only; the output is a new directory.
#   sbatch --array=0-2 --export=ALL,WT=<pinned worktree>,EXPECT_HEAD=<sha>,SPLIT=train|heldout,NSHARD=3 extend_corpus_run.sbatch
#SBATCH --job-name=ext-run
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/ext-run-%A_%a.out
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=24:00:00
set -euo pipefail
WT=${WT:?FAIL LOUD: set WT}; EXPECT_HEAD=${EXPECT_HEAD:?}; SPLIT=${SPLIT:?}; NSHARD=${NSHARD:?}
cd "$WT"
for v in $(compgen -e | grep -E '^(HEROSIM_|GNN_|COSIM_|LIVE_AUDIT_|GATE_|SIM_|QUEUE_FEATURE|MAX_PLACEMENT|NEAR_RTT_|PARTIAL_STATE|WF1_)'); do unset "$v"; done
if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"; micromamba activate gnn
HEAD=$(git rev-parse HEAD); echo "HEAD $HEAD node $(hostname) job=${SLURM_JOB_ID:-} task=${SLURM_ARRAY_TASK_ID:-0}/$NSHARD split=$SPLIT"
[[ "$HEAD" == "$EXPECT_HEAD"* ]] || { echo "FAIL LOUD: worktree HEAD $HEAD is not $EXPECT_HEAD"; exit 1; }
export PYTHONPATH="$WT" PYTHONUNBUFFERED=1 EXTEND_CODE_COMMIT=$HEAD
export COSIM_SUPPRESS_SIM_PRINTS=1 GNN_CAPTURE_DATASET_STATE=0 HEROSIM_PEER_EXCHANGE=1 HEROSIM_PY=python3 HEROSIM_REPLICA_RELEASE=1 HEROSIM_RETAIN_PEER_STATS=1 \
  HEROSIM_RETAIN_TASK_TIMES=1 HEROSIM_SCALEOUT=kpa HEROSIM_SERVER_ONLY_REPLICAS=1 HEROSIM_SHARED_AUTOSCALER=0 HEROSIM_SNAPSHOT_FIDELITY=1 \
  HEROSIM_TRANSFER_MODEL=pipelined HEROSIM_WARMTH_PHYSICS=node_disk_v2 MAX_PLACEMENT_COMBINATIONS_SKIP=2000000 PYTHONHASHSEED=0 COSIM_PLACEMENT_TIMEOUT_S=120
SRC=/share/nikola.lukic/workload_fix_v1/accel_replica_corpus/gnn_datasets_wf1_$SPLIT
OUT=/share/nikola.lukic/accel_uncap/${HEAD:0:8}/$SPLIT
mkdir -p $OUT
python3 scripts_cosim/extend_corpus_slate.py --src $SRC --out $OUT --shard ${SLURM_ARRAY_TASK_ID:-0}/$NSHARD --workers 64 --verify 20 --max-plans 100000 --link-untouched
echo "shard done"
