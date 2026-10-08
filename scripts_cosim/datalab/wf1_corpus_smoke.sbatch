#!/bin/bash
# r1_attribution_v1 preparation (smoke, not a production run): 5 datasets of the unmodified corpus generator on two
# calibration topologies, under the R1 environment. WT and EXPECT_HEAD name the pinned detached worktree.
#SBATCH --job-name=wf1cs
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/wf1cs-%j.out
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=04:00:00
set -euo pipefail
WT=${WT:?FAIL LOUD: set WT}
EXPECT_HEAD=${EXPECT_HEAD:?FAIL LOUD: set EXPECT_HEAD}
OUT=${OUT:?FAIL LOUD: set OUT (under $WT/simulation_data/; the generator writes relative to its repo root)}
cd "$WT"
if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3 PYTHONPATH="$WT" PYTHONUNBUFFERED=1
HEAD=$(git rev-parse HEAD)
echo "HEAD $HEAD node $(hostname) OUT=$OUT"
[[ "$HEAD" == "$EXPECT_HEAD"* ]] || { echo "FAIL LOUD: worktree HEAD $HEAD is not $EXPECT_HEAD"; exit 1; }
[[ ! -e "$OUT" ]] || { echo "FAIL LOUD: $OUT exists"; exit 1; }
export HEROSIM_TRANSFER_MODEL=pipelined HEROSIM_REPLICA_RELEASE=1 HEROSIM_SCALEOUT=kpa HEROSIM_SHARED_AUTOSCALER=0
export GATE_FIXED_POLICY_TIME_SCALE=1.0
export HEROSIM_PEER_EXCHANGE=1 HEROSIM_RETAIN_TASK_TIMES=1 HEROSIM_RETAIN_PEER_STATS=1
export MAX_PLACEMENT_COMBINATIONS_SKIP=2000000 GNN_CAPTURE_DATASET_STATE=0 COSIM_SUPPRESS_SIM_PRINTS=1
export HEROSIM_TRACE_DATASET_ERRORS=1
python3 -c "import sys,torch;print('[INTERP]',sys.executable,torch.__version__)"
SUB=${OUT#$WT/simulation_data/}
python3 -u scripts_cosim/wf1_corpus_smoke.py --no-quiet --grid wf1_corpus_smoke --num-tasks "${NUM_TASKS:-6}" \
  --max-datasets 5 --allow-non-unique-replicas --warmth-physics node_disk_v2 --output-subdir "$SUB" \
  --progress-log-name progress_wf1_corpus_smoke.txt --workers 30
echo "=== done: $(ls -d "$OUT"/ds_* 2>/dev/null | wc -l) datasets, $(ls "$OUT"/ds_*/placements/placements.jsonl 2>/dev/null | wc -l) with placements.jsonl ==="
