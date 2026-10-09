#!/bin/bash
# r1_attribution_v1 PRODUCTION corpus, step 2 of 3 -- PREPARED, NOT SUBMITTED (rungs come from S4).
# Live capture of the peer-group snapshots the corpus is built from, WITH HEROSIM_SNAPSHOT_FIDELITY=1 (the corpus build in step 3 needs
# the same flag and refuses a snapshot without the block). One array task per training topology; its rungs x windows in parallel.
# Behaviour policy as small_batch_v1_capture.sbatch (the batched rule in the decoder's seat), R1.1 environment.
#   sbatch --array=0-<n_topologies-1> --export=ALL,WT=...,ROOT=...,TOPOS="...",TAGS="lo hi" wf1_corpus_prod_capture.sbatch
#SBATCH --job-name=wf1p-cap
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/wf1p-cap-%A_%a.out
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=06:00:00
set -euo pipefail
WT=${WT:?FAIL LOUD: set WT (pinned worktree)}
ROOT=${ROOT:?FAIL LOUD: set ROOT (the step-1 output root)}
TOPOS=${TOPOS:?FAIL LOUD: set TOPOS}
TAGS=${TAGS:?FAIL LOUD: set TAGS (the rung tags of step 1)}
WINDOWS=${WINDOWS:-"g0 g1 g2 g3"}
T=${SLURM_ARRAY_TASK_ID:?FAIL LOUD: submit as an array}
TOPO_ARR=($TOPOS); SEED=${TOPO_ARR[$T]:?FAIL LOUD: array index beyond TOPOS}
cd "$WT"
if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3 PYTHONPATH="$WT" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=""
echo "HEAD $(git rev-parse HEAD) topology $SEED node $(hostname)"
export HEROSIM_TRANSFER_MODEL=pipelined HEROSIM_REPLICA_RELEASE=1 HEROSIM_SCALEOUT=kpa HEROSIM_SHARED_AUTOSCALER=0 GATE_FIXED_POLICY_TIME_SCALE=1.0
export HEROSIM_PEER_EXCHANGE=1 HEROSIM_SERVER_ONLY_REPLICAS=1 HEROSIM_WARMTH_PHYSICS=node_disk_v2 PYTHONHASHSEED=0 HEROSIM_GNN_DEVICE=cpu
export GNN_DECODE_MODE=masked_topo GNN_BATCH_BY_PEER_GROUP=1
export HEROSIM_SNAPSHOT_FIDELITY=1
unset GNN_MODEL_PATH GNN_DISABLE_MESSAGE_PASSING || true
export LIVE_AUDIT_STRIDE=${LIVE_AUDIT_STRIDE:-31} LIVE_AUDIT_MAX_SNAPSHOTS=${LIVE_AUDIT_MAX_SNAPSHOTS:-300} LIVE_AUDIT_MIN_BATCH_SIZE=2 LIVE_AUDIT_MIN_CANDIDATES=1
export HEROSIM_RAW_DIR=/tmp/herosim_raw_${SLURM_JOB_ID:-$$}; mkdir -p "$HEROSIM_RAW_DIR"; trap 'rm -rf "$HEROSIM_RAW_DIR"' EXIT
mkdir -p "$ROOT/snapshots" "$ROOT/capture_logs"
pids=()
for R in $TAGS; do
  for W in $WINDOWS; do
    CELL=cc40s${SEED}; SNAP="$ROOT/snapshots/${CELL}_${R}_${W}.jsonl"
    [[ -s "$SNAP" ]] && { echo "[$CELL $R $W] done already"; continue; }
    ( export LIVE_AUDIT_SNAPSHOT_PATH="$SNAP"; rm -f "$SNAP"
      python3 src/executesimulation.py --config "$ROOT/inputs/wf1_$R/cfg/$CELL.json" --workload "$ROOT/inputs/wf1_$R/wl/grounded_${W}_n50000.json" \
        --policy peer_greedy_network_batch --output "$HEROSIM_RAW_DIR/${CELL}_${R}_${W}.raw.json" > "$ROOT/capture_logs/${CELL}_${R}_${W}.log" 2>&1
      echo "[$CELL $R $W] rc=$? snapshots=$(wc -l < "$SNAP")" ) &
    pids+=($!)
  done
done
rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done
# a capture that hung (starved-client spin) wrote no snapshots: the build skips it, and the count is the evidence
exit $rc
