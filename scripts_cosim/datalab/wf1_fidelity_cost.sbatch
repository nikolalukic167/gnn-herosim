#!/bin/bash
# r1_attribution_v1 preparation: cost of a co-sim plan under the I11-validated autoscaler (snapshot fidelity).
# 1. recapture the peer-group snapshots of the two calibration topologies with HEROSIM_SNAPSHOT_FIDELITY=1;
# 2. replay plans of the 6 smoke datasets (wf1_corpus_smoke_b.sbatch) from those snapshots and report cost / failures.
#   sbatch --export=ALL,WT=<worktree>,CORP=<corpus_check_b dir>[,SAMPLE=60][,ALL=ds_00002] wf1_fidelity_cost.sbatch
#SBATCH --job-name=wf1fid
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/wf1fid-%j.out
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=08:00:00
set -euo pipefail
WT=${WT:?FAIL LOUD: set WT}
CORP=${CORP:?FAIL LOUD: set CORP}
SAMPLE=${SAMPLE:-0}
ALL=${ALL:-}
TOPOS=${TOPOS:-"9601 9607"}
OUT=${OUT:-$CORP/fidelity_cost}
cd "$WT"
if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3 PYTHONPATH="$WT" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
echo "HEAD $(git rev-parse HEAD) node $(hostname)"
[[ ! -e "$OUT" ]] || { echo "FAIL LOUD: $OUT exists"; exit 1; }
mkdir -p "$OUT/snapshots"
export HEROSIM_TRANSFER_MODEL=pipelined HEROSIM_REPLICA_RELEASE=1 HEROSIM_SCALEOUT=kpa HEROSIM_SHARED_AUTOSCALER=0 GATE_FIXED_POLICY_TIME_SCALE=1.0
export HEROSIM_PEER_EXCHANGE=1 HEROSIM_SERVER_ONLY_REPLICAS=1 HEROSIM_WARMTH_PHYSICS=node_disk_v2 PYTHONHASHSEED=0 HEROSIM_GNN_DEVICE=cpu
export GNN_DECODE_MODE=masked_topo GNN_BATCH_BY_PEER_GROUP=1
unset GNN_MODEL_PATH GNN_DISABLE_MESSAGE_PASSING || true
export LIVE_AUDIT_STRIDE=31 LIVE_AUDIT_MAX_SNAPSHOTS=300 LIVE_AUDIT_MIN_BATCH_SIZE=2 LIVE_AUDIT_MIN_CANDIDATES=1
export HEROSIM_SNAPSHOT_FIDELITY=1
export HEROSIM_RAW_DIR=/tmp/herosim_raw_${SLURM_JOB_ID:-$$}; mkdir -p "$HEROSIM_RAW_DIR"; trap 'rm -rf "$HEROSIM_RAW_DIR"' EXIT
echo "=== [1/2] capture with fidelity"
pids=(); T0=$(date +%s)
for T in $TOPOS; do
  ( export LIVE_AUDIT_SNAPSHOT_PATH="$OUT/snapshots/cc40s${T}_g0.jsonl"
    python3 src/executesimulation.py --config "$CORP/inputs/wf1_hi/cfg/cc40s${T}.json" --workload "$CORP/inputs/wf1_hi/wl/grounded_g0_n50000.json" \
      --policy peer_greedy_network_batch --output "$HEROSIM_RAW_DIR/cc40s${T}_g0.raw.json" > "$OUT/capture_${T}.log" 2>&1; echo "capture $T rc=$?" ) &
  pids+=($!)
done
rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done
echo "capture wall $(( $(date +%s) - T0 )) s"
for T in $TOPOS; do wc -l "$OUT/snapshots/cc40s${T}_g0.jsonl"; ls -la "$OUT/snapshots/cc40s${T}_g0.jsonl" | awk '{print $5/1e6 " MB"}'; done
[[ $rc == 0 ]] || { echo "FAIL LOUD: capture failed"; tail -5 "$OUT"/capture_*.log; exit 1; }
echo "=== [2/2] plan replays under fidelity"
python3 -u scripts_cosim/wf1_fidelity_cost.py --datasets "$CORP/gnn_datasets_wf1_smoke_b" \
  --snapshots "$OUT"/snapshots/*.jsonl --enumerate --sample "$SAMPLE" ${ALL:+--also-all $ALL} --workers 28 --out "$OUT/cost.jsonl"
