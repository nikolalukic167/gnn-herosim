#!/bin/bash
# peer_affinity_v1 -- stage 3 live gate on a real production trace (docs/lineages/peer_affinity_v1.md).
# One cell (corpus shape, unseen topology seed), one peer-augmented production trace, 34 arms:
#   task 0      knative_network          (reactive, per arrival, peer-blind, charged)
#   task 1      knative_network_batch    (reactive, 10-task window like the GNN's, peer-blind)
#   tasks 2-17  gnn   lr2e3 seed 1..16   (prefix-conditioned masked_topo, peer-group batching)
#   tasks 18-33 mpoff lr2e3 seed 1..16   (same, PeerConv never runs)
# Every arm: HEROSIM_PEER_EXCHANGE=1, server-only replicas, warmth node_disk_v2, no --seed
# (the cell config carries the topology seed; --seed would override it -- gate-tools 2026-08-20).
#
#SBATCH --job-name=pa-s3-live
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/pa-s3-live-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/pa-s3-live-%A_%a.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --array=0-33

set -euo pipefail
cd /home/nikola.lukic/gnn-herosim

if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3
export PYTHONPATH="$PWD"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=""

python3 -c "import sys,torch;print('[INTERP]',sys.executable,torch.__version__)"
echo "[CODE] $(git describe --dirty --always) on $(git rev-parse --abbrev-ref HEAD)"

GATE=simulation_data/peer_affinity_live_gate
WORKLOAD="${WORKLOAD:-$GATE/workloads/workload-150-150-peer_p2_x200.json}"
CFG="${CFG:-$GATE/configs/cell_s7901.json}"
TAG="$(basename "$WORKLOAD" .json)__$(basename "$CFG" .json)"
OUT_DIR="$GATE/results/$TAG"
mkdir -p "$OUT_DIR" logs

[[ -f "$WORKLOAD" ]] || { echo "FAIL LOUD: workload missing: $WORKLOAD (rsync it)" >&2; exit 1; }
[[ -f "$CFG" ]] || { echo "FAIL LOUD: cell config missing: $CFG (rsync it)" >&2; exit 1; }
echo "[INPUTS] workload $(md5sum "$WORKLOAD" | cut -c1-12) cfg $(md5sum "$CFG" | cut -c1-12)"

# physics + gate contract, identical for every arm
export HEROSIM_PEER_EXCHANGE=1
export HEROSIM_SERVER_ONLY_REPLICAS=1
export HEROSIM_WARMTH_PHYSICS=node_disk_v2
export PYTHONHASHSEED=0
export HEROSIM_GNN_DEVICE=cpu

T=${SLURM_ARRAY_TASK_ID}
if [[ $T -eq 0 ]]; then
  POLICY=knative_network; ARM=knative_network
elif [[ $T -eq 1 ]]; then
  POLICY=knative_network_batch; ARM=knative_network_batch
  export KNATIVE_BATCH_SIZE=10 KNATIVE_BATCH_TIMEOUT=0.02
else
  POLICY=gnn
  if [[ $T -le 17 ]]; then ARMNAME=gnn; SEED=$(( T - 1 )); else ARMNAME=mpoff; SEED=$(( T - 17 )); fi
  ARM="${ARMNAME}_s${SEED}"
  CK="models/peer-affinity-v1-t1b-${ARMNAME}-lr2e3-seed${SEED}.pt"
  [[ -f "$CK" ]] || { echo "FAIL LOUD: checkpoint missing: $CK" >&2; exit 1; }
  [[ -f "${CK%.pt}.contract.json" ]] || { echo "FAIL LOUD: sidecar missing for $CK" >&2; exit 1; }
  export GNN_MODEL_PATH="$CK"
  export GNN_DECODE_MODE=masked_topo
  export GNN_BATCH_BY_PEER_GROUP=1
  if [[ $ARMNAME == mpoff ]]; then export GNN_DISABLE_MESSAGE_PASSING=1; fi
fi
OUT="$OUT_DIR/${ARM}.json"
echo "[TASK $T] policy=$POLICY arm=$ARM -> $OUT"
if [[ -f "$OUT" ]]; then echo "[TASK $T] result exists, not re-running: $OUT"; exit 0; fi

${HEROSIM_PY:-pipenv run python3} src/executesimulation.py \
  --config "$CFG" --workload "$WORKLOAD" --policy "$POLICY" --output "$OUT.partial"
mv "$OUT.partial" "$OUT"
python3 - "$OUT" <<'PY'
import json, sys
d = json.load(open(sys.argv[1])); st = d.get("stats") or d
print("[RESULT]", json.dumps({k: st.get(k) for k in (
    "policy", "num_tasks", "total_rtt", "averageElapsedTime", "totalPeerExchangeTime",
    "totalPeerRendezvousWait", "endTime", "schedulerCounters")}))
print("[PROVENANCE env]", json.dumps({k: v for k, v in (d.get("run_provenance") or {}).get("env", {}).items() if v}))
PY
