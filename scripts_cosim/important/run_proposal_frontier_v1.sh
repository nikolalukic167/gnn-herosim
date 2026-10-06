#!/usr/bin/env bash
set -euo pipefail

seed="${1:?usage: run_proposal_frontier_v1.sh SEED ARM [OUTPUT_ROOT]}"
arm="${2:?usage: run_proposal_frontier_v1.sh SEED ARM [OUTPUT_ROOT]}"
output_root="${3:-/tmp/proposal-frontier-v1}"
case "$arm" in
  gnn) mp=0; model=/tmp/g32train/models/g32-sampled-slate-v1-gnn-seed1-final.pt ;;
  mpoff) mp=1; model=/tmp/g32train/models/g32-sampled-slate-v1-mpoff-seed1-final.pt ;;
  handmulti) mp=1; model=/tmp/jb2-mpoff-seed1.pt ;;
  *) echo "unknown frontier arm: $arm" >&2; exit 2 ;;
esac

config="/tmp/fixed-g32-config-${seed}.json"
workload="/tmp/fixed-g32-weighted_bridge-5120.json"
for path in "$config" "$workload" "$model" "${model%.pt}.contract.json" /tmp/jb2-mpoff-seed1.pt /tmp/jb2-mpoff-seed1.contract.json; do
  test -s "$path" || { echo "missing or empty input: $path" >&2; exit 2; }
done
mkdir -p "$output_root"
prefix="$output_root/seed-${seed}-${arm}"
for suffix in .json .log -plans.jsonl -manifest.json -inputs.sha256; do
  test ! -e "${prefix}${suffix}" || { echo "output already exists: ${prefix}${suffix}" >&2; exit 2; }
done
sha256sum "$config" "$workload" "$model" "${model%.pt}.contract.json" \
  /tmp/jb2-mpoff-seed1.pt /tmp/jb2-mpoff-seed1.contract.json \
  scripts_cosim/important/proposal_frontier_v1.py \
  scripts_cosim/important/run_proposal_frontier_v1.sh \
  > "${prefix}-inputs.sha256"
export PIPENV_IGNORE_VIRTUALENVS=1 VIRTUAL_ENV= PYTHONPATH=/root/projects/my-herosim
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED=0
export HEROSIM_PG_EXCHANGE_SCALE=4 HEROSIM_PG_CD_PASSES=6 HEROSIM_PEER_EXCHANGE=1
export HEROSIM_SERVER_ONLY_REPLICAS=1 HEROSIM_WARMTH_PHYSICS=node_disk_v2
export GNN_MODEL_PATH="$model" GNN_DECODE_MODE=masked_topo GNN_QUEUE_NORM_MODE=scheduler_adaptive
export GNN_BATCH_BY_PEER_GROUP=1 GNN_BATCH_SIZE=32 GNN_PREFIX_ALPHA_KEY=inf
export GNN_DISABLE_MESSAGE_PASSING="$mp" GNN_MP_PLATFORM_EDGES_OFF=0
export PARTIAL_STATE_CONTRACT=partial_state_v3 PARTIAL_STATE_PEER_MASS=1
export INFERENCE_FEATURE_LAYOUT=dim22 TOPOLOGY_FEATURE_CONTRACT=src_index_v0
export QUEUE_FEATURE_CONTRACT=legacy_v0 HEROSIM_GNN_DEVICE=cpu
export FRONTIER_ARM="$arm" FRONTIER_OLD_MPOFF_PATH=/tmp/jb2-mpoff-seed1.pt
export FIXED_REPLICA_MANIFEST_PATH="${prefix}-manifest.json"
export SELECTOR_TRACE_PATH="${prefix}-plans.jsonl"
pipenv run python3 scripts_cosim/important/proposal_frontier_v1.py \
  --config "$config" \
  --workload "$workload" \
  --policy gnn_seeded_cd --queue-length 100 \
  --output "${prefix}.json" \
  > "${prefix}.log" 2>&1
