#!/bin/bash
# usage: run_cell.sh <topo> <win> <rung> <policy> <out.json> [max_events]   (out.json: absolute path or relative to CWD)
# env: HEROSIM_AUDIT_INPUTS (dir holding grounded_<rung>/{cfg,wl}), HEROSIM_PY (default python3), PHYSICS=r1|old (default r1), TS (policy time scale, default 1.0), MEMCAP (GB, I6 cell), HS (PYTHONHASHSEED)
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
IN=${HEROSIM_AUDIT_INPUTS:?set HEROSIM_AUDIT_INPUTS}; PY=${HEROSIM_PY:-python3}
topo=$1; win=$2; rung=$3; pol=$4; out=$(realpath -m "$5"); mx=${6:-}
cfg=$IN/grounded_$rung/cfg/cc40s$topo.json; wl=$IN/grounded_$rung/wl/grounded_${win}_n50000.json
if [ "${PHYSICS:-r1}" = r1 ]; then export HEROSIM_TRANSFER_MODEL=pipelined HEROSIM_REPLICA_RELEASE=1 HEROSIM_SCALEOUT=kpa
else unset HEROSIM_TRANSFER_MODEL HEROSIM_REPLICA_RELEASE HEROSIM_SCALEOUT; fi
export HEROSIM_PEER_EXCHANGE=1 HEROSIM_SERVER_ONLY_REPLICAS=1 HEROSIM_WARMTH_PHYSICS=node_disk_v2 PYTHONHASHSEED=${HS:-0} \
  HEROSIM_GNN_DEVICE=cpu SIM_FORCE_FULL_STATS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONPATH=$REPO
export HEROSIM_POLICY_TIME_SCALE=${TS:-1.0}
case $pol in peer_greedy_network_cd|peer_greedy_network_batch) export GNN_DECODE_MODE=masked_topo GNN_BATCH_BY_PEER_GROUP=1;; esac
[ -n "$mx" ] && export HEROSIM_MAX_EVENTS=$mx
ENTRY=src/executesimulation.py; [ -n "$MEMCAP" ] && { export HEROSIM_AUDIT_NODE_MEMORY_GB=$MEMCAP; ENTRY=scripts_cosim/physics_audit/run_reduced_memory.py; }
cd "$REPO" && $PY $ENTRY --config "$cfg" --workload "$wl" --policy "$pol" --output "$out"
