#!/bin/bash
# usage: i11run.sh <params oracle|live> <n> <out.jsonl> <cfg> <workload> <trace> <snapshots>   (pass-1 single-cell I11 replay, no isolated truth)
# env: HEROSIM_PY (default python3), TS (default 1.0), FID (fidelity, default 1), ONLY (comma list of batch[0] ids)
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd); PY=${HEROSIM_PY:-python3}
export HEROSIM_TRANSFER_MODEL=pipelined HEROSIM_REPLICA_RELEASE=1 HEROSIM_SCALEOUT=kpa HEROSIM_PEER_EXCHANGE=1 HEROSIM_SERVER_ONLY_REPLICAS=1 HEROSIM_WARMTH_PHYSICS=node_disk_v2 PYTHONHASHSEED=0 SIM_FORCE_FULL_STATS=1 OMP_NUM_THREADS=1 PYTHONPATH=$REPO HEROSIM_POLICY_TIME_SCALE=${TS:-1.0} GNN_CAPTURE_DATASET_STATE=0 HEROSIM_SNAPSHOT_FIDELITY=${FID:-1}
cd "$REPO" && $PY scripts_cosim/physics_audit/i11_replay.py --cfg "$4" --workload "$5" --trace "$6" --snapshots "$7" --out "$3" --n $2 --params $1 ${ONLY:+--only $ONLY}
