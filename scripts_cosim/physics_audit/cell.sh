#!/bin/bash
# usage: cell.sh <topo> <rung> <events> <n_uniform> <n_targeted> <outdir>   (capture if needed, then I11 replay -> <outdir>/<topo>_<rung>/replay.jsonl)
# env as run_cell.sh; FID=0 replays without the fidelity block, REPLAY=original, TAG, ONLY, TW (truth workers, default 4). Run over i11_cells.txt with xargs -P 5 -L1.
D=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); REPO=$(cd "$D/../.." && pwd)
IN=${HEROSIM_AUDIT_INPUTS:?set HEROSIM_AUDIT_INPUTS}; PY=${HEROSIM_PY:-python3}
topo=$1; rung=$2; ev=$3; out=$(realpath -m "$6")/${topo}_${rung}; mkdir -p $out
cfg=$IN/grounded_$rung/cfg/cc40s$topo.json; wl=$IN/grounded_$rung/wl/grounded_g0_n50000.json
[ -s $out/snap.jsonl ] || "$D/capture.sh" $topo $rung $ev $out
export GNN_DECODE_MODE=masked_topo GNN_BATCH_BY_PEER_GROUP=1 HEROSIM_TRANSFER_MODEL=pipelined HEROSIM_REPLICA_RELEASE=1 HEROSIM_SCALEOUT=kpa HEROSIM_PEER_EXCHANGE=1 HEROSIM_SERVER_ONLY_REPLICAS=1 HEROSIM_WARMTH_PHYSICS=node_disk_v2 PYTHONHASHSEED=0 SIM_FORCE_FULL_STATS=1 OMP_NUM_THREADS=1 PYTHONPATH=$REPO GNN_CAPTURE_DATASET_STATE=0 HEROSIM_SNAPSHOT_FIDELITY=${FID:-1}
export HEROSIM_POLICY_TIME_SCALE=${TS:-1.0}
cd "$REPO" && $PY scripts_cosim/physics_audit/i11_replay.py --cfg $cfg --workload $wl --trace $out/t.jsonl --snapshots $out/snap.jsonl --out $out/replay${TAG:-}.jsonl --n $4 --targeted $5 ${ONLY:+--only $ONLY} ${REPLAY:+--replay $REPLAY} --params live --workers ${TW:-4} --cell ${topo}_${rung} > $out/replay${TAG:-}.log 2>&1
echo "cell ${topo}_${rung}: $(grep -E '^all ' $out/replay${TAG:-}.log)"
