#!/bin/bash
# usage: audit_run.sh <topo> <rung> <arm cd|reactive> <events> <outdir>   (one traced run -> <outdir>/<topo>_<rung>_<arm>.{json,log,trace.jsonl})
# env as run_cell.sh; PHYSICS=old for the old-physics runs. Run over audit_cells.txt: xargs -P 14 -L1 bash -c './audit_run.sh $0 $1 $2 12000 OUT' < audit_cells.txt
D=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
topo=$1; rung=$2; arm=$3; ev=$4; mkdir -p "$5"; out=$(realpath -m "$5")/${topo}_${rung}_${arm}
case $arm in cd) pol=peer_greedy_network_cd;; reactive) pol=knative_network;; esac
HEROSIM_AUDIT_TRACE=$out.trace.jsonl "$D/run_cell.sh" $topo g0 $rung $pol $out.json $ev > $out.log 2>&1
echo "$topo $rung $arm rc=$?"
