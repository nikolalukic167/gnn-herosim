#!/bin/bash
# usage: capture.sh <topo> <rung> <max_events> <outdir>   (live CD run with trace + fidelity snapshots -> <outdir>/{t.jsonl,snap.jsonl,live.json})
# env as run_cell.sh. Called by cell.sh.
D=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); mkdir -p "$4"; O=$(realpath "$4"); rm -f $O/t.jsonl $O/snap.jsonl
export HEROSIM_SNAPSHOT_FIDELITY=1 HEROSIM_AUDIT_TRACE=$O/t.jsonl LIVE_AUDIT_SNAPSHOT_PATH=$O/snap.jsonl LIVE_AUDIT_MIN_BATCH_SIZE=1 LIVE_AUDIT_MIN_CANDIDATES=1 LIVE_AUDIT_MAX_SNAPSHOTS=100000
"$D/run_cell.sh" $1 g0 $2 peer_greedy_network_cd $O/live.json $3 > $O/live.log 2>&1; echo "capture $1 $2 rc=$?"
