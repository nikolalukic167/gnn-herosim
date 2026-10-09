#!/bin/bash
# Scale go/no-go (b): I11 (co-sim replay vs isolated live) on one scale-probe cell, N uniform states t >= 360 s, capture policy CD
# (physics_audit/cell.sh, as i11_finals_fix.sbatch). Bars signed by the coordinator before the data: median <= 1 %, p95 <= 1 %, max <= 5 %.
#   sbatch --export=ALL,WT=...,CELLDIR=<scale_probe/<cell>>,TOPO=9101,RUNG=heavy,N=40,OUT=<dir> scale_probe_i11.sbatch
#SBATCH --job-name=sp-i11
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/sp-i11-%j.out
#SBATCH --partition=CPU-amd
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=06:00:00
set -euo pipefail
WT=${WT:?}; CELLDIR=${CELLDIR:?}; TOPO=${TOPO:?}; RUNG=${RUNG:-heavy}; N=${N:-40}; OUT=${OUT:?}
cd "$WT"; eval "$(micromamba shell hook --shell bash)"; micromamba activate gnn
for v in $(compgen -e | grep -E '^(HEROSIM_|GNN_|COSIM_|LIVE_AUDIT_|GATE_|SIM_|QUEUE_FEATURE|MAX_PLACEMENT|DECLARED_SLATE)'); do unset "$v"; done
echo "HEAD $(git rev-parse HEAD) cell $CELLDIR topo $TOPO rung $RUNG N $N node $(hostname) $(date -Is)"
mkdir -p "$OUT/inputs"; ln -sfn "$CELLDIR/inputs/wf1_$RUNG" "$OUT/inputs/grounded_$RUNG"
W=/tmp/sp_i11_$SLURM_JOB_ID; mkdir -p $W
export HEROSIM_PY=python3 PYTHONPATH=$WT HEROSIM_AUDIT_INPUTS=$OUT/inputs TS=1.0 HEROSIM_TRANSFER_MODEL=pipelined HEROSIM_REPLICA_RELEASE=1 HEROSIM_SCALEOUT=kpa GATE_FIXED_POLICY_TIME_SCALE=1.0 HEROSIM_SHARED_AUTOSCALER=0 TW=4
TMIN=360 scripts_cosim/physics_audit/cell.sh $TOPO $RUNG 50000 $N 0 $W
C=$W/${TOPO}_$RUNG; mkdir -p "$OUT/${TOPO}_$RUNG"; cp $C/replay*.jsonl $C/replay*.log $C/live.log "$OUT/${TOPO}_$RUNG/" 2>/dev/null || true
python3 - "$OUT/${TOPO}_$RUNG/replay.jsonl" <<'E' | tee "$OUT/summary.txt"
import json,math,statistics as st,sys
rs=[json.loads(l) for l in open(sys.argv[1])]
ok=[r for r in rs if r.get("rel_err") is not None]; es=sorted(abs(r["rel_err"]) for r in ok); n=len(es)
print("states",len(rs),"with truth",n,"unreplayed",sum(1 for r in rs if r.get("replay") is None))
if n: print("median %.3f%% p95 %.3f%% max %.3f%%"%(100*st.median(es),100*es[min(n-1,math.ceil(.95*n)-1)],100*es[-1]))
for r in sorted(ok,key=lambda r:-abs(r["rel_err"]))[:5]: print("  t=%7.1f err=%.3f%% live=%.4f replay=%.4f"%(r["t"],100*r["rel_err"],r["live"],r["replay"]))
E
rm -rf $W; echo "done $(date -Is)"
