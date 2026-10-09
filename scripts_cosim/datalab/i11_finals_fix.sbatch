#!/bin/bash
# I11 (co-sim replay vs isolated live) on the load_recalibration_v1 finals rungs, calibration topologies, window g0: per cell 5 uniform states
# (t >= 360 s), 5 from the last third, and up to 5 states with a compute-lock backlog > 100 s. Capture policy CD (cell.sh). Audit traces stay on
# node-local /tmp. usage: WT=<worktree> EXPECT_HEAD=<sha> OUT_NAME=<dir under load_recalibration_v1> sbatch --export=ALL ...
#SBATCH --job-name=i11fin
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/i11fin-%j.out
#SBATCH --partition=CPU-amd
#SBATCH --cpus-per-task=48
#SBATCH --mem=190G
#SBATCH --time=12:00:00
set -euo pipefail
WT=${WT:?set WT}; EXPECT_HEAD=${EXPECT_HEAD:?set EXPECT_HEAD}; OUT_NAME=${OUT_NAME:?set OUT_NAME}; cd $WT
eval "$(micromamba shell hook --shell bash)"; micromamba activate gnn
[[ $(git rev-parse HEAD) == $EXPECT_HEAD* ]] || { echo "HEAD is not $EXPECT_HEAD"; exit 1; }; [[ -z "$(git status --porcelain -- src scripts_cosim)" ]] || { echo "worktree dirty"; exit 1; }
L=$HOME/gnn-herosim/simulation_data/load_recalibration_v1; F=$L/bisect_eff/final
OUT=$L/$OUT_NAME; mkdir -p $OUT/inputs
ln -sfn $F/m0p2666/wf1_m0p2666 $OUT/inputs/grounded_z; ln -sfn $F/m1p2584/wf1_m1p2584 $OUT/inputs/grounded_l
ln -sfn $F/m5p9402/wf1_m5p9402 $OUT/inputs/grounded_m; ln -sfn $F/m11p6139/wf1_m11p6139 $OUT/inputs/grounded_h
W=/tmp/i11fin_$SLURM_JOB_ID; mkdir -p $W
export HEROSIM_PY=python3 PYTHONPATH=$WT HEROSIM_AUDIT_INPUTS=$OUT/inputs TS=1.0 HEROSIM_TRANSFER_MODEL=pipelined HEROSIM_REPLICA_RELEASE=1 HEROSIM_SCALEOUT=kpa GATE_FIXED_POLICY_TIME_SCALE=1.0 HEROSIM_SHARED_AUTOSCALER=0 TW=4
cellrun() { t=${1%_*}; r=${1#*_}; C=$W/${t}_$r
  TMIN=360 scripts_cosim/physics_audit/cell.sh $t $r 50000 5 0 $W
  T3=$(python3 - $C/t.jsonl <<'E'
import sys
sys.path.insert(0,'.')
from scripts_cosim.physics_audit.i11_replay import live_states
ts=[s["t"] for s in live_states(sys.argv[1])]; print(min(ts)+(max(ts)-min(ts))*2/3)
E
)
  TAG=_late TMIN=$T3 scripts_cosim/physics_audit/cell.sh $t $r 50000 5 0 $W
  python3 scripts_cosim/physics_audit/bl2.py $C/t.jsonl $C/bl.json 360 5 > $C/bl_summary.txt
  IDS=$(cat $C/bl.json.ids)
  if [ -n "$IDS" ]; then ONLY=$IDS TAG=_bl scripts_cosim/physics_audit/cell.sh $t $r 50000 5 0 $W; fi
}
export -f cellrun; export W
for r in z l m h; do for t in 9601 9602 9607 9608; do echo ${t}_$r; done; done | xargs -P 8 -I{} bash -c "cellrun {}"
for r in z l m h; do for t in 9601 9602 9607 9608; do C=$W/${t}_$r; mkdir -p $OUT/${t}_$r; cp $C/replay*.jsonl $C/replay*.log $C/bl.json $C/bl_summary.txt $OUT/${t}_$r/ 2>/dev/null || true; done; done
python3 - $OUT <<'E2' | tee $OUT/summary.txt
import json,glob,math,statistics as st,sys
O=sys.argv[1]
def stat(es):
    es=sorted(es); n=len(es)
    return "n=%d median %.3f%% p95 %.3f%% max %.3f%%"%(n,100*st.median(es),100*es[min(n-1,math.ceil(.95*n)-1)],100*es[-1]) if n else "n=0"
for rung in "zlmh":
  for name,pat in (("uniform t>=360","replay.jsonl"),("last third","replay_late.jsonl"),("backlog>100 s","replay_bl.jsonl")):
    allr=[]
    for t in (9601,9602,9607,9608):
        d=f"{O}/{t}_{rung}"
        try: rs=[json.loads(l) for l in open(f"{d}/{pat}")]
        except FileNotFoundError: rs=[]
        for r in rs: r["topo"]=t
        allr+=rs
    ok=[r for r in allr if r.get("rel_err") is not None]
    print("RUNG",rung,name,stat([abs(r["rel_err"]) for r in ok]),"no-truth",len(allr)-len(ok),"unreplayed",sum(1 for r in allr if r.get("replay") is None))
    for r in sorted(ok,key=lambda r:-abs(r["rel_err"]))[:3]:
        print("   %s t=%7.1f err=%.3f%% live=%.4f replay=%.4f abs miss %.4f"%(r["topo"],r["t"],100*r["rel_err"],r["live"],r["replay"],abs(r["replay"]-r["live"])))
E2
echo done
