#!/bin/bash
# I11 on the load_recalibration_v1 finals rungs (calibration topologies, window g0), one live capture per cell, the same states replayed under
# four replay variants against the same isolated live truth (the truth runs are cached in <cell>/truth and shared):
#   A d28d1bb0                       (1 ms batch poll, ghosts by platform name)
#   B f7711324                       (exact_batch, ghosts by platform name)
#   C WT_CD with HEROSIM_REPLAY_EXACT_BATCH=0   (1 ms poll, ghosts by task id with ordered net timers)
#   D WT_CD with HEROSIM_REPLAY_EXACT_BATCH=1   (exact_batch, ghosts by task id with ordered net timers)
#   E WT_E (default settings)                    (D + a net-stage ghost's due time = pop + latency, bit-equal to a same-instant batch task's)
# usage: WT_E=<worktree at EXPECT_HEAD> WT_CD=<8ada0140 worktree> EXPECT_HEAD=<sha of WT_E> WT_A=<d28d1bb0 worktree with the I11 drivers> WT_B=<f7711324 worktree> OUT_NAME=<dir> sbatch --export=ALL ...
#SBATCH --job-name=i11var
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/i11var-%j.out
#SBATCH --partition=CPU-amd
#SBATCH --cpus-per-task=48
#SBATCH --mem=190G
#SBATCH --time=14:00:00
set -euo pipefail
WT_E=${WT_E:?}; WT_CD=${WT_CD:?}; EXPECT_HEAD=${EXPECT_HEAD:?}; WT_A=${WT_A:?}; WT_B=${WT_B:?}; OUT_NAME=${OUT_NAME:?}; cd $WT_E
eval "$(micromamba shell hook --shell bash)"; micromamba activate gnn
[[ $(git rev-parse HEAD) == $EXPECT_HEAD* ]] || { echo "HEAD is not $EXPECT_HEAD"; exit 1; }; [[ -z "$(git status --porcelain -- src scripts_cosim)" ]] || { echo "worktree dirty"; exit 1; }
L=$HOME/gnn-herosim/simulation_data/load_recalibration_v1; F=$L/bisect_eff/final
OUT=$L/$OUT_NAME; mkdir -p $OUT/inputs
ln -sfn $F/m0p2666/wf1_m0p2666 $OUT/inputs/grounded_z; ln -sfn $F/m1p2584/wf1_m1p2584 $OUT/inputs/grounded_l
ln -sfn $F/m5p9402/wf1_m5p9402 $OUT/inputs/grounded_m; ln -sfn $F/m11p6139/wf1_m11p6139 $OUT/inputs/grounded_h
W=/tmp/i11var_$SLURM_JOB_ID; mkdir -p $W
export HEROSIM_PY=python3 HEROSIM_AUDIT_INPUTS=$OUT/inputs TS=1.0 HEROSIM_TRANSFER_MODEL=pipelined HEROSIM_REPLICA_RELEASE=1 HEROSIM_SCALEOUT=kpa GATE_FIXED_POLICY_TIME_SCALE=1.0 HEROSIM_SHARED_AUTOSCALER=0 TW=4
export WT_E WT_CD WT_A WT_B W OUT
cellrun() { t=${1%_*}; r=${1#*_}; C=$W/${t}_$r
  cd $WT_E; export PYTHONPATH=$WT_E
  # capture + variant E replay of the uniform set (cell.sh = capture, then replay); truth is cached under $C/truth
  TMIN=360 scripts_cosim/physics_audit/cell.sh $t $r 50000 5 0 $W
  mv $C/replay.jsonl $C/replay_E_uni.jsonl; mv $C/replay.log $C/replay_E_uni.log
  T3=$(python3 - $C/t.jsonl <<'E'
import sys
sys.path.insert(0,'.')
from scripts_cosim.physics_audit.i11_replay import live_states
ts=[s["t"] for s in live_states(sys.argv[1])]; print(min(ts)+(max(ts)-min(ts))*2/3)
E
)
  TAG=_late TMIN=$T3 scripts_cosim/physics_audit/cell.sh $t $r 50000 5 0 $W; mv $C/replay_late.jsonl $C/replay_E_late.jsonl; mv $C/replay_late.log $C/replay_E_late.log
  python3 scripts_cosim/physics_audit/bl2.py $C/t.jsonl $C/bl.json 360 5 > $C/bl_summary.txt
  IDS=$(cat $C/bl.json.ids)
  if [ -n "$IDS" ]; then ONLY=$IDS TAG=_bl scripts_cosim/physics_audit/cell.sh $t $r 50000 5 0 $W; mv $C/replay_bl.jsonl $C/replay_E_bl.jsonl; mv $C/replay_bl.log $C/replay_E_bl.log; fi
  # the same state sets under A, B, C (cell.sh with REPLAY overrides is not used: call the replay of each worktree directly)
  CFG=$HEROSIM_AUDIT_INPUTS/grounded_$r/cfg/cc40s$t.json; WL=$HEROSIM_AUDIT_INPUTS/grounded_$r/wl/grounded_g0_n50000.json
  export GNN_DECODE_MODE=masked_topo GNN_BATCH_BY_PEER_GROUP=1 HEROSIM_PEER_EXCHANGE=1 HEROSIM_SERVER_ONLY_REPLICAS=1 HEROSIM_WARMTH_PHYSICS=node_disk_v2 PYTHONHASHSEED=0 SIM_FORCE_FULL_STATS=1 OMP_NUM_THREADS=1 GNN_CAPTURE_DATASET_STATE=0 HEROSIM_SNAPSHOT_FIDELITY=1 HEROSIM_POLICY_TIME_SCALE=1.0
  for V in A B C D; do
    case $V in A) WT=$WT_A; EX=1;; B) WT=$WT_B; EX=1;; C) WT=$WT_CD; EX=0;; D) WT=$WT_CD; EX=1;; esac
    cd $WT; export PYTHONPATH=$WT HEROSIM_REPLAY_EXACT_BATCH=$EX
    for S in uni late bl; do
      case $S in uni) ARGS="--tmin 360 --n 5 --targeted 0";; late) ARGS="--tmin $T3 --n 5 --targeted 0";; bl) [ -n "$IDS" ] || continue; ARGS="--only $IDS --n 5 --targeted 0";; esac
      python3 scripts_cosim/physics_audit/i11_replay.py --cfg $CFG --workload $WL --trace $C/t.jsonl --snapshots $C/snap.jsonl --out $C/replay_${V}_$S.jsonl $ARGS --params live --workers 4 --cell ${t}_$r > $C/replay_${V}_$S.log 2>&1 || true
    done
  done
}
export -f cellrun
for r in z l m h; do for t in 9601 9602 9607 9608; do echo ${t}_$r; done; done | xargs -P 8 -I{} bash -c "cellrun {}"
for r in z l m h; do for t in 9601 9602 9607 9608; do C=$W/${t}_$r; mkdir -p $OUT/${t}_$r; cp $C/replay_*.jsonl $C/bl_summary.txt $OUT/${t}_$r/ 2>/dev/null || true; done; done
cd $WT_E; python3 scripts_cosim/physics_audit/summarize_variants.py $OUT | tee $OUT/summary.txt
echo done
