#!/bin/bash
# physics_audit_v1 I11 on the workload_fix_v1 W2+W3+W4 environment: does co-simulation reproduce live latency?
# 4 calibration topologies x 2 provisional rungs (lo x0.2666, hi x11.6139), CD arm, 1 s window, S7's builders.
# Per cell: live CD capture (trace + fidelity snapshots), then 12 uniform states from t >= 360 s among the first 3000 arrivals
# and 20 states from arrival 3000 on, each replayed through the co-sim path against an isolated live run.
#SBATCH --job-name=i11wf1
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/i11wf1-%j.out
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=24:00:00
set -euo pipefail
WT=${WT:?set WT}; EXPECT_HEAD=${EXPECT_HEAD:?set EXPECT_HEAD}
D=/home/nikola.lukic/gnn-herosim/simulation_data
W=$D/workload_fix_v1; OUT=${OUT:-$W/i11_wf1}
X1=$D/small_batch_confirm_v1/inputs/grounded/wl
TOPOS="9601 9602 9607 9608"; EVENTS=${EVENTS:-6000}
cd "$WT"
if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"; micromamba activate gnn
export HEROSIM_PY=python3 PYTHONPATH="$WT"
python3 -c "import sys,torch;print('[INTERP]',sys.executable,torch.__version__)"
HEAD=$(git rev-parse HEAD); echo "HEAD $HEAD node $(hostname) OUT=$OUT"
[[ "$HEAD" == "$EXPECT_HEAD"* ]] || { echo "FAIL LOUD: HEAD $HEAD is not $EXPECT_HEAD"; exit 1; }
[[ -z "$(git status --porcelain -- src scripts_cosim)" ]] || { echo "FAIL LOUD: worktree dirty"; exit 1; }
[[ ! -e "$OUT" ]] || { echo "FAIL LOUD: $OUT exists"; exit 1; }
mkdir -p "$OUT/inputs"
python3 -c "import json;json.dump({'topologies':[int(t) for t in '$TOPOS'.split()]},open('$OUT/sel_cal.json','w'))"
python3 scripts_cosim/workload_fix_v1_stage_topologies.py --cfg-dir $W/inputs/cfg_cal --selected $OUT/sel_cal.json --out $OUT/topos
for spec in lo=0.2666 hi=11.6139; do
  tag=${spec%%=*}; m=${spec##*=}
  python3 scripts_cosim/workload_fix_v1_build.py --grounded-wl "$X1" --cfg-dir $OUT/topos/cfg_w4 --topologies $TOPOS \
    --rung "$tag=$(python3 -c "print(repr(1.0/$m))")" --payload-sampler wf1_v1 --batch-timeout-fixed 1.0 --task-mix wf1_v1 --out "$OUT/_b_$tag"
  mv "$OUT/_b_$tag/wf1_$tag" "$OUT/inputs/grounded_$tag"; cp "$OUT/_b_$tag/manifest.json" "$OUT/manifest_$tag.json"; rm -rf "$OUT/_b_$tag"
done
# W3 and W4 are in the configs the replay and the live run both read
python3 - "$OUT/inputs" <<'E2'
import json,sys
for tag in ("lo","hi"):
    c=json.load(open(f"{sys.argv[1]}/grounded_{tag}/cfg/cc40s9601.json"))
    assert c["network"]["backbone"]["access_classes"] and c["network"]["reachability_repair"], tag
    assert c["scheduler"]["batch_timeout"] == 1.0, tag
print("[preflight] cfg carries access_classes, reachability_repair, batch_timeout 1.0")
E2
export HEROSIM_AUDIT_INPUTS=$OUT/inputs TS=1.0 HEROSIM_TRANSFER_MODEL=pipelined HEROSIM_REPLICA_RELEASE=1 HEROSIM_SCALEOUT=kpa
export GATE_FIXED_POLICY_TIME_SCALE=1.0 HEROSIM_SHARED_AUTOSCALER=0
export TMIN=360 LATE_AFTER=3000 LATE_N=20 TW=${TW:-6}
WORK=/tmp/i11wf1_${SLURM_JOB_ID:-$$}; mkdir -p $WORK; trap 'rm -rf $WORK' EXIT
for t in $TOPOS; do for r in lo hi; do echo "$t $r"; done; done > $OUT/cells.txt
set +e
xargs -P 8 -L1 bash -c "timeout 20h scripts_cosim/physics_audit/cell.sh \$0 \$1 $EVENTS 12 0 $WORK" < $OUT/cells.txt
set -e
mkdir -p $OUT/cells
for d in $WORK/*/; do n=$(basename $d); mkdir -p $OUT/cells/$n
  cp $d/replay*.jsonl $d/replay*.log $d/live.log $OUT/cells/$n/ 2>/dev/null || true
  ls -la $d | head -20 > $OUT/cells/$n/ls.txt
  python3 - $d/t.jsonl > $OUT/cells/$n/trace_tail.txt <<'E3' || true
import sys,json
last=None
for l in open(sys.argv[1]):
    if l.startswith('{"k":"end"') or l.startswith('{"k":"header"'): print(l[:300].strip())
E3
done
echo "=== i11_wf1 done: $(ls $OUT/cells/*/replay.jsonl 2>/dev/null | wc -l) of 8 cells replayed ==="
