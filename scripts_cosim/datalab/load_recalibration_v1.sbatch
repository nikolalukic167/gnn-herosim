#!/bin/bash
# load_recalibration_v1 on R1.1 + the reservation fix, WF1 (W2 payloads + W3 access classes + W4 mix/repair), batching window 1 s.
# LRC_JOB selects the job; WT is the pinned detached worktree and EXPECT_HEAD its commit (no defaults; never move a worktree while jobs run).
#   cfg        W3/W4 configs for the calibration topologies (workload_fix_v1_stage_topologies.py) into $R/cfg
#   identity   instrumentation identity check: CD and Knative on 2 calibration topologies at one multiplier, run from WT;
#              run it once per worktree (OUT names the directory) and compare with load_recalibration_v1_identity.py
#   bisect     the registered bisection (load_recalibration_v1_bisect.py), then the two mapped provisional multipliers
#SBATCH --job-name=lrc
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/lrc-%j.out
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=24:00:00
set -euo pipefail
WT=${WT:?FAIL LOUD: set WT (pinned detached worktree)}
EXPECT_HEAD=${EXPECT_HEAD:?FAIL LOUD: set EXPECT_HEAD}
JOB=${LRC_JOB:?FAIL LOUD: set LRC_JOB}
D=/home/nikola.lukic/gnn-herosim/simulation_data
W=$D/workload_fix_v1
R=$D/load_recalibration_v1
X1=$D/small_batch_confirm_v1/inputs/grounded/wl
CAL_TOPOS=${CAL_TOPOS:-9601 9602 9607 9608}
P=${PAR:-$(( ${SLURM_CPUS_PER_TASK:-64} - 2 ))}
cd "$WT"
if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3 PYTHONPATH="$WT"
export HEROSIM_RAW_DIR=/tmp/herosim_raw_${SLURM_JOB_ID:-$$}
mkdir -p "$HEROSIM_RAW_DIR"; trap 'rm -rf "$HEROSIM_RAW_DIR"' EXIT
python3 -c "import sys,torch;print('[INTERP]',sys.executable,torch.__version__)"
HEAD=$(git rev-parse HEAD)
echo "HEAD $HEAD  node $(hostname)  job=$JOB"
[[ "$HEAD" == "$EXPECT_HEAD"* ]] || { echo "FAIL LOUD: worktree HEAD $HEAD is not $EXPECT_HEAD"; exit 1; }
[[ -z "$(git status --porcelain -- src scripts_cosim experiments run_experiment.py)" ]] || { echo "FAIL LOUD: worktree dirty"; exit 1; }
[[ -f src/placement/request_timeout.py ]] || { echo "FAIL LOUD: this worktree has no R1.1 request timeout"; exit 1; }
export HEROSIM_TRANSFER_MODEL=pipelined HEROSIM_REPLICA_RELEASE=1 HEROSIM_SCALEOUT=kpa GATE_FIXED_POLICY_TIME_SCALE=1.0
export HEROSIM_SHARED_AUTOSCALER=0
mkdir -p "$R"
case "$JOB" in
  cfg)
    [[ ! -e "$R/cfg" ]] || { echo "FAIL LOUD: $R/cfg exists"; exit 1; }
    python3 -c "import json;print(json.dumps({'topologies':[int(x) for x in '$CAL_TOPOS'.split()]}))" > "$R/cal_select.json"
    python3 scripts_cosim/workload_fix_v1_stage_topologies.py --cfg-dir "$W/inputs/cfg_cal" --selected "$R/cal_select.json" --out "$R/cfg"
    ;;
  identity)
    OUT=${OUT:?FAIL LOUD: set OUT}
    M=${LRC_M:-3}; tag=m${M//./p}
    [[ ! -e "$OUT" ]] || { echo "FAIL LOUD: $OUT exists"; exit 1; }
    mkdir -p "$OUT"
    python3 scripts_cosim/workload_fix_v1_build.py --grounded-wl "$X1" --cfg-dir "$R/cfg/cfg_w4" --topologies 9601 9607 \
      --rung "$tag=$(python3 -c "print(repr(1.0/$M))")" --payload-sampler wf1_v1 --task-mix wf1_v1 --batch-timeout-fixed 1.0 --out "$OUT/inputs"
    WF1_RUNGS=$tag WF1_TOPOS=9601,9607 WF1_CAL_WINDOWS=g0 WF1_CAL_KINDS=cd,reactive \
      python3 scripts_cosim/fresh_topo_burst_v1_gate.py wf1cal --inputs "$OUT/inputs" --out "$OUT/gate" \
      --parallel 4 --mem 4G --timeout "${TIMEOUT:-5400}" --no-scope | grep -v "^\[skip"
    ;;
  identity_lock)
    # amendment 2026-10-09: the lock-wait stamps change no simulated field. Three CD cells re-run from WT against the same
    # cells from the earlier bisection (bisect/<tag>/gate), which carry the earlier instrumentation only.
    OUT=${OUT:?FAIL LOUD: set OUT}
    [[ ! -e "$OUT" ]] || { echo "FAIL LOUD: $OUT exists"; exit 1; }
    mkdir -p "$OUT/base" "$OUT/new"
    for spec in m24p7105:9601:g0 m11p6139:9607:g0 m5p9402:9608:g0; do
      IFS=: read -r tag topo win <<<"$spec"
      cp "$R/bisect/$tag/gate/cc40s${topo}__${win}${tag}__cd_s0.summary.json" "$OUT/base/"
      WF1_RUNGS=$tag WF1_TOPOS=$topo WF1_CAL_WINDOWS=$win WF1_CAL_KINDS=cd \
        python3 scripts_cosim/fresh_topo_burst_v1_gate.py wf1cal --inputs "$R/bisect/$tag" --out "$OUT/new" \
        --parallel 1 --mem 4G --timeout "${TIMEOUT:-5400}" --no-scope | grep -v "^\[skip" &
    done
    wait
    python3 scripts_cosim/load_recalibration_v1_identity.py "$OUT/base" "$OUT/new"
    ;;
  points)
    # amendment 2026-10-09: CD-only steering evaluations of the given multipliers into a fresh work directory, with the earlier
    # runs' rung inputs linked (same windows and configs), so the effective queue share exists at every step
    BW=${BISECT_NEW:-bisect_eff}
    [[ ! -e "$R/$BW" ]] || { echo "FAIL LOUD: $R/$BW exists"; exit 1; }
    mkdir -p "$R/$BW"
    for m in ${LRC_POINTS:?FAIL LOUD: set LRC_POINTS}; do
      tag=$(python3 -c "print('m' + f'{float(\"$m\"):.4f}'.replace('.', 'p'))")
      mkdir -p "$R/$BW/$tag"
      # an earlier run's rung inputs are linked; a multiplier not built before is built by the driver
      [[ -d "$R/bisect/$tag/wf1_$tag" ]] && ln -s "$R/bisect/$tag/wf1_$tag" "$R/$BW/$tag/wf1_$tag" || true
    done
    python3 scripts_cosim/load_recalibration_v1_bisect.py --work "$R/$BW" --grounded-wl "$X1" --cfg-dir "$R/cfg/cfg_w4" \
      --topologies $CAL_TOPOS --windows g0 g1 --parallel 8 --points $LRC_POINTS --log "$R/${BW}_points.json"
    ;;
  bisect)
    # LRC_RESUME=1 (amendment 2026-10-09): continue in the existing $R/bisect; built rungs and finished cells are reused
    BW=${LRC_WORK:-bisect}
    [[ -n "${LRC_RESUME:-}" || ( ! -e "$R/$BW" && ! -e "$R/${BW}_steps.json" ) ]] || { echo "FAIL LOUD: $R/$BW exists"; exit 1; }
    [[ -d "$R/cfg/cfg_w4" ]] || { echo "FAIL LOUD: run the cfg job first"; exit 1; }
    python3 scripts_cosim/load_recalibration_v1_bisect.py --work "$R/$BW" --grounded-wl "$X1" --cfg-dir "$R/cfg/cfg_w4" \
      --topologies $CAL_TOPOS --windows g0 g1 --parallel "${LRC_PAR:-12}" --map 0.2666 11.6139 --log "$R/${BW}_steps.json" ${LRC_ARGS:-}
    find "$R/$BW" -name "*.log" -size +50M -exec gzip -1 {} \;
    ;;
  *) echo "FAIL LOUD: LRC_JOB=$JOB"; exit 1 ;;
esac
