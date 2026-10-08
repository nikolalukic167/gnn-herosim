#!/bin/bash
# workload_fix_v1 on R1. WF1_JOB selects the job; WT is the pinned detached worktree and EXPECT_HEAD its commit
# (neither has a default: a job must name the code it runs, and the worktree is never moved while jobs run).
#   verify  the legacy build chain reproduces client_local_v1/wl_so/grounded_x20 byte for byte
#   replay  TP1 smoke arms on R1, for comparison between two worktrees (OUT names the directory)
#   calcheck  CD and Knative on the calibration topologies at one multiplier (admission look)
#   tune  amendment WB2: CD at fixed batching windows 1,2,4,8,16 s on the calibration topologies
#   bisect  provisional load on the calibration topologies (workload_fix_v1_bisect.py)
#   stage   build the stage's rungs (WF1_SAMPLER) and run the five classical arms on the 19 test topologies
#SBATCH --job-name=wf1
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/wf1-%j.out
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=24:00:00
set -euo pipefail
WT=${WT:?FAIL LOUD: set WT (pinned detached worktree)}
EXPECT_HEAD=${EXPECT_HEAD:?FAIL LOUD: set EXPECT_HEAD}
JOB=${WF1_JOB:?FAIL LOUD: set WF1_JOB}
D=/home/nikola.lukic/gnn-herosim/simulation_data
W=$D/workload_fix_v1
X1=$D/small_batch_confirm_v1/inputs/grounded/wl
SELECT=$D/small_batch_confirm_v1/inputs/selected.json
CAL_TOPOS=${CAL_TOPOS:-9601 9602 9603 9604}
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
# R1 (physics_audit_v1)
export HEROSIM_TRANSFER_MODEL=pipelined HEROSIM_REPLICA_RELEASE=1 HEROSIM_SCALEOUT=kpa GATE_FIXED_POLICY_TIME_SCALE=1.0
export HEROSIM_SHARED_AUTOSCALER=0
mkdir -p "$W"
case "$JOB" in
  verify)
    python3 scripts_cosim/workload_fix_v1_build.py --grounded-wl "$X1" --cfg-dir $D/small_batch_confirm_v1/inputs/cfg \
      --topologies 9483 --rung x20=0.5 --payload-sampler legacy --out "$W/verify_x20" \
      --verify-against $D/client_local_v1/wl_so/grounded_x20
    ;;
  replay)
    OUT=${OUT:?FAIL LOUD: set OUT}
    IN=$D/client_local_v1/inputs_so_server
    mkdir -p "$OUT"
    TP1_COND=pipe_release TP1_SMOKE=${TP1_SMOKE:-cd,batched,reactive,selfpredict,locality} \
      python3 scripts_cosim/fresh_topo_burst_v1_gate.py tp1 --inputs "$IN" --out "$OUT" --selection "$IN/selected.json" \
      --parallel 5 --mem 4G --timeout 5400 --no-scope | grep -v "^\[skip"
    ;;
  calcheck)
    # admission look at the calibration topologies: CD and Knative at one multiplier, short timeout
    M=${WF1_M:-2}; tag=m${M//./p}
    python3 scripts_cosim/workload_fix_v1_build.py --grounded-wl "$X1" --cfg-dir "$W/inputs/cfg_cal" \
      --topologies $CAL_TOPOS --rung "$tag=$(python3 -c "print(repr(1.0/$M))")" --payload-sampler legacy --out "$W/calcheck/$tag"
    WF1_RUNGS=$tag WF1_TOPOS=${CAL_TOPOS// /,} WF1_CAL_WINDOWS=g0,g1,g2,g3 WF1_CAL_KINDS=cd,reactive \
      python3 scripts_cosim/fresh_topo_burst_v1_gate.py wf1cal --inputs "$W/calcheck/$tag" --out "$W/calcheck/$tag/gate" \
      --parallel "$P" --mem 4G --timeout "${TIMEOUT:-900}" --no-scope | grep -v "^\[skip"
    ;;
  tune)
    # amendment WB2: CD at each fixed batching window on the calibration topologies, both provisional rungs, W2 workload
    TR=$W/stage_w2_wb2/tune; IN=$TR/inputs; mkdir -p "$IN"
    TAGS_ALL=""
    for spec in lo=0.2666 hi=11.6139; do
      rung=${spec%%=*}; m=${spec##*=}
      for bw in ${WF1_WINDOWS:-1 2 4 8 16}; do
        tag=${rung}b${bw}
        if [[ ! -d "$IN/wf1_$tag" ]]; then
          python3 scripts_cosim/workload_fix_v1_build.py --grounded-wl "$X1" --cfg-dir "$W/inputs/cfg_cal" --topologies $CAL_TOPOS \
            --rung "$tag=$(python3 -c "print(repr(1.0/$m))")" --payload-sampler wf1_v1 --batch-timeout-fixed "$bw" --out "$IN/_b_$tag"
          mv "$IN/_b_$tag/wf1_$tag" "$IN/wf1_$tag"; cp "$IN/_b_$tag/manifest.json" "$IN/manifest_$tag.json"; rm -rf "$IN/_b_$tag"
        fi
        TAGS_ALL="$TAGS_ALL,$tag"
      done
    done
    TAGS_ALL=${TAGS_ALL#,}
    mkdir -p "$TR/gate"
    set +e
    WF1_RUNGS=$TAGS_ALL WF1_TOPOS=${CAL_TOPOS// /,} WF1_CAL_WINDOWS=g0,g1,g2,g3 WF1_CAL_KINDS=cd \
      python3 scripts_cosim/fresh_topo_burst_v1_gate.py wf1cal --inputs "$IN" --out "$TR/gate" --parallel "$P" --mem 4G \
      --timeout "${TIMEOUT:-1800}" --no-scope | grep -v "^\[skip"
    set -e
    echo "=== tune: $(ls "$TR"/gate/*.summary.json 2>/dev/null | wc -l) summaries, $(ls "$TR"/gate/*.failed.json 2>/dev/null | wc -l) failed ==="
    ;;
  bisect)
    BDIR=${BISECT_NAME:-bisect}
    python3 scripts_cosim/workload_fix_v1_bisect.py --work "$W/$BDIR" --grounded-wl "$X1" --cfg-dir "$W/inputs/cfg_cal" \
      --topologies $CAL_TOPOS --windows g0 g1 --parallel "$P" --log "$W/${BDIR}_steps.json" ${BISECT_ARGS:-}
    ;;
  stage)
    STAGE=${WF1_STAGE:?FAIL LOUD: set WF1_STAGE (w2)}
    SAMPLER=${WF1_SAMPLER:?FAIL LOUD: set WF1_SAMPLER}
    TAGS=${WF1_TAGS:?FAIL LOUD: set WF1_TAGS (tag=multiplier,...)}
    if [[ -n "${WF1_OUTROOT:-}" ]]; then IN=${WF1_IN:-$WF1_OUTROOT/inputs}; OUT=$WF1_OUTROOT/${WF1_GATE:-gate}
    else IN=${WF1_IN:-$W/inputs_$STAGE}; OUT=$W/gate_$STAGE; fi
    SELECT=${WF1_SELECT:-$SELECT}
    BT_FLAG=""; [[ -n "${WF1_BATCH_FIXED:-}" ]] && BT_FLAG="--batch-timeout-fixed $WF1_BATCH_FIXED"
    export WF1_ARMS=${WF1_ARMS:-}
    RUNGS=""
    for spec in ${TAGS//,/ }; do
      tag=${spec%%=*}; m=${spec##*=}
      [[ -d "$IN/wf1_$tag" ]] || python3 scripts_cosim/workload_fix_v1_build.py --grounded-wl "$X1" \
        --cfg-dir $D/small_batch_confirm_v1/inputs/cfg --topologies $(python3 -c "import json;print(*json.load(open('$SELECT'))['topologies'])") \
        --rung "$tag=$(python3 -c "print(repr(1.0/$m))")" --payload-sampler "$SAMPLER" $BT_FLAG --out "$IN/_b_$tag"
      [[ -d "$IN/wf1_$tag" ]] || { mkdir -p "$IN"; mv "$IN/_b_$tag/wf1_$tag" "$IN/wf1_$tag"; cp "$IN/_b_$tag/manifest.json" "$IN/manifest_$tag.json"; rm -rf "$IN/_b_$tag"; }
      RUNGS="$RUNGS,$tag"
    done
    RUNGS=${RUNGS#,}
    mkdir -p "$OUT"
    set +e
    WF1_RUNGS=$RUNGS python3 scripts_cosim/fresh_topo_burst_v1_gate.py wf1 --inputs "$IN" --out "$OUT" --selection "$SELECT" \
      --parallel "$P" --mem 4G --timeout "${TIMEOUT:-2700}" --no-scope | grep -v "^\[skip"
    rc=${PIPESTATUS[0]}
    # registered failure rule (transfer_physics_v1): one rerun of a failed run at 3x the timeout; a second failure stands
    if ls "$OUT"/*.failed.json >/dev/null 2>&1; then
      echo "=== first pass: $(ls "$OUT"/*.failed.json | wc -l) failed; rerunning once at 3x timeout ==="
      mkdir -p "$OUT/first_pass_failed" && mv "$OUT"/*.failed.json "$OUT/first_pass_failed/"
      WF1_RUNGS=$RUNGS python3 scripts_cosim/fresh_topo_burst_v1_gate.py wf1 --inputs "$IN" --out "$OUT" --selection "$SELECT" \
        --parallel "$P" --mem 4G --timeout "$(( ${TIMEOUT:-2700} * 3 ))" --no-scope | grep -v "^\[skip"
      rc=${PIPESTATUS[0]}
    fi
    set -e
    echo "=== $STAGE: $(ls "$OUT"/*.summary.json 2>/dev/null | wc -l) summaries, $(ls "$OUT"/*.failed.json 2>/dev/null | wc -l) failed ==="
    find "$OUT" -name "*.log" -size +50M -exec gzip -1 {} \;
    echo "driver rc=$rc"
    ;;
  *) echo "FAIL LOUD: WF1_JOB=$JOB"; exit 1 ;;
esac
