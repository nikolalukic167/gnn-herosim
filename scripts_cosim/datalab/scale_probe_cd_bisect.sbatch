#!/bin/bash
# Scale go/no-go (c): a 2-step CD bisection toward the heavy band (CD effective queue share 0.40-0.50, load_recalibration_v1) on fresh
# scaled topologies. One array task per seed: mint the cell, then evaluate CD (window g0, full stats) at m0 = heavy x servers/6; if the
# effective share is below the band double m (up to two more steps); if a step overshoots, the next step is the geometric midpoint.
#   SEEDS="9901 9902 ..." sbatch --array=0-<n-1> --export=ALL,WT=...,OUT=...,SEEDS="...",NC=160,NS=24,P=0.6 scale_probe_cd_bisect.sbatch
#SBATCH --job-name=sp-cdbis
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/sp-cdbis-%A_%a.out
#SBATCH --partition=CPU-amd
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
set -euo pipefail
WT=${WT:?}; OUT=${OUT:?}; SEEDS=${SEEDS:?}; NC=${NC:-160}; NS=${NS:-24}; P=${P:-0.6}; M0=${M0:-11.6139}
ARR=($SEEDS); SEED=${ARR[${SLURM_ARRAY_TASK_ID:?}]:?}
X1=/home/nikola.lukic/gnn-herosim/simulation_data/small_batch_confirm_v1/inputs/grounded/wl
BASECFG=/home/nikola.lukic/gnn-herosim/simulation_data/workload_fix_v1/corpus_prod_prov/cfg_src/cc40s9101.json
cd "$WT"; eval "$(micromamba shell hook --shell bash)"; micromamba activate gnn
for v in $(compgen -e | grep -E '^(HEROSIM_|GNN_|COSIM_|LIVE_AUDIT_|GATE_|SIM_|QUEUE_FEATURE|MAX_PLACEMENT|DECLARED_SLATE)'); do unset "$v"; done
export HEROSIM_PY=python3 PYTHONPATH=$WT PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2
CELL=c${NC}s${NS}p${P}s${SEED}; ROOT=$OUT/$CELL; mkdir -p $ROOT/raw $ROOT/inputs; RAW=/tmp/sp_cdbis_${SLURM_JOB_ID}_$SEED; mkdir -p $RAW
echo "HEAD $(git rev-parse HEAD) cell $CELL node $(hostname) $(date -Is)"
python3 scripts_cosim/scale_probe_cfg.py "$BASECFG" "$ROOT/cfg_src" --clients "$NC" --servers "$NS" --p "$P" --seed "$SEED" >/dev/null
evaluate() {  # $1 = multiplier; prints the effective share (or plain share if lock wait is unavailable)
  m=$1; tag=m$(python3 -c "print('%.4f'%$m)" | tr . p)
  if [[ ! -d "$ROOT/inputs/wf1_$tag" ]]; then
    factor=$(python3 -c "print(repr(1.0/$m))")   # m is the FINAL multiplier (the first run of this driver scaled twice: 4x the capture's load; retracted)
    python3 scripts_cosim/workload_fix_v1_build.py --grounded-wl "$X1" --cfg-dir "$ROOT/cfg_src" --topologies "$SEED" --rung "$tag=$factor" \
      --payload-sampler wf1_v1 --task-mix wf1_v1 --batch-timeout-fixed 1 --out "$ROOT/_b_$tag" >/dev/null
    mkdir -p "$ROOT/inputs_w24"; mv "$ROOT/_b_$tag/wf1_$tag" "$ROOT/inputs_w24/wf1_$tag"; cp "$ROOT/_b_$tag/manifest.json" "$ROOT/inputs_w24/manifest_$tag.json"
    python3 scripts_cosim/workload_fix_v1_w3_build.py --w2-inputs "$ROOT/inputs_w24" --tags $tag --out "$ROOT/inputs_$tag" --diff-out "$ROOT/w3_vs_w2_$tag.json" >/dev/null
    mv "$ROOT/inputs_$tag/wf1_$tag" "$ROOT/inputs/wf1_$tag"
    f="$ROOT/inputs/wf1_$tag/wl/grounded_g0_n50000.json"; python3 scripts_cosim/scale_probe_remap_clients.py "$f" "$f.tmp" --clients "$NC" >/dev/null && mv "$f.tmp" "$f"
    ln -sfn "$ROOT/inputs/wf1_$tag" "$ROOT/inputs/grounded_$tag"
  fi
  ( export HEROSIM_AUDIT_INPUTS=$ROOT/inputs TS=1.0 SIM_FORCE_FULL_STATS=1 HEROSIM_SHARED_AUTOSCALER=0 GATE_FIXED_POLICY_TIME_SCALE=1.0
    t0=$(date +%s); timeout --kill-after=60 2700 scripts_cosim/physics_audit/run_cell.sh $SEED g0 $tag peer_greedy_network_cd $RAW/${CELL}_${tag}_g0.raw.json 50000 > $ROOT/cd_${tag}.log 2>&1 || echo "rc=$? (timeout 124 = hung)" >> $ROOT/cd_${tag}.log
    echo "wall_s $(( $(date +%s) - t0 ))" >> $ROOT/cd_${tag}.log )
  python3 scripts_cosim/scale_probe_summary.py $RAW $ROOT/inputs > $ROOT/summary_$tag.json 2>>$ROOT/cd_${tag}.log || true
  rm -f $RAW/${CELL}_${tag}_g0.raw.json
  python3 - $ROOT/summary_$tag.json $tag <<'E'
import json,sys
d=json.load(open(sys.argv[1])); v=d.get(sys.argv[2]+"_g0") or {}
lw=v.get("lock_wait") or {}; eff=lw.get("effective_queue_share"); plain=v.get("queue_share")
print(json.dumps(dict(tag=sys.argv[2], effective_share=eff, plain_share=plain, elapsed=v.get("elapsed_mean"), failures=v.get("request_failures"), end_over_arrival=v.get("run_end_over_last_arrival"), p95=(v.get("latency_percentiles") or {}).get("p95"))), file=sys.stderr)
print(eff if eff is not None else (plain if plain is not None else "nan"))
E
}
m=$(python3 -c "print($M0*$NS/6.0)"); prev=""
for step in 0 1 2; do
  s=$(evaluate $m 2>>$ROOT/steps.jsonl); echo "step $step m=$m share=$s"
  inband=$(python3 -c "s=float('$s'); print(1 if 0.40<=s<=0.50 else (2 if s>0.50 else 0))")
  [[ $inband == 1 ]] && { echo "BAND REACHED at m=$m"; break; }
  if [[ $inband == 2 ]]; then [[ -z "$prev" ]] && { echo "OVERSHOOT at m0; the band is below m0"; break; }; next=$(python3 -c "import math; print(math.sqrt($prev*$m))"); else next=$(python3 -c "print(2*$m)"); fi
  prev=$m; m=$next
done
cat $ROOT/steps.jsonl; rm -rf $RAW; echo "done $CELL $(date -Is)"
