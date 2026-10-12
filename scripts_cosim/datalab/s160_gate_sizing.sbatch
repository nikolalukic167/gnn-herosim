#!/bin/bash
# scale_160_v1 gate sizing (descriptive, resources only): one SPARE topology (16020, never a test topology) at g0 on the moderate and heavy
# rungs, CD, self-predict and a smoke-weights gnn_eng cell, one cell at a time so the peak RSS and the wall time belong to one cell.
# The latencies are not read: only wall time, peak RSS and decision cost are extracted, and the summaries are deleted afterwards.
#   sbatch --export=ALL,WT=<pinned worktree>,EXPECT_HEAD=<sha>,ROOT=<new dir>,M_MODERATE=16.4245,M_HEAVY=23.2278 s160_gate_sizing.sbatch
#SBATCH --job-name=s160-size
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/s160-size-%j.out
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=06:00:00
set -euo pipefail
WT=${WT:?}; EXPECT_HEAD=${EXPECT_HEAD:?}; ROOT=${ROOT:?}; M_MODERATE=${M_MODERATE:?}; M_HEAVY=${M_HEAVY:?}; TOPO=${TOPO:-16020}
D=/home/nikola.lukic/gnn-herosim/simulation_data
SMOKE=$D/workload_fix_v1/r1a_arms_servesmoke_inputs
cd "$WT"
for v in $(compgen -e | grep -E '^(HEROSIM_|GNN_|COSIM_|LIVE_AUDIT_|GATE_|SIM_|QUEUE_FEATURE|MAX_PLACEMENT|NEAR_RTT_|PARTIAL_STATE|WF1_|R1A_)'); do unset "$v"; done
if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3 PYTHONPATH="$WT" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export HEROSIM_RAW_DIR=/tmp/herosim_raw_${SLURM_JOB_ID:-$$}; mkdir -p "$HEROSIM_RAW_DIR"; trap 'rm -rf "$HEROSIM_RAW_DIR"' EXIT
HEAD=$(git rev-parse HEAD); echo "HEAD $HEAD node $(hostname) cpus=$(nproc) ROOT=$ROOT topo=$TOPO"
[[ "$HEAD" == "$EXPECT_HEAD"* ]] || { echo "FAIL LOUD: worktree HEAD $HEAD is not $EXPECT_HEAD"; exit 1; }
[[ "$TOPO" -ge 16013 && "$TOPO" -le 16020 ]] || { echo "FAIL LOUD: sizing runs on a spare (16013-16020), not $TOPO"; exit 1; }
[[ ! -e "$ROOT" ]] || { echo "FAIL LOUD: $ROOT exists"; exit 1; }
export HEROSIM_TRANSFER_MODEL=pipelined HEROSIM_REPLICA_RELEASE=1 HEROSIM_SCALEOUT=kpa GATE_FIXED_POLICY_TIME_SCALE=1.0 HEROSIM_SHARED_AUTOSCALER=0
mkdir -p "$ROOT"
python3 scripts_cosim/scale_probe_cfg.py "$D/workload_fix_v1/corpus_prod_prov/cfg_src/cc40s9101.json" "$ROOT/cfg_src" --clients 160 --servers 24 --p 0.6 --seed "$TOPO" >/dev/null
for spec in moderate=$M_MODERATE heavy=$M_HEAVY; do
  tag=${spec%%=*}; m=${spec##*=}
  python3 scripts_cosim/workload_fix_v1_build.py --grounded-wl "$D/small_batch_confirm_v1/inputs/grounded/wl" --cfg-dir "$ROOT/cfg_src" --topologies "$TOPO" \
    --rung "$tag=$(python3 -c "print(repr(1.0/$m))")" --payload-sampler wf1_v1 --task-mix wf1_v1 --batch-timeout-fixed 1.0 --out "$ROOT/_b_$tag" >/dev/null
  mkdir -p "$ROOT/inputs_w24"; mv "$ROOT/_b_$tag/wf1_$tag" "$ROOT/inputs_w24/wf1_$tag"; cp "$ROOT/_b_$tag/manifest.json" "$ROOT/inputs_w24/manifest_$tag.json"; rm -rf "$ROOT/_b_$tag"
done
python3 scripts_cosim/workload_fix_v1_w3_build.py --w2-inputs "$ROOT/inputs_w24" --tags moderate heavy --out "$ROOT/inputs" --diff-out "$ROOT/w3_vs_w2_diff.json" >/dev/null
for tag in moderate heavy; do f="$ROOT/inputs/wf1_$tag/wl/grounded_g0_n50000.json"; python3 scripts_cosim/scale_probe_remap_clients.py "$f" "$f.tmp" --clients 160 >/dev/null && mv "$f.tmp" "$f"; done
ln -s "$SMOKE/models" "$ROOT/inputs/models"; cp "$SMOKE/r1_attribution_v1_split.json" "$ROOT/inputs/"
echo '{"verdict":"DESIGN-READY","topologies":['$TOPO']}' > "$ROOT/sel.json"
# peak RSS of the simulation process, sampled every 3 s (one cell at a time, so the series belongs to one cell)
( while true; do ps -eo rss,etimes,args | awk '/executesimulation.py --config/ && !/timeout/ {print strftime("%T"), $1, $2}' >> "$ROOT/rss.log"; sleep 3; done ) &
SAMP=$!
export WF1_RUNGS=moderate,heavy R1A_TOPOS=$TOPO R1A_WINDOWS=g0 R1A_SEEDS=1 R1A_ARMS=cd,selfpredict,ra_gnn_eng
python3 scripts_cosim/fresh_topo_burst_v1_gate.py r1a --inputs "$ROOT/inputs" --out "$ROOT/gate" --selection "$ROOT/sel.json" \
  --parallel 1 --mem 64G --timeout 8100 --no-scope | grep -v "^\[skip" || true
kill $SAMP 2>/dev/null || true
python3 - "$ROOT" <<'PY'
import glob, json, os, sys
root = sys.argv[1]
peak = max([int(l.split()[1]) for l in open(root + "/rss.log")] or [0]) / 1048576
print(f"peak simulation RSS over the whole job: {peak:.2f} GiB")
for f in sorted(glob.glob(root + "/gate/*.summary.json")) + sorted(glob.glob(root + "/gate/*.failed.json")):
    s = json.load(open(f)); t = s.get("decisionTiming") or {}
    print(os.path.basename(f).replace(".json", ""), "wall_s", s.get("wallclock_s"), "per_task_median_us", None if t.get("per_task_median_s") is None else round(1e6 * t["per_task_median_s"]), s.get("why", ""))
    if f.endswith(".summary.json"):
        os.remove(f)
PY
