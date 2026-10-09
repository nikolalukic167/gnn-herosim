#!/bin/bash
# r1_attribution_v1 live gate DRY RUN: 1 test topology x 3 rungs x window g0 x every arm (15), served from the smoke checkpoints (seed 1), plus
#   (a) the watchdog: Knative at the heavy rung with a 60 s limit and a 20 s grace must be killed and counted (failed.json "watchdog: ..."),
#   (b) identity: one CD cell run through the watchdog path and through the old wf1cal path (no progress file): every simulated field equal.
#   sbatch --export=ALL,WT=<pinned worktree>,EXPECT_HEAD=<sha>,IN=<inputs from r1a_gate_inputs>,OUT=<new dir> r1a_gate_dry.sbatch
#SBATCH --job-name=r1a-dry
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/r1a-dry-%j.out
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=240G
#SBATCH --time=06:00:00
set -euo pipefail
WT=${WT:?}; EXPECT_HEAD=${EXPECT_HEAD:?}; IN=${IN:?}; OUT=${OUT:?}; TOPO=${TOPO:-9483}
D=/home/nikola.lukic/gnn-herosim/simulation_data
SMOKE=$D/workload_fix_v1/r1a_arms_servesmoke_inputs
cd "$WT"
for v in $(compgen -e | grep -E '^(HEROSIM_|GNN_|COSIM_|LIVE_AUDIT_|GATE_|SIM_|QUEUE_FEATURE|MAX_PLACEMENT|NEAR_RTT_|PARTIAL_STATE|WF1_)'); do unset "$v"; done
if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3 PYTHONPATH="$WT" PYTHONUNBUFFERED=1
export HEROSIM_RAW_DIR=/tmp/herosim_raw_${SLURM_JOB_ID:-$$}; mkdir -p "$HEROSIM_RAW_DIR"; trap 'rm -rf "$HEROSIM_RAW_DIR"' EXIT
python3 -c "import sys,torch;print('[INTERP]',sys.executable,torch.__version__)"
HEAD=$(git rev-parse HEAD); echo "HEAD $HEAD node $(hostname) OUT=$OUT"
[[ "$HEAD" == "$EXPECT_HEAD"* ]] || { echo "FAIL LOUD: worktree HEAD $HEAD is not $EXPECT_HEAD"; exit 1; }
[[ -z "$(git status --porcelain -- src scripts_cosim experiments run_experiment.py)" ]] || { echo "FAIL LOUD: worktree dirty"; exit 1; }
[[ ! -e "$OUT" ]] || { echo "FAIL LOUD: $OUT exists"; exit 1; }
mkdir -p "$OUT/in"
for r in light moderate heavy; do ln -s "$IN/wf1_$r" "$OUT/in/wf1_$r"; done
ln -s "$SMOKE/models" "$OUT/in/models"; cp "$SMOKE/r1_attribution_v1_split.json" "$OUT/in/"
SEL=$D/small_batch_confirm_v1/inputs/selected.json
export HEROSIM_TRANSFER_MODEL=pipelined HEROSIM_REPLICA_RELEASE=1 HEROSIM_SCALEOUT=kpa GATE_FIXED_POLICY_TIME_SCALE=1.0 HEROSIM_SHARED_AUTOSCALER=0
G="python3 scripts_cosim/fresh_topo_burst_v1_gate.py"
# (a) watchdog on Knative at heavy, in the background
( WF1_RUNGS=heavy R1A_TOPOS=$TOPO R1A_WINDOWS=g0 R1A_ARMS=reactive R1A_GRACE_S=20 \
  $G r1a --inputs "$OUT/in" --out "$OUT/watchdog" --selection "$SEL" --parallel 1 --mem 8G --timeout 60 --no-scope > "$OUT/watchdog.log" 2>&1 ) &
# (b) the old path for the identity cell, in the background
( WF1_RUNGS=moderate WF1_TOPOS=$TOPO WF1_CAL_WINDOWS=g0 WF1_CAL_KINDS=cd \
  $G wf1cal --inputs "$OUT/in" --out "$OUT/identity_old" --parallel 1 --mem 8G --timeout 7200 --no-scope > "$OUT/identity_old.log" 2>&1 ) &
# the dry run proper
WF1_RUNGS=light,moderate,heavy R1A_TOPOS=$TOPO R1A_WINDOWS=g0 R1A_SEEDS=1 \
  $G r1a --inputs "$OUT/in" --out "$OUT/gate" --selection "$SEL" --parallel "${PAR:-30}" --mem 8G --timeout "${LIMIT:-8100}" --no-scope | grep -v "^\[skip"
wait
echo "== watchdog =="; cat "$OUT/watchdog.log"; cat "$OUT"/watchdog/*.failed.json 2>/dev/null; echo
echo "== identity =="
python3 - "$OUT" "$TOPO" <<'PY'
import json, sys
out, topo = sys.argv[1], sys.argv[2]
name = f"cc40s{topo}__g0moderate__cd_s0.summary.json"
a, b = json.load(open(f"{out}/identity_old/{name}")), json.load(open(f"{out}/gate/{name}"))
skip = {"wallclock_s", "code", "total_rtt_plus_inference", "env"}
diff = [k for k in sorted(set(a) | set(b)) if k not in skip and a.get(k) != b.get(k)]
envd = {k: (a["env"].get(k), b["env"].get(k)) for k in set(a["env"]) | set(b["env"]) if a["env"].get(k) != b["env"].get(k)}
print("[identity] differing simulated fields:", diff or "none", "| env keys differing:", envd)
sys.exit(1 if diff else 0)
PY
echo "failed cells: $(ls "$OUT"/gate/*.failed.json 2>/dev/null | wc -l)  summaries: $(ls "$OUT"/gate/*.summary.json 2>/dev/null | wc -l)"
