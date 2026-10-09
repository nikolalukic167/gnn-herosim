#!/bin/bash
# r1_attribution_v1 decision-cost side-study (descriptive): 1 test topology x 3 rungs x g0 x every classical arm, CD<-random and the CD<-X / learned
# arms served from the smoke checkpoints (architecture, not weights, sets the decode cost; labelled "smoke weights"), on a dedicated node
# (--exclusive, OMP_NUM_THREADS=1, PAR cells in parallel) so the per-decision wall times compare across arms.
#   sbatch --export=ALL,WT=<pinned worktree>,EXPECT_HEAD=<sha>,IN=<inputs from r1a_gate_inputs>,OUT=<new dir> r1a_gate_timing.sbatch
#SBATCH --job-name=r1a-time
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/r1a-time-%j.out
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=04:00:00
set -euo pipefail
WT=${WT:?}; EXPECT_HEAD=${EXPECT_HEAD:?}; IN=${IN:?}; OUT=${OUT:?}; TOPO=${TOPO:-9483}
D=/home/nikola.lukic/gnn-herosim/simulation_data
SMOKE=$D/workload_fix_v1/r1a_arms_servesmoke_inputs
cd "$WT"
for v in $(compgen -e | grep -E '^(HEROSIM_|GNN_|COSIM_|LIVE_AUDIT_|GATE_|SIM_|QUEUE_FEATURE|MAX_PLACEMENT|NEAR_RTT_|PARTIAL_STATE|WF1_|R1A_)'); do unset "$v"; done
if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3 PYTHONPATH="$WT" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export HEROSIM_RAW_DIR=/tmp/herosim_raw_${SLURM_JOB_ID:-$$}; mkdir -p "$HEROSIM_RAW_DIR"; trap 'rm -rf "$HEROSIM_RAW_DIR"' EXIT
HEAD=$(git rev-parse HEAD); echo "HEAD $HEAD node $(hostname) cpus=$(nproc) OUT=$OUT"
[[ "$HEAD" == "$EXPECT_HEAD"* ]] || { echo "FAIL LOUD: worktree HEAD $HEAD is not $EXPECT_HEAD"; exit 1; }
# a wedged Ceph client can hang `git status` (datalab-pitfalls 14): bounded `git diff --quiet`, a warning rather than a stop
rc=0; timeout 300 git diff --quiet HEAD -- src scripts_cosim experiments run_experiment.py || rc=$?
[[ $rc == 0 ]] || echo "WARN: worktree check rc=$rc (1 = tracked files differ from HEAD, 124 = timed out after 300 s)"
[[ ! -e "$OUT" ]] || { echo "FAIL LOUD: $OUT exists"; exit 1; }
mkdir -p "$OUT/in"
for r in light moderate heavy; do ln -s "$IN/wf1_$r" "$OUT/in/wf1_$r"; done
ln -s "$SMOKE/models" "$OUT/in/models"; cp "$SMOKE/r1_attribution_v1_split.json" "$OUT/in/"
export HEROSIM_TRANSFER_MODEL=pipelined HEROSIM_REPLICA_RELEASE=1 HEROSIM_SCALEOUT=kpa GATE_FIXED_POLICY_TIME_SCALE=1.0 HEROSIM_SHARED_AUTOSCALER=0
export WF1_RUNGS=light,moderate,heavy R1A_TOPOS=$TOPO R1A_WINDOWS=g0 R1A_SEEDS=1
python3 scripts_cosim/fresh_topo_burst_v1_gate.py r1a --inputs "$OUT/in" --out "$OUT/gate" --selection "$D/small_batch_confirm_v1/inputs/selected.json" \
  --parallel "${PAR:-8}" --mem 8G --timeout 8100 --no-scope | grep -v "^\[skip"
python3 - "$OUT/gate" <<'PY'
import glob, json, sys
rows = []
for f in sorted(glob.glob(sys.argv[1] + "/*.summary.json")):
    s = json.load(open(f)); t = s.get("decisionTiming") or {}
    rows.append((s["arm"].split("__")[2], s["arm"].split("__")[1], t))
print(f"{'cell':44s} {'calls':>6s} {'tasks':>6s} {'call med ms':>11s} {'call p95 ms':>11s} {'task med ms':>11s} {'task p95 ms':>11s} {'total s':>8s} {'moves/call':>10s} {'load s':>7s}")
f = lambda t, k, m=1000.0: "-" if t.get(k) is None else f"{t[k] * m:.2f}"
for kind, win, t in rows:
    print(f"{kind + ' ' + win:44s} {t.get('calls', 0):6d} {t.get('tasks', 0):6d} {f(t, 'call_median_s'):>11s} {f(t, 'call_p95_s'):>11s} {f(t, 'per_task_median_s'):>11s} {f(t, 'per_task_p95_s'):>11s} {f(t, 'total_s', 1):>8s} {f(t, 'refine_moves_per_call_mean', 1):>10s} {f(t, 'model_load_s', 1):>7s}")
PY
