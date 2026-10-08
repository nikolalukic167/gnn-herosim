#!/bin/bash
# r1a-features acceptance: the default live path is unchanged. Re-run 3 cells of the R1.1 W2 stage (reactive and CD, rung
# lo, window g0, topologies 9483-9485, 1 s window inputs) on this worktree and compare every numeric summary field with
# the finished stage_w2_r11 run (WORKTREE = the code under test; REF = the reference gate directory).
#SBATCH --job-name=r1aid
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/r1aid-%j.out
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
set -euo pipefail
WT=${WT:?FAIL LOUD: set WT}
W=/home/nikola.lukic/gnn-herosim/simulation_data/workload_fix_v1
IN=${IN:-$W/stage_w2_wb2/inputs}
REF=${REF:-$W/stage_w2_r11}
OUT=${OUT:?FAIL LOUD: set OUT}
cd "$WT"
if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3 PYTHONPATH="$WT" PYTHONUNBUFFERED=1
export HEROSIM_RAW_DIR=/tmp/herosim_raw_${SLURM_JOB_ID:-$$}; mkdir -p "$HEROSIM_RAW_DIR"; trap 'rm -rf "$HEROSIM_RAW_DIR"' EXIT
export HEROSIM_TRANSFER_MODEL=pipelined HEROSIM_REPLICA_RELEASE=1 HEROSIM_SCALEOUT=kpa GATE_FIXED_POLICY_TIME_SCALE=1.0
export HEROSIM_SHARED_AUTOSCALER=0 WF1_ARMS=
echo "HEAD $(git rev-parse HEAD)"
[[ ! -e "$OUT" ]] || { echo "FAIL LOUD: $OUT exists"; exit 1; }
WF1_RUNGS=lo WF1_TOPOS=9483,9484,9485 WF1_CAL_WINDOWS=g0 WF1_CAL_KINDS=reactive,cd \
  python3 scripts_cosim/fresh_topo_burst_v1_gate.py wf1cal --inputs "$IN" --out "$OUT" --parallel 6 --mem 4G --timeout 2700 --no-scope | grep -v "^\[skip"
python3 - "$OUT" "$REF" <<'PY'
import glob, json, os, sys
out, ref = sys.argv[1:3]
SKIP = {"wallclock_s", "code", "env", "run_provenance", "corpus", "arm", "cell", "checkpoint_seed", "policy_name", "traceFile"}
def flat(d, p=""):
    for k, v in d.items():
        if k in SKIP: continue
        if isinstance(v, dict): yield from flat(v, p + k + ".")
        elif isinstance(v, (int, float)) and not isinstance(v, bool): yield p + k, v
n = bad = 0
for f in sorted(glob.glob(out + "/*.summary.json")):
    g = os.path.join(ref, os.path.basename(f))
    if not os.path.exists(g): print("NO REFERENCE", os.path.basename(f)); bad += 1; continue
    a, b = dict(flat(json.load(open(f)))), dict(flat(json.load(open(g))))
    diff = {k: (a.get(k), b.get(k)) for k in set(a) | set(b) if a.get(k) != b.get(k)}
    n += 1
    print(os.path.basename(f), "fields", len(a), "differing", len(diff), dict(list(diff.items())[:4]))
    bad += bool(diff)
print(f"cells compared {n}, with differences {bad}")
sys.exit(1 if bad or not n else 0)
PY
