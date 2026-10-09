#!/bin/bash
# r1_attribution_v1 live gate, inputs: the 19 test topologies at the three recalibrated rungs (light x1.2584, moderate x5.9402,
# heavy x11.6139) as WF1 (W2 payloads + W3 access classes + W4 mix/repair, 1 s window), single origin, windows g0-g3.
# The heavy rung is checked against the stage inputs already used for these topologies (stage_w234_wb2, x11.6139): window files by sha256,
# cell configs by content. One job; ROOT must not exist.
#   sbatch --export=ALL,WT=<pinned worktree>,EXPECT_HEAD=<sha>,ROOT=<dir> r1a_gate_inputs.sbatch
#SBATCH --job-name=r1a-in
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/r1a-in-%j.out
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
set -euo pipefail
WT=${WT:?FAIL LOUD: set WT (pinned detached worktree)}
EXPECT_HEAD=${EXPECT_HEAD:?FAIL LOUD: set EXPECT_HEAD}
ROOT=${ROOT:?FAIL LOUD: set ROOT}
D=/home/nikola.lukic/gnn-herosim/simulation_data
X1=$D/small_batch_confirm_v1/inputs/grounded/wl
SELECT=$D/small_batch_confirm_v1/inputs/selected.json
STAGE=$D/workload_fix_v1/stage_w234_wb2/inputs
cd "$WT"
if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3 PYTHONPATH="$WT"
HEAD=$(git rev-parse HEAD)
echo "HEAD $HEAD node $(hostname) ROOT=$ROOT"
[[ "$HEAD" == "$EXPECT_HEAD"* ]] || { echo "FAIL LOUD: worktree HEAD $HEAD is not $EXPECT_HEAD"; exit 1; }
[[ -z "$(git status --porcelain -- src scripts_cosim experiments run_experiment.py)" ]] || { echo "FAIL LOUD: worktree dirty"; exit 1; }
[[ ! -e "$ROOT" ]] || { echo "FAIL LOUD: $ROOT exists"; exit 1; }
export HEROSIM_TRANSFER_MODEL=pipelined HEROSIM_REPLICA_RELEASE=1 HEROSIM_SCALEOUT=kpa GATE_FIXED_POLICY_TIME_SCALE=1.0 HEROSIM_SHARED_AUTOSCALER=0
mkdir -p "$ROOT"
python3 scripts_cosim/workload_fix_v1_stage_topologies.py --cfg-dir "$D/small_batch_confirm_v1/inputs/cfg" --selected "$SELECT" --out "$ROOT/cfg"
TOPOS=$(python3 -c "import json;print(*json.load(open('$SELECT'))['topologies'])")
for spec in light=1.2584 moderate=5.9402 heavy=11.6139; do
  tag=${spec%%=*}; m=${spec##*=}
  python3 scripts_cosim/workload_fix_v1_build.py --grounded-wl "$X1" --cfg-dir "$ROOT/cfg/cfg_w4" --topologies $TOPOS \
    --rung "$tag=$(python3 -c "print(repr(1.0/$m))")" --payload-sampler wf1_v1 --task-mix wf1_v1 --batch-timeout-fixed 1.0 --out "$ROOT/_b_$tag"
  mkdir -p "$ROOT/inputs"; mv "$ROOT/_b_$tag/wf1_$tag" "$ROOT/inputs/wf1_$tag"; cp "$ROOT/_b_$tag/manifest.json" "$ROOT/inputs/manifest_$tag.json"; rm -rf "$ROOT/_b_$tag"
done
python3 - "$ROOT/inputs" "$STAGE" <<'PY'
import hashlib, json, sys
inp, stage = sys.argv[1], sys.argv[2]
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
m_new, m_old = json.load(open(f"{inp}/manifest_heavy.json")), json.load(open(f"{stage}/manifest_hi.json"))
bad = []
for name, want in m_old["rungs"]["hi"]["wl_sha256"].items():
    got = m_new["rungs"]["heavy"]["wl_sha256"][name]
    if got != want:
        bad.append(f"window {name}")
for t in m_old["topologies"]:
    a, b = json.load(open(f"{inp}/wf1_heavy/cfg/cc40s{t}.json")), json.load(open(f"{stage}/wf1_hi/cfg/cc40s{t}.json"))
    if a != b:
        bad.append(f"cfg {t}: " + ",".join(sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))))
print("[check] heavy vs stage_w234_wb2 wf1_hi:", "IDENTICAL" if not bad else bad)
sys.exit(1 if bad else 0)
PY
echo "inputs ready: $ROOT/inputs (rungs light, moderate, heavy)"
