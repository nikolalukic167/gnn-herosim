#!/bin/bash
# long_horizon_s0_v1 inputs (docs/lineages/long_horizon_s0_v1.md): the stationary 24 h window L24 at the accel moderate and heavy rungs
# (scripts_cosim/long_horizon_mint.py; cycles 0-3 must equal the accel gate windows g0-g3 at each rung, so L24 starts with g0), the
# topology's cfg at each rung (from the big_groups G=8 gate inputs: the same accel cell, workload-independent), the accel models and split,
# and a one-topology selection.
#   WT, EXPECT_HEAD   pinned detached worktree and its commit
#   ROOT              new directory; <ROOT>/inputs is the gate's IN
#   TOPO              default 16900
#   sbatch --export=ALL,WT=<wt>,EXPECT_HEAD=<sha>,ROOT=<dir> long_horizon_inputs.sbatch
#SBATCH --job-name=lh-in
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/lh-in-%j.out
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=03:00:00
set -euo pipefail
WT=${WT:?FAIL LOUD: set WT}; EXPECT_HEAD=${EXPECT_HEAD:?FAIL LOUD: set EXPECT_HEAD}; ROOT=${ROOT:?FAIL LOUD: set ROOT}
TOPO=${TOPO:-16900}
D=/home/nikola.lukic/gnn-herosim/simulation_data
ACC=/share/nikola.lukic/accel_gate/inputs_41ef627b/inputs
BG=/share/nikola.lukic/big_groups/gate_inputs_G8/inputs
MODELS=/share/nikola.lukic/accel_prod/gate_inputs/models
X1=$D/small_batch_confirm_v1/inputs/grounded/wl
BASE=$D/peer_affinity_live_gate/workloads
cd "$WT"
for v in $(compgen -e | grep -E '^(HEROSIM_|GNN_|COSIM_|LIVE_AUDIT_|GATE_|SIM_|QUEUE_FEATURE|MAX_PLACEMENT|NEAR_RTT_|PARTIAL_STATE|WF1_|R1A_)'); do unset "$v"; done
if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3 PYTHONPATH="$WT" PYTHONUNBUFFERED=1 PYTHONHASHSEED=0
python3 -c "import sys,torch;print('[INTERP]',sys.executable,torch.__version__)"
HEAD=$(git rev-parse HEAD)
echo "HEAD $HEAD node $(hostname) job=${SLURM_JOB_ID:-} ROOT=$ROOT TOPO=$TOPO"
[[ "$HEAD" == "$EXPECT_HEAD"* ]] || { echo "FAIL LOUD: worktree HEAD $HEAD is not $EXPECT_HEAD"; exit 1; }
[[ ! -e "$ROOT/inputs/wf1_moderate" && ! -e "$ROOT/inputs/wf1_heavy" ]] || { echo "FAIL LOUD: $ROOT already holds minted windows"; exit 1; }
mkdir -p "$ROOT/inputs"
for r in moderate heavy; do
  python3 scripts_cosim/long_horizon_mint.py --rung $r --manifest "$ACC/manifest_$r.json" --gate-wl "$ACC/wf1_$r/wl" --w2-wl "$ACC/../inputs_w24/wf1_$r/wl" --x1 "$X1" \
    --lib-dir "$WT/docs/lineages/grounded_workload_v1" --base-dir "$BASE" --cfg "$BG/wf1_$r/cfg/cc40s$TOPO.json" --out "$ROOT/inputs/wf1_$r"
done
ln -s "$ACC/accel_replica_v1_split.json" "$ROOT/inputs/accel_replica_v1_split.json"
ln -s "$MODELS" "$ROOT/inputs/models"
md5sum "$ROOT/inputs/models/accel-replica-v1-gnn-eng-seed1.pt"
python3 - "$ROOT/inputs/models/accel-replica-v1-gnn-eng-seed1.pt" <<'PY'
import hashlib, sys
h = hashlib.md5(open(sys.argv[1], "rb").read()).hexdigest()
assert h.startswith("0f915a36"), f"FAIL LOUD: g4 seed-1 checkpoint md5 {h}, want 0f915a36..."
print("[ckpt] g4 seed1 md5", h)
PY
[[ -f "$ROOT/inputs/models/accel-replica-v1-gnn-eng-seed1.contract.json" ]] || { echo "FAIL LOUD: g4 seed-1 sidecar missing"; exit 1; }
python3 - "$ROOT/inputs/selection.json" "$TOPO" <<'PY'
import json, sys
json.dump({"verdict": "DESIGN-READY", "lineage": "long_horizon_s0_v1 probe", "topologies": [int(sys.argv[2])],
           "note": "one spent topology (big_groups_s0_v1 G=8 gate set 16900-16911), run on the accel K=1 L24 mint; disclosed"},
          open(sys.argv[1], "w"), indent=1)
PY
sha256sum "$ROOT"/inputs/wf1_*/wl/long_horizon_L24.json
echo "inputs ready: $ROOT/inputs"
