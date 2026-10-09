#!/bin/bash
# r1_attribution_v1: one live cell per ra_* kind (topology WF1_TOPOS, moderate rung, window g0) through the gate driver (phase wf1cal), served from
# the smoke checkpoints. Checks that every cell finishes, queue share < 1, and the provenance names the right arm.
#   sbatch --export=ALL,WT=<pinned worktree>,IN=<inputs dir with wf1_moderate/, models/, r1_attribution_v1_split.json>,OUT=<new dir> r1a_arms_servesmoke.sbatch
#SBATCH --job-name=r1a-serve
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/r1a-serve-%j.out
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
set -euo pipefail
WT=${WT:?}; IN=${IN:?}; OUT=${OUT:?}; TOPO=${TOPO:-9101}
cd "$WT"
for v in $(compgen -e | grep -E '^(HEROSIM_|GNN_|COSIM_|LIVE_AUDIT_|GATE_|SIM_|QUEUE_FEATURE|MAX_PLACEMENT|NEAR_RTT_|PARTIAL_STATE|WF1_)'); do unset "$v"; done
if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3 PYTHONPATH="$WT" PYTHONUNBUFFERED=1
export HEROSIM_RAW_DIR=/tmp/herosim_raw_${SLURM_JOB_ID:-$$}; mkdir -p "$HEROSIM_RAW_DIR"; trap 'rm -rf "$HEROSIM_RAW_DIR"' EXIT
echo "HEAD $(git rev-parse HEAD) node $(hostname)"
[[ -z "$(git status --porcelain -- src scripts_cosim experiments run_experiment.py)" ]] || { echo "FAIL LOUD: worktree dirty"; exit 1; }
[[ ! -e "$OUT" ]] || { echo "FAIL LOUD: $OUT exists"; exit 1; }
export HEROSIM_TRANSFER_MODEL=pipelined HEROSIM_REPLICA_RELEASE=1 HEROSIM_SCALEOUT=kpa GATE_FIXED_POLICY_TIME_SCALE=1.0 HEROSIM_SHARED_AUTOSCALER=0
export WF1_RUNGS=moderate WF1_TOPOS=$TOPO WF1_CAL_WINDOWS=g0
export WF1_CAL_KINDS=ra_gnn_eng,ra_twin_eng,ra_mlp_same,ra_gnn_raw,ra_twin_raw,ra_gnn_eng_physmp,ra_set_transformer
python3 scripts_cosim/fresh_topo_burst_v1_gate.py wf1cal --inputs "$IN" --out "$OUT" --parallel 7 --mem 8G --timeout 7200 --no-scope
python3 - "$OUT" <<'PY'
import glob, json, sys
bad = 0
for f in sorted(glob.glob(sys.argv[1] + "/*.summary.json")):
    s = json.load(open(f)); name = f.split("/")[-1].replace(".summary.json", "")
    prov = json.dumps(s.get("run_provenance") or s.get("env") or {})
    print(name, "queue_share", s.get("queue_share"), "avg_elapsed", s.get("averageElapsedTime"), "| arm in provenance:", [k for k in ("gnn_eng","twin_eng","mlp_same","gnn_raw","twin_raw","gnn_eng_physmp","set_transformer") if f"r1-attribution-v1-{k.replace('_','-')}-seed" in prov])
PY
echo "failed cells: $(ls "$OUT"/*.failed.json 2>/dev/null | wc -l)"
