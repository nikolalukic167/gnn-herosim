#!/bin/bash
# plan_rerank_v1 Phase 1 smoke (wiring only, never a gate cell): (1) a smoke fit of the variant-a selector on the first 64 train /
# 32 val pools of the g4 pool, 1 epoch, marked smoke in its sidecar; (2) it is installed under the gate name in the smoke inputs;
# (3) one r1a cell of ra_gnn_eng_prsela on a NON-gate topology (default 16301, accel_replica_v1's gate set; never 19300-19305 or
# 19100-19123) with the smoke permission and the per-batch diag on. Read: the prsel_* counters and the instrument checks only.
#   WT, EXPECT_HEAD   pinned detached worktree and its commit
#   IN                smoke inputs: wf1_<rung>/{cfg,wl} models/ accel_replica_v1_split.json selection.json (models/ must be this smoke's own)
#   OUT               result directory
#   TOPO RUNG WINDOW  the cell (defaults 16301 moderate g0); LIMIT watchdog limit in s (default 36000)
#   SEL_FROM          optional: serve this earlier smoke selector (path without .pt) instead of fitting; SELNAME / ARM its gate name / arm
#SBATCH --job-name=prsel-smoke
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/prsel-smoke-%j.out
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
set -euo pipefail
WT=${WT:?FAIL LOUD: set WT}; EXPECT_HEAD=${EXPECT_HEAD:?FAIL LOUD: set EXPECT_HEAD}; IN=${IN:?}; OUT=${OUT:?}
TOPO=${TOPO:-16301}; RUNG=${RUNG:-moderate}; WINDOW=${WINDOW:-g0}
case "$TOPO" in 1930[0-5]|191[0-1][0-9]|1912[0-3]) echo "FAIL LOUD: $TOPO is a sealed plan_rerank_v1 gate topology"; exit 1;; esac
cd "$WT"
for v in $(compgen -e | grep -E '^(HEROSIM_|GNN_|COSIM_|LIVE_AUDIT_|GATE_|SIM_|QUEUE_FEATURE|MAX_PLACEMENT|NEAR_RTT_|PARTIAL_STATE|WF1_|PRSEL_)'); do unset "$v"; done
if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3 PYTHONPATH="$WT" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 PYTHONHASHSEED=0
export HEROSIM_RAW_DIR=/tmp/herosim_raw_${SLURM_JOB_ID:-$$}; mkdir -p "$HEROSIM_RAW_DIR"; trap 'rm -rf "$HEROSIM_RAW_DIR"' EXIT
python3 -c "import sys,torch;print('[INTERP]',sys.executable,torch.__version__)"
HEAD=$(git rev-parse HEAD)
echo "HEAD $HEAD node $(hostname) job=${SLURM_JOB_ID:-} IN=$IN OUT=$OUT cell=$TOPO/$RUNG/$WINDOW"
[[ "$HEAD" == "$EXPECT_HEAD"* ]] || { echo "FAIL LOUD: worktree HEAD $HEAD is not $EXPECT_HEAD"; exit 1; }
mkdir -p "$OUT"
# (1) smoke fit, unless SEL_FROM names an earlier smoke selector (<path stem>) to serve again (a replay of the same cell)
if [[ -n "${SEL_FROM:-}" ]]; then
  CK=$SEL_FROM
else
  python3 run_experiment.py experiments/plan_rerank_v1_gnn_eng_sela_smoke.yaml --seed 1
  CK=models/plan-rerank-v1-gnn-eng-sela-smoke-seed1
fi
python3 -c "import json,sys; c=json.load(open('$CK.contract.json')); assert c['smoke'] is True, c; print('[SMOKE CKPT]', '$CK', c['variant'], c['selected_epoch'], c['base_checkpoint'], c['base_checkpoint_md5'])"
md5sum "$CK.pt"
# (2) install under the gate name, in this smoke's own models/ only
[[ ! -L "$IN/models" ]] || { echo "FAIL LOUD: $IN/models is a link; the smoke installs into its own models/"; exit 1; }
SELNAME=${SELNAME:-plan-rerank-v1-gnn-eng-sela-seed1}
cp "$CK.pt" "$IN/models/$SELNAME.pt"
cp "$CK.contract.json" "$IN/models/$SELNAME.contract.json"
# (3) the cell
export HEROSIM_TRANSFER_MODEL=pipelined HEROSIM_REPLICA_RELEASE=1 HEROSIM_SCALEOUT=kpa GATE_FIXED_POLICY_TIME_SCALE=1.0 HEROSIM_SHARED_AUTOSCALER=0
export WF1_RUNGS=$RUNG R1A_STEM=accel-replica-v1 R1A_SPLIT=accel_replica_v1_split.json R1A_TOPOS=$TOPO R1A_WINDOWS=$WINDOW \
  R1A_ARMS=${ARM:-ra_gnn_eng_prsela} R1A_SEEDS=1 R1A_PRSEL_SMOKE=1 R1A_PRSEL_DIAG=1
python3 scripts_cosim/fresh_topo_burst_v1_gate.py r1a --inputs "$IN" --out "$OUT" --selection "$IN/selection.json" --parallel 1 --mem 32G \
  --timeout "${LIMIT:-36000}" --no-scope | grep -v "^\[skip"
python3 - "$OUT" <<'EOF'
import glob, json, sys
for f in sorted(glob.glob(sys.argv[1] + "/*.failed.json")):
    print("[FAILED]", f, json.load(open(f)).get("why"))
for f in sorted(glob.glob(sys.argv[1] + "/*.summary.json")):
    c = json.load(open(f)).get("schedulerCounters") or {}
    print("[COUNTERS]", f, json.dumps({k: v for k, v in sorted(c.items()) if k.startswith(("prsel_", "slate_declared", "prefix_batches", "prefix_tasks"))}))
EOF
