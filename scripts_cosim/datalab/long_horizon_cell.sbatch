#!/bin/bash
# long_horizon_s0_v1 probe (docs/lineages/long_horizon_s0_v1.md): ONE cell per job -- topology TOPO, window L24 at RUNG, arm ARM -- through
# phase r1a of fresh_topo_burst_v1_gate.py (progress watchdog; a kill is a recorded failure, never rerun). The cell records hourly bins.
#   WT, EXPECT_HEAD   pinned detached worktree and its commit
#   IN                long_horizon_inputs.sbatch's <ROOT>/inputs
#   OUT               result directory (shared by the 7 cells)
#   RUNG ARM          moderate|heavy and one of cd, cd_exactS (moderate only), ra_gnn_eng_nosplit, ra_gnn_eng_cdxapply
#   TOPO LIMIT        default 16900 and 72000 s
#   /home is near quota: submit with --output=/share/nikola.lukic/long_horizon/logs/lh-cell-%j.out (the cell's own log is in OUT)
#SBATCH --job-name=lh-cell
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/lh-cell-%j.out
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
set -euo pipefail
WT=${WT:?FAIL LOUD: set WT}; EXPECT_HEAD=${EXPECT_HEAD:?FAIL LOUD: set EXPECT_HEAD}; IN=${IN:?}; OUT=${OUT:?}
RUNG=${RUNG:?}; ARM=${ARM:?}; TOPO=${TOPO:-16900}
case "$RUNG/$ARM" in
  moderate/cd|moderate/cd_exactS|moderate/ra_gnn_eng_nosplit|moderate/ra_gnn_eng_cdxapply|heavy/cd|heavy/ra_gnn_eng_nosplit|heavy/ra_gnn_eng_cdxapply) ;;
  *) echo "FAIL LOUD: $RUNG/$ARM is not one of the 7 registered cells"; exit 1;;
esac
cd "$WT"
for v in $(compgen -e | grep -E '^(HEROSIM_|GNN_|COSIM_|LIVE_AUDIT_|GATE_|SIM_|QUEUE_FEATURE|MAX_PLACEMENT|NEAR_RTT_|PARTIAL_STATE|WF1_|R1A_)'); do unset "$v"; done
if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3 PYTHONPATH="$WT" PYTHONUNBUFFERED=1
export HEROSIM_RAW_DIR=/tmp/herosim_raw_${SLURM_JOB_ID:-$$}; mkdir -p "$HEROSIM_RAW_DIR"; trap 'rm -rf "$HEROSIM_RAW_DIR"' EXIT
python3 -c "import sys,torch;print('[INTERP]',sys.executable,torch.__version__)"
HEAD=$(git rev-parse HEAD)
echo "HEAD $HEAD node $(hostname) job=${SLURM_JOB_ID:-} cell=$TOPO/L24$RUNG/$ARM IN=$IN OUT=$OUT"
[[ "$HEAD" == "$EXPECT_HEAD"* ]] || { echo "FAIL LOUD: worktree HEAD $HEAD is not $EXPECT_HEAD"; exit 1; }
rc=0; timeout 300 git diff --quiet HEAD -- src scripts_cosim experiments run_experiment.py || rc=$?
[[ $rc == 0 ]] || echo "WARN: worktree check rc=$rc (1 = tracked files differ from HEAD, 124 = timed out after 300 s)"
[[ -f "$IN/wf1_$RUNG/wl/long_horizon_L24.json" ]] || { echo "FAIL LOUD: $IN/wf1_$RUNG/wl/long_horizon_L24.json missing"; exit 1; }
mkdir -p "$OUT"
export HEROSIM_TRANSFER_MODEL=pipelined HEROSIM_REPLICA_RELEASE=1 HEROSIM_SCALEOUT=kpa GATE_FIXED_POLICY_TIME_SCALE=1.0 HEROSIM_SHARED_AUTOSCALER=0
export WF1_RUNGS=$RUNG R1A_STEM=accel-replica-v1 R1A_SPLIT=accel_replica_v1_split.json R1A_TOPOS=$TOPO R1A_WINDOWS=L24 R1A_ARMS=$ARM R1A_SEEDS=1
python3 scripts_cosim/fresh_topo_burst_v1_gate.py r1a --inputs "$IN" --out "$OUT" --selection "$IN/selection.json" --parallel 1 --mem 120G \
  --timeout "${LIMIT:-72000}" --no-scope | grep -v "^\[skip"
python3 - "$OUT" "$TOPO" "$RUNG" "$ARM" <<'EOF'
import glob, json, sys
out, topo, rung, arm = sys.argv[1:]
for f in sorted(glob.glob(f"{out}/cc40s{topo}__L24{rung}__{arm}_s*.failed.json")):
    print("[FAILED]", f, json.load(open(f)).get("why"))
for f in sorted(glob.glob(f"{out}/cc40s{topo}__L24{rung}__{arm}_s*.summary.json")):
    s = json.load(open(f)); hb = s.get("hourlyBins") or {}
    print("[CELL]", f, "num_tasks", s.get("num_tasks"), "wall", s.get("wallclock_s"), "endTime", s.get("endTime"),
          "bins", len(hb.get("bins") or []), "bin n sum", sum(b["n"] for b in hb.get("bins") or []))
EOF
