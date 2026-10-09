#!/bin/bash
# r1_attribution_v1 diagnostic (3b), DESCRIPTIVE ONLY: an early checkpoint of one learned arm served on 2 TRAIN-split topologies x 3 rungs x window g0 next to CD and
# reactive Knative on the same cells, at the gate commit. Never a result: the checkpoint is a mid-training snapshot, the cells are training topologies.
# The script refuses any cell that is a held-out topology (corpus_prod/split.json) or a gate topology (selected.json).
#   sbatch --export=ALL,WT=<worktree at the gate commit>,IN=<scratch inputs: models/, split, wf1_<rung>/>,OUT=<new dir>,TOPOS=9101,9103 r1a_diag_smoke.sbatch
#SBATCH --job-name=r1a-diag
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/r1a-diag-%j.out
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
set -uo pipefail
WT=${WT:?}; IN=${IN:?}; OUT=${OUT:?}; TOPOS=${TOPOS:-9101,9103}; ARM=${ARM:-ra_gnn_eng}
CP=${CORPUS:-/share/nikola.lukic/workload_fix_v1/corpus_prod}
SEL=${SELECTED:-/home/nikola.lukic/gnn-herosim/simulation_data/small_batch_confirm_v1/inputs/selected.json}
cd "$WT"
python3 - "$TOPOS" "$CP/split.json" "$SEL" <<'PY' || exit 1
import json, sys
topos = {t for t in sys.argv[1].split(",") if t}
corpus = json.load(open(sys.argv[2])); gate = {str(t) for t in json.load(open(sys.argv[3]))["topologies"]}
held, train = {str(t) for t in corpus["heldout"]}, {str(t) for t in corpus["train"]}
bad = sorted((topos & held) | (topos & gate)); not_train = sorted(topos - train)
if bad or not_train:
    sys.exit(f"FAIL LOUD: cells {bad} are held-out or gate topologies, {not_train} are not train-split topologies")
print(f"topologies {sorted(topos)}: all train-split, none held-out ({len(held)}) or gate ({len(gate)})")
PY
for v in $(compgen -e | grep -E '^(HEROSIM_|GNN_|COSIM_|LIVE_AUDIT_|GATE_|SIM_|QUEUE_FEATURE|MAX_PLACEMENT|NEAR_RTT_|PARTIAL_STATE|WF1_)'); do unset "$v"; done
if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3 PYTHONPATH="$WT" PYTHONUNBUFFERED=1
export HEROSIM_RAW_DIR=/tmp/herosim_raw_${SLURM_JOB_ID:-$$}; mkdir -p "$HEROSIM_RAW_DIR"; trap 'rm -rf "$HEROSIM_RAW_DIR"' EXIT
echo "HEAD $(git rev-parse HEAD) node $(hostname)"
[[ ! -e "$OUT" ]] || { echo "FAIL LOUD: $OUT exists"; exit 1; }
mkdir -p "$OUT"
export HEROSIM_TRANSFER_MODEL=pipelined HEROSIM_REPLICA_RELEASE=1 HEROSIM_SCALEOUT=kpa GATE_FIXED_POLICY_TIME_SCALE=1.0 HEROSIM_SHARED_AUTOSCALER=0
export WF1_TOPOS=$TOPOS WF1_CAL_WINDOWS=g0
# classical arms on every cell
( WF1_RUNGS=light,moderate,heavy WF1_CAL_KINDS=reactive,cd python3 scripts_cosim/fresh_topo_burst_v1_gate.py wf1cal --inputs "$IN" --out "$OUT/classical" --parallel 6 --mem 8G --timeout 7200 --no-scope | grep -v "^\[skip" ) > "$OUT/classical.log" 2>&1 &
# the learned arm, one invocation per cell so each has its own decode trace
pids=()
for T in ${TOPOS//,/ }; do for R in light moderate heavy; do
  ( WF1_TOPOS=$T WF1_RUNGS=$R WF1_CAL_KINDS=$ARM GNN_PREFIX_TRACE_PATH="$OUT/trace_${T}_${R}.pkl" \
    python3 scripts_cosim/fresh_topo_burst_v1_gate.py wf1cal --inputs "$IN" --out "$OUT/learned_${T}_${R}" --parallel 1 --mem 8G --timeout 7200 --no-scope | grep -v "^\[skip" ) > "$OUT/learned_${T}_${R}.log" 2>&1 &
  pids+=($!)
done; done
wait
tail -n 3 "$OUT"/classical.log "$OUT"/learned_*.log | cut -c1-200
python3 "${READER:-scripts_cosim/r1a_diag_smoke_read.py}" --out "$OUT" --arm "${ARM#ra_}" --topos "$TOPOS" | tee "$OUT/read.txt"
