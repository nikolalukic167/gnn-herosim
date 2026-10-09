#!/bin/bash
# r1_attribution_v1 live gate (docs/lineages/r1_attribution_v1.md): arms x the 19 test topologies x light/moderate/heavy x g0-g3, on R1.1 + the T-fix.
# Phase r1a of fresh_topo_burst_v1_gate.py, progress-rate watchdog instead of fixed reruns (a cell projected past LIMIT after 10 min is killed and
# counted), production pauses for a rung when more than 5 % of its decided non-Knative cells hang. Array shards: R1A_SHARD=<task>/<count>.
#   WT, EXPECT_HEAD   pinned detached worktree and its commit (never move it while jobs run)
#   IN                inputs: wf1_light/ wf1_moderate/ wf1_heavy/ models/ r1_attribution_v1_split.json
#   OUT               result directory (shared by every shard; may exist when resuming)
#   R1A_TOPOS R1A_WINDOWS R1A_ARMS R1A_SEEDS   restrict the cell list (unset = the registered grid); PAR = parallel cells per task
#   LIMIT             watchdog / hard limit in s (default 8100 = the former 3x rerun)
#   sbatch --array=0-N%K ... for the full gate; no --array for the dry run
#SBATCH --job-name=r1a-gate
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/r1a-gate-%A_%a.out
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=240G
#SBATCH --time=24:00:00
set -euo pipefail
WT=${WT:?FAIL LOUD: set WT}; EXPECT_HEAD=${EXPECT_HEAD:?FAIL LOUD: set EXPECT_HEAD}; IN=${IN:?}; OUT=${OUT:?}
D=/home/nikola.lukic/gnn-herosim/simulation_data
SELECT=$D/small_batch_confirm_v1/inputs/selected.json
cd "$WT"
for v in $(compgen -e | grep -E '^(HEROSIM_|GNN_|COSIM_|LIVE_AUDIT_|GATE_|SIM_|QUEUE_FEATURE|MAX_PLACEMENT|NEAR_RTT_|PARTIAL_STATE|WF1_)'); do unset "$v"; done
if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3 PYTHONPATH="$WT" PYTHONUNBUFFERED=1
export HEROSIM_RAW_DIR=/tmp/herosim_raw_${SLURM_JOB_ID:-$$}; mkdir -p "$HEROSIM_RAW_DIR"; trap 'rm -rf "$HEROSIM_RAW_DIR"' EXIT
python3 -c "import sys,torch;print('[INTERP]',sys.executable,torch.__version__)"
HEAD=$(git rev-parse HEAD)
echo "HEAD $HEAD node $(hostname) job=${SLURM_JOB_ID:-} task=${SLURM_ARRAY_TASK_ID:-none}/${SLURM_ARRAY_TASK_COUNT:-1} OUT=$OUT"
[[ "$HEAD" == "$EXPECT_HEAD"* ]] || { echo "FAIL LOUD: worktree HEAD $HEAD is not $EXPECT_HEAD"; exit 1; }
# a wedged Ceph client can hang `git status` (datalab-pitfalls 14): bounded `git diff --quiet`, a warning rather than a stop
rc=0; timeout 300 git diff --quiet HEAD -- src scripts_cosim experiments run_experiment.py || rc=$?
[[ $rc == 0 ]] || echo "WARN: worktree check rc=$rc (1 = tracked files differ from HEAD, 124 = timed out after 300 s)"
for r in light moderate heavy; do [[ -d "$IN/wf1_$r" ]] || { echo "FAIL LOUD: $IN/wf1_$r missing"; exit 1; }; done
mkdir -p "$OUT"
export HEROSIM_TRANSFER_MODEL=pipelined HEROSIM_REPLICA_RELEASE=1 HEROSIM_SCALEOUT=kpa GATE_FIXED_POLICY_TIME_SCALE=1.0 HEROSIM_SHARED_AUTOSCALER=0
export WF1_RUNGS=light,moderate,heavy
[[ -z "${SLURM_ARRAY_TASK_ID:-}" ]] || export R1A_SHARD="${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT}"
python3 scripts_cosim/fresh_topo_burst_v1_gate.py r1a --inputs "$IN" --out "$OUT" --selection "$SELECT" --parallel "${PAR:-60}" --mem 8G \
  --timeout "${LIMIT:-8100}" --no-scope | grep -v "^\[skip"
echo "failed cells: $(ls "$OUT"/*.failed.json 2>/dev/null | wc -l)  summaries: $(ls "$OUT"/*.summary.json 2>/dev/null | wc -l)"
