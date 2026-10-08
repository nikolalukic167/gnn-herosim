#!/bin/bash
# kpa_scaleout_v1 gates re-measured on R1.1 (docs/lineages/workload_fix_v1.md, R1.1 re-measure scope): the PUBLISHED
# settings -- per-rung policy time scale, GATE_FIXED_POLICY_TIME_SCALE unset -- with only the code changed.
#   TP1_COND   replay | pipe | release | pipe_release           (transfer model x release, as transfer_physics_v1_gate.sbatch)
#   KIND       kpa (default; HEROSIM_SCALEOUT=kpa) | legacy_shared (legacy scale-out, shared autoscaler, reactive + selfpredict)
#   WT, EXPECT_HEAD   the pinned detached worktree and its commit (no defaults; never move the worktree while jobs run)
# Output: simulation_data/kpa_scaleout_v1/gate_<cond>_kpa_r11 or gate_<cond>_legacy_shared_r11. An existing OUT is refused.
#SBATCH --job-name=kpar11
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/kpar11-%j.out
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=24:00:00
set -euo pipefail
WT=${WT:?FAIL LOUD: set WT (pinned detached worktree)}
EXPECT_HEAD=${EXPECT_HEAD:?FAIL LOUD: set EXPECT_HEAD}
COND=${TP1_COND:?FAIL LOUD: set TP1_COND}
KIND=${KIND:-kpa}
TIMEOUT=${TIMEOUT:-2700}
D=/home/nikola.lukic/gnn-herosim/simulation_data
IN=$D/client_local_v1/inputs_so_server
case "$KIND" in
  kpa) export HEROSIM_SCALEOUT=kpa HEROSIM_SHARED_AUTOSCALER=0 TP1_ARMS=; OUT=$D/kpa_scaleout_v1/gate_${COND}_kpa_r11 ;;
  legacy_shared) export HEROSIM_SCALEOUT=legacy HEROSIM_SHARED_AUTOSCALER=1 TP1_ARMS=reactive,selfpredict
                 OUT=$D/kpa_scaleout_v1/gate_${COND}_legacy_shared_r11 ;;
  *) echo "FAIL LOUD: KIND=$KIND"; exit 1 ;;
esac
# the published gates ran at the per-rung time scale; a fixed one would change the physics along with the code
unset GATE_FIXED_POLICY_TIME_SCALE
case "$COND" in
  replay) export HEROSIM_TRANSFER_MODEL=store_forward HEROSIM_REPLICA_RELEASE=0 ;;
  pipe) export HEROSIM_TRANSFER_MODEL=pipelined HEROSIM_REPLICA_RELEASE=0 ;;
  release) export HEROSIM_TRANSFER_MODEL=store_forward HEROSIM_REPLICA_RELEASE=1 ;;
  pipe_release) export HEROSIM_TRANSFER_MODEL=pipelined HEROSIM_REPLICA_RELEASE=1 ;;
  *) echo "FAIL LOUD: TP1_COND=$COND"; exit 1 ;;
esac
SPLIT=experiments/small_batch_so_v1_split.json
P=${PAR:-$(( ${SLURM_CPUS_PER_TASK:-64} - 2 ))}
cd "$WT"
if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3 PYTHONPATH="$WT"
export HEROSIM_RAW_DIR=/tmp/herosim_raw_${SLURM_JOB_ID:-$$}
mkdir -p "$HEROSIM_RAW_DIR"; trap 'rm -rf "$HEROSIM_RAW_DIR"' EXIT
python3 -c "import sys,torch;print('[INTERP]',sys.executable,torch.__version__)"
HEAD=$(git rev-parse HEAD)
echo "HEAD $HEAD  node $(hostname)  cond=$COND kind=$KIND transfer=$HEROSIM_TRANSFER_MODEL release=$HEROSIM_REPLICA_RELEASE scaleout=$HEROSIM_SCALEOUT OUT=$OUT"
[[ "$HEAD" == "$EXPECT_HEAD"* ]] || { echo "FAIL LOUD: worktree HEAD $HEAD is not $EXPECT_HEAD"; exit 1; }
[[ -z "$(git status --porcelain -- src scripts_cosim experiments run_experiment.py)" ]] || { echo "FAIL LOUD: worktree dirty"; exit 1; }
[[ -f src/placement/request_timeout.py ]] || { echo "FAIL LOUD: this worktree has no R1.1 request timeout"; exit 1; }
[[ ! -e "$OUT" ]] || { echo "FAIL LOUD: $OUT exists; a stage directory is not overwritten"; exit 1; }
[[ -z "${GATE_FIXED_POLICY_TIME_SCALE:-}" ]] || { echo "FAIL LOUD: GATE_FIXED_POLICY_TIME_SCALE is set"; exit 1; }
for R in x20 x30 x50; do
  [[ -f "$IN/grounded_$R/.built" ]] || { echo "FAIL LOUD: $IN/grounded_$R not built"; exit 1; }
  cmp "$SPLIT" "$IN/grounded_$R/$(basename "$SPLIT")"
  for s in 1 2; do
    for ext in .pt .contract.json; do
      f="$IN/grounded_$R/models/small-batch-so-v1-so1load-lr2e3-seed$s$ext"
      [ -f "$f" ] || { echo "FAIL LOUD: missing $f"; exit 1; }
    done
  done
done
# published sizes: 5 rules + zero-shot so1load x 2 seeds over 19 topologies x 3 rungs x 4 windows = 1,596; reactive + selfpredict = 456
[[ "$KIND" == kpa ]] && EXPECTED=1596 || EXPECTED=456
mkdir -p "$OUT"
set +e
TP1_COND=$COND TP1_SMOKE= python3 scripts_cosim/fresh_topo_burst_v1_gate.py tp1 --inputs "$IN" --out "$OUT" \
  --selection "$IN/selected.json" --parallel "$P" --mem 4G --timeout "$TIMEOUT" --no-scope | grep -v "^\[skip"
rc=${PIPESTATUS[0]}
set -e
echo "=== r11 $COND $KIND: $(ls "$OUT"/*.summary.json 2>/dev/null | wc -l) of $EXPECTED summaries, $(ls "$OUT"/*.failed.json 2>/dev/null | wc -l) failed ==="
echo "driver rc=$rc"
