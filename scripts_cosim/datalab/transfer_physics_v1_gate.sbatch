#!/bin/bash
# transfer_physics_v1: one physics condition (TP1_COND = replay | pipe | release | pipe_release) of the 2x2
# transfer model x replica release, on the single-origin server-only cells of client_local_v1 (inputs_so_server):
# the five rules and zero-shot so1load (seeds 1-2). The store-and-forward / held-replica cell is
# client_local_v1/gate_so_server and small_batch_so_v1/gate; `replay` reruns 24 of those runs at defaults and
# must reproduce them exactly.
#SBATCH --job-name=tp1-gate
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/tp1-gate-%j.out
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=24:00:00
set -euo pipefail
WT=${WT:-/home/nikola.lukic/gnn-herosim-wt/sbso}
TIMEOUT=${TIMEOUT:-2700}
COND=${TP1_COND:?FAIL LOUD: set TP1_COND}
D=/home/nikola.lukic/gnn-herosim/simulation_data
IN=$D/client_local_v1/inputs_so_server
# kpa_scaleout_v1: HEROSIM_SCALEOUT=kpa runs the same condition under KPA scale-out into its own directory
export HEROSIM_SCALEOUT=${HEROSIM_SCALEOUT:-legacy}
export HEROSIM_SHARED_AUTOSCALER=${HEROSIM_SHARED_AUTOSCALER:-0} TP1_ARMS=${TP1_ARMS:-} GATE_FIXED_POLICY_TIME_SCALE=${GATE_FIXED_POLICY_TIME_SCALE:-}
case "$HEROSIM_SCALEOUT" in
  legacy) if [ "$HEROSIM_SHARED_AUTOSCALER" = 1 ]; then OUT=${OUT:-$D/kpa_scaleout_v1/gate_${COND}_legacy_shared}; else OUT=${OUT:-$D/transfer_physics_v1/gate_$COND}; fi ;;
  kpa) if [ -n "$GATE_FIXED_POLICY_TIME_SCALE" ]; then OUT=${OUT:?FAIL LOUD: a fixed-time-scale (R1) run names its own OUT}; else OUT=${OUT:-$D/kpa_scaleout_v1/gate_${COND}_kpa}; fi ;;
  *) echo "FAIL LOUD: HEROSIM_SCALEOUT=$HEROSIM_SCALEOUT"; exit 1 ;;
esac
SPLIT=experiments/small_batch_so_v1_split.json
P=${PAR:-$(( ${SLURM_CPUS_PER_TASK:-64} - 2 ))}
case "$COND" in
  replay) export HEROSIM_TRANSFER_MODEL=store_forward HEROSIM_REPLICA_RELEASE=0 ;;
  pipe) export HEROSIM_TRANSFER_MODEL=pipelined HEROSIM_REPLICA_RELEASE=0 ;;
  release) export HEROSIM_TRANSFER_MODEL=store_forward HEROSIM_REPLICA_RELEASE=1 ;;
  pipe_release) export HEROSIM_TRANSFER_MODEL=pipelined HEROSIM_REPLICA_RELEASE=1 ;;
  *) echo "FAIL LOUD: TP1_COND=$COND"; exit 1 ;;
esac
cd "$WT"
if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3 PYTHONPATH="$WT"
export HEROSIM_RAW_DIR=/tmp/herosim_raw_${SLURM_JOB_ID:-$$}
mkdir -p "$HEROSIM_RAW_DIR"; trap 'rm -rf "$HEROSIM_RAW_DIR"' EXIT
python3 -c "import sys,torch;print('[INTERP]',sys.executable,torch.__version__)"
echo "HEAD $(git rev-parse HEAD)  node $(hostname)  P=$P  cond=$COND transfer=$HEROSIM_TRANSFER_MODEL release=$HEROSIM_REPLICA_RELEASE scaleout=$HEROSIM_SCALEOUT"
[[ -z "$(git status --porcelain -- src scripts_cosim experiments run_experiment.py)" ]] || { echo "FAIL LOUD: worktree dirty"; exit 1; }
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
mkdir -p "$OUT"
set +e
TP1_COND=$COND TP1_SMOKE=${TP1_SMOKE:-} python3 scripts_cosim/fresh_topo_burst_v1_gate.py tp1 --inputs "$IN" --out "$OUT" \
  --selection "$IN/selected.json" --parallel "$P" --mem 4G --timeout "$TIMEOUT" --no-scope | grep -v "^\[skip"
rc=${PIPESTATUS[0]}
set -e
echo "=== tp1 $COND: $(ls "$OUT"/*.summary.json 2>/dev/null | wc -l) summaries, $(ls "$OUT"/*.failed.json 2>/dev/null | wc -l) failed ==="
find "$OUT" -name "*.log" -size +50M -exec gzip -1 {} \;
echo "driver rc=$rc"
