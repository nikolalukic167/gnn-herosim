#!/bin/bash
# Promotion gate: should `tempfix` replace the deployed siv1 checkpoint?
#
# On workload-150-100 at a fixed dim22 layout, tempfix wins 5/5 vs Knative in both
# conditions and beats the deployed checkpoint on mean margin in both (-14.0% vs -9.4%
# without a backbone, -34.1% vs -24.0% with). That reverses the "corrected-cache retrain
# FAILS 0/15" verdict, which is now understood as a serving-layout artifact
# (LINEAGES 2026-08-23). But it is ONE trace -- and the whole point of that entry is that
# one trace is not a verdict. This re-runs the same three arms on workload-175-100.
#
# All arms pinned to INFERENCE_FEATURE_LAYOUT=dim22. tempfix declares no layout in its
# sidecar (so without this it would serve atomic21 and reproduce the original confound);
# the deployed checkpoint declares dim22, which this AGREES with rather than overrides.
# Knative never touches the feature builder, so the pin is inert for it and kept only so
# every arm records the same serving configuration.
#
# 30 tasks: {knative, deployed, tempfix} x {backbone, no-backbone} x 5 cells.
#SBATCH --job-name=tempfix-promo
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/tempfix-promo-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/tempfix-promo-%A_%a.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=16:00:00
#SBATCH --array=0-29

set -euo pipefail
cd /home/nikola.lukic/gnn-herosim

if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TORCH_NUM_THREADS=4

# run_simulation.py re-spawns the simulator; since 2026-08-23 it honors HEROSIM_PY, so
# this now really is the activated env end to end. Print both to make it checkable.
python3 -c "import sys,torch;print('[INTERP]',sys.executable,torch.__version__)"

CELLS=(cell01_p25_s9001 cell02_p35_s9002 cell03_p15_s9003 cell04_p50_s9004 cell05_p20_s9005)
I="$SLURM_ARRAY_TASK_ID"
CELL="${CELLS[$(( I % 5 ))]}"
COND_IDX=$(( (I / 5) % 2 ))     # 0 = backbone, 1 = no-backbone
ARM_IDX=$(( I / 10 ))           # 0 = knative, 1 = deployed, 2 = tempfix

if [[ "$COND_IDX" == "0" ]]; then
  COND="backbone"
  SRC_SWEEP="simulation_data/normal_sim_sweeps/a1_backbone_bw1p5"
  export PARITY_EXTRA_ARGS="--allow-backbone-latency-divergence"
else
  COND="nobackbone"
  SRC_SWEEP="simulation_data/normal_sim_sweeps/full_corpus_siv1_live_gate_20260820"
  export PARITY_EXTRA_ARGS=""
fi

case "$ARM_IDX" in
  0) ARM="knative";  POLICY="knative"; MODEL="models/near-rtt-v2-full-corpus-siv1-dim14-ce-only.pt" ;;
  1) ARM="deployed"; POLICY="gnn";     MODEL="models/near-rtt-v2-full-corpus-siv1-dim14-ce-only.pt" ;;
  2) ARM="tempfix";  POLICY="gnn";     MODEL="models/near-rtt-v2-full-corpus-siv1-dim14-ce-only-tempfix.pt" ;;
  *) echo "FAIL LOUD: bad ARM_IDX $ARM_IDX" >&2; exit 1 ;;
esac

export SWEEP_DIR="simulation_data/normal_sim_sweeps/promo175_${COND}_${ARM}"
export GNN_MODEL="$MODEL"
export WORKLOAD="data/nofs-ids/traces/workload-175-100.json"
export INFERENCE_FEATURE_LAYOUT=dim22

[[ -f "$GNN_MODEL" ]] || { echo "FAIL LOUD: missing $GNN_MODEL" >&2; exit 1; }
[[ -f "$WORKLOAD" ]] || { echo "FAIL LOUD: missing $WORKLOAD" >&2; exit 1; }
[[ -d "${SRC_SWEEP}/cell_infrastructure/${CELL}" ]] || {
  echo "FAIL LOUD: missing cell ${SRC_SWEEP}/cell_infrastructure/${CELL}" >&2; exit 1; }

# mkdir -p + rsync, never `[[ -d ]] || cp -r` (not atomic; killed 3 tasks of job 710315).
mkdir -p "${SWEEP_DIR}/results" "${SWEEP_DIR}/cell_infrastructure" "${SWEEP_DIR}/configs" logs
for d in cell_infrastructure configs; do
  rsync -a "${SRC_SWEEP}/${d}/" "${SWEEP_DIR}/${d}/"
done

echo "=== task=${I} arm=${ARM} cond=${COND} cell=${CELL} trace=175-100 layout=dim22 ==="
bash scripts_cosim/important/run_full_corpus_siv1_live_gate.sh "$POLICY" "$CELL"
echo "=== task=${I} arm=${ARM} cond=${COND} cell=${CELL} COMPLETE ==="
