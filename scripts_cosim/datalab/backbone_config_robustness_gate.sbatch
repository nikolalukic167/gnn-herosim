#!/bin/bash
# Is the GNN's contention win specific to n_core=4 / bw=1.5 MB/s, or does it hold across
# backbone configurations?
#
# The recorded win (LINEAGES link_contention_v1) uses one backbone config, and n_core=4 was
# chosen because it was that lineage's own measured interior peak -- i.e. selected for being
# where the effect was largest. A win that only appears at the tuned point is not a claim
# about network contention. This runs two different configurations:
#
#   bb_core8_bw1p5 : n_core=8, bw=1.5 MB/s  -- different core topology, same bandwidth
#   bb_core4_bw0p5 : n_core=4, bw=0.5 MB/s  -- same topology, more binding bandwidth
#
# Cells for both were minted with `rng_stream: independent_v1`, so their preflight passes
# with **no --allow-backbone-latency-divergence waiver** -- verified 10/10 locally,
# including the three cells whose non-empty repair sets forced the waiver on the legacy a1
# cells. These are the first backbone cells in the repo that are parity-exact by
# construction rather than by relaxation.
#
# All arms pinned to INFERENCE_FEATURE_LAYOUT=dim22 (tempfix's sidecar declares none; the
# deployed checkpoint's declares dim22, which this agrees with).
#
# 30 tasks: {knative, deployed, tempfix} x {core8_bw1p5, core4_bw0p5} x 5 cells.
#SBATCH --job-name=bb-config-rob
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/bb-config-rob-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/bb-config-rob-%A_%a.err
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

python3 -c "import sys,torch;print('[INTERP]',sys.executable,torch.__version__)"

CELLS=(cell01_p25_s9001 cell02_p35_s9002 cell03_p15_s9003 cell04_p50_s9004 cell05_p20_s9005)
I="$SLURM_ARRAY_TASK_ID"
CELL="${CELLS[$(( I % 5 ))]}"
CFG_IDX=$(( (I / 5) % 2 ))
ARM_IDX=$(( I / 10 ))

CFGS=(bb_core8_bw1p5 bb_core4_bw0p5)
BBCFG="${CFGS[$CFG_IDX]}"
SRC_SWEEP="simulation_data/normal_sim_sweeps/${BBCFG}"

case "$ARM_IDX" in
  0) ARM="knative";  POLICY="knative"; MODEL="models/near-rtt-v2-full-corpus-siv1-dim14-ce-only.pt" ;;
  1) ARM="deployed"; POLICY="gnn";     MODEL="models/near-rtt-v2-full-corpus-siv1-dim14-ce-only.pt" ;;
  2) ARM="tempfix";  POLICY="gnn";     MODEL="models/near-rtt-v2-full-corpus-siv1-dim14-ce-only-tempfix.pt" ;;
  *) echo "FAIL LOUD: bad ARM_IDX $ARM_IDX" >&2; exit 1 ;;
esac

export SWEEP_DIR="simulation_data/normal_sim_sweeps/bbrob_${BBCFG}_${ARM}"
export GNN_MODEL="$MODEL"
export WORKLOAD="data/nofs-ids/traces/workload-150-100.json"
export INFERENCE_FEATURE_LAYOUT=dim22
# Deliberately EMPTY: these cells are parity-exact by construction. If the preflight ever
# needs the waiver here, the rng_stream fix has regressed and the gate should fail, not
# be relaxed.
export PARITY_EXTRA_ARGS=""

[[ -f "$GNN_MODEL" ]] || { echo "FAIL LOUD: missing $GNN_MODEL" >&2; exit 1; }
[[ -d "${SRC_SWEEP}/cell_infrastructure/${CELL}" ]] || {
  echo "FAIL LOUD: missing cell ${SRC_SWEEP}/cell_infrastructure/${CELL}" >&2; exit 1; }

mkdir -p "${SWEEP_DIR}/results" "${SWEEP_DIR}/cell_infrastructure" "${SWEEP_DIR}/configs" logs
for d in cell_infrastructure configs; do
  rsync -a "${SRC_SWEEP}/${d}/" "${SWEEP_DIR}/${d}/"
done

echo "=== task=${I} arm=${ARM} backbone=${BBCFG} cell=${CELL} layout=dim22 ==="
bash scripts_cosim/important/run_full_corpus_siv1_live_gate.sh "$POLICY" "$CELL"
echo "=== task=${I} arm=${ARM} backbone=${BBCFG} cell=${CELL} COMPLETE ==="
