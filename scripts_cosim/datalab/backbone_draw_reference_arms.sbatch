#!/bin/bash
# Reference arms for the draw-robustness gate (companion to
# backbone_draw_robustness_gate.sbatch): knative and the DEPLOYED checkpoint, on the same
# cells / trace / venue / commit as the two alternate draws.
#
# Why re-run numbers that already exist. The recorded backbone arm (LINEAGES
# link_contention_v1, 2026-08-21) was produced LOCALLY, on a dirty working tree, before
# run_provenance carried a code stamp. The alternate draws run on datalab at a committed
# clean tree. Venue itself is proven not to be a variable (PARITY.md, job 709163: 0.03-0.40%
# of total_rtt, and logits bit-identical), and the only serving-path change since is
# network_graph.py's access-latency fix, which these checkpoints never reach
# (network_graph_contract=off) and which was verified to move no logits. So the expectation
# is that the old numbers reproduce -- but a draw comparison whose baseline came from a
# different venue AND a different, unstamped working tree would carry an avoidable
# confound, and this session exists to remove those, not to argue they are probably small.
#
# Layout: the deployed checkpoint's sidecar declares dim22, so the export below AGREES with
# it (apply_checkpoint_inference_feature_layout raises on a conflict, not on a match). It is
# set explicitly so all four GNN arms of this gate record the same serving configuration.
#
# 20 tasks: {knative, deployed} x {backbone, no-backbone} x 5 cells.
#SBATCH --job-name=bb-draw-ref
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/bb-draw-ref-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/bb-draw-ref-%A_%a.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --array=0-19

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
COND_IDX=$(( (I / 5) % 2 ))     # 0 = backbone, 1 = no-backbone
ARM_IDX=$(( I / 10 ))           # 0 = knative, 1 = deployed gnn

if [[ "$COND_IDX" == "0" ]]; then
  COND="backbone"
  SRC_SWEEP="simulation_data/normal_sim_sweeps/a1_backbone_bw1p5"
  export PARITY_EXTRA_ARGS="--allow-backbone-latency-divergence"
else
  COND="nobackbone"
  SRC_SWEEP="simulation_data/normal_sim_sweeps/full_corpus_siv1_live_gate_20260820"
  export PARITY_EXTRA_ARGS=""
fi

if [[ "$ARM_IDX" == "0" ]]; then
  ARM="knative"; POLICY="knative"
else
  ARM="deployed"; POLICY="gnn"
fi

export SWEEP_DIR="simulation_data/normal_sim_sweeps/drawgate_${COND}_${ARM}"
export GNN_MODEL="models/near-rtt-v2-full-corpus-siv1-dim14-ce-only.pt"
export WORKLOAD="data/nofs-ids/traces/workload-150-100.json"
export INFERENCE_FEATURE_LAYOUT=dim22

[[ -f "$GNN_MODEL" ]] || { echo "FAIL LOUD: missing $GNN_MODEL" >&2; exit 1; }
[[ -d "${SRC_SWEEP}/cell_infrastructure/${CELL}" ]] || {
  echo "FAIL LOUD: missing cell ${SRC_SWEEP}/cell_infrastructure/${CELL}" >&2; exit 1; }

# mkdir -p + rsync, not `[[ -d ]] || cp -r`: the latter is not atomic and killed three
# tasks of job 710315 when an arm's five tasks started together (datalab-pitfalls #7).
mkdir -p "${SWEEP_DIR}/results" "${SWEEP_DIR}/cell_infrastructure" "${SWEEP_DIR}/configs" logs
for d in cell_infrastructure configs; do
  rsync -a "${SRC_SWEEP}/${d}/" "${SWEEP_DIR}/${d}/"
done

echo "=== task=${I} arm=${ARM} cond=${COND} cell=${CELL} layout=${INFERENCE_FEATURE_LAYOUT} ==="
bash scripts_cosim/important/run_full_corpus_siv1_live_gate.sh "$POLICY" "$CELL"
echo "=== task=${I} arm=${ARM} cond=${COND} cell=${CELL} COMPLETE ==="
