#!/bin/bash
# link_contention_v1 draw-robustness gate: is the GNN's backbone win a property of the
# REGIME or of the lucky training draw? And is the "lottery" itself partly a serving
# artifact?
#
# The deployed checkpoint beats Knative 5/5 (mean -24.0%) under a binding-bandwidth
# backbone (LINEAGES.md, link_contention_v1 2026-08-21), vs -9.4% without it. But the
# variance control (2026-08-22) showed a val-acc-identical retrain can differ from the
# deployed draw by up to 36% of live total_rtt, so a single-checkpoint verdict is a claim
# about one draw, not about the recipe.
#
# 2x2x5 = 20 tasks: {prefixctl, tempfix} x {backbone, no-backbone} x 5 cells. Both draws
# LOST without the backbone in their original gates (prefixctl 1/5, tempfix 0/5). The
# no-backbone arm is therefore also a re-measurement of the lottery table itself under a
# corrected serving configuration -- see the layout note below -- and the backbone arm
# asks whether the regime rescues draws that otherwise lose.
#
# INFERENCE_FEATURE_LAYOUT is pinned to dim22 on purpose. Both checkpoints declare
# inference_feature_layout=null in their sidecars, and
# feature_builder._inference_feature_layout defaults an UNSET env var to atomic21 -- which
# is what their original gates served under (run_provenance.env layout=None, and the
# loader banner "Using atomic21 inference layout"), while every deployed-checkpoint gate
# served dim22. dim22 vs atomic21 toggles use_norm_queue: same tensor shapes, different
# platform-feature meanings. Leaving it unset would repeat that mismatch and confound the
# draw comparison with a serving difference. Neither sidecar declares a layout, so this
# export cannot conflict with a contract.
#
# Cells: the backbone arm uses the legacy_v0 a1 cells, so its parity preflight still needs
# --allow-backbone-latency-divergence (the rng_stream=independent_v1 fix applies to newly
# minted cells; re-minting here would break comparability with the existing arms). The
# no-backbone arm uses the siv1 gate cells, verified this session to be physically
# identical across venues (network_maps / replica_placements / queue_distributions /
# link_topology all byte-identical; only metadata path+timestamp differ).
#SBATCH --job-name=bb-draw-gate
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/bb-draw-gate-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/bb-draw-gate-%A_%a.err
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

# Prove which interpreter actually serves (datalab-pitfalls #8).
python3 -c "import sys,torch;print('[INTERP]',sys.executable,torch.__version__)"

CELLS=(cell01_p25_s9001 cell02_p35_s9002 cell03_p15_s9003 cell04_p50_s9004 cell05_p20_s9005)
I="$SLURM_ARRAY_TASK_ID"
CELL="${CELLS[$(( I % 5 ))]}"
COND_IDX=$(( (I / 5) % 2 ))     # 0 = backbone, 1 = no-backbone
DRAW_IDX=$(( I / 10 ))          # 0 = prefixctl, 1 = tempfix
DRAWS=(prefixctl tempfix)
DRAW="${DRAWS[$DRAW_IDX]}"

if [[ "$COND_IDX" == "0" ]]; then
  COND="backbone"
  SRC_SWEEP="simulation_data/normal_sim_sweeps/a1_backbone_bw1p5"
  export PARITY_EXTRA_ARGS="--allow-backbone-latency-divergence"
else
  COND="nobackbone"
  SRC_SWEEP="simulation_data/normal_sim_sweeps/full_corpus_siv1_live_gate_20260820"
  export PARITY_EXTRA_ARGS=""
fi

export SWEEP_DIR="simulation_data/normal_sim_sweeps/drawgate_${COND}_${DRAW}"
export GNN_MODEL="models/near-rtt-v2-full-corpus-siv1-dim14-ce-only-${DRAW}.pt"
export WORKLOAD="data/nofs-ids/traces/workload-150-100.json"
export INFERENCE_FEATURE_LAYOUT=dim22

[[ -f "$GNN_MODEL" ]] || { echo "FAIL LOUD: missing $GNN_MODEL" >&2; exit 1; }
[[ -d "${SRC_SWEEP}/cell_infrastructure/${CELL}" ]] || {
  echo "FAIL LOUD: missing cell ${SRC_SWEEP}/cell_infrastructure/${CELL}" >&2; exit 1; }

# Cells are shared and pre-minted, but `[[ -d ]] || cp -r` is NOT atomic: the five tasks
# of an arm start together, all see the directory missing, and all four losers die on
# "cp: cannot create directory ... File exists" (pitfalls #7 — cost tasks 16/17/19 of job
# 710315). Populate with rsync into a pre-created directory instead: idempotent, safe to
# run concurrently, and a no-op once the files are there.
mkdir -p "${SWEEP_DIR}/results" "${SWEEP_DIR}/cell_infrastructure" "${SWEEP_DIR}/configs" logs
for d in cell_infrastructure configs; do
  rsync -a "${SRC_SWEEP}/${d}/" "${SWEEP_DIR}/${d}/"
done

echo "=== task=${I} draw=${DRAW} cond=${COND} cell=${CELL} layout=${INFERENCE_FEATURE_LAYOUT} ==="
bash scripts_cosim/important/run_full_corpus_siv1_live_gate.sh gnn "$CELL"
echo "=== task=${I} draw=${DRAW} cond=${COND} cell=${CELL} COMPLETE ==="
