#!/bin/bash
# link_mp_v1 — gate all three arm families on the 20 BACKBONE cells, AT THE PINNED COMMIT.
# Registration: docs/lineages/link_mp_v1.md.
#
# 960 tasks = 3 families x 16 seeds x 20 cells, throttled to 60 concurrent. The cell ->
# (workload, PARITY_EXTRA_ARGS) mapping is byte-identical to the four BACKBONE blocks of
# mp_ablation_gate.sbatch (blocks 0/2/4/5 there): a cell that differs cannot be compared
# against the frozen Knative baselines already in gate_stats_summary.json. The two FLAT
# blocks are deliberately absent — a core_v1 checkpoint fails loud on a fabric-less graph
# by design, and the registered question lives on backbone cells.
#
# Submit with:  --dependency=afterok:<linkmp-train array job id>
#SBATCH --job-name=linkmp-gate
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/linkmp-gate-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/linkmp-gate-%A_%a.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=16:00:00
#SBATCH --array=0-959%60

set -euo pipefail

PIN_DIR=/home/nikola.lukic/gnn-herosim-pin-__LINK_MP_V1_PIN_SHORT__
PIN_COMMIT=__LINK_MP_V1_PIN__
cd "$PIN_DIR"

if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TORCH_NUM_THREADS=4

# Serving device is explicit; cuda is the only axis PARITY.md finds that moves GNN logits.
export HEROSIM_GNN_DEVICE=cpu

HEAD_NOW=$(git rev-parse HEAD)
[[ "$HEAD_NOW" == "$PIN_COMMIT" ]] || {
  echo "FAIL LOUD: pinned worktree at $HEAD_NOW, registration requires $PIN_COMMIT" >&2; exit 1; }
DIRTY=$(git status --porcelain -- src scripts_cosim experiments run_experiment.py)
[[ -z "$DIRTY" ]] || {
  echo "FAIL LOUD: pinned worktree dirty under CODE_PATHS:" >&2; echo "$DIRTY" >&2; exit 1; }
echo "[PIN] $PIN_DIR at $PIN_COMMIT, CODE_PATHS clean"

python3 -c "import sys,torch;print('[INTERP]',sys.executable,torch.__version__)"

CELLS=(cell01_p25_s9001 cell02_p35_s9002 cell03_p15_s9003 cell04_p50_s9004 cell05_p20_s9005)
FAMS=(lgctrl lgmpoff lgon)
I="$SLURM_ARRAY_TASK_ID"

FAM="${FAMS[$(( I / 320 ))]}"
R=$(( I % 320 ))
SEED=$(( R / 20 + 1 ))
J=$(( R % 20 ))
CELL="${CELLS[$(( J % 5 ))]}"
BLOCK=$(( J / 5 ))
[[ "$SEED" -ge 1 && "$SEED" -le 16 ]] || { echo "FAIL LOUD: bad seed $SEED (array must be 0-959)" >&2; exit 1; }

ARM="${FAM}${SEED}"
export GNN_MODEL="models/gnn-linkmp-${FAM}-s${SEED}.pt"

# Serve-time family levers — MUST mirror training exactly (a mismatch is the 2026-08-16
# 12.4x failure). Both land in run_provenance.env, which the scorer VOID-gates per family.
case "$FAM" in
  lgctrl)  : ;;
  lgmpoff) export GNN_DISABLE_MESSAGE_PASSING=1 ;;
  lgon)    export NETWORK_GRAPH_CONTRACT=core_v1 ;;
  *) echo "FAIL LOUD: bad family $FAM" >&2; exit 1 ;;
esac

BACKBONE_CELLS="simulation_data/normal_sim_sweeps/a1_backbone_bw1p5"
case "$BLOCK" in
  0) PREFIX=drawgate;  COND=backbone;        SRC_SWEEP="$BACKBONE_CELLS"; WL=150-100
     export PARITY_EXTRA_ARGS="--allow-backbone-latency-divergence" ;;
  1) PREFIX=promo175;  COND=backbone;        SRC_SWEEP="$BACKBONE_CELLS"; WL=175-100
     export PARITY_EXTRA_ARGS="--allow-backbone-latency-divergence" ;;
  2) PREFIX=bbrob;     COND=bb_core8_bw1p5;  SRC_SWEEP="simulation_data/normal_sim_sweeps/bb_core8_bw1p5"; WL=150-100
     export PARITY_EXTRA_ARGS="" ;;
  3) PREFIX=bbrob;     COND=bb_core4_bw0p5;  SRC_SWEEP="simulation_data/normal_sim_sweeps/bb_core4_bw0p5"; WL=150-100
     export PARITY_EXTRA_ARGS="" ;;
  *) echo "FAIL LOUD: bad block $BLOCK" >&2; exit 1 ;;
esac

export SWEEP_DIR="simulation_data/normal_sim_sweeps/${PREFIX}_${COND}_${ARM}"
export WORKLOAD="data/nofs-ids/traces/workload-${WL}.json"

[[ -f "$GNN_MODEL" ]] || { echo "FAIL LOUD: missing $GNN_MODEL" >&2; exit 1; }
[[ -f "$WORKLOAD" ]] || { echo "FAIL LOUD: missing $WORKLOAD" >&2; exit 1; }
[[ -d "${SRC_SWEEP}/cell_infrastructure/${CELL}" ]] || {
  echo "FAIL LOUD: missing cell ${SRC_SWEEP}/cell_infrastructure/${CELL}" >&2; exit 1; }
[[ "$SWEEP_DIR" == *"_lg"* ]] || {
  echo "FAIL LOUD: SWEEP_DIR must be lineage-private: $SWEEP_DIR" >&2; exit 1; }

# The filename is not evidence; the contract is — and the family lever must match it.
python3 - "$GNN_MODEL" "$SEED" "$FAM" <<'PY'
import json, sys
from pathlib import Path
path, want_seed, fam = sys.argv[1], int(sys.argv[2]), sys.argv[3]
sidecar = Path(path).with_suffix(".contract.json")
if not sidecar.is_file():
    raise SystemExit(f"FAIL LOUD: {sidecar} missing — serving would silently adopt defaults")
c = json.load(open(sidecar))
if int(c.get("train_seed", -1)) != want_seed:
    raise SystemExit(f"FAIL LOUD: {path} contract train_seed={c.get('train_seed')} != {want_seed}")
if not c.get("deterministic_algorithms"):
    raise SystemExit(f"FAIL LOUD: {path} is not a reproducible draw")
ngc = c.get("network_graph_contract")
if fam == "lgon" and ngc != "core_v1":
    raise SystemExit(f"FAIL LOUD: lgon checkpoint records network_graph_contract={ngc!r}")
if fam != "lgon" and ngc not in (None, "off"):
    raise SystemExit(f"FAIL LOUD: {fam} checkpoint records network_graph_contract={ngc!r}")
print(f"[CKPT] {path} fam={fam} seed={want_seed} layout={c.get('inference_feature_layout')} "
      f"queue={c.get('queue_feature_contract')} network_graph={ngc}")
PY

mkdir -p "${SWEEP_DIR}/results" "${SWEEP_DIR}/cell_infrastructure" "${SWEEP_DIR}/configs" logs
# rsync, not `[[ -d ]] || cp -r`: the non-atomic form killed three tasks of job 710315
# when an arm's tasks started together (datalab-pitfalls #7).
for d in cell_infrastructure configs; do
  rsync -a "${SRC_SWEEP}/${d}/" "${SWEEP_DIR}/${d}/"
done

echo "=== task=${I} gate=${PREFIX} cond=${COND} cell=${CELL} arm=${ARM} fam=${FAM} seed=${SEED} (pinned ${PIN_COMMIT:0:12}) ==="
# No --seed is passed anywhere in this chain: it would OVERRIDE the config's topology seed
# and turn a seed study into a topology study.
bash scripts_cosim/important/run_full_corpus_siv1_live_gate.sh gnn "$CELL"
echo "=== task=${I} arm=${ARM} cell=${CELL} COMPLETE ==="
