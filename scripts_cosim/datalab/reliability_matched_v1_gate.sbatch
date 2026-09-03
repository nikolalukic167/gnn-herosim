#!/bin/bash
# reliability_matched_v1 — the registered gate.
#
# Registration: docs/lineages/reliability_matched_v1.md, committed BEFORE this ran.
# Does the GNN's reliability edge survive training both model classes on the SAME corpus?
#
# One array task per BACKBONE block (4 blocks x 5 cells = the 20 cells score_link_mp_v1
# names). Per block: 16 corpus-matched MLP arms + 4 GNN venue-control arms + Knative,
# all argmax, on the block's own configs and workload. 21 arms x 5 cells x 4 blocks.
#
# The GNN comparison group is FROZEN (link_mp_v1's 16 lgon arms at pin 8aef27a); the four
# arms re-gated here are the venue control the registration requires, not new evidence.
#
#SBATCH --job-name=relmatch-gate
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/relmatch-gate-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/relmatch-gate-%A_%a.err
#SBATCH --partition=CPU-amd
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=120G
#SBATCH --time=16:00:00

set -euo pipefail
cd /home/nikola.lukic/gnn-herosim

if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3
export PYTHONPATH="$PWD"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONHASHSEED=0

python3 -c "import sys,torch;print('[INTERP]',sys.executable,torch.__version__)"
echo "[CODE] $(git describe --dirty --always) on $(git rev-parse --abbrev-ref HEAD)"

BLOCK="${SLURM_ARRAY_TASK_ID}"
BACKBONE_CELLS="simulation_data/normal_sim_sweeps/a1_backbone_bw1p5"
case "$BLOCK" in
  0) NAME=drawgate_backbone;      SRC="$BACKBONE_CELLS"; WL=150-100 ;;
  1) NAME=promo175_backbone;      SRC="$BACKBONE_CELLS"; WL=175-100 ;;
  2) NAME=bbrob_bb_core8_bw1p5;   SRC="simulation_data/normal_sim_sweeps/bb_core8_bw1p5"; WL=150-100 ;;
  3) NAME=bbrob_bb_core4_bw0p5;   SRC="simulation_data/normal_sim_sweeps/bb_core4_bw0p5"; WL=150-100 ;;
  *) echo "FAIL LOUD: bad block $BLOCK" >&2; exit 1 ;;
esac
WORKLOAD="data/nofs-ids/traces/workload-${WL}.json"
[[ -f "$WORKLOAD" ]] || { echo "FAIL LOUD: missing $WORKLOAD" >&2; exit 1; }
[[ -d "$SRC/configs" ]] || { echo "FAIL LOUD: missing $SRC/configs" >&2; exit 1; }

OUT=simulation_data/reliability_matched_v1
mkdir -p "$OUT"

ARGS=(
  --sweep-dir "$SRC"
  --cells cell01_p25_s9001 cell02_p35_s9002 cell03_p15_s9003 cell04_p50_s9004 cell05_p20_s9005
  --workload "$WORKLOAD"
  --workers 20
  --timeout 14400
  --arm "knative:knative"
)

# The 16 corpus-matched MLP arms — the new evidence.
for s in $(seq 1 16); do
  CK="models/tabular/batch_edge_mlp_link_mp_v1_dim22_batchcache_seed${s}.pt"
  [[ -f "$CK" ]] || { echo "FAIL LOUD: missing $CK — a partial group cannot be ranked" >&2; exit 1; }
  ARGS+=(--arm "mlp:mlp_matched_s${s}=$CK")
done

# The venue control: 4 frozen GNN arms re-gated at HEAD (registration §Venue control).
# NETWORK_GRAPH_CONTRACT is NOT exported — load_gnn_model adopts core_v1 from each
# checkpoint's .contract.json and raises on a conflicting export.
for s in 1 5 9 13; do
  CK="models/gnn-linkmp-lgon-s${s}.pt"
  [[ -f "$CK" ]] || { echo "FAIL LOUD: missing $CK" >&2; exit 1; }
  [[ -f "models/gnn-linkmp-lgon-s${s}.contract.json" ]] || {
    echo "FAIL LOUD: $CK has no contract sidecar" >&2; exit 1; }
  ARGS+=(--arm "gnn:gnn_venuectl_s${s}=$CK")
done

python3 -u scripts_cosim/closed_loop/evaluate_policy.py "${ARGS[@]}" \
  --out "$OUT/eval_${NAME}.json"
echo "=== reliability_matched_v1 block ${BLOCK} (${NAME}) complete ==="
