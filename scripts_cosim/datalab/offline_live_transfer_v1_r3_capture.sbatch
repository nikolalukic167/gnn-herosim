#!/bin/bash
# offline_live_transfer_v1 -- R3 step 1: re-run the BEST and WORST seed of the V=1 gnn arm
# with per-task records retained, so the within-arm spread can be located in TIME.
#
# R1 showed the offline score cannot rank checkpoints; R4 showed it can rank epochs within a
# run. What neither says is where the 1.7x within-arm live spread comes from. R3 asks the one
# question that separates the two candidate mechanisms:
#
#   PRESENT-FROM-THE-START  the gap is already there in the first decile  -> the live states
#                           are outside the trained regime from the first decision
#   COMPOUNDS               the gap is absent early and grows             -> the model builds
#                           its own bad states (closed loop)
#
# Seeds chosen by live latency in results/dobj_f4000_pg16, which is already on the record:
#   seed 14 = 43.99 s (best)      seed 8 = 74.76 s (worst)
# Naming them here rather than recomputing keeps the choice auditable.
#
#   sbatch --array=0-1 scripts_cosim/datalab/offline_live_transfer_v1_r3_capture.sbatch
#
#SBATCH --job-name=olt-r3cap
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/olt-r3cap-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/olt-r3cap-%A_%a.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --array=0-1

set -euo pipefail
cd /home/nikola.lukic/gnn-herosim

if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3
export PYTHONPATH="$PWD"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=""

python3 -c "import sys,torch;print('[INTERP]',sys.executable,torch.__version__)"
echo "[CODE] $(git describe --dirty --always) on $(git rev-parse --abbrev-ref HEAD)"

GATE=simulation_data/peer_affinity_live_gate
WORKLOAD="$GATE/workloads/drainable_f4000_n50000.json"
CFG="$GATE/configs/cell_s7901_f4000_pg16.json"
OUT_DIR="$GATE/results/olt_r3_capture"
mkdir -p "$OUT_DIR"
[[ -f "$WORKLOAD" ]] || { echo "FAIL LOUD: workload missing" >&2; exit 1; }
[[ -f "$CFG" ]] || { echo "FAIL LOUD: cell config missing" >&2; exit 1; }

# Byte-identical contract to the gate these seeds' latencies came from.
export HEROSIM_PEER_EXCHANGE=1
export HEROSIM_SERVER_ONLY_REPLICAS=1
export HEROSIM_WARMTH_PHYSICS=node_disk_v2
export PYTHONHASHSEED=0
export HEROSIM_GNN_DEVICE=cpu

SEEDS=(14 8)
LABELS=(best worst)
T=${SLURM_ARRAY_TASK_ID}
SEED=${SEEDS[$T]}
LABEL=${LABELS[$T]}

CK="models/drainable-objective-v1-v1-gnn-lr2e3-seed${SEED}.pt"
[[ -f "$CK" ]] || { echo "FAIL LOUD: missing $CK" >&2; exit 1; }
[[ -f "${CK%.pt}.contract.json" ]] || { echo "FAIL LOUD: no sidecar for $CK" >&2; exit 1; }

export GNN_MODEL_PATH="$CK"
export GNN_DECODE_MODE=masked_topo
export GNN_BATCH_BY_PEER_GROUP=1
export SIM_FORCE_FULL_STATS=1
export GNN_CAPTURE_DATASET_STATE=1

OUT="$OUT_DIR/${LABEL}_gnn_s${SEED}.json"
echo "[TASK $T] $LABEL seed=$SEED -> $OUT"
if [[ -f "$OUT" ]]; then echo "[TASK $T] exists, not re-running"; exit 0; fi

python3 src/executesimulation.py --config "$CFG" --workload "$WORKLOAD" --policy gnn --output "$OUT.partial"
mv "$OUT.partial" "$OUT"
echo "[RAW KEPT] $OUT ($(du -h "$OUT" | cut -f1))"

# The re-run must reproduce the latency this seed was selected on, or the capture is of a
# different run and the decile comparison is meaningless.
python3 - "$OUT" "$GATE/results/dobj_f4000_pg16/v1_gnn_s${SEED}.summary.json" <<'PY'
import json, sys
raw = json.load(open(sys.argv[1])); st = raw.get("stats") or raw
got = float(st["averageElapsedTime"])
want = float(json.load(open(sys.argv[2]))["averageElapsedTime"])
if abs(got - want) > 0.01:
    raise SystemExit(f"FAIL LOUD: re-run gave {got:.4f} s, the gate recorded {want:.4f} s")
print(f"[reproduced] {got:.4f} s matches the gate")
PY
echo "=== task $T done: $LABEL seed $SEED ==="
