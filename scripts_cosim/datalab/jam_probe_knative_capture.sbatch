#!/bin/bash
# JAM PROBE -- scoping capture, NO BARS, NOT a registered lineage.
#
# offline_live_transfer_v1 R3 measured that 50.5 % of the 1.7x gap between two training seeds
# of one recipe is ONE decile of one trace, with full recovery afterwards. The open question
# that decides whether that is a lever or a curiosity:
#
#     is the learned arms' loss to reactive Knative ALSO concentrated in excursions,
#     or is it spread uniformly across the trace?
#
#   concentrated  -> the graph arm is not steadily worse at placement, it is jam-prone, and
#                    the fix is a different problem from better placement scoring
#   uniform       -> the graph arm is simply worse; the excursion is a side story and this
#                    goes no further
#
# This capture only produces the missing input: the gate retained raw per-task records for
# arms named *_s1 / *_s2, so `knative_network` has none. The GNN side already exists
# (results/olt_r3_capture/{best_gnn_s14,worst_gnn_s8}.json).
#
# Scoping status is deliberate and disclosed: this read has no pre-registered bar, so it can
# ORDER work and cannot close anything. If it comes back jam-shaped, the follow-ups get a
# registration with bars signed before their data, and this read is named there as the
# motivating measurement -- the same handling offline_live_transfer_v1 gave its own.
#
#   sbatch scripts_cosim/datalab/jam_probe_knative_capture.sbatch
#
#SBATCH --job-name=jam-knb
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/jam-knb-%j.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/jam-knb-%j.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00

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

# Byte-identical contract to the gate whose knative_network latency this must reproduce, and
# to the two GNN captures it will be compared against.
export HEROSIM_PEER_EXCHANGE=1
export HEROSIM_SERVER_ONLY_REPLICAS=1
export HEROSIM_WARMTH_PHYSICS=node_disk_v2
export PYTHONHASHSEED=0
export SIM_FORCE_FULL_STATS=1

OUT="$OUT_DIR/reactive_knative_network.json"
if [[ -f "$OUT" ]]; then echo "exists, not re-running"; exit 0; fi

python3 src/executesimulation.py --config "$CFG" --workload "$WORKLOAD" \
  --policy knative_network --output "$OUT.partial"
mv "$OUT.partial" "$OUT"
echo "[RAW KEPT] $OUT ($(du -h "$OUT" | cut -f1))"

# Must reproduce the gate's own knative_network number, or this is a capture of a different
# run and every decile comparison against it is meaningless.
python3 - "$OUT" "$GATE/results/dobj_f4000_pg16/knative_network.summary.json" <<'PY'
import json, sys
raw = json.load(open(sys.argv[1])); st = raw.get("stats") or raw
got = float(st["averageElapsedTime"])
want = float(json.load(open(sys.argv[2]))["averageElapsedTime"])
if abs(got - want) > 0.01:
    raise SystemExit(f"FAIL LOUD: re-run gave {got:.4f} s, the gate recorded {want:.4f} s")
print(f"[reproduced] {got:.4f} s matches the gate")
PY
echo "=== jam probe capture done ==="
