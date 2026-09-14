#!/bin/bash
# drainable_debug_v1 -- D2, the per-task read. Re-runs nothing; parses the retained raw
# results the KEEP_RAW=1 hook kept and applies the registered bars.
#
#   D2_DIR=simulation_data/peer_affinity_live_gate/results/dbg_D2_f4000_pg16 \
#   sbatch --export=ALL scripts_cosim/datalab/drainable_debug_v1_d2_read.sbatch
#
# Why a job and not `ssh datalab python3 ...`: the raw results are ~780 MB each and the read
# parses every one of them. That is exactly the shape of work the login-node watchdog kills
# silently (datalab-pitfalls #2), and the login node's python3 is too old to parse this repo
# besides.
#
#SBATCH --job-name=dbg-d2-read
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/dbg-d2-read-%j.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/dbg-d2-read-%j.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=04:00:00

set -euo pipefail
cd /home/nikola.lukic/gnn-herosim
if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3 PYTHONPATH="$PWD" PYTHONUNBUFFERED=1
echo "[CODE] $(git describe --dirty --always) on $(git rev-parse --abbrev-ref HEAD)"

G=simulation_data/peer_affinity_live_gate
D2_DIR="${D2_DIR:-$G/results/dbg_D2_f4000_pg16}"
WORKLOAD="${WORKLOAD:-$G/workloads/drainable_f4000_n50000.json}"
OUT="${OUT:-simulation_data/drainable_regime_v1/d2_read.json}"
ARMS="${ARMS:-knative_network knative_network_batch gnn_s1 gnn_s2 mpoff_s1 mpoff_s2}"

RAW_ARGS=()
for A in $ARMS; do
  F="$D2_DIR/$A.json"
  [[ -f "$F" ]] || { echo "FAIL LOUD: $F missing -- was the arm run with KEEP_RAW=1?" >&2; exit 1; }
  RAW_ARGS+=(--raw "$A=$F")
done
echo "[INPUTS] $D2_DIR : $(du -sh "$D2_DIR" | cut -f1) across ${#RAW_ARGS[@]} arms"

python3 scripts_cosim/drainable_debug_pertask_read.py \
  "${RAW_ARGS[@]}" --workload "$WORKLOAD" --reference knative_network --out "$OUT"
echo "=== D2 read done -> $OUT ==="
