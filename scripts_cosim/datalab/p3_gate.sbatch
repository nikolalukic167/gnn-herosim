#!/bin/bash
# objective_pivot_v1 Phase 3 — the registered gate. THIS is the verdict.
#
# Held out from everything: `bbrob_bb_core8_bw1p5`, all 5 cells. A different network fabric
# from the one the pilot trained on (core4/bw0.5) and different cells from the dev pair
# that selected the hyperparameters. Nothing in this sweep has been touched by training or
# by selection.
#
# 35 arms x 5 cells = 175 greedy episodes. Every arm runs argmax -- the configuration the
# live gates serve. Nothing that produces a reported number ever samples.
#
# The verdict is the PAIRED CL-minus-Frozen difference over the 16 training seeds
# (Amendment D1: the replication unit is the training run), by one-sided exact Wilcoxon.
# analyze_gate.py applies it with no discretion left in the script, and prints the kill
# criterion itself if the numbers call for it.
#
#SBATCH --job-name=p3-gate
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/p3-gate-%j.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/p3-gate-%j.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00

set -euo pipefail
cd /home/nikola.lukic/gnn-herosim

if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3
export PYTHONPATH="$PWD"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

python3 -c "import sys,torch;print('[INTERP]',sys.executable,torch.__version__)"

GATE_SWEEP=simulation_data/normal_sim_sweeps/live_audit_capture_bbrob_bb_core8_bw1p5
WORKLOAD=data/nofs-ids/traces/workload-150-100-30k.json
FROZEN_GNN=models/gnn-linkmp-lgon-s8.pt
FROZEN_MLP=models/tabular/batch_edge_mlp_full_corpus_siv1_dim22_batchcache_tempfix.pt
OUT=simulation_data/p3_gate

ARGS=(
  --sweep-dir "$GATE_SWEEP"
  --cells cell01_p25_s9001 cell02_p35_s9002 cell03_p15_s9003 cell04_p50_s9004 cell05_p20_s9005
  --workload "$WORKLOAD"
  --arm "gnn:frozen_gnn=$FROZEN_GNN"
  --arm "mlp:frozen_mlp=$FROZEN_MLP"
  --arm "knative:knative"
)
for s in $(seq 1 16); do
  CK="simulation_data/p3_pilot/gnn_s${s}/checkpoints/cl_gnn_step020.pt"
  [[ -f "$CK" ]] || { echo "FAIL LOUD: missing $CK — a partial gate cannot be paired" >&2; exit 1; }
  ARGS+=(--arm "gnn:cl_gnn_s${s}=$CK")
done
for s in $(seq 1 16); do
  CK="simulation_data/p3_pilot/mlp_s${s}/checkpoints/cl_mlp_step020.pt"
  [[ -f "$CK" ]] || { echo "FAIL LOUD: missing $CK — a partial gate cannot be paired" >&2; exit 1; }
  ARGS+=(--arm "mlp:cl_mlp_s${s}=$CK")
done

python3 scripts_cosim/closed_loop/evaluate_policy.py "${ARGS[@]}" \
  --out "$OUT/gate_eval.json" --workers 16

echo ""
echo "############ CL-GNN vs Frozen-GNN ############"
python3 scripts_cosim/closed_loop/analyze_gate.py \
  --gate-json "$OUT/gate_eval.json" \
  --cl-prefix cl_gnn_s --frozen-label frozen_gnn \
  --out "$OUT/verdict_gnn.json"

echo ""
echo "############ CL-MLP vs Frozen-MLP ############"
python3 scripts_cosim/closed_loop/analyze_gate.py \
  --gate-json "$OUT/gate_eval.json" \
  --cl-prefix cl_mlp_s --frozen-label frozen_mlp \
  --out "$OUT/verdict_mlp.json"

echo "=== p3 gate COMPLETE ==="
