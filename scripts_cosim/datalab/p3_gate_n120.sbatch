#!/bin/bash
# objective_pivot_v1 Phase 3, Amendment E — the powered gate. THIS is the verdict.
#
# 120 fresh CL-GNN seeds (17-136) against Frozen-GNN on the held-out `bb_core8_bw1p5`
# fabric: a different fabric from the one they trained on, and different cells from the
# dev pair that selected the hyperparameters. Nothing here has been touched by training
# or by selection.
#
# The original 16 pilot seeds are NOT in this gate (E1). They were looked at before the
# decision to extend, and they are reported as a pooled secondary from the earlier gate's
# artifact, never mixed into this primary.
#
# Frozen-MLP and Knative ride along for context. Both are deterministic on a fixed cell
# and trace, so their numbers must reproduce the earlier gate's EXACTLY -- a free
# cross-run reproducibility check on the whole serving path.
#
# 123 arms x 5 cells = 615 greedy episodes, all argmax. Nothing that produces a reported
# number ever samples.
#
#SBATCH --job-name=p3-gate-n120
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/p3-gate-n120-%j.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/p3-gate-n120-%j.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
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
OUT=simulation_data/p3_gate_n120

ARGS=(
  --sweep-dir "$GATE_SWEEP"
  --cells cell01_p25_s9001 cell02_p35_s9002 cell03_p15_s9003 cell04_p50_s9004 cell05_p20_s9005
  --workload "$WORKLOAD"
  --arm "gnn:frozen_gnn=$FROZEN_GNN"
  --arm "mlp:frozen_mlp=$FROZEN_MLP"
  --arm "knative:knative"
)
MISSING=0
for s in $(seq 17 136); do
  CK="simulation_data/p3_n120/gnn_s${s}/checkpoints/cl_gnn_step020.pt"
  if [[ -f "$CK" ]]; then
    ARGS+=(--arm "gnn:cl_gnn_s${s}=$CK")
  else
    echo "FAIL LOUD: missing $CK" >&2
    MISSING=$((MISSING + 1))
  fi
done
if [[ "$MISSING" -gt 0 ]]; then
  echo "FAIL LOUD: $MISSING of 120 registered seeds have no checkpoint. A gate run on a" >&2
  echo "subset is a gate at an n the registration did not authorise; fix the runs first." >&2
  exit 1
fi

python3 scripts_cosim/closed_loop/evaluate_policy.py "${ARGS[@]}" \
  --out "$OUT/gate_eval.json" --workers 32

echo ""
echo "######## PRIMARY (Amendment E1): CL-GNN vs Frozen-GNN, 120 fresh seeds ########"
python3 scripts_cosim/closed_loop/analyze_gate.py \
  --gate-json "$OUT/gate_eval.json" \
  --cl-prefix cl_gnn_s --frozen-label frozen_gnn \
  --min-seeds 120 --primary \
  --out "$OUT/verdict_primary.json"

echo "=== p3 n120 gate COMPLETE ==="
