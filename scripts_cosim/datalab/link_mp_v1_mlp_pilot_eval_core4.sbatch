#!/bin/bash
# link_mp_v1 corpus-matched MLP pilot — second fabric (2026-09-03).
#
# Evaluates the four MLP checkpoints trained by link_mp_v1_mlp_pilot.sbatch on the OTHER
# bbrob fabric, bb_core4_bw0p5 (5 cells), so the bb_core8_bw1p5 tie with the promoted
# GNN can be checked on a fabric the corpus also covers. Reference arms (the promoted GNN,
# the fabric-blind MLP, Knative) come from the same evaluator so the numbers are paired.
#
#SBATCH --job-name=lmp-mlp-eval4
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/lmp-mlp-eval4-%j.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/lmp-mlp-eval4-%j.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=06:00:00

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
echo "[CODE] $(git describe --dirty --always) on $(git rev-parse --abbrev-ref HEAD)"

GATE_SWEEP=simulation_data/normal_sim_sweeps/live_audit_capture_bbrob_bb_core4_bw0p5
WORKLOAD=data/nofs-ids/traces/workload-150-100-30k.json
OUT=simulation_data/link_mp_v1_mlp_pilot
mkdir -p "$OUT"
ARGS=(
  --sweep-dir "$GATE_SWEEP"
  --cells cell01_p25_s9001 cell02_p35_s9002 cell03_p15_s9003 cell04_p50_s9004 cell05_p20_s9005
  --workload "$WORKLOAD"
  --workers 16
  --arm "gnn:frozen_gnn=models/gnn-linkmp-lgon-s8.pt"
  --arm "mlp:frozen_mlp=models/tabular/batch_edge_mlp_full_corpus_siv1_dim22_batchcache_tempfix.pt"
  --arm "knative:knative"
)
for s in 1 2 3 4; do
  CK="models/tabular/batch_edge_mlp_link_mp_v1_dim22_batchcache_seed${s}.pt"
  [[ -f "$CK" ]] || { echo "FAIL LOUD: missing $CK" >&2; exit 1; }
  ARGS+=(--arm "mlp:mlp_linkmp_s${s}=$CK")
done
python3 -u scripts_cosim/closed_loop/evaluate_policy.py "${ARGS[@]}" --out "$OUT/eval_bb_core4_bw0p5.json"
echo "=== link_mp_v1 MLP pilot core4 eval complete ==="
