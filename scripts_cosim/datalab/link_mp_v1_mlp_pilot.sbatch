#!/bin/bash
# link_mp_v1 — the corpus-matched MLP control, as a PILOT (2026-09-03).
#
# Every one of link_mp_v1's 48 arms is a TaskPlacementGNN variant trained on the
# binding-backbone corpus. The MLP those GNNs are compared against on the bbrob cells was
# trained fabric-blind on full_corpus_siv1. The corpus is the largest lever ever measured
# in this repo, so "GNN -44.8% vs MLP -29.2%" on bb_core8_bw1p5 confounds model class with
# training corpus. This job trains the deployed MLP recipe on the SAME cache the promoted
# GNN trained on (experiments/link_mp_v1_mlp.yaml), four seeds, then evaluates every seed
# on the same five held-out cells with the same greedy evaluator the Phase 3 gate used.
#
# Not a registered gate: no verdict is read from it, no threshold exists for it. It tells
# us whether a registration is worth writing.
#
#SBATCH --job-name=lmp-mlp-pilot
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/lmp-mlp-pilot-%j.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/lmp-mlp-pilot-%j.err
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
export WANDB_ENTITY=nikolalukic167-tu-wien

python3 -c "import sys,torch;print('[INTERP]',sys.executable,torch.__version__)"
echo "[CODE] $(git describe --dirty --always) on $(git rev-parse --abbrev-ref HEAD)"

CACHE=simulation_data/graphs_cache_link_mp_v1_core_v1_dim14
[[ -f "$CACHE/metadata.json" ]] || { echo "FAIL LOUD: $CACHE has no metadata.json (the real cache lives in the pin worktree; symlink it)" >&2; exit 1; }

SEEDS="1 2 3 4"
# --- train: one process per seed, sequential (the MLP trainer is minutes, not hours)
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
mkdir -p logs models/tabular
for s in $SEEDS; do
  python3 -u run_experiment.py experiments/link_mp_v1_mlp.yaml --seed "$s"
  CK="models/tabular/batch_edge_mlp_link_mp_v1_dim22_batchcache_seed${s}.pt"
  [[ -f "$CK" ]] || { echo "FAIL LOUD: training seed $s produced no $CK" >&2; exit 1; }
done

# --- evaluate: the same five held-out cells and the same evaluator as the Phase 3 gate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
GATE_SWEEP=simulation_data/normal_sim_sweeps/live_audit_capture_bbrob_bb_core8_bw1p5
WORKLOAD=data/nofs-ids/traces/workload-150-100-30k.json
OUT=simulation_data/link_mp_v1_mlp_pilot
mkdir -p "$OUT"
ARGS=(
  --sweep-dir "$GATE_SWEEP"
  --cells cell01_p25_s9001 cell02_p35_s9002 cell03_p15_s9003 cell04_p50_s9004 cell05_p20_s9005
  --workload "$WORKLOAD"
  --workers 16
)
for s in $SEEDS; do
  ARGS+=(--arm "mlp:mlp_linkmp_s${s}=models/tabular/batch_edge_mlp_link_mp_v1_dim22_batchcache_seed${s}.pt")
done
python3 -u scripts_cosim/closed_loop/evaluate_policy.py "${ARGS[@]}" --out "$OUT/eval_bb_core8_bw1p5.json"
echo "=== link_mp_v1 MLP pilot complete ==="
