#!/bin/bash
# reliability_matched_v1 — train the 16 corpus-matched MLP arms.
#
# Registration: docs/lineages/reliability_matched_v1.md (committed before any gate data).
# Seeds 1-4 already exist from the 2026-09-03 latency pilot (job 734411) and are reused
# byte-for-byte; this job fills in 5-16 and then asserts all sixteen are present.
#
#SBATCH --job-name=relmatch-train
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/relmatch-train-%j.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/relmatch-train-%j.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=08:00:00

set -euo pipefail
cd /home/nikola.lukic/gnn-herosim

if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3
export PYTHONPATH="$PWD"
export PYTHONUNBUFFERED=1
export WANDB_ENTITY=nikolalukic167-tu-wien
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16

python3 -c "import sys,torch;print('[INTERP]',sys.executable,torch.__version__)"
echo "[CODE] $(git describe --dirty --always) on $(git rev-parse --abbrev-ref HEAD)"

CACHE=simulation_data/graphs_cache_link_mp_v1_core_v1_dim14
[[ -f "$CACHE/metadata.json" ]] || { echo "FAIL LOUD: $CACHE has no metadata.json" >&2; exit 1; }
mkdir -p logs models/tabular

for s in $(seq 5 16); do
  CK="models/tabular/batch_edge_mlp_link_mp_v1_dim22_batchcache_seed${s}.pt"
  if [[ -f "$CK" ]]; then echo "[skip] seed $s already trained"; continue; fi
  python3 -u run_experiment.py experiments/link_mp_v1_mlp.yaml --seed "$s"
  [[ -f "$CK" ]] || { echo "FAIL LOUD: seed $s produced no $CK" >&2; exit 1; }
done

# The registration fixes n=16; a partial group cannot be ranked, so fail here rather
# than at scoring time after the gate has burned its episodes.
missing=()
for s in $(seq 1 16); do
  CK="models/tabular/batch_edge_mlp_link_mp_v1_dim22_batchcache_seed${s}.pt"
  [[ -f "$CK" ]] || missing+=("$s")
done
if (( ${#missing[@]} )); then
  echo "FAIL LOUD: missing MLP seeds: ${missing[*]}" >&2; exit 1
fi
echo "=== reliability_matched_v1: all 16 MLP arms present ==="
