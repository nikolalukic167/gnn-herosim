#!/bin/bash
# peer_affinity_warm_v1 -- W0 screen: SSC rewrite, alpha pre-scan, cache, structure scorer, read.
# Runs after both W0 generation arrays completed (docs/lineages/peer_affinity_warm_v1.md).
#   sbatch scripts_cosim/datalab/peer_affinity_warm_v1_w0_read.sbatch
#
#SBATCH --job-name=pa-warm-w0-read
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/pa-warm-w0-read-%j.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/pa-warm-w0-read-%j.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00

set -euo pipefail
cd /home/nikola.lukic/gnn-herosim
if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3
export PYTHONPATH="$PWD:$PWD/src/notebooks"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export DAG_PRIMARY_ALPHA_KEY=2.5 PARTIAL_STATE_CONTRACT=partial_state_v2 HEROSIM_PEER_EXCHANGE=1
python3 -c "import sys,torch;print('[INTERP]',sys.executable,torch.__version__)"
echo "[CODE] $(git describe --dirty --always) on $(git rev-parse --abbrev-ref HEAD)"

W0=simulation_data/peer_affinity_warm_v1
KNB=simulation_data/gnn_datasets_peer_affinity_warm_v1_w0_knb
GNN=simulation_data/gnn_datasets_peer_affinity_warm_v1_w0_gnn
CACHE=simulation_data/graphs_cache_peer_affinity_warm_v1_w0
mkdir -p "$W0" logs
for d in "$KNB" "$GNN"; do
  n=$(find "$d" -mindepth 3 -maxdepth 3 -path '*/placements/placements.jsonl' -size +0 | wc -l)
  echo "[corpus] $d: $n complete sweeps"
  [[ "$n" -ge 40 ]] || { echo "FAIL LOUD: fewer than 40 complete sweeps under $d"; exit 1; }
  bad=$(find "$d" -mindepth 1 -maxdepth 1 -type d -name 'ds_*' | while read -r ds; do [[ -s "$ds/placements/placements.jsonl" && -s "$ds/best.json" ]] || echo "$ds"; done | wc -l)
  [[ "$bad" == "0" ]] || { echo "FAIL LOUD: $bad dataset dirs under $d are incomplete"; exit 1; }
done

echo "=== [1/5] SSC rewrite"
for d in "$KNB" "$GNN"; do python3 scripts_cosim/refresh_optimal_full_stats.py --base-dir "$d" --rewrite-ssc | tail -1; done

echo "=== [2/5] alpha pre-scan (report only; W0 is diagnostic)"
python3 scripts_cosim/peer_affinity_alpha_prescan.py "$KNB" "$GNN" --alpha 2.0 | tail -6

echo "=== [3/5] cache -> $CACHE"
rm -rf "$CACHE"
python3 -u src/notebooks/prepare_graphs_cache.py --base-dirs "$KNB" "$GNN" --cache-dir "$CACHE" \
  --dag-partial-state --platform-feature-dim 14 2>&1 | grep -v "it/s\]" | tail -12

echo "=== [4/5] structure scorer (alpha 2.5, rtt)"
# --allow-replica-reuse: two types x five tasks over ~3 candidates per type -- the sweep and
# the serving decoder both allow reuse; the no-reuse mask has no plan on these datasets.
python3 scripts_cosim/score_route_b_contention.py --corpus "$KNB" --objective rtt --alphas 2.5 \
  --allow-replica-reuse --include-per-dataset --out "$W0/w0_score_knb.json" | tail -3
python3 scripts_cosim/score_route_b_contention.py --corpus "$GNN" --objective rtt --alphas 2.5 \
  --allow-replica-reuse --include-per-dataset --out "$W0/w0_score_gnn.json" | tail -3

echo "=== [5/5] read"
python3 scripts_cosim/peer_affinity_warm_w0_read.py --cache-dir "$CACHE" \
  --scorer-json "$W0/w0_score_knb.json" "$W0/w0_score_gnn.json" --out "$W0/w0_read.json"
echo "=== W0 read done ==="
