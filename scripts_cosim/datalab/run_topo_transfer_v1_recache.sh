#!/usr/bin/env bash
# Repair + build the size_invariant_v1/core_v1 graph cache for the topology_transfer_v1
# corpus (server_node_count 20/28/40/60/80 ladder; train {20,28,40}, held-out {60,80}).
#
# Runs via sbatch (topo_transfer_v1_recache.sbatch), never directly over ssh on the login
# node -- a raw ssh+nohup run of this exact repair+recache combo was silently killed ~6
# minutes into the graph-build phase on slurm-head-1 (no traceback, no OOM evidence; the
# login node enforces limits on sustained multi-worker CPU jobs). Every other recache
# script in this directory (full_corpus_siv1_recache, warmth_features_recache) already
# goes through sbatch for the same reason.
#
# Corpus generation (SLURM job 704238, netc-style array) may still be adding completed
# ds_* directories while this runs. That's expected: the repair pass here closes most of
# the race, and --allow-missing-queue-data on prepare_graphs_cache.py loudly skips (not
# silently drops) any straggler that finishes in the remaining window rather than crashing
# the whole build. See docs/notes/placements_jsonl_required.md for why placements.jsonl,
# not best.json, is the completion signal prepare_graphs_cache.py itself already keys on.
#
# NETWORK_GRAPH_CONTRACT=core_v1 is load-bearing, not optional: it's what makes the cache
# include the network/core-link (backbone) graph entities instead of building a
# topology-blind cache, which would defeat the whole point of this corpus's topology-SIZE
# axis. TOPOLOGY_FEATURE_CONTRACT=size_invariant_v1 makes task feature dim 2 a size-
# invariant reachability fraction instead of the size-entangled node-index formula
# (src_index_v0), which is required for the topology-transfer question (train on
# {20,28,40}, evaluate on {60,80}) to be answerable at all -- see
# src/placement/topology_features.py docstring.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/nikola.lukic/gnn-herosim}"
cd "$PROJECT_ROOT"

ENV_NAME="${HEROSIM_ENV_NAME:-gnn}"
CORPUS_DIR="simulation_data/gnn_datasets_4tasks_topo_transfer_v1"
CACHE_DIR="simulation_data/graphs_cache_topo_transfer_v1"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="logs/topo_transfer_v1_recache_${TS}.log"

mkdir -p logs "$CACHE_DIR"

exec > >(tee -a "$LOG") 2>&1
echo "=== topo_transfer_v1 recache ${TS} ==="
echo "Host: $(hostname)"
echo "Corpus: ${CORPUS_DIR}"
echo "Cache: ${CACHE_DIR}"

if ! command -v micromamba >/dev/null 2>&1; then
  source "${HOME}/.bashrc"
fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate "${ENV_NAME}"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src/notebooks"

n_jsonl=$(find "$CORPUS_DIR" -path '*/placements/placements.jsonl' -size +0 2>/dev/null | wc -l)
echo "placements.jsonl present right now: ${n_jsonl}"

echo "=== Step 1: refresh_optimal_full_stats --repair ==="
python3 -u scripts_cosim/refresh_optimal_full_stats.py \
  --base-dir "$CORPUS_DIR" \
  --repair

echo "=== Step 2: prepare_graphs_cache (core_v1 network graph, size_invariant_v1 topology feature) ==="
NETWORK_GRAPH_CONTRACT=core_v1 TOPOLOGY_FEATURE_CONTRACT=size_invariant_v1 python3 -u \
  src/notebooks/prepare_graphs_cache.py \
  --base-dirs "$CORPUS_DIR" \
  --cache-dir "$CACHE_DIR" \
  --queue-feature-contract scale_invariant_v1 \
  --allow-missing-queue-data

n_graphs=$(python3 -c "import pickle; print(len(pickle.load(open('${CACHE_DIR}/graphs.pkl','rb'))))")
echo "Graphs in cache: ${n_graphs}"
echo "=== topo_transfer_v1 recache complete === log=${LOG}"
