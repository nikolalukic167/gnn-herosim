#!/bin/bash
# link_mp_v1 — build BOTH graph caches from the binding-backbone corpus, AT THE PINNED
# COMMIT. Registration: docs/lineages/link_mp_v1.md.
#
#   graphs_cache_link_mp_v1_dim14          NETWORK_GRAPH_CONTRACT unset ("off")
#   graphs_cache_link_mp_v1_core_v1_dim14  NETWORK_GRAPH_CONTRACT=core_v1
#
# Same corpus, same flags, same dataset order — the caches differ in exactly the network
# entities, which is the one factor the lgon arms add. The core cache is then asserted to
# carry net_* attrs on EVERY graph: one fabric-less graph would crash training hours in
# (TaskPlacementGNN fails loud on a missing block, by design), so fail here instead.
#
# Submit with:  --dependency=afterok:<linkmp-cosim array job id>
#SBATCH --job-name=linkmp-recache
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/linkmp-recache-%j.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/linkmp-recache-%j.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00

set -euo pipefail

PIN_DIR=/home/nikola.lukic/gnn-herosim-pin-__LINK_MP_V1_PIN_SHORT__
PIN_COMMIT=__LINK_MP_V1_PIN__
cd "$PIN_DIR"

if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3
export PYTHONPATH="$PWD:$PWD/src/notebooks"
export PYTHONUNBUFFERED=1

HEAD_NOW=$(git rev-parse HEAD)
[[ "$HEAD_NOW" == "$PIN_COMMIT" ]] || {
  echo "FAIL LOUD: pinned worktree at $HEAD_NOW, registration requires $PIN_COMMIT" >&2; exit 1; }
DIRTY=$(git status --porcelain -- src scripts_cosim experiments run_experiment.py)
[[ -z "$DIRTY" ]] || {
  echo "FAIL LOUD: pinned worktree dirty under CODE_PATHS:" >&2; echo "$DIRTY" >&2; exit 1; }
echo "[PIN] $PIN_DIR at $PIN_COMMIT, CODE_PATHS clean"

CORPUS_DIRS=(
  "simulation_data/gnn_datasets_4tasks_link_mp_v1_core4_bw0p5"
  "simulation_data/gnn_datasets_4tasks_link_mp_v1_core4_bw1p5"
  "simulation_data/gnn_datasets_4tasks_link_mp_v1_core8_bw1p5"
)
CACHE_OFF="simulation_data/graphs_cache_link_mp_v1_dim14"
CACHE_CORE="simulation_data/graphs_cache_link_mp_v1_core_v1_dim14"
# Floor against catastrophic generation loss; the two caches must additionally be EQUAL.
MIN_GRAPHS=1500

total_jsonl=0
for d in "${CORPUS_DIRS[@]}"; do
  n=$(find "$d" -path '*/placements/placements.jsonl' -size +0 2>/dev/null | wc -l)
  echo "  ${d}: ${n} placements.jsonl"
  total_jsonl=$((total_jsonl + n))
done
echo "Total placements.jsonl: ${total_jsonl}"
[[ "$total_jsonl" -ge "$MIN_GRAPHS" ]] || {
  echo "FAIL LOUD: only ${total_jsonl} complete datasets (< ${MIN_GRAPHS})" >&2; exit 1; }

echo "=== Step 1: refresh_optimal_full_stats --repair per corpus ==="
for d in "${CORPUS_DIRS[@]}"; do
  python3 -u scripts_cosim/refresh_optimal_full_stats.py --base-dir "$d" --repair
done

echo "=== Step 2: cache OFF (no network entities) ==="
rm -rf "$CACHE_OFF" && mkdir -p "$CACHE_OFF"
python3 -u src/notebooks/prepare_graphs_cache.py \
  --base-dirs "${CORPUS_DIRS[@]}" \
  --cache-dir "$CACHE_OFF" \
  --queue-feature-contract scale_invariant_v1 \
  --platform-feature-dim 14

echo "=== Step 3: cache core_v1 (network entities in the graph) ==="
rm -rf "$CACHE_CORE" && mkdir -p "$CACHE_CORE"
NETWORK_GRAPH_CONTRACT=core_v1 python3 -u src/notebooks/prepare_graphs_cache.py \
  --base-dirs "${CORPUS_DIRS[@]}" \
  --cache-dir "$CACHE_CORE" \
  --queue-feature-contract scale_invariant_v1 \
  --platform-feature-dim 14

echo "=== Step 4: verify the caches pair up and the core cache has the fabric ==="
python3 - "$CACHE_OFF" "$CACHE_CORE" "$MIN_GRAPHS" <<'PY'
import pickle, sys
off_dir, core_dir, min_graphs = sys.argv[1], sys.argv[2], int(sys.argv[3])
off = pickle.load(open(f"{off_dir}/graphs.pkl", "rb"))
core = pickle.load(open(f"{core_dir}/graphs.pkl", "rb"))
if len(off) != len(core):
    raise SystemExit(f"FAIL LOUD: cache sizes differ: off={len(off)} core={len(core)}")
if len(core) < min_graphs:
    raise SystemExit(f"FAIL LOUD: only {len(core)} graphs (< {min_graphs})")
def graph_of(entry):
    return entry["graph"] if isinstance(entry, dict) else entry
missing = sum(
    1 for e in core
    if getattr(graph_of(e), "net_node_features", None) is None
    or getattr(graph_of(e), "net_link_features", None) is None
    or getattr(graph_of(e), "net_edge_index", None) is None
)
if missing:
    raise SystemExit(
        f"FAIL LOUD: {missing}/{len(core)} core_v1 graphs are missing network entities. "
        "A fabric-less graph crashes lgon training mid-epoch; fix the corpus first.")
leaked = sum(1 for e in off if getattr(graph_of(e), "net_node_features", None) is not None)
if leaked:
    raise SystemExit(f"FAIL LOUD: {leaked} OFF-cache graphs carry network entities")
g = graph_of(core[0])
print(f"[OK] {len(core)} graphs in both caches; core sample: "
      f"net_node_features {tuple(g.net_node_features.shape)}, "
      f"net_link_features {tuple(g.net_link_features.shape)}, "
      f"net_edge_index {tuple(g.net_edge_index.shape)}")
PY
echo "=== recache complete ==="
