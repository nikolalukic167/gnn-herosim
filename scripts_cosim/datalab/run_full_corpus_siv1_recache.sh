#!/usr/bin/env bash
# Build a merged scale_invariant_v1 graph cache from the full healthy legacy_v0_node_disk_v2_4task
# training group (contention_v2/v3/v4_pilot, 1060_warmth_v2, sparse_warmth_v2, highq_safe_20260606)
# -- 2,816 completed datasets, all coupling-validated PASS per DATASET_HEALTH_REPORT.json.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/nikola.lukic/gnn-herosim}"
cd "$PROJECT_ROOT"

CORPUS_DIRS=(
  "simulation_data/gnn_datasets_4tasks_contention_v2"
  "simulation_data/gnn_datasets_4tasks_contention_v3"
  "simulation_data/gnn_datasets_4tasks_contention_v4_pilot"
  "simulation_data/gnn_datasets_4tasks_1060_warmth_v2"
  "simulation_data/gnn_datasets_4tasks_sparse_warmth_v2"
  "simulation_data/gnn_datasets_4tasks_highq_safe_20260606"
)
# Known-bad dataset IDs discovered by prepare_graphs_cache.py's training-contract-5.5 integrity
# check (sweep-min labels absent from scheduling-time candidate edges). Excluded via an
# oversample manifest (weight-1 allowlist) rather than deleted, so the corpus dirs stay intact.
BAD_DATASET_IDS=(
  "gnn_datasets_4tasks_1060_warmth_v2/ds_00252"
  "gnn_datasets_4tasks_1060_warmth_v2/ds_00254"
  "gnn_datasets_4tasks_contention_v2/ds_00646"
  "gnn_datasets_4tasks_contention_v2/ds_00654"
  "gnn_datasets_4tasks_contention_v2/ds_00656"
  "gnn_datasets_4tasks_contention_v2/ds_00657"
  "gnn_datasets_4tasks_contention_v2/ds_00658"
  "gnn_datasets_4tasks_contention_v2/ds_00661"
  "gnn_datasets_4tasks_contention_v2/ds_00663"
  "gnn_datasets_4tasks_contention_v2/ds_00729"
  "gnn_datasets_4tasks_contention_v2/ds_00737"
  "gnn_datasets_4tasks_contention_v2/ds_00747"
  "gnn_datasets_4tasks_contention_v2/ds_00748"
  "gnn_datasets_4tasks_contention_v2/ds_00750"
  "gnn_datasets_4tasks_contention_v2/ds_00765"
  "gnn_datasets_4tasks_contention_v2/ds_00766"
  "gnn_datasets_4tasks_contention_v2/ds_00768"
  "gnn_datasets_4tasks_contention_v2/ds_00770"
  "gnn_datasets_4tasks_contention_v2/ds_00771"
  "gnn_datasets_4tasks_contention_v2/ds_00835"
  "gnn_datasets_4tasks_contention_v2/ds_00841"
  "gnn_datasets_4tasks_contention_v2/ds_00842"
  "gnn_datasets_4tasks_contention_v2/ds_00861"
  "gnn_datasets_4tasks_contention_v2/ds_00872"
  "gnn_datasets_4tasks_contention_v2/ds_00874"
  "gnn_datasets_4tasks_contention_v2/ds_00877"
  "gnn_datasets_4tasks_contention_v2/ds_00878"
  "gnn_datasets_4tasks_contention_v2/ds_00899"
  "gnn_datasets_4tasks_sparse_warmth_v2/ds_00105"
  "gnn_datasets_4tasks_sparse_warmth_v2/ds_00265"
  "gnn_datasets_4tasks_sparse_warmth_v2/ds_00289"
)
OVERSAMPLE_MANIFEST="logs/full_corpus_siv1_pipeline/oversample_manifest_exclude_bad31.json"
CACHE_DIR="simulation_data/graphs_cache_full_corpus_siv1_dim14"
PHASE_DIR="logs/full_corpus_siv1_pipeline"
MIN_GRAPHS="${MIN_GRAPHS:-2600}"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="logs/full_corpus_siv1_recache_${TS}.log"
ENV_NAME="${HEROSIM_ENV_NAME:-gnn}"

mkdir -p logs "$PHASE_DIR" "$CACHE_DIR"

exec > >(tee -a "$LOG") 2>&1
echo "=== full_corpus siv1 recache ${TS} ==="
echo "Host: $(hostname)"
echo "Corpora: ${CORPUS_DIRS[*]}"
echo "Cache: ${CACHE_DIR}"

if ! command -v micromamba >/dev/null 2>&1; then
  source "${HOME}/.bashrc"
fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate "${ENV_NAME}"

export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src/notebooks"

total_jsonl=0
for d in "${CORPUS_DIRS[@]}"; do
  n=$(find "$d" -path '*/placements/placements.jsonl' -size +0 2>/dev/null | wc -l)
  echo "  ${d}: ${n} placements.jsonl"
  total_jsonl=$((total_jsonl + n))
done
echo "Total placements.jsonl across corpora: ${total_jsonl}"
if [[ "$total_jsonl" -lt "$MIN_GRAPHS" ]]; then
  echo "ERROR: expected >= ${MIN_GRAPHS} total datasets, got ${total_jsonl}" >&2
  exit 1
fi

if [[ -f "${CACHE_DIR}/graphs.pkl" && "${FORCE_RECACHE:-0}" != "1" ]]; then
  n_graphs=$(python3 -c "import pickle; print(len(pickle.load(open('${CACHE_DIR}/graphs.pkl','rb'))))")
  if [[ "$n_graphs" -ge "$MIN_GRAPHS" ]]; then
    echo "SKIP recache: ${CACHE_DIR} already has ${n_graphs} graphs"
    touch "${PHASE_DIR}/phase_recache.done"
    exit 0
  fi
  echo "WARN: cache exists but only ${n_graphs} graphs (< ${MIN_GRAPHS}); rebuilding"
fi

echo "=== refresh_optimal_full_stats (--rewrite-ssc) per corpus ==="
for d in "${CORPUS_DIRS[@]}"; do
  echo "--- ${d} ---"
  python3 -u scripts_cosim/refresh_optimal_full_stats.py --base-dir "$d" --rewrite-ssc
done

rm -rf "$CACHE_DIR"
mkdir -p "$CACHE_DIR"

echo "=== building oversample manifest (allowlist, excludes ${#BAD_DATASET_IDS[@]} known-bad datasets) ==="
python3 - "$OVERSAMPLE_MANIFEST" "${CORPUS_DIRS[@]}" <<PYEOF
import json, sys
from pathlib import Path

out_path = Path(sys.argv[1])
corpus_dirs = [Path(p) for p in sys.argv[2:]]
bad = set("""${BAD_DATASET_IDS[@]}""".split())

weights = {}
for d in corpus_dirs:
    for ds in sorted(d.glob("ds_*")):
        jsonl = ds / "placements" / "placements.jsonl"
        if not jsonl.exists() or jsonl.stat().st_size == 0:
            continue
        key = f"{d.name}/{ds.name}"
        if key in bad:
            continue
        weights[key] = 1

out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(weights))
print(f"Wrote {len(weights)} dataset weights (excluded {len(bad)}) to {out_path}")
PYEOF

echo "=== prepare_graphs_cache (full corpus, scale_invariant_v1, dim14) ==="
python3 -u src/notebooks/prepare_graphs_cache.py \
  --base-dirs "${CORPUS_DIRS[@]}" \
  --cache-dir "$CACHE_DIR" \
  --queue-feature-contract scale_invariant_v1 \
  --platform-feature-dim 14 \
  --oversample-manifest "$OVERSAMPLE_MANIFEST"

n_graphs=$(python3 -c "import pickle; print(len(pickle.load(open('${CACHE_DIR}/graphs.pkl','rb'))))")
echo "Graphs in cache: ${n_graphs}"
if [[ "$n_graphs" -lt "$MIN_GRAPHS" ]]; then
  echo "ERROR: cache too small (${n_graphs} < ${MIN_GRAPHS})" >&2
  exit 1
fi

touch "${PHASE_DIR}/phase_recache.done"
echo "=== recache complete === log=${LOG}"
