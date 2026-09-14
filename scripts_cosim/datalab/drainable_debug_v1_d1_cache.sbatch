#!/bin/bash
# drainable_debug_v1 -- D1, step 3: one graphs cache over all three source corpora, so both
# checkpoints can be decoded on every captured state. Same recipe as the warm W1 cache; no
# split artifact is built because nothing here is trained on -- these are diagnostic states
# from the GATE cell, exactly as peer_affinity_warm_v1's W0 was.
#
#   sbatch scripts_cosim/datalab/drainable_debug_v1_d1_cache.sbatch
#
#SBATCH --job-name=dbg-d1-cache
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/dbg-d1-cache-%j.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/dbg-d1-cache-%j.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00

set -euo pipefail
cd /home/nikola.lukic/gnn-herosim
if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3 PYTHONPATH="$PWD" PYTHONUNBUFFERED=1
export DAG_PRIMARY_ALPHA_KEY=2.5 PARTIAL_STATE_CONTRACT=partial_state_v2 PARTIAL_STATE_PEER_MASS=1
export HEROSIM_PEER_EXCHANGE=1
echo "[CODE] $(git describe --dirty --always) on $(git rev-parse --abbrev-ref HEAD)"

CORPORA=()
for S in knb gnn mpoff; do CORPORA+=("simulation_data/gnn_datasets_drainable_debug_v1_d1_${S}"); done
CACHE="${CACHE:-simulation_data/graphs_cache_drainable_debug_v1_d1}"

# A truncated sweep biases every statistic downstream; refuse before spending the cache build.
for d in "${CORPORA[@]}"; do
  [[ -d "$d" ]] || { echo "FAIL LOUD: corpus missing: $d" >&2; exit 1; }
  read -r n bad <<< "$(python3 - "$d" <<'PY'
import json, sys, pathlib
root = pathlib.Path(sys.argv[1]); n = bad = 0
for ds in sorted(root.glob("ds_*")):
    n += 1
    meta = ds / "placement_metadata.json"
    if not meta.exists() or not json.loads(meta.read_text()).get("sweep_complete", False):
        bad += 1
print(n, bad)
PY
)"
  echo "[corpus] $d: $n datasets, $bad incomplete"
  [[ "$bad" == "0" ]] || { echo "FAIL LOUD: $bad incomplete sweeps under $d" >&2; exit 1; }
done

echo "=== [1/3] SSC rewrite"
for d in "${CORPORA[@]}"; do
  python3 scripts_cosim/refresh_optimal_full_stats.py --base-dir "$d" --rewrite-ssc | tail -2
  [[ $(ls "$d"/ds_*/system_state_captured_unique.json | wc -l) -eq $(ls -d "$d"/ds_* | wc -l) ]] \
    || { echo "FAIL LOUD: SSC missing under $d" >&2; exit 1; }
done

echo "=== [2/3] cache -> $CACHE"
rm -rf "$CACHE"
python3 -u src/notebooks/prepare_graphs_cache.py --base-dirs "${CORPORA[@]}" --cache-dir "$CACHE" \
  --dag-partial-state --platform-feature-dim 14 2>&1 | grep -v "it/s\]" | tail -30
python3 - "$CACHE" <<'PY'
import json, sys
m = json.load(open(sys.argv[1] + "/metadata.json"))
need = {"partial_state_contract": "partial_state_v2", "dag_primary_alpha_key": "2.5",
        "peer_exchange_block": True, "platform_feature_dim": 14}
for k, v in need.items():
    assert m.get(k) == v, f"FAIL LOUD: cache {k}={m.get(k)!r} != {v!r}"
print("[cache] contract ok, num_datasets =", m.get("num_datasets"), "rtt rows =", m.get("num_rtt_combo_rows"))
PY

echo "=== [3/3] near-RTT sidecar"
python3 -u src/notebooks/prepare_graphs_cache_near_rtt.py --cache-dir "$CACHE" --skip-cache-build 2>&1 | grep -v "it/s\]" | tail -4
[[ -f "$CACHE/valid_combos_near_rtt_capped.pkl" ]] || { echo "FAIL LOUD: no near-RTT sidecar" >&2; exit 1; }
du -sh "$CACHE"
echo "=== D1 cache done ==="
