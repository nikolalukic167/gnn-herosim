#!/bin/bash
# live_headroom_v1 -- the graph cache on the CD-induced live-snapshot corpus (docs/lineages/live_headroom_v1.md): backlog_corpus_v1's
# cache recipe (partial_state_v4, alpha inf, rtt_drift:1 at 0.46/s on the measured drain table), one corpus, single-node groups
# quarantined, and a diagnostic-only split (val = every parent but two placeholders; nothing is trained on this cache).
#   sbatch --export=ALL,WT=/home/nikola.lukic/gnn-herosim-wt/live_headroom scripts_cosim/datalab/live_headroom_v1_cache.sbatch
#SBATCH --job-name=lh1-cache
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/lh1-cache-%j.out
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=06:00:00
set -euo pipefail
WT=${WT:-/home/nikola.lukic/gnn-herosim-wt/live_headroom}
cd "$WT"
if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3 PYTHONPATH="$WT:$WT/src/notebooks" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
export DAG_PRIMARY_ALPHA_KEY=inf PARTIAL_STATE_CONTRACT=partial_state_v4 NEAR_RTT_LABEL_BACKLOG_CLOCK=seeded
echo "[CODE] $(git describe --dirty --always) at $WT"
[[ -z "$(git status --porcelain -- src scripts_cosim | grep -v '^??' || true)" ]] || { echo "FAIL LOUD: worktree dirty" >&2; exit 1; }
ln -sfn live_headroom_v1/gnn_datasets_live_headroom_v1 simulation_data/gnn_datasets_live_headroom_v1_train
TRAIN=simulation_data/gnn_datasets_live_headroom_v1_train
CACHE=simulation_data/graphs_cache_live_headroom_v1_psv4_inf
SPLIT=experiments/live_headroom_v1_split.json
DRAIN=simulation_data/drainable_objective_v1/drain_table.json
[[ -f "$DRAIN" ]] || { echo "FAIL LOUD: drain table missing"; exit 1; }
python3 - "$TRAIN" <<'PY'
import json, os, sys
d = sys.argv[1]; ok = 0; bad = []
for ds in sorted(x for x in os.listdir(d) if x.startswith("ds_")):
    try:
        m = json.load(open(os.path.join(d, ds, "placement_metadata.json")))
        rows = sum(1 for _ in open(os.path.join(d, ds, "placements", "placements.jsonl")))
        good = m.get("sweep_complete") is True and rows == m["num_placements"] and os.path.getsize(os.path.join(d, ds, "best.json")) > 0
    except Exception:
        good = False
    if good: ok += 1
    else: bad.append(ds)
print(f"[corpus] {ok} complete sweeps, {len(bad)} incomplete")
if bad: raise SystemExit(f"FAIL LOUD: incomplete datasets {bad[:10]}")
PY
python3 scripts_cosim/refresh_optimal_full_stats.py --base-dir "$TRAIN" --rewrite-ssc | tail -2
build_cache() {
rm -rf "$CACHE"
python3 -u src/notebooks/prepare_graphs_cache.py --base-dirs "$TRAIN" --cache-dir "$CACHE" \
  --dag-partial-state --platform-feature-dim 14 \
  --label-objective "rtt_drift:1" --label-arrival-rate 0.46 --label-drain-table "$DRAIN" 2>&1 | grep -v "it/s\]" | tail -12
}
build_cache
python3 - "$CACHE" simulation_data/live_headroom_v1/quarantine_single_node <<'PY'
import os, pickle, shutil, sys
cache, q = sys.argv[1:3]
graphs = pickle.load(open(cache + "/graphs.pkl", "rb")); ids = pickle.load(open(cache + "/dataset_ids.pkl", "rb"))
bad = sorted({i.split("@seq", 1)[0] for g, i in zip(graphs, ids) if float(g.partial_state_ctx["peer_norm"]) <= 0.0})
os.makedirs(q, exist_ok=True)
for i in bad:
    shutil.move(os.path.join("simulation_data", i), os.path.join(q, i.replace("/", "__")))
open(os.path.join(q, "list.txt"), "w").write("\n".join(bad) + "\n")
print(f"[quarantine] {len(bad)} of {len(graphs)} datasets have peer_norm 0 and moved to {q}")
open("/tmp/lh1_quarantined", "w").write(str(len(bad)))
PY
if [[ "$(cat /tmp/lh1_quarantined)" != "0" ]]; then build_cache; fi
python3 -u src/notebooks/prepare_graphs_cache_near_rtt.py --cache-dir "$CACHE" --skip-cache-build 2>&1 | grep -v "it/s\]" | tail -3
python3 - "$CACHE" "$SPLIT" <<'PY'
import json, pickle, sys
sys.path.insert(0, "src/notebooks")
from non_unique_lib.training_contract import SPLIT_ARTIFACT_SCHEMA, canonical_parent_id
cache, out = sys.argv[1:3]
m = json.load(open(cache + "/metadata.json"))
assert m.get("partial_state_contract") == "partial_state_v4" and m.get("dag_primary_alpha_key") == "inf"
ids = sorted({canonical_parent_id(i) for i in pickle.load(open(cache + "/dataset_ids.pkl", "rb"))})
payload = {"schema": SPLIT_ARTIFACT_SCHEMA, "cache_dir": cache, "n_parents": len(ids), "random_state": 0,
           "heldout_corpus": "none", "val_fraction": 1.0, "train": ids[:1], "test": ids[1:2], "val": ids[2:],
           "note": "diagnostic only: train and test are one placeholder parent each; nothing is trained on this cache"}
json.dump(payload, open(out, "w"), indent=1)
print(f"[split] {len(ids)} parents, val {len(ids) - 2}")
PY
sha256sum "$SPLIT"
echo "=== cache done ==="
