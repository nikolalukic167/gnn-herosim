#!/bin/bash
# r1_attribution_v1: two builds of the same snapshots must be bit-identical (best.json, optimal_result.json, placements, cache tensors).
#   sbatch --export=ALL,WT=<worktree>,SNAPS=<snapshot jsonl>,TRACE=<trace>,CELLCFG=<cell cfg>,OUT=<new dir> r1a_determinism.sbatch
#SBATCH --job-name=r1aud-det
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/r1aud-det-%j.out
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=02:00:00
set -euo pipefail
WT=${WT:?}; SNAPS=${SNAPS:?}; TRACE=${TRACE:?}; CELLCFG=${CELLCFG:?}; OUT=${OUT:?}
BATCHES=${BATCHES:-2}
cd "$WT"
for v in $(compgen -e | grep -E '^(HEROSIM_|GNN_|COSIM_|LIVE_AUDIT_|GATE_|SIM_|QUEUE_FEATURE|MAX_PLACEMENT)'); do unset "$v"; done
if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3 PYTHONPATH="$WT:$WT/src/notebooks" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
echo "HEAD $(git rev-parse HEAD)"
[[ ! -e "$OUT" ]] || { echo "FAIL LOUD: $OUT exists"; exit 1; }
mkdir -p "$OUT"
export HEROSIM_TRANSFER_MODEL=pipelined HEROSIM_REPLICA_RELEASE=1 HEROSIM_SCALEOUT=kpa HEROSIM_SHARED_AUTOSCALER=0 GATE_FIXED_POLICY_TIME_SCALE=1.0
export HEROSIM_PEER_EXCHANGE=1 HEROSIM_SERVER_ONLY_REPLICAS=1 HEROSIM_WARMTH_PHYSICS=node_disk_v2 PYTHONHASHSEED=0 HEROSIM_SNAPSHOT_FIDELITY=1
export MAX_PLACEMENT_COMBINATIONS_SKIP=2000000 GNN_CAPTURE_DATASET_STATE=0 COSIM_SUPPRESS_SIM_PRINTS=1 HEROSIM_RETAIN_TASK_TIMES=1 HEROSIM_RETAIN_PEER_STATS=1
export COSIM_PLACEMENT_TIMEOUT_S=120
python3 scripts_cosim/wf1_shard_snapshots.py --snapshots "$SNAPS" --out "$OUT/shard.jsonl" --min-time 360
for X in A B; do
  python3 scripts_cosim/make_warm_corpus.py --snapshots "$OUT/shard.jsonl" --trace "$TRACE" --cell-config "$CELLCFG" \
    --output-dir "$OUT/$X/gnn_datasets" --source-tag det --limit-batches "$BATCHES" --require-limit-batches --min-time 360 --target-combos 20000 \
    --workers 15 --no-cap-filter --min-choice-fraction 0.2 --quiet --variable-group-min 2 --fidelity --require-connected-batch &
done
wait
# resume: the same cell resubmitted into the same directory must build nothing new
before=$(ls -d "$OUT/A/gnn_datasets"/ds_* | wc -l)
python3 scripts_cosim/make_warm_corpus.py --snapshots "$OUT/shard.jsonl" --trace "$TRACE" --cell-config "$CELLCFG" \
  --output-dir "$OUT/A/gnn_datasets" --source-tag det --limit-batches "$BATCHES" --require-limit-batches --min-time 360 --target-combos 20000 \
  --workers 15 --no-cap-filter --min-choice-fraction 0.2 --quiet --variable-group-min 2 --fidelity --require-connected-batch
after=$(ls -d "$OUT/A/gnn_datasets"/ds_* | wc -l)
echo "resume: datasets before $before after $after"
[[ "$before" -eq "$after" ]] || { echo "FAIL LOUD: a resumed cell built $((after-before)) new dataset(s)"; exit 1; }
export DAG_PRIMARY_ALPHA_KEY=inf PARTIAL_STATE_CONTRACT=partial_state_v5 NEAR_RTT_LABEL_BACKLOG_CLOCK=seeded
for X in A B; do
  python3 scripts_cosim/refresh_optimal_full_stats.py --base-dir "$OUT/$X/gnn_datasets" --rewrite-ssc | tail -1
  python3 -u src/notebooks/prepare_graphs_cache.py --base-dirs "$OUT/$X/gnn_datasets" --cache-dir "$OUT/$X/cache" --dag-partial-state --platform-feature-dim 14 2>&1 | grep -v "it/s\]" | tail -3
done
python3 - "$OUT" <<'PY'
import hashlib, json, pickle, sys
from pathlib import Path
import torch
o = Path(sys.argv[1]); bad = 0
def h(p): return hashlib.sha1(Path(p).read_bytes()).hexdigest()
for a in sorted((o / "A/gnn_datasets").glob("ds_*")):
    b = o / "B/gnn_datasets" / a.name
    ra = sorted(l for l in open(a / "placements/placements.jsonl") if l.strip()); rb = sorted(l for l in open(b / "placements/placements.jsonl") if l.strip())
    same = ra == rb; bad += not same
    print(a.name, "placements (as a set of rows; file order is completion order)", "identical" if same else "DIFFERS")
    same = h(a / "best.json") == h(b / "best.json"); bad += not same
    print(a.name, "best.json", "identical" if same else "DIFFERS")
    for f in ("optimal_result.json", "system_state_captured_unique.json"):  # informational: wall-clock schedulingTime and set-order lists
        print(a.name, f, "byte-identical" if h(a / f) == h(b / f) else "bytes differ (informational; the cache tensors below are the criterion)")
ga = pickle.load(open(o / "A/cache/graphs.pkl", "rb")); gb = pickle.load(open(o / "B/cache/graphs.pkl", "rb"))
for i, (x, y) in enumerate(zip(ga, gb)):
    for k in ("task_features", "platform_features", "edge_index", "edge_attr", "y"):
        if hasattr(x, k):
            same = torch.equal(getattr(x, k), getattr(y, k)); bad += not same
            print("graph", i, k, "identical" if same else "DIFFERS")
ok = not bad and len(ga) == len(gb)
print("graphs", len(ga), len(gb), "DIFFERENCES:", bad)
(o / "determinism_result.json").write_text(json.dumps({"passed": ok, "resume_new_datasets": 0, "graphs": [len(ga), len(gb)], "differences": bad}))
sys.exit(0 if ok else 1)
PY
