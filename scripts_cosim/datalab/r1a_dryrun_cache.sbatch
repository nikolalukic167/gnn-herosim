#!/bin/bash
# r1_attribution_v1 downstream dry run, step 1: a v5 cache from the production pieces that are finished, split by topology, and the cache checks.
# Reads the production corpus through a symlink view (nothing is written into it); writes only under OUT.
#   sbatch --export=ALL,WT=<pinned worktree>,OUT=<new dir>[,MAX_INDEX=48000,CORPUS=...] r1a_dryrun_cache.sbatch
#SBATCH --job-name=r1a-dcache
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/r1a-dcache-%j.out
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=12:00:00
set -euo pipefail
WT=${WT:?}; OUT=${OUT:?}
CP=${CORPUS:-/share/nikola.lukic/workload_fix_v1/corpus_prod}
SUF=${SUF:-E8894a08}
MAX_INDEX=${MAX_INDEX:-48000}   # datasets of the first 20 topologies (index = ((T*3+rung)*4+window)*200 + k): the finished piece
cd "$WT"
for v in $(compgen -e | grep -E '^(HEROSIM_|GNN_|COSIM_|LIVE_AUDIT_|GATE_|SIM_|QUEUE_FEATURE|MAX_PLACEMENT|NEAR_RTT_|PARTIAL_STATE)'); do unset "$v"; done
if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3 PYTHONPATH="$WT:$WT/src/notebooks" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
echo "HEAD $(git rev-parse HEAD) node $(hostname)"
[[ ! -e "$OUT" ]] || { echo "FAIL LOUD: $OUT exists"; exit 1; }
mkdir -p "$OUT/view"
export HEROSIM_TRANSFER_MODEL=pipelined HEROSIM_REPLICA_RELEASE=1 HEROSIM_SCALEOUT=kpa HEROSIM_SHARED_AUTOSCALER=0 GATE_FIXED_POLICY_TIME_SCALE=1.0
export DAG_PRIMARY_ALPHA_KEY=inf PARTIAL_STATE_CONTRACT=partial_state_v5 NEAR_RTT_LABEL_BACKLOG_CLOCK=seeded
python3 - "$CP" "$SUF" "$OUT/view" "$MAX_INDEX" <<'PY'
import os, shutil, sys
from pathlib import Path
cp, suf, view, mx = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), int(sys.argv[4])
for split in ("train", "heldout"):
    src = cp / f"gnn_datasets_wf1_{split}_{suf}"
    dst = view / src.name; dst.mkdir()
    n = 0
    for d in sorted(src.glob("ds_*")):
        if int(d.name[3:]) < mx:
            # a real directory of symlinks: refresh_optimal_full_stats --rewrite-ssc rewrites optimal_result.json and writes the SSC file, so
            # those two are private copies and nothing is ever written through a link into the production corpus
            t = dst / d.name; t.mkdir()
            for f in d.iterdir():
                if f.name == "optimal_result.json":
                    shutil.copyfile(f, t / f.name)
                else:
                    os.symlink(f, t / f.name)
            n += 1
    print(f"view {dst.name}: {n} of {len(list(src.glob('ds_*')))} datasets (index < {mx})")
PY
S0=$(date +%s)
for B in "$OUT"/view/*/; do [[ -n "$(ls -A "$B")" ]] && python3 scripts_cosim/refresh_optimal_full_stats.py --base-dir "$B" --rewrite-ssc | tail -1; done
echo "ssc rewrite wall seconds: $(( $(date +%s) - S0 ))"
BASES=$(for B in "$OUT"/view/*/; do [[ -n "$(ls -A "$B")" ]] && echo -n "$B "; done)
N=$(find "$OUT"/view -mindepth 2 -maxdepth 2 | wc -l)
echo "=== cache build over $N datasets"
START=$(date +%s)
python3 -u src/notebooks/prepare_graphs_cache.py --base-dirs $BASES --cache-dir "$OUT/cache" --dag-partial-state --platform-feature-dim 14 > "$OUT/cache.log" 2>&1 || { tail -30 "$OUT/cache.log"; echo "FAIL LOUD: cache build failed"; exit 1; }
END=$(date +%s)
echo "cache build wall seconds: $((END-START)) for $N linked datasets"
grep -vE "it/s\]|^\s*$" "$OUT/cache.log" | tail -25 | cut -c1-250
python3 scripts_cosim/make_r1a_split.py --cache-dir "$OUT/cache" --corpus-split "$CP/split.json" --out "$OUT/split.json" --dry-run-test 3
python3 - "$OUT" "$START" "$END" <<'PY'
import json, pickle, sys
from pathlib import Path
import torch
out = Path(sys.argv[1]); cache = out / "cache"
m = json.loads((cache / "metadata.json").read_text()); g = pickle.load(open(cache / "graphs.pkl", "rb"))
ids = pickle.load(open(cache / "dataset_ids.pkl", "rb"))
print("physics_env", m.get("physics_env")); print("candidate_slate", m.get("candidate_slate"), "| contract", m.get("partial_state_contract"),
      "| alpha", m.get("dag_primary_alpha_key"), "| platform_feature_dim", m.get("platform_feature_dim"), "| peer_exchange_block", m.get("peer_exchange_block"))
assert m["physics_env"].get("inflight_capture") == "legacy", "FAIL LOUD: inflight_capture is not legacy"
assert m["candidate_slate"] == "declared_pruning_v1" and m["partial_state_contract"] == "partial_state_v5"
print("graphs", len(g), "datasets in metadata", m.get("num_datasets"), "dataset_ids", len(ids))
x = g[0]
print("widths: task", tuple(x.task_features.shape), "platform", tuple(x.platform_features.shape), "edge_attr", tuple(x.edge_attr.shape),
      "onehot4", tuple(x.task_type_onehot4.shape), "peer_edge_index", tuple(x.peer_edge_index.shape))
ok_flags = all(set(torch.unique(h.platform_features[:, -2:]).tolist()) <= {0.0, 1.0} for h in g[:500])
print("exposure flags (platform last two columns are 0/1) on first 500 graphs:", ok_flags)
print("tied_optimal_logit_plans alphas:", sorted(x.tied_optimal_logit_plans) if hasattr(x, "tied_optimal_logit_plans") else None)
print("partial_state_ctx keys:", sorted(x.partial_state_ctx)[:30])
PY
echo "=== done; outputs in $OUT"
