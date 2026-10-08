#!/bin/bash
# partial_state_v5 train/serve parity on WF1 smoke datasets (r1_attribution_v1 prep, not a production run).
# Input: the datasets of wf1_corpus_smoke_b.sbatch (route B: WF1 inputs -> capture -> make_warm_corpus). Steps: build the
# v5 cache under the R1 environment, write an UNTRAINED v5 checkpoint, replay each dataset through the live serving path
# and compare it with the cache graph attribute by attribute (peer_affinity_live_serve_check.py), then check that the
# only differences are the documented peer_norm of single-candidate-node datasets.
#   sbatch --export=ALL,WT=$HOME/gnn-herosim-wt/r1a,CORP=<dir holding gnn_datasets_wf1_smoke_b>,TEMPLATE=<v4 .contract.json> ...
#SBATCH --job-name=ps5par
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/ps5par-%j.out
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
set -euo pipefail
WT=${WT:?FAIL LOUD: set WT}
CORP=${CORP:?FAIL LOUD: set CORP (the directory that holds gnn_datasets_wf1_smoke_b)}
TEMPLATE=${TEMPLATE:-/home/nikola.lukic/gnn-herosim/models/load-repr-v1-v4load-gnnedge0-lr2e3-seed1.contract.json}
DATASETS=${DATASETS:-gnn_datasets_wf1_smoke_b}
cd "$WT"
if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3 PYTHONPATH="$WT:$WT/src/notebooks" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=""
echo "HEAD $(git rev-parse HEAD)"
export HEROSIM_TRANSFER_MODEL=pipelined HEROSIM_REPLICA_RELEASE=1 HEROSIM_SCALEOUT=kpa HEROSIM_SHARED_AUTOSCALER=0
export HEROSIM_PEER_EXCHANGE=1 DAG_PRIMARY_ALPHA_KEY=inf PARTIAL_STATE_CONTRACT=partial_state_v5 NEAR_RTT_LABEL_BACKLOG_CLOCK=seeded
OUT=$CORP/parity_v5; rm -rf "$OUT"; mkdir -p "$OUT" simulation_data models
ln -sfn "$CORP/$DATASETS" "simulation_data/$DATASETS"
python3 -u src/notebooks/prepare_graphs_cache.py --base-dirs "$CORP/$DATASETS" --cache-dir "$OUT/cache" --dag-partial-state --platform-feature-dim 14 > "$OUT/cache.log" 2>&1
python3 -c "
import json,pickle
ids=pickle.load(open('$OUT/cache/dataset_ids.pkl','rb')); json.dump({'test':ids},open('$OUT/split.json','w'))"
python3 scripts_cosim/partial_state_v5_untrained_checkpoint.py --cache-dir "$OUT/cache" --template "$TEMPLATE" --out models/ps_v5_untrained_seed0.pt
export PARTIAL_STATE_LOAD_SECONDS=1 PARTIAL_STATE_PEER_MASS=1 GNN_PREFIX_ALPHA_KEY=inf
unset GNN_DISABLE_MESSAGE_PASSING
set +e
python3 scripts_cosim/peer_affinity_live_serve_check.py --checkpoint models/ps_v5_untrained_seed0.pt --cache-dir "$OUT/cache" \
  --split-artifact "$OUT/split.json" --split test --workers 3 --output "$OUT/servecheck.json" > "$OUT/servecheck.log" 2>&1
set -e
python3 - "$OUT/servecheck.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
bad = [(r["dataset_id"], m) for r in d["per_dataset"] for m in r["mismatches"] if not m.startswith("partial_state_ctx.peer_norm: live 1.0 != cache 0.0")]
print(f"datasets {d['n_datasets']}, bit-identical {d['n_clean']}, unexpected mismatches {len(bad)}")
for ds, m in bad:
    print("  ", ds, m[:240])
sys.exit(1 if bad else 0)
PY
