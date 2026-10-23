#!/bin/bash
# live_link_features_v1: the full v7 cache build and the live-vs-cache check on the FIRST dataset of the recaptured corpus, before any training. A one-dataset mirror (SSC rewritten), its v7 cache, then
# wf1_fidelity_parity.py: the dataset's fidelity snapshot is replayed under the real GNN scheduler with its stored flows restored (HEROSIM_REPLAY_STORED_FLOWS=1, option D's exchange physics), the batch is
# decided by a v7 checkpoint through the live builder, and the served graph (platform columns incl. P1-P3, link_counts, link_routes, every context ingredient) is compared with the cache graph.
#   sbatch --export=ALL,PIN=<sha8>,CORPUS=<dir>,ARM=<arm name>,DS=ds_00000,CKPT=<v7 .pt with its .contract.json> v7_first_check.sbatch
#SBATCH --job-name=link-first
#SBATCH --output=/share/nikola.lukic/uncut_groups/flowcap/link_corpus/logs/%x-%j.out
#SBATCH --partition=CPU-amd
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=02:00:00
#SBATCH --exclude=os-cpu-slurm-[21,15]
set -euo pipefail
PIN=${PIN:?}; CORPUS=${CORPUS:?}; ARM=${ARM:?}; DS=${DS:-ds_00000}; CKPT=${CKPT:?}   # DS: one dataset name or a space-separated list
WT=$HOME/gnn-herosim-wt/v7corpus_${PIN}; cd "$WT"
for v in $(compgen -e | grep -E '^(HEROSIM_|GNN_|COSIM_|LIVE_AUDIT_|GATE_|SIM_|QUEUE_FEATURE|MAX_PLACEMENT|NEAR_RTT_|PARTIAL_STATE)'); do unset "$v"; done
eval "$(micromamba shell hook --shell bash)"; micromamba activate gnn
export HEROSIM_PY=python3 PYTHONPATH="$WT:$WT/src/notebooks" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=""
[[ "$(git rev-parse HEAD)" == ${PIN}* ]] || { echo "FAIL LOUD: HEAD $(git rev-parse HEAD)"; exit 1; }
[[ -f "$CKPT" && -f "${CKPT%.pt}.contract.json" ]] || { echo "FAIL LOUD: $CKPT or its sidecar missing"; exit 1; }
L=/share/nikola.lukic/uncut_groups/flowcap/link_corpus; F=$L/first_check_$SLURM_JOB_ID; B=$F/gnn_datasets_wf1_$ARM
mkdir -p "$B"
python3 - "$CORPUS/$ARM" "$B" $DS <<'PY'
import os, shutil, sys
from pathlib import Path
src_root, dst_root = Path(sys.argv[1]), Path(sys.argv[2])
for name in sys.argv[3:]:
    src, dst = src_root / name, dst_root / name
    dst.mkdir()
    for f in src.iterdir():
        if f.name == "optimal_result.json":
            shutil.copyfile(f, dst / f.name)
        else:
            os.symlink(f.resolve(), dst / f.name)
PY
export HEROSIM_TRANSFER_MODEL=pipelined HEROSIM_REPLICA_RELEASE=1 HEROSIM_SCALEOUT=kpa HEROSIM_SHARED_AUTOSCALER=0 GATE_FIXED_POLICY_TIME_SCALE=1.0
python3 scripts_cosim/refresh_optimal_full_stats.py --base-dir "$B" --rewrite-ssc | tail -1
export HEROSIM_PEER_EXCHANGE=1 HEROSIM_SERVER_ONLY_REPLICAS=1 HEROSIM_WARMTH_PHYSICS=node_disk_v2 PYTHONHASHSEED=0 HEROSIM_SNAPSHOT_FIDELITY=1
export HEROSIM_EXCHANGE_FAIRSHARE=1 HEROSIM_EXCHANGE_TCP=1 HEROSIM_REPLAY_STORED_FLOWS=1
export SIM_FORCE_FULL_STATS=1 DAG_PRIMARY_ALPHA_KEY=inf PARTIAL_STATE_CONTRACT=partial_state_v7 PARTIAL_STATE_LOAD_SECONDS=1 PARTIAL_STATE_PEER_MASS=1 PARTIAL_STATE_EXCHANGE_SECONDS=1
export NEAR_RTT_LABEL_BACKLOG_CLOCK=seeded GNN_PREFIX_ALPHA_KEY=inf GNN_SERVE_CANDIDATE_SLATE=declared_pruning_v1_uncapped R1A_SLATE=declared_pruning_v1_uncapped PARITY_SERVE_SLATE=declared_pruning_v1_uncapped
unset GNN_DISABLE_MESSAGE_PASSING || true
OUT=$F/parity; mkdir -p "$OUT"
python3 -u src/notebooks/prepare_graphs_cache.py --base-dirs "$B" --cache-dir "$OUT/cache" --dag-partial-state --platform-feature-dim 14 > "$OUT/cache.log" 2>&1 || { tail -30 "$OUT/cache.log"; echo "FAIL LOUD: cache build failed"; exit 1; }
python3 -c "
import pickle,json
g=pickle.load(open('$OUT/cache/graphs.pkl','rb'))
for x in g:
    print('cache graph:', int(x.n_tasks), 'tasks, platform width', int(x.platform_features.size(-1)), ', link columns nonzero on', int((x.platform_features[:, -3:] != 0).any(1).sum()), 'of', int(x.platform_features.size(0)), 'platforms; link_counts', len(x.partial_state_ctx['link_counts']), 'links, flow-hops', int(sum(x.partial_state_ctx['link_counts'].values())), '; link_routes', len(x.partial_state_ctx['link_routes']), 'pairs')
    assert int(x.platform_features.size(-1)) == 21"
python3 scripts_cosim/wf1_fidelity_parity.py --datasets "$B" --cache-dir "$OUT/cache" --checkpoint "$CKPT" --out "$OUT/parity.json"
python3 - "$OUT/parity.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
bad = [r for r in d["per_dataset"] if r["mismatches"]]
print(f"parity: {d['clean']}/{d['n']} clean")
for r in bad:
    print("MISMATCH", r["dataset"], r["mismatches"][:8])
sys.exit(1 if bad else 0)
PY
echo "=== first-dataset check passed; report $OUT/parity.json"
