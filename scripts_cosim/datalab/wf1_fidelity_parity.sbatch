#!/bin/bash
# r1_attribution_v1: fidelity-aware train/serve parity (wf1_fidelity_parity.py) on a fidelity corpus built by
# wf1_fidelity_corpus_smoke.sbatch (or the production build): v5 cache under the R1 environment, an UNTRAINED v5 checkpoint, and
# the live scheduler's served graph for each dataset's batch decision compared with the cache graph.
#   sbatch --export=ALL,WT=<worktree>,FC=<fidelity_corpus dir holding gnn_datasets_wf1_fidelity> wf1_fidelity_parity.sbatch
#SBATCH --job-name=wf1fidp
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/wf1fidp-%j.out
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
set -euo pipefail
WT=${WT:?FAIL LOUD: set WT}
FC=${FC:?FAIL LOUD: set FC}
DS=${DS:-gnn_datasets_wf1_fidelity}
TEMPLATE=${TEMPLATE:-/home/nikola.lukic/gnn-herosim/models/load-repr-v1-v4load-gnnedge0-lr2e3-seed1.contract.json}
cd "$WT"
if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3 PYTHONPATH="$WT:$WT/src/notebooks" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=""
echo "HEAD $(git rev-parse HEAD)"
export HEROSIM_TRANSFER_MODEL=pipelined HEROSIM_REPLICA_RELEASE=1 HEROSIM_SCALEOUT=kpa HEROSIM_SHARED_AUTOSCALER=0 GATE_FIXED_POLICY_TIME_SCALE=1.0
export HEROSIM_PEER_EXCHANGE=1 HEROSIM_SERVER_ONLY_REPLICAS=1 HEROSIM_WARMTH_PHYSICS=node_disk_v2 PYTHONHASHSEED=0 HEROSIM_SNAPSHOT_FIDELITY=1
export SIM_FORCE_FULL_STATS=1 DAG_PRIMARY_ALPHA_KEY=inf PARTIAL_STATE_CONTRACT=partial_state_v5 PARTIAL_STATE_LOAD_SECONDS=1 PARTIAL_STATE_PEER_MASS=1
export NEAR_RTT_LABEL_BACKLOG_CLOCK=seeded GNN_PREFIX_ALPHA_KEY=inf GNN_SERVE_CANDIDATE_SLATE=declared_pruning_v1
unset GNN_DISABLE_MESSAGE_PASSING || true
OUT=$FC/parity_fidelity; rm -rf "$OUT"; mkdir -p "$OUT" models
python3 -u src/notebooks/prepare_graphs_cache.py --base-dirs "$FC/$DS" --cache-dir "$OUT/cache" --dag-partial-state --platform-feature-dim 14 > "$OUT/cache.log" 2>&1
python3 scripts_cosim/partial_state_v5_untrained_checkpoint.py --cache-dir "$OUT/cache" --template "$TEMPLATE" --out models/ps_v5_untrained_fid.pt
python3 scripts_cosim/wf1_fidelity_parity.py --datasets "$FC/$DS" --cache-dir "$OUT/cache" --checkpoint models/ps_v5_untrained_fid.pt --out "$OUT/parity.json"
