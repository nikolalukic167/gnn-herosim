#!/bin/bash
# r1_attribution_v1 PRODUCTION corpus, step 3 of 3 -- PREPARED, NOT SUBMITTED (rungs come from S4).
# make_warm_corpus --fidelity over the step-2 snapshots: every candidate plan replayed as i11_replay replays a live decision (queued
# tasks are state, label = sum over the batch of done - scheduled). HEROSIM_SNAPSHOT_FIDELITY=1 at build as at capture. One array task per
# training topology. Train / heldout by topology (HELDOUT_TOPOS). Single-candidate-node batches are rejected and counted in
# warm_summary_<tag>.json; no synthetic backlog (the replayed queue is the state).
#   NOT IMPLEMENTED HERE: the node's declared candidate pruning (top 5 per task by standalone cost, sub-batches <= 4 tasks above
#   100,000 plans). make_warm_corpus draws its slate with choose_candidates (balanced subset, <= TARGET plans); see the hand-off.
#   sbatch --array=0-<n-1> --export=ALL,WT=...,ROOT=...,TOPOS="...",TAGS="lo hi",HELDOUT_TOPOS="..." wf1_corpus_prod_build.sbatch
#SBATCH --job-name=wf1p-bld
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/wf1p-bld-%A_%a.out
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=24:00:00
set -euo pipefail
WT=${WT:?FAIL LOUD: set WT (pinned worktree)}
ROOT=${ROOT:?FAIL LOUD: set ROOT}
TOPOS=${TOPOS:?FAIL LOUD: set TOPOS}
TAGS=${TAGS:?FAIL LOUD: set TAGS}
HELDOUT_TOPOS=${HELDOUT_TOPOS:-}
WINDOWS=${WINDOWS:-"g0 g1 g2 g3"}
LIMIT=${LIMIT:-90}; TARGET=${TARGET:-20000}; STRIDE=${STRIDE:-200}
T=${SLURM_ARRAY_TASK_ID:?FAIL LOUD: submit as an array}
TOPO_ARR=($TOPOS); SEED=${TOPO_ARR[$T]:?FAIL LOUD: array index beyond TOPOS}
cd "$WT"
if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3 PYTHONPATH="$WT" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
echo "HEAD $(git rev-parse HEAD) topology $SEED node $(hostname)"
export HEROSIM_TRANSFER_MODEL=pipelined HEROSIM_REPLICA_RELEASE=1 HEROSIM_SCALEOUT=kpa HEROSIM_SHARED_AUTOSCALER=0 GATE_FIXED_POLICY_TIME_SCALE=1.0
export HEROSIM_PEER_EXCHANGE=1 HEROSIM_SERVER_ONLY_REPLICAS=1 HEROSIM_WARMTH_PHYSICS=node_disk_v2 PYTHONHASHSEED=0
export HEROSIM_SNAPSHOT_FIDELITY=1
export MAX_PLACEMENT_COMBINATIONS_SKIP=2000000 GNN_CAPTURE_DATASET_STATE=0 COSIM_SUPPRESS_SIM_PRINTS=1 HEROSIM_RETAIN_TASK_TIMES=1 HEROSIM_RETAIN_PEER_STATS=1
export COSIM_PLACEMENT_TIMEOUT_S=${COSIM_PLACEMENT_TIMEOUT_S:-120}
SPLIT=train; for h in $HELDOUT_TOPOS; do [[ "$h" == "$SEED" ]] && SPLIT=heldout; done
OUT="$ROOT/gnn_datasets_wf1_$SPLIT"; mkdir -p "$OUT"
NR=$(echo $TAGS | wc -w); NW=$(echo $WINDOWS | wc -w)
ri=0
for R in $TAGS; do
  wi=0
  for W in $WINDOWS; do
    CELL=cc40s${SEED}; SNAP="$ROOT/snapshots/${CELL}_${R}_${W}.jsonl"
    START=$(( ((T * NR + ri) * NW + wi) * STRIDE ))
    if [[ ! -s "$SNAP" ]]; then echo "[$CELL $R $W] no snapshots (capture hung or failed); skipped"; wi=$((wi+1)); continue; fi
    python3 - "$SNAP" "$ROOT/snapshots/shard_${CELL}_${R}_${W}.jsonl" "$(( LIMIT * 3 / 2 ))" <<'PY'
import json, sys
from scripts_cosim.make_warm_corpus import SnapshotRejected, check_consecutive_batch
src, dst, want = sys.argv[1], sys.argv[2], int(sys.argv[3])
ok = []
for line in open(src):
    if line.strip():
        try: check_consecutive_batch(json.loads(line), 2); ok.append(line)
        except SnapshotRejected: pass
step = max(1, len(ok) // want)
open(dst, "w").writelines(ok[::step][:want])
print(f"[shard] {len(ok)} eligible snapshots, offering {len(ok[::step][:want])}")
PY
    python3 scripts_cosim/make_warm_corpus.py --snapshots "$ROOT/snapshots/shard_${CELL}_${R}_${W}.jsonl" \
      --trace "$ROOT/inputs/wf1_$R/wl/grounded_${W}_n50000.json" --cell-config "$ROOT/inputs/wf1_$R/cfg/$CELL.json" \
      --output-dir "$OUT" --source-tag "wf1p_${CELL}_${R}_${W}" --start-index "$START" --limit "$LIMIT" --target-combos "$TARGET" \
      --workers $(( SLURM_CPUS_PER_TASK - 1 )) --no-cap-filter --min-choice-fraction 0.2 --quiet --variable-group-min 2 --fidelity
    wi=$((wi+1))
  done
  ri=$((ri+1))
done
echo "=== topology $SEED ($SPLIT): $(ls -d "$OUT"/ds_* 2>/dev/null | wc -l) datasets in $OUT (all sources of this split so far)"
