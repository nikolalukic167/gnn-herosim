#!/bin/bash
# r1_attribution_v1 PRODUCTION corpus, step 3 of 3 -- submit only on the coordinator's word, aftercorr on the capture array (see the run procedure below).
# make_warm_corpus --fidelity over the step-2 snapshots: every candidate plan replayed as i11_replay replays a live decision (queued
# tasks are state, label = sum over the batch of done - scheduled). HEROSIM_SNAPSHOT_FIDELITY=1 at build as at capture. One array task
# per topology. Train / heldout by topology (HELDOUT_TOPOS). The node's declared pruning (top 5 candidates per task by standalone cost,
# sub-batches of <= 4 tasks above 100,000 plans; src/placement/declared_slate.py) decides each batch's candidate sets. Single-candidate-node,
# disconnected, no-choice and no-peer-pair batches are rejected and counted in warm_summary_<tag>.json; a truncated or failed sweep is
# archived to <split>_discarded and counted. No synthetic backlog (the replayed queue is the state).
# Volume: BATCHES_TARGET (5,000) BATCHES over TOPOS_ALL x rungs x windows cells. A sub-batch counts toward its parent batch. Each cell
# offers every eligible snapshot in a spread order (wf1_shard_snapshots.py) and stops at its share, so a rejection is replaced by the
# next snapshot of the same spread; a cell whose snapshots run out is a shortfall and fails the task at the end. Snapshots before
# t = 360 s are dropped (I11 validated t >= 360 s).
# For 100+ topologies submit via wf1_corpus_prod_submit_builds.sh (pieces under the 50-job limit, check afterany on all pieces).
# Run procedure: chain the build with --dependency=aftercorr:<capture> and the SAME --array range, so topology t builds as soon as capture
# task t ends instead of waiting for the slowest topology. aftercorr needs the matching capture task to exit 0: a hung or failed CELL does
# not fail the task (it is a sentinel the build counts), a cap-hit cell does (exit 1). A capture task that fails leaves its build task
# DependencyNeverSatisfied; scancel it (squeue -u $USER -t PD -o "%i %r" | grep DependencyNeverSatisfied), because the afterany check on
# the whole build waits for it otherwise. The check (wf1_corpus_prod_check.sbatch) stays --dependency=afterany:<build>.
#   sbatch --dependency=aftercorr:<capture> --array=0-<n-1> --export=ALL,WT=...,ROOT=...,TOPOS="...",TAGS="light moderate heavy",HELDOUT_TOPOS="..." wf1_corpus_prod_build.sbatch
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
TARGET=${TARGET:-20000}; STRIDE=${STRIDE:-200}; MIN_TIME=${MIN_TIME:-360}
BATCHES_TARGET=${BATCHES_TARGET:-5000}; TOPOS_ALL=${TOPOS_ALL:-$TOPOS}
# the account allows 50 queued jobs (AssocMaxSubmitJobLimit), so a long TOPOS list is submitted in array chunks: chunk k is --array=0-<n-1> with TOPO_OFFSET=<first index>, and T is the index into the FULL TOPOS list
T=$(( ${SLURM_ARRAY_TASK_ID:?FAIL LOUD: submit as an array} + ${TOPO_OFFSET:-0} ))
TOPO_ARR=($TOPOS); SEED=${TOPO_ARR[$T]:?FAIL LOUD: array index beyond TOPOS}
cd "$WT"
for v in $(compgen -e | grep -E '^(HEROSIM_|GNN_|COSIM_|LIVE_AUDIT_|GATE_|SIM_|QUEUE_FEATURE|MAX_PLACEMENT)'); do unset "$v"; done
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
# disk guard: fail loudly before writing if the filesystem this task writes to, or the /home quota the job logs live on, has < GUARD_GB free
GUARD_GB=${GUARD_GB:-15}
free_gb() { case "$(readlink -f "$1")" in /home/*) echo $(( (268435456000 - $(getfattr --only-values -n ceph.dir.rbytes "$HOME" 2>/dev/null)) / 1000000000 ));;
  *) df -P -BG "$1" | awk 'NR==2{gsub("G","",$4); print $4}';; esac; }
for d in "$HOME" "$ROOT"; do
  g=$(free_gb "$d"); echo "free on $d: ${g} GB (guard ${GUARD_GB})"
  [[ "$g" -ge "$GUARD_GB" ]] || { echo "FAIL LOUD: only ${g} GB free for $d, below the ${GUARD_GB} GB guard"; exit 1; }
done
SPLIT=train; for h in $HELDOUT_TOPOS; do [[ "$h" == "$SEED" ]] && SPLIT=heldout; done
# OUT_TAG: a code variant's own datasets and build logs beside another variant's (gnn_datasets_wf1_<split>_<tag>, build_logs_<tag>)
OUT_TAG=${OUT_TAG:-}; BLOGS="$ROOT/build_logs${OUT_TAG:+_$OUT_TAG}"
OUT="$ROOT/gnn_datasets_wf1_${SPLIT}${OUT_TAG:+_$OUT_TAG}"; mkdir -p "$OUT" "$BLOGS"
NR=$(echo $TAGS | wc -w); NW=$(echo $WINDOWS | wc -w); NT=$(echo $TOPOS_ALL | wc -w)
PER_CELL=$(( (BATCHES_TARGET + NT * NR * NW - 1) / (NT * NR * NW) ))
echo "target $BATCHES_TARGET batches over $NT topologies x $NR rungs x $NW windows = $PER_CELL batches per cell"
# count check: the capture must have left one sentinel per cell of this topology
have=$(ls "$ROOT"/snapshots/cc40s${SEED}_*.done 2>/dev/null | wc -l) || true
[[ "$have" -eq $(( NR * NW )) ]] || { echo "FAIL LOUD: topology $SEED has $have capture sentinels, expected $(( NR * NW )) (capture unfinished or crashed)"; exit 1; }
ri=0; shortfall=0; cells_not_ok=""; cap_cells=""
for R in $TAGS; do
  wi=0
  for W in $WINDOWS; do
    CELL=cc40s${SEED}; SNAP="$ROOT/snapshots/${CELL}_${R}_${W}.jsonl"; DONE="$ROOT/snapshots/${CELL}_${R}_${W}.done"
    START=$(( ((T * NR + ri) * NW + wi) * STRIDE ))
    if grep -q '"cap_hit": true' "$DONE"; then cap_cells="$cap_cells ${R}_${W}"; echo "[$CELL $R $W] CAP HIT: the capture stopped at the snapshot cap, the late trace was never offered"; fi
    if ! grep -q '"status": "ok"' "$DONE"; then
      echo "[$CELL $R $W] capture $(cat "$DONE"); no corpus from this cell"; cells_not_ok="$cells_not_ok ${R}_${W}"; wi=$((wi+1)); continue
    fi
    python3 scripts_cosim/wf1_shard_snapshots.py --snapshots "$SNAP" --out "$ROOT/snapshots/shard_${CELL}_${R}_${W}.jsonl" --min-time "$MIN_TIME"
    python3 scripts_cosim/make_warm_corpus.py --manifest-per-source --snapshots "$ROOT/snapshots/shard_${CELL}_${R}_${W}.jsonl" \
      --trace "$ROOT/inputs/wf1_$R/wl/grounded_${W}_n50000.json" --cell-config "$ROOT/inputs/wf1_$R/cfg/$CELL.json" \
      --output-dir "$OUT" --source-tag "wf1p_${CELL}_${R}_${W}" --start-index "$START" --limit-batches "$PER_CELL" --require-limit-batches \
      --min-time "$MIN_TIME" --target-combos "$TARGET" \
      --workers $(( SLURM_CPUS_PER_TASK - 1 )) --no-cap-filter --min-choice-fraction 0.2 --quiet --variable-group-min 2 --fidelity --require-connected-batch \
      > "$BLOGS/${CELL}_${R}_${W}.log" 2>&1 || { echo "[$CELL $R $W] SHORTFALL or error (rc=$?), see build_logs"; shortfall=$((shortfall+1)); }
    wi=$((wi+1))
  done
  ri=$((ri+1))
done
echo "=== topology $SEED ($SPLIT): $(ls -d "$OUT"/ds_* 2>/dev/null | wc -l) datasets in $OUT (all sources of this split so far); capture-not-ok cells:${cells_not_ok:- none}; cap-hit cells:${cap_cells:- none}; shortfall cells: $shortfall"
echo "{\"topology\": $SEED, \"split\": \"$SPLIT\", \"cells_not_ok\": \"${cells_not_ok# }\", \"cap_hit_cells\": \"${cap_cells# }\", \"shortfall_cells\": $shortfall}" > "$BLOGS/topology_${SEED}.json"
[[ $shortfall -eq 0 ]] || exit 1
