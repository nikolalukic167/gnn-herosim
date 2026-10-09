#!/bin/bash
# Scale probe (2026-10-09, descriptive, coordinator-approved: "one or two cheap CPU jobs"). One array task = one (clients, servers, p,
# seed) cell: mint WF1 inputs at two load rungs (moderate / heavy multipliers scaled by servers/6), spread the single-origin windows
# over the clients, capture peer-group snapshots live (same capture policy and environment as wf1_corpus_prod_capture.sbatch), build
# a small fidelity corpus at the declared slate (top 5) and at a wider slate (top 10, probe hook DECLARED_SLATE_TOP_K), then run the
# joint-headroom screen and the candidate stats on both. Everything is written under OUT=/share/...; nothing in the repo or in
# the production corpus is touched. The load is NOT calibrated: the achieved queue share is reported with the cell.
#   SIZES="40:6:0.6:9101 80:12:0.6:9101 ..." sbatch --array=0-<n-1> --export=ALL,WT=...,OUT=...,SIZES="..." scale_probe.sbatch
#SBATCH --job-name=scaleprobe
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/scaleprobe-%A_%a.out
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=10:00:00
set -euo pipefail
WT=${WT:?FAIL LOUD: set WT}; OUT=${OUT:?FAIL LOUD: set OUT}; SIZES=${SIZES:?FAIL LOUD: set SIZES}
T=${SLURM_ARRAY_TASK_ID:?FAIL LOUD: submit as an array}
ARR=($SIZES); SPEC=${ARR[$T]:?FAIL LOUD: array index beyond SIZES}
IFS=: read -r NC NS P SEED <<< "$SPEC"
RUNGS=${RUNGS:-"moderate=5.9402 heavy=11.6139"}; WINDOWS=${WINDOWS:-"g0 g1"}; BATCHES=${BATCHES:-12}
X1=/home/nikola.lukic/gnn-herosim/simulation_data/small_batch_confirm_v1/inputs/grounded/wl
BASECFG=/home/nikola.lukic/gnn-herosim/simulation_data/workload_fix_v1/corpus_prod_prov/cfg_src/cc40s9101.json
cd "$WT"
for v in $(compgen -e | grep -E '^(HEROSIM_|GNN_|COSIM_|LIVE_AUDIT_|GATE_|SIM_|QUEUE_FEATURE|MAX_PLACEMENT|DECLARED_SLATE)'); do unset "$v"; done
if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3 PYTHONPATH="$WT" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
CELL=c${NC}s${NS}p${P}s${SEED}; ROOT="$OUT/$CELL"; mkdir -p "$ROOT"
echo "HEAD $(git rev-parse HEAD) cell $CELL node $(hostname) $(date -Is)"

# 1. inputs: one cfg (counts, p, seed changed), rung multipliers scaled by servers/6, W2 payloads + W4 mix + single origin, W3 classes
if [[ ! -f "$ROOT/inputs.done" ]]; then
  rm -rf "$ROOT/cfg_src" "$ROOT/_b_"* "$ROOT/inputs_w24" "$ROOT/inputs"
  python3 scripts_cosim/scale_probe_cfg.py "$BASECFG" "$ROOT/cfg_src" --clients "$NC" --servers "$NS" --p "$P" --seed "$SEED"
  TAGS=""
  for spec in $RUNGS; do
    tag=${spec%%=*}; m=${spec##*=}
    factor=$(python3 -c "print(repr(6.0/($m*$NS)))")
    python3 scripts_cosim/workload_fix_v1_build.py --grounded-wl "$X1" --cfg-dir "$ROOT/cfg_src" --topologies "$SEED" --rung "$tag=$factor" \
      --payload-sampler wf1_v1 --task-mix wf1_v1 --batch-timeout-fixed 1 --out "$ROOT/_b_$tag"
    mkdir -p "$ROOT/inputs_w24"; mv "$ROOT/_b_$tag/wf1_$tag" "$ROOT/inputs_w24/wf1_$tag"
    TAGS="$TAGS $tag"
  done
  python3 scripts_cosim/workload_fix_v1_w3_build.py --w2-inputs "$ROOT/inputs_w24" --tags $TAGS --out "$ROOT/inputs" --diff-out "$ROOT/w3_vs_w2_diff.json"
  for tag in $TAGS; do for W in g0 g1 g2 g3; do
    f="$ROOT/inputs/wf1_$tag/wl/grounded_${W}_n50000.json"
    python3 scripts_cosim/scale_probe_remap_clients.py "$f" "$f.tmp" --clients "$NC" && mv "$f.tmp" "$f"
  done; done
  date -Is > "$ROOT/inputs.done"
fi
TAGS=$(for spec in $RUNGS; do echo -n "${spec%%=*} "; done)

# 2. capture (the production capture environment, verbatim)
export HEROSIM_TRANSFER_MODEL=pipelined HEROSIM_REPLICA_RELEASE=1 HEROSIM_SCALEOUT=kpa HEROSIM_SHARED_AUTOSCALER=0 GATE_FIXED_POLICY_TIME_SCALE=1.0
export HEROSIM_PEER_EXCHANGE=1 HEROSIM_SERVER_ONLY_REPLICAS=1 HEROSIM_WARMTH_PHYSICS=node_disk_v2 PYTHONHASHSEED=0 HEROSIM_GNN_DEVICE=cpu
export GNN_DECODE_MODE=masked_topo GNN_BATCH_BY_PEER_GROUP=1 HEROSIM_SNAPSHOT_FIDELITY=1 HEROSIM_RETAIN_TASK_TIMES=1
export LIVE_AUDIT_STRIDE=31 LIVE_AUDIT_MAX_SNAPSHOTS=600 LIVE_AUDIT_MIN_BATCH_SIZE=2 LIVE_AUDIT_MIN_CANDIDATES=1
CAPTURE_TIMEOUT_S=${CAPTURE_TIMEOUT_S:-7200}
mkdir -p "$ROOT/snapshots" "$ROOT/capture_logs" "$ROOT/raw"
pids=()
for R in $TAGS; do for W in $WINDOWS; do
  SNAP="$ROOT/snapshots/${CELL}_${R}_${W}.jsonl"; DONE="$ROOT/snapshots/${CELL}_${R}_${W}.done"
  [[ -s "$DONE" ]] && continue
  ( export LIVE_AUDIT_SNAPSHOT_PATH="$SNAP" OMP_NUM_THREADS=2; rm -f "$SNAP" "$DONE"; rc=0; t0=$(date +%s)
    timeout --kill-after=60 "$CAPTURE_TIMEOUT_S" python3 src/executesimulation.py --config "$ROOT/inputs/wf1_$R/cfg/cc40s$SEED.json" \
      --workload "$ROOT/inputs/wf1_$R/wl/grounded_${W}_n50000.json" --policy peer_greedy_network_batch \
      --output "$ROOT/raw/${CELL}_${R}_${W}.raw.json" > "$ROOT/capture_logs/${CELL}_${R}_${W}.log" 2>&1 || rc=$?
    n=0; [[ -f "$SNAP" ]] && n=$(wc -l < "$SNAP")
    status=ok; [[ $rc -eq 124 || $rc -eq 137 ]] && status=hung; [[ $rc -ne 0 && $status == ok ]] && status=failed; [[ $status == ok && $n -eq 0 ]] && status=failed
    echo "{\"status\": \"$status\", \"rc\": $rc, \"snapshots\": $n, \"wall_s\": $(( $(date +%s) - t0 ))}" > "$DONE"; echo "[$CELL $R $W] $(cat "$DONE")" ) &
  pids+=($!)
done; done
for p in "${pids[@]}"; do wait "$p" || true; done
python3 scripts_cosim/scale_probe_summary.py "$ROOT/raw" "$ROOT/inputs" > "$ROOT/capture_summary.json" || echo "summary failed"
rm -rf "$ROOT/raw"

# 3. build: declared slate (K=5) and the wider slate (K=10); same build environment as wf1_corpus_prod_build.sbatch
export MAX_PLACEMENT_COMBINATIONS_SKIP=2000000 GNN_CAPTURE_DATASET_STATE=0 COSIM_SUPPRESS_SIM_PRINTS=1 HEROSIM_RETAIN_PEER_STATS=1 COSIM_PLACEMENT_TIMEOUT_S=120
unset GNN_DECODE_MODE GNN_BATCH_BY_PEER_GROUP HEROSIM_GNN_DEVICE LIVE_AUDIT_STRIDE LIVE_AUDIT_MAX_SNAPSHOTS LIVE_AUDIT_MIN_BATCH_SIZE LIVE_AUDIT_MIN_CANDIDATES
for K in 5 10; do
  export DECLARED_SLATE_TOP_K=$K; D="$ROOT/ds_k$K"; mkdir -p "$D" "$ROOT/build_logs"; i=0
  for R in $TAGS; do for W in $WINDOWS; do
    SNAP="$ROOT/snapshots/${CELL}_${R}_${W}.jsonl"; DONE="$ROOT/snapshots/${CELL}_${R}_${W}.done"
    grep -q '"status": "ok"' "$DONE" || { echo "[$CELL $R $W] capture $(cat "$DONE"); skipped"; continue; }
    python3 scripts_cosim/wf1_shard_snapshots.py --snapshots "$SNAP" --out "$ROOT/snapshots/shard_${CELL}_${R}_${W}.jsonl" --min-time 360
    python3 scripts_cosim/make_warm_corpus.py --snapshots "$ROOT/snapshots/shard_${CELL}_${R}_${W}.jsonl" \
      --trace "$ROOT/inputs/wf1_$R/wl/grounded_${W}_n50000.json" --cell-config "$ROOT/inputs/wf1_$R/cfg/cc40s$SEED.json" \
      --output-dir "$D" --source-tag "probe_${CELL}_${R}_${W}" --start-index $(( i * 1000 )) --limit-batches "$BATCHES" \
      --min-time 360 --target-combos 20000 --workers 31 --no-cap-filter --min-choice-fraction 0.2 --quiet --variable-group-min 2 \
      --fidelity --require-connected-batch > "$ROOT/build_logs/k${K}_${CELL}_${R}_${W}.log" 2>&1 || echo "[$CELL $R $W K=$K] build rc=$? (shortfall or error)"
    i=$((i+1))
  done; done
  python3 scripts_cosim/joint_headroom_screen.py "$ROOT/headroom_k$K.jsonl" "$D" --workers 16
  python3 scripts_cosim/corpus_candidate_stats.py "$ROOT/candstats_k$K.jsonl" "$D" --workers 16
done
echo "done $CELL $(date -Is)"
