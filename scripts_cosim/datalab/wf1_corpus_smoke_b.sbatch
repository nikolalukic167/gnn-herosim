#!/bin/bash
# r1_attribution_v1 preparation (smoke, not a production run), route B: WF1 inputs for two calibration topologies at the
# hi rung (W2 payloads, W4 mix, single origin, 1 s window, W3 access classes) -> live capture of peer-group snapshots
# (window g0, behaviour policy as small_batch_v1_capture.sbatch) -> make_warm_corpus co-sim datasets -> SSC rewrite -> cache.
# WT and EXPECT_HEAD name the pinned detached worktree; OUT must not exist.
#SBATCH --job-name=wf1csb
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/wf1csb-%j.out
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=06:00:00
set -euo pipefail
WT=${WT:?FAIL LOUD: set WT}
EXPECT_HEAD=${EXPECT_HEAD:?FAIL LOUD: set EXPECT_HEAD}
OUT=${OUT:?FAIL LOUD: set OUT}
D=/home/nikola.lukic/gnn-herosim/simulation_data
W=$D/workload_fix_v1
X1=$D/small_batch_confirm_v1/inputs/grounded/wl
TOPOS=${TOPOS:-"9601 9607"}
FACTOR=$(python3 -c "print(repr(1.0/11.6139))")
cd "$WT"
if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3 PYTHONPATH="$WT:$WT/src/notebooks" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=""
HEAD=$(git rev-parse HEAD)
echo "HEAD $HEAD node $(hostname) OUT=$OUT"
[[ "$HEAD" == "$EXPECT_HEAD"* ]] || { echo "FAIL LOUD: worktree HEAD $HEAD is not $EXPECT_HEAD"; exit 1; }
[[ ! -e "$OUT" ]] || { echo "FAIL LOUD: $OUT exists"; exit 1; }
mkdir -p "$OUT"
echo "=== [1/5] W2/W4 inputs at the hi rung, then W3 access classes"
python3 scripts_cosim/workload_fix_v1_build.py --grounded-wl "$X1" --cfg-dir "$W/inputs/cfg_cal" --topologies $TOPOS \
  --rung "hi=$FACTOR" --payload-sampler wf1_v1 --task-mix wf1_v1 --batch-timeout-fixed 1 --out "$OUT/_b"
mkdir -p "$OUT/inputs_w24"; mv "$OUT/_b/wf1_hi" "$OUT/inputs_w24/wf1_hi"; cp "$OUT/_b/manifest.json" "$OUT/inputs_w24/manifest_hi.json"
python3 scripts_cosim/workload_fix_v1_w3_build.py --w2-inputs "$OUT/inputs_w24" --tags hi --out "$OUT/inputs" --diff-out "$OUT/w3_vs_w2_diff.json"
# R1.1 environment of the gate, plus the capture recipe's
export HEROSIM_TRANSFER_MODEL=pipelined HEROSIM_REPLICA_RELEASE=1 HEROSIM_SCALEOUT=kpa HEROSIM_SHARED_AUTOSCALER=0 GATE_FIXED_POLICY_TIME_SCALE=1.0
export HEROSIM_PEER_EXCHANGE=1 HEROSIM_SERVER_ONLY_REPLICAS=1 HEROSIM_WARMTH_PHYSICS=node_disk_v2 PYTHONHASHSEED=0 HEROSIM_GNN_DEVICE=cpu
export GNN_DECODE_MODE=masked_topo GNN_BATCH_BY_PEER_GROUP=1
unset GNN_MODEL_PATH GNN_DISABLE_MESSAGE_PASSING || true
export LIVE_AUDIT_STRIDE=31 LIVE_AUDIT_MAX_SNAPSHOTS=300 LIVE_AUDIT_MIN_BATCH_SIZE=2 LIVE_AUDIT_MIN_CANDIDATES=1
export HEROSIM_RAW_DIR=/tmp/herosim_raw_${SLURM_JOB_ID:-$$}; mkdir -p "$HEROSIM_RAW_DIR"; trap 'rm -rf "$HEROSIM_RAW_DIR"' EXIT
mkdir -p "$OUT/snapshots" "$OUT/capture"
echo "=== [2/5] capture (g0)"
pids=()
for T in $TOPOS; do
  ( export LIVE_AUDIT_SNAPSHOT_PATH="$OUT/snapshots/cc40s${T}_g0.jsonl"
    python3 src/executesimulation.py --config "$OUT/inputs/wf1_hi/cfg/cc40s${T}.json" \
      --workload "$OUT/inputs/wf1_hi/wl/grounded_g0_n50000.json" --policy peer_greedy_network_batch --output "$HEROSIM_RAW_DIR/cc40s${T}_g0.raw.json" \
      > "$OUT/capture/cc40s${T}_g0.log" 2>&1
    echo "capture $T rc=$?" ) &
  pids+=($!)
done
rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done
for T in $TOPOS; do tail -2 "$OUT/capture/cc40s${T}_g0.log" | cut -c1-200; wc -l "$OUT/snapshots/cc40s${T}_g0.jsonl"; done
[[ $rc == 0 ]] || { echo "FAIL LOUD: a capture failed"; exit 1; }
echo "=== [3/5] make_warm_corpus"
export MAX_PLACEMENT_COMBINATIONS_SKIP=2000000 GNN_CAPTURE_DATASET_STATE=0 COSIM_SUPPRESS_SIM_PRINTS=1 HEROSIM_RETAIN_TASK_TIMES=1 HEROSIM_RETAIN_PEER_STATS=1
export COSIM_AUTOSCALER_RECONCILE_INTERVAL=1e12 HEROSIM_TRACE_DATASET_ERRORS=1
DRAIN=$D/drainable_objective_v1/drain_table.json
export HEROSIM_BACKLOG_DRAIN_TABLE="$DRAIN"
CORP="$OUT/gnn_datasets_wf1_smoke_b"; i=0
for T in $TOPOS; do
  python3 scripts_cosim/make_warm_corpus.py --snapshots "$OUT/snapshots/cc40s${T}_g0.jsonl" --trace "$OUT/inputs/wf1_hi/wl/grounded_g0_n50000.json" \
    --cell-config "$OUT/inputs/wf1_hi/cfg/cc40s${T}.json" --output-dir "$CORP" --source-tag "wf1smoke_cc40s${T}_g0" --start-index $(( i * 100 )) --limit 3 \
    --target-combos 20000 --workers 30 --no-cap-filter --min-choice-fraction 0.2 --quiet --variable-group-min 2 \
    --synthetic-backlog-rungs 0 3 --synthetic-backlog-busy-prob 0.5 --synthetic-backlog-task-seconds 4.5 --synthetic-backlog-seed 7001 || echo "make_warm_corpus $T FAILED"
  i=$(( i + 1 ))
done
echo "datasets: $(ls -d "$CORP"/ds_* 2>/dev/null | wc -l), with placements.jsonl: $(ls "$CORP"/ds_*/placements/placements.jsonl 2>/dev/null | wc -l)"
echo "=== [4/5] SSC rewrite"
python3 scripts_cosim/refresh_optimal_full_stats.py --base-dir "$CORP" --rewrite-ssc | tail -2
echo "=== [5/5] cache (load_repr recipe, seeded clock)"
export DAG_PRIMARY_ALPHA_KEY=inf PARTIAL_STATE_CONTRACT=partial_state_v4 NEAR_RTT_LABEL_BACKLOG_CLOCK=seeded
set +e
python3 -u src/notebooks/prepare_graphs_cache.py --base-dirs "$CORP" --cache-dir "$OUT/graphs_cache" --dag-partial-state --platform-feature-dim 14 \
  --label-objective "rtt_drift:1" --label-arrival-rate 0.46 --label-drain-table "$DRAIN" 2>&1 | grep -v "it/s\]" | tail -25 | cut -c1-260
echo "cache rc=${PIPESTATUS[0]}"
ls "$OUT/graphs_cache" 2>&1 | head
