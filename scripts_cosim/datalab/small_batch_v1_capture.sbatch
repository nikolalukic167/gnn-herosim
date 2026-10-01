#!/bin/bash
# small_batch_v1 -- capture the batches the live seat actually dispatches: whole peer groups of the grounded workload
# (library fan-out sizes, mean about 4) on the 24 TRAINING topologies, windows g0-g3 at x1. Behaviour policy as
# joint_burst_v1_capture.sbatch (the batched rule in the decoder's seat). docs/lineages/small_batch_v1.md.
#
#   sbatch --export=ALL,WT=/home/nikola.lukic/gnn-herosim-wt/small_batch scripts_cosim/datalab/small_batch_v1_capture.sbatch
#
#SBATCH --job-name=sb1-capture
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/sb1-capture-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/sb1-capture-%A_%a.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#SBATCH --array=0-23

set -euo pipefail
WT=${WT:-/home/nikola.lukic/gnn-herosim-wt/small_batch}
cd "$WT"
if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3
export PYTHONPATH="$WT"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=""
echo "[CODE] $(git describe --dirty --always) at $WT"

GATE=/home/nikola.lukic/gnn-herosim/simulation_data/peer_affinity_live_gate
WLDIR=/home/nikola.lukic/gnn-herosim/simulation_data/backlog_corpus_v1/gates/inputs/grounded/wl
SB=simulation_data/small_batch_v1
mkdir -p "$SB/snapshots" "$SB/capture_summaries"
SEEDS=($(seq 9201 9224))
T=${SLURM_ARRAY_TASK_ID}
SEED=${SEEDS[$T]}
unit() {
WINDOW=$1
CELL="cc40s${SEED}"; CFG="$GATE/configs/${CELL}.json"
WORKLOAD="$WLDIR/grounded_${WINDOW}_n50000.json"
[[ -f "$CFG" && -f "$WORKLOAD" ]] || { echo "FAIL LOUD: $CFG or $WORKLOAD missing" >&2; exit 1; }
SNAP="$SB/snapshots/${CELL}_${WINDOW}.jsonl"
SUMMARY="$SB/capture_summaries/${CELL}_${WINDOW}.summary.json"
if [[ -f "$SUMMARY" && -s "$SNAP" ]]; then echo "[TASK $T $WINDOW] done already"; return 0; fi
echo "[TASK $T $WINDOW] $CELL $WINDOW -> $SNAP"

export HEROSIM_PEER_EXCHANGE=1 HEROSIM_SERVER_ONLY_REPLICAS=1 HEROSIM_WARMTH_PHYSICS=node_disk_v2
export PYTHONHASHSEED=0 HEROSIM_GNN_DEVICE=cpu
export GNN_DECODE_MODE=masked_topo GNN_BATCH_BY_PEER_GROUP=1
unset GNN_MODEL_PATH GNN_DISABLE_MESSAGE_PASSING || true
# stride keyed on the batch's first task id; groups start at arbitrary ids here, so 31 keeps about 1 group in 31
# (roughly 400 candidate snapshots over the trace), capped at 300
export LIVE_AUDIT_SNAPSHOT_PATH="$SNAP" LIVE_AUDIT_STRIDE=31 LIVE_AUDIT_MAX_SNAPSHOTS=300
export LIVE_AUDIT_MIN_BATCH_SIZE=2 LIVE_AUDIT_MIN_CANDIDATES=1
rm -f "$SNAP"
RAW="$SB/capture_summaries/${CELL}_${WINDOW}.raw.json"
python3 src/executesimulation.py --config "$CFG" --workload "$WORKLOAD" --policy peer_greedy_network_batch --output "$RAW"
python3 - "$RAW" "$SUMMARY" "$SNAP" "$CELL" "$WINDOW" <<'PY'
import json, sys
raw, summary, snap, cell, window = sys.argv[1:6]
d = json.load(open(raw)); st = d.get("stats") or d
s = {k: st.get(k) for k in ("num_tasks", "total_rtt", "averageElapsedTime", "averageQueueTime", "averageWaitTime")}
c = st.get("schedulerCounters") or {}
s["counters"] = {k: v for k, v in c.items() if not isinstance(v, list)}
sizes = [len(json.loads(l)["tasks"]) for l in open(snap)]
s.update(cell=cell, window=window, policy="peer_greedy_network_batch", snapshots=len(sizes),
         batch_sizes={str(k): sizes.count(k) for k in sorted(set(sizes))},
         code=(d.get("run_provenance") or {}).get("code"))
if int(s["num_tasks"] or 0) != 50000:
    raise SystemExit(f"FAIL LOUD: num_tasks={s['num_tasks']!r}")
json.dump(s, open(summary + ".partial", "w"), indent=1)
print("[RESULT]", json.dumps({k: s[k] for k in ("cell", "window", "snapshots", "batch_sizes")}))
PY
mv "$SUMMARY.partial" "$SUMMARY"; rm -f "$RAW"
[[ -s "$SNAP" ]] || { echo "FAIL LOUD: no snapshots captured in $SNAP" >&2; exit 1; }
echo "=== task $T done: $(wc -l < "$SNAP") snapshots ==="

}
pids=()
for W in g0 g1 g2 g3; do unit $W > "$SB/capture_summaries/${CELL:-cc40s${SEED}}_$W.log" 2>&1 & pids+=($!); done
rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done
for W in g0 g1 g2 g3; do tail -3 "$SB/capture_summaries/cc40s${SEED}_$W.log"; done
exit $rc
