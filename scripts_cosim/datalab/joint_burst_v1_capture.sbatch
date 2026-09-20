#!/bin/bash
# joint_burst_v1 -- capture whole-group snapshots on the TRAINING topologies under bursts.
# docs/lineages/joint_burst_v1.md.
#
# Behaviour policy: the batched rule (`peer_greedy_network_batch`), i.e. the learned arms' own
# serving stack with a rule in the decoder's seat, so every snapshot is a state the served model
# will meet: loaded queues on the live clock, the whole peer group as the batch (bursts make the
# group arrive together). 24 training topologies x 2 windows (w0, w1) = 48 tasks. A task that
# hangs (the starved-client spin) leaves no snapshot file and is dropped from the corpus.
#
#   sbatch --array=0-47 scripts_cosim/datalab/joint_burst_v1_capture.sbatch
#
#SBATCH --job-name=jb1-capture
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/jb1-capture-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/jb1-capture-%A_%a.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --array=0-47

set -euo pipefail
cd /home/nikola.lukic/gnn-herosim
if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3
export PYTHONPATH="$PWD"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=""
echo "[CODE] $(git describe --dirty --always) on $(git rev-parse --abbrev-ref HEAD)"

GATE=simulation_data/peer_affinity_live_gate
JB=simulation_data/joint_burst_v1
mkdir -p "$JB/snapshots" "$JB/capture_summaries"
SEEDS=($(seq 9201 9224)); WINDOWS=(w0 w1)
T=${SLURM_ARRAY_TASK_ID}
SEED=${SEEDS[$(( T / 2 ))]}; WINDOW=${WINDOWS[$(( T % 2 ))]}
CELL="cc40s${SEED}"; CFG="$GATE/configs/${CELL}.json"
if [[ $WINDOW == w0 ]]; then WL_NAME=burst_drainable_f4000_n50000; else WL_NAME=burst_drainable_${WINDOW}_n50000; fi
WORKLOAD="$GATE/workloads/${WL_NAME}.json"
[[ -f "$CFG" && -f "$WORKLOAD" ]] || { echo "FAIL LOUD: $CFG or $WORKLOAD missing" >&2; exit 1; }
SNAP="$JB/snapshots/${CELL}_${WINDOW}.jsonl"
SUMMARY="$JB/capture_summaries/${CELL}_${WINDOW}.summary.json"
if [[ -f "$SUMMARY" && -s "$SNAP" ]]; then echo "[TASK $T] done already"; exit 0; fi
echo "[TASK $T] $CELL $WINDOW -> $SNAP"

export HEROSIM_PEER_EXCHANGE=1 HEROSIM_SERVER_ONLY_REPLICAS=1 HEROSIM_WARMTH_PHYSICS=node_disk_v2
export PYTHONHASHSEED=0 HEROSIM_GNN_DEVICE=cpu
export GNN_DECODE_MODE=masked_topo GNN_BATCH_BY_PEER_GROUP=1
unset GNN_MODEL_PATH GNN_DISABLE_MESSAGE_PASSING || true
# stride keyed on the batch's first task id; groups start at multiples of 10, so 300 keeps
# every 30th group -> ~166 candidate snapshots over the trace, capped at 150
export LIVE_AUDIT_SNAPSHOT_PATH="$SNAP" LIVE_AUDIT_STRIDE=300 LIVE_AUDIT_MAX_SNAPSHOTS=150
export LIVE_AUDIT_MIN_BATCH_SIZE=1 LIVE_AUDIT_MIN_CANDIDATES=1
rm -f "$SNAP"
RAW="$JB/capture_summaries/${CELL}_${WINDOW}.raw.json"
python3 src/executesimulation.py --config "$CFG" --workload "$WORKLOAD" --policy peer_greedy_network_batch --output "$RAW"
python3 - "$RAW" "$SUMMARY" "$SNAP" "$CELL" "$WINDOW" <<'PY'
import json, sys
raw, summary, snap, cell, window = sys.argv[1:6]
d = json.load(open(raw)); st = d.get("stats") or d
s = {k: st.get(k) for k in ("num_tasks", "total_rtt", "averageElapsedTime", "averageQueueTime", "averageWaitTime", "totalPeerExchangeTime", "totalPeerRendezvousWait")}
c = st.get("schedulerCounters") or {}
s["counters"] = {k: v for k, v in c.items() if not isinstance(v, list)}
s.update(cell=cell, window=window, policy="peer_greedy_network_batch", lever="burst",
         snapshots=sum(1 for _ in open(snap)),
         env={k: v for k, v in (d.get("run_provenance") or {}).get("env", {}).items() if v},
         code=(d.get("run_provenance") or {}).get("code"))
if int(s["num_tasks"] or 0) != 50000:
    raise SystemExit(f"FAIL LOUD: num_tasks={s['num_tasks']!r}")
json.dump(s, open(summary + ".partial", "w"), indent=1)
print("[RESULT]", json.dumps({k: s[k] for k in ("cell", "window", "averageElapsedTime", "averageWaitTime", "snapshots")}))
PY
mv "$SUMMARY.partial" "$SUMMARY"; rm -f "$RAW"
[[ -s "$SNAP" ]] || { echo "FAIL LOUD: no snapshots captured in $SNAP" >&2; exit 1; }
echo "=== task $T done: $(wc -l < "$SNAP") snapshots ==="
