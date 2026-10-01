#!/bin/bash
# live_headroom_v1 -- capture the live states CD itself produces at the grounded x2 / x3 / x5 rungs on the 12 development
# topologies (docs/lineages/live_headroom_v1.md). Behaviour policy: the CD greedy (peer_greedy_network_cd), so each snapshot is
# a batch plus the standing queue CD left behind. One array task per topology; rungs in sequence, windows g0-g3 in parallel.
# A unit that hangs (the starved-client spin) is killed at 20 minutes and dropped.
#   sbatch --export=ALL,WT=/home/nikola.lukic/gnn-herosim-wt/small_batch scripts_cosim/datalab/live_headroom_v1_capture.sbatch
#SBATCH --job-name=lh1-capture
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/lh1-capture-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/lh1-capture-%A_%a.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#SBATCH --array=0-11
set -euo pipefail
WT=${WT:-/home/nikola.lukic/gnn-herosim-wt/small_batch}
cd "$WT"
if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3 PYTHONPATH="$WT" PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""
echo "[CODE] $(git describe --dirty --always) at $WT"
IN=/home/nikola.lukic/gnn-herosim/simulation_data/backlog_corpus_v1/gates/inputs
LH=simulation_data/live_headroom_v1
mkdir -p "$LH/snapshots" "$LH/capture_summaries"
TOPOS=($(python3 -c "import json;print(*json.load(open('$IN/selected.json'))['topologies'])"))
T=${SLURM_ARRAY_TASK_ID}; TOPO=${TOPOS[$T]}
unit() {
  local RUNG=$1 WINDOW=$2
  local CFG="$IN/grounded_${RUNG}/cfg/cc40s${TOPO}.json" WL="$IN/grounded/wl/grounded_${WINDOW}_n50000.json"
  local TAG="cc40s${TOPO}_${RUNG}_${WINDOW}"
  local SNAP="$LH/snapshots/${TAG}.jsonl" SUMMARY="$LH/capture_summaries/${TAG}.summary.json"
  [[ -f "$CFG" && -f "$WL" ]] || { echo "FAIL LOUD: $CFG or $WL missing" >&2; return 1; }
  if [[ -f "$SUMMARY" && -s "$SNAP" ]]; then echo "[$TAG] done already"; return 0; fi
  export HEROSIM_PEER_EXCHANGE=1 HEROSIM_SERVER_ONLY_REPLICAS=1 HEROSIM_WARMTH_PHYSICS=node_disk_v2
  export PYTHONHASHSEED=0 HEROSIM_GNN_DEVICE=cpu SIM_FORCE_FULL_STATS=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
  export GNN_DECODE_MODE=masked_topo GNN_BATCH_BY_PEER_GROUP=1
  unset GNN_MODEL_PATH GNN_DISABLE_MESSAGE_PASSING HEROSIM_POLICY_TIME_SCALE || true
  local TS; TS=$(python3 -c "import json;print((json.load(open('$CFG')).get('cd_gap_v1_rate_scale') or {}).get('factor',1.0))")
  [[ "$TS" == "1.0" ]] || export HEROSIM_POLICY_TIME_SCALE="$TS"
  export LIVE_AUDIT_SNAPSHOT_PATH="$SNAP" LIVE_AUDIT_STRIDE=31 LIVE_AUDIT_MAX_SNAPSHOTS=300
  export LIVE_AUDIT_MIN_BATCH_SIZE=2 LIVE_AUDIT_MIN_CANDIDATES=1
  rm -f "$SNAP"
  local RAW="$LH/capture_summaries/${TAG}.raw.json"
  timeout 1200 python3 src/executesimulation.py --config "$CFG" --workload "$WL" --policy peer_greedy_network_cd --output "$RAW" \
    || { echo "[$TAG] hung or failed (rc $?); dropped"; return 1; }
  python3 - "$RAW" "$SUMMARY" "$SNAP" "$TAG" <<'PY'
import json, sys
raw, summary, snap, tag = sys.argv[1:5]
d = json.load(open(raw)); st = d.get("stats") or d
sizes = [len(json.loads(l)["tasks"]) for l in open(snap)]
s = {k: st.get(k) for k in ("num_tasks", "total_rtt", "averageElapsedTime", "averageQueueTime")}
s.update(tag=tag, policy="peer_greedy_network_cd", snapshots=len(sizes), batch_sizes={str(k): sizes.count(k) for k in sorted(set(sizes))},
         code=(d.get("run_provenance") or {}).get("code"))
if int(s["num_tasks"] or 0) != 50000:
    raise SystemExit(f"FAIL LOUD: num_tasks={s['num_tasks']!r}")
json.dump(s, open(summary + ".partial", "w"), indent=1)
print("[RESULT]", tag, s["snapshots"], "snapshots, mean latency", round(float(s["averageElapsedTime"]), 1))
PY
  mv "$SUMMARY.partial" "$SUMMARY"; rm -f "$RAW"
}
for RUNG in x20 x30 x50; do
  pids=()
  for W in g0 g1 g2 g3; do unit $RUNG $W > "$LH/capture_summaries/cc40s${TOPO}_${RUNG}_${W}.log" 2>&1 & pids+=($!); done
  for p in "${pids[@]}"; do wait "$p" || true; done
  for W in g0 g1 g2 g3; do tail -1 "$LH/capture_summaries/cc40s${TOPO}_${RUNG}_${W}.log"; done
done
echo "=== topology $TOPO done: $(ls $LH/capture_summaries/cc40s${TOPO}_*.summary.json 2>/dev/null | wc -l)/12 units ==="
