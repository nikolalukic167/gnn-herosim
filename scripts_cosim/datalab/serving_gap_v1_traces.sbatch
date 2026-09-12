#!/bin/bash
# serving_gap_v1 stage 1 — H1 traces (docs/lineages/serving_gap_v1.md).
# One live run per (corpus, arm, seed) on the matched production trace, writing a SLIM decode
# trace (GNN_PREFIX_TRACE_SLIM=1 drops the served graph: 45,375 batches x ~50 kB would be
# 2.3 GB per run and 96 of them do not fit the quota). The ~400 MB result JSON is deleted the
# moment the run lands; only the trace and a one-line summary are kept.
#
# Condition: UNCAPPED. The platform cap is itself an anti-concentration constraint, so
# measuring herding under it would mask the pathology it is meant to test. Stated here rather
# than left implicit.
#
#   T = 32*corpus + 16*arm + (seed-1);  corpora x200p2, x800p2, x800p3;  arms gnn, mpoff
#
#SBATCH --job-name=sg-traces
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/sg-traces-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/sg-traces-%A_%a.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --array=0-95

set -euo pipefail
cd /home/nikola.lukic/gnn-herosim
if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3 PYTHONPATH="$PWD" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=""
echo "[CODE] $(git describe --dirty --always) on $(git rev-parse --abbrev-ref HEAD)"

G=simulation_data/peer_affinity_live_gate
T=${SLURM_ARRAY_TASK_ID}
CORPORA=(x200p2 x800p2 x800p3); ARMS=(gnn mpoff)
CORPUS=${CORPORA[$(( T / 32 ))]}; ARM=${ARMS[$(( (T % 32) / 16 ))]}; SEED=$(( T % 16 + 1 ))
case $CORPUS in
  x200p2) CKPREFIX=models/peer-affinity-v1-t1b;    WL=$G/workloads/workload-150-150-peer_p2_x200.json ;;
  x800p2) CKPREFIX=models/peer-affinity-v1-x800p2; WL=$G/workloads/workload-150-150-peer_p2_x800.json ;;
  x800p3) CKPREFIX=models/peer-affinity-v1-x800p3; WL=$G/workloads/workload-150-150-peer_p3_x800.json ;;
esac
CK="${CKPREFIX}-${ARM}-lr2e3-seed${SEED}.pt"
# H1 wants every batch and no graph (SLIM); H2 wants ~500 graph-carrying records, i.e.
# TRACE_EVERY=90 over 45,375 batches with the graph kept. Both are the same 96 runs.
TRACE_SET="${TRACE_SET:-h1}"
OUT_DIR=simulation_data/serving_gap_v1/traces_${TRACE_SET}/$CORPUS
if [[ $TRACE_SET == h1 ]]; then export GNN_PREFIX_TRACE_SLIM=1; else export GNN_PREFIX_TRACE_EVERY="${TRACE_EVERY:-90}"; fi
TRACE=$OUT_DIR/${ARM}_s${SEED}.pkl
mkdir -p "$OUT_DIR" logs
echo "[TASK $T] corpus=$CORPUS arm=$ARM seed=$SEED ck=$CK -> $TRACE"
if [[ -s "$TRACE" ]]; then echo "[skip] trace exists"; exit 0; fi
[[ -f "$CK" ]] || { echo "FAIL LOUD: checkpoint missing: $CK" >&2; exit 1; }
[[ -f "${CK%.pt}.contract.json" ]] || { echo "FAIL LOUD: sidecar missing for $CK" >&2; exit 1; }
[[ -f "$WL" ]] || { echo "FAIL LOUD: workload missing: $WL" >&2; exit 1; }

export HEROSIM_PEER_EXCHANGE=1 HEROSIM_SERVER_ONLY_REPLICAS=1 HEROSIM_WARMTH_PHYSICS=node_disk_v2
export PYTHONHASHSEED=0 HEROSIM_GNN_DEVICE=cpu
export GNN_MODEL_PATH="$CK" GNN_DECODE_MODE=masked_topo GNN_BATCH_BY_PEER_GROUP=1
export GNN_PREFIX_TRACE_PATH="$TRACE.partial"
if [[ $ARM == mpoff ]]; then export GNN_DISABLE_MESSAGE_PASSING=1; fi

RAW=$OUT_DIR/${ARM}_s${SEED}.result.json
python3 src/executesimulation.py --config "$G/configs/cell_s7901.json" --workload "$WL" \
  --policy gnn --output "$RAW"
python3 - "$RAW" "$OUT_DIR/${ARM}_s${SEED}.summary.json" <<'PY'
import json, sys
raw, out = sys.argv[1:3]
d = json.load(open(raw)); st = d.get("stats") or d
keep = ("num_tasks","total_rtt","endTime","averageCommunicationsTime","averageExecutionTime",
        "averageQueueTime","totalPeerExchangeTime","schedulerCounters")
s = {k: st.get(k) for k in keep}
s["env"] = {k: v for k, v in (d.get("run_provenance") or {}).get("env", {}).items() if v}
json.dump(s, open(out, "w"))
print("[RESULT]", json.dumps({k: s[k] for k in s if k != "env"}))
PY
rm -f "$RAW"
mv "$TRACE.partial" "$TRACE"
ls -la "$TRACE"
echo "=== task $T done ==="
