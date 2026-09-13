#!/bin/bash
# peer_affinity_warm_v1 -- W1 warm-up capture (docs/lineages/peer_affinity_warm_v1.md, W1 + Amendment 1).
# 28 cells (s9001-s9024 train, s9101-s9104 held-out) x 2 behaviour sources on the warm-up trace
# workload-150-100-peer_p2_x200.json (301k events, peer seed 7301). Each run captures peer-group-aligned
# LIVE_AUDIT snapshots at stride 2,000 (<= 150). The raw result (~300 MB) is summarised and deleted.
#   task T: cell = CELLS[T / 2], source = (T % 2 == 0 ? knb : gnn)
#   sbatch --array=0-55%20 scripts_cosim/datalab/peer_affinity_warm_v1_w1_capture.sbatch
#
#SBATCH --job-name=pa-warm-w1-cap
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/pa-warm-w1-cap-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/pa-warm-w1-cap-%A_%a.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=08:00:00

set -euo pipefail
cd /home/nikola.lukic/gnn-herosim
if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3
export PYTHONPATH="$PWD"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=""
python3 -c "import sys,torch;print('[INTERP]',sys.executable,torch.__version__)"
echo "[CODE] $(git describe --dirty --always) on $(git rev-parse --abbrev-ref HEAD)"

CELLS=($(seq 9001 9024) $(seq 9101 9104))
T=${SLURM_ARRAY_TASK_ID}
SEED=${CELLS[$(( T / 2 ))]}
if [[ $(( T % 2 )) -eq 0 ]]; then SRC=knb; else SRC=gnn; fi
G=simulation_data/peer_affinity_live_gate
W1=simulation_data/peer_affinity_warm_v1/w1
WORKLOAD=$G/workloads/workload-150-100-peer_p2_x200.json
CFG=$G/configs/cell_s${SEED}.json
mkdir -p "$W1/snapshots" "$W1/capture_summaries" logs
SNAP="$W1/snapshots/cell_s${SEED}_${SRC}.jsonl"
SUMMARY="$W1/capture_summaries/cell_s${SEED}_${SRC}.summary.json"
[[ -f "$WORKLOAD" ]] || { echo "FAIL LOUD: workload missing: $WORKLOAD" >&2; exit 1; }
[[ -f "$CFG" ]] || { echo "FAIL LOUD: cell config missing: $CFG" >&2; exit 1; }
if [[ -f "$SUMMARY" && -s "$SNAP" ]]; then echo "[TASK $T] done already: $SUMMARY"; exit 0; fi
echo "[TASK $T] cell_s$SEED source=$SRC -> $SNAP"
echo "[INPUTS] workload $(md5sum "$WORKLOAD" | cut -c1-12) cfg $(md5sum "$CFG" | cut -c1-12)"

# physics + gate contract, identical to the W0 captures and the stage-3 gate
export HEROSIM_PEER_EXCHANGE=1 HEROSIM_SERVER_ONLY_REPLICAS=1 HEROSIM_WARMTH_PHYSICS=node_disk_v2 PYTHONHASHSEED=0
export HEROSIM_GNN_DEVICE=cpu
export LIVE_AUDIT_SNAPSHOT_PATH="$SNAP" LIVE_AUDIT_STRIDE=2000 LIVE_AUDIT_MAX_SNAPSHOTS=150
export LIVE_AUDIT_MIN_BATCH_SIZE=1 LIVE_AUDIT_MIN_CANDIDATES=1
rm -f "$SNAP"
if [[ $SRC == knb ]]; then
  POLICY=knative_network_batch
  export KNATIVE_BATCH_SIZE=10 KNATIVE_BATCH_TIMEOUT=0.02 KNATIVE_BATCH_BY_PEER_GROUP=1
else
  POLICY=gnn
  CK=models/peer-affinity-v1-t1b-gnn-lr2e3-seed1.pt
  [[ -f "$CK" && -f "${CK%.pt}.contract.json" ]] || { echo "FAIL LOUD: $CK or its sidecar missing" >&2; exit 1; }
  export GNN_MODEL_PATH="$CK" GNN_DECODE_MODE=masked_topo GNN_BATCH_BY_PEER_GROUP=1 GNN_PREFIX_PLATFORM_CAP=1
fi
RAW="$W1/capture_summaries/cell_s${SEED}_${SRC}.raw.json"
${HEROSIM_PY:-pipenv run python3} src/executesimulation.py --config "$CFG" --workload "$WORKLOAD" --policy "$POLICY" --output "$RAW"
python3 - "$RAW" "$SUMMARY" "$SNAP" "$SRC" "$SEED" <<'PY'
import json, sys
raw, summary, snap, src, seed = sys.argv[1:6]
d = json.load(open(raw)); st = d.get("stats") or d
s = {k: st.get(k) for k in ("policy", "num_tasks", "total_rtt", "averageElapsedTime", "totalPeerExchangeTime", "endTime", "averageQueueTime")}
s.update({"cell": f"cell_s{seed}", "source": src, "snapshots": sum(1 for _ in open(snap)),
          "env": {k: v for k, v in (d.get("run_provenance") or {}).get("env", {}).items() if v},
          "code": (d.get("run_provenance") or {}).get("code")})
json.dump(s, open(summary + ".partial", "w"))
print("[RESULT]", json.dumps({k: s[k] for k in s if k not in ("env", "code")}))
PY
mv "$SUMMARY.partial" "$SUMMARY"; rm -f "$RAW"
[[ -s "$SNAP" ]] || { echo "FAIL LOUD: no snapshots captured in $SNAP" >&2; exit 1; }
echo "=== task $T done: $(wc -l < "$SNAP") snapshots ==="
