#!/bin/bash
# partial_state_v3 -- P3, the PRIMARY: does a checkpoint trained at 6 servers survive 12 /
# 24 / 80 servers, live? docs/lineages/partial_state_v3.md.
#
#   rung R0:  6 servers  drainable_f4000  0.460 arrivals/s   (in support, 3.55 cand/task)
#   rung R1: 12 servers  drainable_f2000  0.921 arrivals/s   (out of support)
#   rung R2: 24 servers  drainable_f1000  1.842 arrivals/s   (out, 14.18 cand/task)
#   rung R3: 80 servers  drainable_f300   6.139 arrivals/s   (out, 47.92 cand/task)
#
# Arrivals per server are held at R0's ratio; the rungs are NOT load-matched
# (cluster_scale_v1 S0.a) and the statistic is the arm's elapsed RELATIVE TO REACTIVE on the
# same cell, defined at any load. Per rung: 4 topology seeds (9001, 9002, 9003, 9005 -- 9004
# hangs on every policy) x (1 knative_network + 4 gnn_v3 + 4 mpoff_v3, checkpoint seeds
# 1, 2, 4, 5) = 36 arms; 144 arms in all. 25 % cell attrition budgeted.
#
# P3-a (blocking): every learned arm on a readable scaled cell must COMPLETE -- a raise in
# krank_node_order or a contract refusal at 12+ servers closes the lineage STILL-PINNED.
# That is the reason the lineage exists, so it is the first thing this job can show.
#
# Task 0 mints the cells (differing from the base in exactly two asserted fields: server
# count and topology seed) and writes the manifest; every other task waits for it
# (datalab-pitfalls #7). Every arm name carries cell, rung, policy and checkpoint seed.
#
#   sbatch --array=0-143%12 scripts_cosim/datalab/partial_state_v3_p3_gate.sbatch
#
#SBATCH --job-name=psv3-p3
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/psv3-p3-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/psv3-p3-%A_%a.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --array=0-143

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
python3 -c "import sys,torch;print('[INTERP]',sys.executable,torch.__version__)"
echo "[CODE] $(git describe --dirty --always) on $(git rev-parse --abbrev-ref HEAD)"

GATE=simulation_data/peer_affinity_live_gate
OUT_DIR="$GATE/results/psv3_p3"
MANIFEST=simulation_data/partial_state_v3/p3_cells.json
BASE="$GATE/configs/cell_s9001_f4000_pg16.json"
mkdir -p "$OUT_DIR" simulation_data/partial_state_v3

RUNGS=(R0 R1 R2 R3)
SERVERS=(6 12 24 80)
WORKLOADS=(drainable_f4000_n50000 drainable_f2000_n50000 drainable_f1000_n50000 drainable_f300_n50000)
TSEEDS=(9001 9002 9003 9005)
CKSEEDS=(1 2 4 5)

export HEROSIM_PEER_EXCHANGE=1
export HEROSIM_SERVER_ONLY_REPLICAS=1
export HEROSIM_WARMTH_PHYSICS=node_disk_v2
export PYTHONHASHSEED=0
export HEROSIM_GNN_DEVICE=cpu
export SIM_FORCE_FULL_STATS=1

T=${SLURM_ARRAY_TASK_ID}

# --- cells: task 0 mints, everyone else waits (datalab-pitfalls #7) ---------------------
if [[ $T -eq 0 ]]; then
  if [[ ! -f "$MANIFEST" ]]; then
    echo "[mint] $MANIFEST"
    python3 scripts_cosim/cluster_scale_v1_cells.py --base "$BASE" --sim-inputs data/nofs-ids \
      --rungs 6,12,24,80 --seeds 9001,9002,9003,9005 --out-dir "$GATE/configs" \
      --manifest "$MANIFEST.partial" --prefix cs
    mv "$MANIFEST.partial" "$MANIFEST"
  fi
else
  for i in $(seq 1 120); do [[ -f "$MANIFEST" ]] && break; sleep 5; done
  [[ -f "$MANIFEST" ]] || { echo "FAIL LOUD: $MANIFEST never appeared (task 0 did not mint)" >&2; exit 1; }
fi

R=$(( T / 36 )); I=$(( T % 36 )); CI=$(( I / 9 )); A=$(( I % 9 ))
RUNG="${RUNGS[$R]}"; NS="${SERVERS[$R]}"; TSEED="${TSEEDS[$CI]}"
WORKLOAD="$GATE/workloads/${WORKLOADS[$R]}.json"
[[ -f "$WORKLOAD" ]] || { echo "FAIL LOUD: workload missing: $WORKLOAD" >&2; exit 1; }
CELL="cs${NS}s${TSEED}"
CFG="$GATE/configs/${CELL}.json"
[[ -f "$CFG" ]] || { echo "FAIL LOUD: cell config missing: $CFG" >&2; exit 1; }
python3 - "$CFG" "$NS" <<'PY'
import json, sys
cfg, want = json.load(open(sys.argv[1])), int(sys.argv[2])
got = int(cfg["nodes"]["server_nodes"]["count"])
if got != want:
    raise SystemExit(f"FAIL LOUD: {sys.argv[1]} declares {got} servers, rung expects {want}")
bt = float(cfg["scheduler"]["batch_timeout"])
if bt != 16.0:
    raise SystemExit(f"FAIL LOUD: batch_timeout {bt} -- every policy constant is held fixed")
PY

if [[ $A -eq 0 ]]; then KIND=reactive; POLICY=knative_network; SEED=0
elif [[ $A -le 4 ]]; then KIND=gnn; POLICY=gnn; SEED="${CKSEEDS[$(( A - 1 ))]}"
else KIND=mpoff; POLICY=gnn; SEED="${CKSEEDS[$(( A - 5 ))]}"; fi
ARM="${CELL}__${RUNG}__${KIND}_s${SEED}"

if [[ $KIND != reactive ]]; then
  CK="models/partial-state-v3-${KIND}-lr2e3-seed${SEED}.pt"
  [[ -f "$CK" ]] || { echo "FAIL LOUD: missing checkpoint $CK" >&2; exit 1; }
  SC="${CK%.pt}.contract.json"
  [[ -f "$SC" ]] || { echo "FAIL LOUD: no sidecar for $CK" >&2; exit 1; }
  python3 - "$SC" "$KIND" <<'PY'
import json, sys
sc, arm = json.load(open(sys.argv[1])), sys.argv[2]
if sc.get("disable_message_passing") is not (arm == "mpoff"):
    raise SystemExit(f"FAIL LOUD: sidecar disable_message_passing={sc.get('disable_message_passing')!r} for arm {arm}")
if sc.get("partial_state_contract") != "partial_state_v3" or int(sc.get("partial_state_feature_dim") or 0) != 22:
    raise SystemExit(f"FAIL LOUD: sidecar contract/dim = {sc.get('partial_state_contract')!r}/"
                     f"{sc.get('partial_state_feature_dim')!r}; P3 serves partial_state_v3 / 22 only")
PY
  unset PARTIAL_STATE_CONTRACT || true
  export GNN_MODEL_PATH="$CK"
  export GNN_DECODE_MODE=masked_topo
  export GNN_BATCH_BY_PEER_GROUP=1
  if [[ $KIND == mpoff ]]; then export GNN_DISABLE_MESSAGE_PASSING=1; fi
fi

SUMMARY="$OUT_DIR/${ARM}.summary.json"
echo "[TASK $T] rung=$RUNG servers=$NS cell=$CELL kind=$KIND seed=$SEED arm=$ARM -> $SUMMARY"
if [[ -f "$SUMMARY" ]]; then echo "[TASK $T] exists, not re-running"; exit 0; fi
RAW="$OUT_DIR/${ARM}.raw.json"
START=$(date +%s)
python3 src/executesimulation.py --config "$CFG" --workload "$WORKLOAD" --policy "$POLICY" --output "$RAW"
WALL=$(( $(date +%s) - START ))

python3 - "$RAW" "$SUMMARY" "$ARM" "$CELL" "$RUNG" "$NS" "${WORKLOADS[$R]}" "$KIND" "$SEED" "$WALL" <<'PY'
import json, sys
sys.path.insert(0, ".")
from scripts_cosim.offline_live_transfer_v1_decile_read import decile_queue_summary
from scripts_cosim.scheduler_residence_v1_read import residence_summary
raw, summary, arm, cell, rung, servers, workload, kind, seed, wall = sys.argv[1:11]
doc = json.load(open(raw)); st = doc.get("stats") or doc
out = {k: st.get(k) for k in (
    "num_tasks", "total_rtt", "averageElapsedTime", "averageQueueTime", "averageWaitTime",
    "totalPeerExchangeTime", "totalPeerRendezvousWait", "endTime", "schedulerCounters")}
_se = st.get("scaleEvents")
out["scaleEventCount"] = len(_se) if isinstance(_se, list) else _se
out["decile_summary"] = decile_queue_summary(st.get("taskResults") or [])
out.update(arm=arm, cell=cell, rung=rung, servers=int(servers), workload=workload, arm_kind=kind,
           checkpoint_seed=int(seed), policy_name=("knative_network" if kind == "reactive" else "gnn"),
           wallclock_s=int(wall), representation=(None if kind == "reactive" else "v3"))
c = out.get("schedulerCounters") or {}
tasks = c.pop("residence_tasks", None); batches = c.pop("residence_batches", None)
c.pop("queue_range_records", None)
if kind != "reactive" and tasks:
    # Descriptive: collection per rung, the mechanism cluster_scale_v1 measured, for free.
    out["residence_summary"] = residence_summary(
        tasks, batches or [], out["decile_summary"]["bounds"],
        float(out["averageWaitTime"]), int(c.get("residence_unstamped", 0)))
env = {k: v for k, v in (doc.get("run_provenance") or {}).get("env", {}).items() if v}
out["env"] = env
out["code"] = (doc.get("run_provenance") or {}).get("code")
served = env.get("PARTIAL_STATE_CONTRACT")
if kind != "reactive" and served and served != "partial_state_v3":
    raise SystemExit(f"FAIL LOUD: run_provenance says PARTIAL_STATE_CONTRACT={served!r}")
json.dump(out, open(summary + ".partial", "w"), indent=1)
extra = ""
if out.get("residence_summary"):
    r = out["residence_summary"]
    extra = f"  collect={r['collection_s']:.3f} batch={r['mean_batch_size']:.2f}"
print(f"[RESULT] {arm} servers={servers} {workload} elapsed={out['averageElapsedTime']:.3f} "
      f"queue={out['averageQueueTime']:.3f} wall={wall}s{extra}")
PY
mv "$SUMMARY.partial" "$SUMMARY"
rm -f "$RAW"
echo "=== task $T done: $ARM ==="
