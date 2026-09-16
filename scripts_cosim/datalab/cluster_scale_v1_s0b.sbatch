#!/bin/bash
# cluster_scale_v1 -- S0.b AMENDED: does peer-group assembly cost fall when arrivals speed up?
#
# Amendment 1 (2026-09-16). S0's scaled rungs are VOID -- they were not load-matched (reactive
# queue 25x and 49x the baseline, because the registered capacity model used a drain measured
# in a different regime and is ~6x too high) -- and every gnn arm above 6 servers FAILED on
#   ValueError: krank_node_order: 7 nodes exceed the registered pad width 6
# (reduced_features.py:252). The served feature layout cannot represent more than SIX
# candidate-hosting nodes, so this axis needs a retrained REPRESENTATION, not just a model.
#
# But S0.b never needed the scaled rungs. Collection waits for a task's peer-group members to
# ARRIVE and exits the moment the group is complete (scheduler.py:443), strictly before any
# decode and never consulting the model. So it is measured at 6 servers -- inside the pad
# width -- by varying ONLY the workload.
#
# DISCLOSED: these rungs are overloaded (6 servers cannot drain 6.139 arrivals/s). Their
# LATENCY numbers are meaningless and are not read. Collection time is the only statistic.
#
# Registered in docs/lineages/cluster_scale_v1.md. Three rungs holding rho ~ 0.16 by adding
# capacity as arrivals speed up. Workloads already exist; no trace is regenerated.
#
#   rung 0: drainable_f4000  0.460 arrivals/s   6 servers   group assembles in ~21.7 s
#   rung 1: drainable_f1000  1.842 arrivals/s  24 servers   ~5.4 s
#   rung 2: drainable_f300   6.139 arrivals/s  80 servers   ~1.6 s
#
# 24 arms = 3 rungs x 4 topology seeds x (1 reactive + 1 gnn). Cells differ from the base in
# exactly two asserted fields (server count, topology seed); EVERY policy time constant is
# held fixed -- window 16 s, poll 1 ms, batch size 10 -- because their cost is what is being
# measured. scheduler_residence_v1 R0 measured that the collector exits the moment the peer
# group is complete, so a fixed window is an upper bound that binds less as arrivals speed up;
# scaling it would rescale the answer to zero by construction. Registered departure.
#
# The `gnn` arm here is an INSTRUMENT, not a performance claim: collection happens strictly
# before any decode and never consults the model, so collection time is model-independent.
# S0.d has already marked rungs 1 and 2 OUT-OF-SUPPORT (14.2x and 47.9x candidates per task
# against a corpus max of 5), so no latency number from this job means anything about quality.
#
# S0.a  load match (blocking): reactive mean queue at rungs 1-2 within [0.33, 3.0] x rung 0's.
# S0.b  primary: median gnn COLLECTION time falls monotonically and reaches <= 2.0 s at the
#       fastest readable rung, from 6.10-6.13 s measured at rung 0.
# S0.c  feasibility: median wall-clock <= 45 min at the fastest readable rung.
#
# 25 % cell attrition is budgeted (scheduler_residence_v1 R3: 5 of 20 draws hang on EVERY
# policy including reactive -- the documented starved-client spin). MEASURED on job 769390:
# seed 9004 hangs at BOTH readable rungs (tasks 3 and 7, TIMEOUT at 45 min) and is dropped;
# 9001-9003 complete in 78-149 s each.
#
# DEFECT FIXED after job 769390: ARM was "${CELL}__${KIND}" with no rung tag, so all three
# rungs resolved to the same summary path and the f300 rung exited on the idempotence guard
# ("exists, not re-running", 1 s per task) having measured nothing. The rung is now in the
# arm name, so a rung's own summary is what the guard tests.
#
#   sbatch scripts_cosim/datalab/cluster_scale_v1_s0b.sbatch
#
#SBATCH --job-name=cs-s0b
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/cs-s0b-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/cs-s0b-%A_%a.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:15:00
#SBATCH --array=0-8

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
OUT_DIR="$GATE/results/cs_s0b"
mkdir -p "$OUT_DIR"

SERVERS=(6 6 6)
WORKLOADS=(drainable_f4000_n50000 drainable_f1000_n50000 drainable_f300_n50000)
SEEDS=(9001 9002 9003)

export HEROSIM_PEER_EXCHANGE=1
export HEROSIM_SERVER_ONLY_REPLICAS=1
export HEROSIM_WARMTH_PHYSICS=node_disk_v2
export PYTHONHASHSEED=0
export HEROSIM_GNN_DEVICE=cpu
export SIM_FORCE_FULL_STATS=1

T=${SLURM_ARRAY_TASK_ID}
R=$(( T / 3 )); I=$(( T % 3 ))
NS="${SERVERS[$R]}"
WORKLOAD="$GATE/workloads/${WORKLOADS[$R]}.json"
[[ -f "$WORKLOAD" ]] || { echo "FAIL LOUD: workload missing: $WORKLOAD" >&2; exit 1; }

KIND=gnn; SEED="${SEEDS[$I]}"; POLICY=gnn

CELL="cs6s${SEED}"
CFG="$GATE/configs/${CELL}.json"
[[ -f "$CFG" ]] || { echo "FAIL LOUD: cell config missing: $CFG" >&2; exit 1; }

# The cell must actually carry the rung's server count -- a mis-paired rung would measure a
# different cluster than the one it is named after.
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

if [[ $KIND == gnn ]]; then
  CK="models/drainable-objective-v1-v1-gnn-lr2e3-seed1.pt"
  [[ -f "$CK" ]] || { echo "FAIL LOUD: missing checkpoint $CK" >&2; exit 1; }
  [[ -f "${CK%.pt}.contract.json" ]] || { echo "FAIL LOUD: no sidecar" >&2; exit 1; }
  export GNN_MODEL_PATH="$CK"
  export GNN_DECODE_MODE=masked_topo
  export GNN_BATCH_BY_PEER_GROUP=1
fi

RUNG="${WORKLOADS[$R]}"; RUNG="${RUNG#drainable_}"; RUNG="${RUNG%_n50000}"
ARM="${CELL}__${KIND}__${RUNG}"
SUMMARY="$OUT_DIR/${ARM}.summary.json"
echo "[TASK $T] rung=$R servers=$NS workload=${WORKLOADS[$R]} arm=$ARM"
if [[ -f "$SUMMARY" ]]; then echo "[TASK $T] exists, not re-running"; exit 0; fi

RAW="$OUT_DIR/${ARM}.raw.json"
START=$(date +%s)
python3 src/executesimulation.py --config "$CFG" --workload "$WORKLOAD" --policy "$POLICY" --output "$RAW"
WALL=$(( $(date +%s) - START ))

python3 - "$RAW" "$SUMMARY" "$ARM" "$CELL" "$POLICY" "$KIND" "$NS" "${WORKLOADS[$R]}" "$WALL" <<'PY'
import json, sys
sys.path.insert(0, ".")
from scripts_cosim.offline_live_transfer_v1_decile_read import decile_queue_summary
from scripts_cosim.scheduler_residence_v1_read import residence_summary

raw, summary, arm, cell, policy, kind, servers, workload, wall = sys.argv[1:10]
doc = json.load(open(raw)); st = doc.get("stats") or doc
out = {k: st.get(k) for k in (
    "num_tasks", "total_rtt", "averageElapsedTime", "averageQueueTime", "averageWaitTime",
    "totalPeerExchangeTime", "totalPeerRendezvousWait", "endTime", "schedulerCounters")}
_se = st.get("scaleEvents")
out["scaleEventCount"] = len(_se) if isinstance(_se, list) else _se
out["decile_summary"] = decile_queue_summary(st.get("taskResults") or [])
out.update(arm=arm, cell=cell, policy_name=policy, arm_kind=kind,
           servers=int(servers), workload=workload, wallclock_s=int(wall))

c = out.get("schedulerCounters") or {}
tasks = c.pop("residence_tasks", None)
batches = c.pop("residence_batches", None)
c.pop("queue_range_records", None)
if kind == "gnn":
    # S0.b's statistic. Fails loud rather than reporting a rung with no collection number.
    if not tasks:
        raise SystemExit("FAIL LOUD: no residence rows on a gnn arm -- S0.b has nothing to read")
    out["residence_summary"] = residence_summary(
        tasks, batches or [], out["decile_summary"]["bounds"],
        float(out["averageWaitTime"]), int(c.get("residence_unstamped", 0)))
    r = out["residence_summary"]
    if r["reconstruction_error"] is None or r["reconstruction_error"] > 0.01:
        raise SystemExit(f"FAIL LOUD: decomposition does not reconcile "
                         f"({r['reconstruction_error']})")
out["env"] = {k: v for k, v in (doc.get("run_provenance") or {}).get("env", {}).items() if v}
out["code"] = (doc.get("run_provenance") or {}).get("code")
json.dump(out, open(summary + ".partial", "w"), indent=1)
extra = ""
if kind == "gnn":
    r = out["residence_summary"]
    extra = (f"  | wait {r['average_wait_time_s']:.3f} = HOL {r['head_of_line_s']:.3f}"
             f" + COLLECT {r['collection_s']:.3f} + place {r['placement_s']:.3f}"
             f"  batches {r['n_batches']} size {r['mean_batch_size']:.2f}")
print(f"[RESULT] {arm} servers={servers} {workload}  elapsed={out['averageElapsedTime']:.3f}"
      f"  queue={out['averageQueueTime']:.3f}  wall={wall}s{extra}")
PY
mv "$SUMMARY.partial" "$SUMMARY"
rm -f "$RAW"
echo "=== task $T done: $ARM ==="
