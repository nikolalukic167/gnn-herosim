#!/bin/bash
# accel_replica_v1: the default rule must reproduce R1.1 exactly. One live run per arm family on the same cell (4000 events) at the base commit (BASE_WT, R1.1)
# and at the new commit (NEW_WT, default rule): same per-task placements and times, same scale events, same totals.
#   sbatch --array=0-4 --export=ALL,BASE_WT=..,NEW_WT=..,OUT=..,INPUTS=<inputs dir with grounded_<RUNG>/>,TOPO=9903,RUNG=m46p4556 accel_default_identity.sbatch
#SBATCH --job-name=accel-ident
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/accel-ident-%A_%a.out
#SBATCH --partition=CPU-amd
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
set -uo pipefail
BASE_WT=${BASE_WT:?}; NEW_WT=${NEW_WT:?}; OUT=${OUT:?}; INPUTS=${INPUTS:?}; TOPO=${TOPO:?}; RUNG=${RUNG:?}; EVENTS=${EVENTS:-4000}
POLS=(knative_network random_network peer_greedy_selfpredict_network peer_greedy_network_batch peer_greedy_network_cd)
POL=${POLS[${SLURM_ARRAY_TASK_ID:?}]}
mkdir -p "$OUT"
eval "$(micromamba shell hook --shell bash)"; micromamba activate gnn
for v in $(compgen -e | grep -E '^(HEROSIM_|GNN_|COSIM_|LIVE_AUDIT_|GATE_|SIM_|QUEUE_FEATURE|MAX_PLACEMENT|DECLARED_SLATE)'); do unset "$v"; done
export HEROSIM_PY=python3 HEROSIM_AUDIT_INPUTS=$INPUTS TS=1.0 SIM_FORCE_FULL_STATS=1 HEROSIM_SHARED_AUTOSCALER=0 GATE_FIXED_POLICY_TIME_SCALE=1.0
for W in base new; do
  WT=$BASE_WT; [[ $W == new ]] && WT=$NEW_WT
  echo "== $POL at $W: $(cd "$WT" && git rev-parse --short HEAD)"
  ( cd "$WT" && timeout --kill-after=60 2400 bash scripts_cosim/physics_audit/run_cell.sh "$TOPO" g0 "$RUNG" "$POL" "$OUT/${POL}_$W.raw.json" "$EVENTS" > "$OUT/${POL}_$W.log" 2>&1 ) || echo "rc=$? $W"
done
python3 - "$OUT/${POL}_base.raw.json" "$OUT/${POL}_new.raw.json" <<'PY'
import json, sys
a, b = (json.load(open(p)) for p in sys.argv[1:3])
def digest(d):
    s = d["stats"]
    tasks = [(t["taskId"], t["executionNode"], t["executionPlatform"], round(t["doneTime"], 9), round(t["startedTime"], 9)) for t in s["taskResults"]]
    ev = [(e["name"], e["action"], round(e["timestamp"], 9), e.get("count"), e.get("cause")) for e in s["scaleEvents"]]
    return {"total_rtt": d["total_rtt"], "num_tasks": d["num_tasks"], "endTime": s["endTime"], "tasks": tasks, "scale_events": ev,
            "averageElapsedTime": s["averageElapsedTime"], "averageQueueTime": s["averageQueueTime"], "unusedPlatforms": s["unusedPlatforms"]}
da, db = digest(a), digest(b)
bad = [k for k in da if da[k] != db[k]]
print("IDENTICAL" if not bad else f"DIFFERENT in {bad}", "| tasks", len(da["tasks"]), "scale events", len(da["scale_events"]), "total_rtt", da["total_rtt"])
sys.exit(1 if bad else 0)
PY
