#!/bin/bash
# scheduler_residence_v1 R3 -- does topology lopsidedness predict the queue blow-up?
#
# Registered in docs/lineages/scheduler_residence_v1.md, Amendment 1, signed before any R3
# cell had been served a task. R1 found that the one cell of three WITHOUT a queue blow-up is
# separated from the other two with no overlap on two structural statistics -- every client
# can reach >= 2 servers (the others have a client that can reach exactly 1), and its
# clients-per-server imbalance is 3 against 8 and 7. Three cells is three cells. This is the
# sample size.
#
# 12 cells x (1 reactive + 4 gnn + 4 mpoff) = 108 arms. The cells were minted by
# scheduler_residence_v1_r3_cells.py, each differing from cell_s9001_f4000_pg16 in EXACTLY one
# flattened field (network.topology.seed, asserted field-by-field), and selected to span the
# structural range BEFORE any of them was served a task. The manifest carries the seeds.
#
# The unit of replication is the CELL: the dependent variable is each cell's median excess
# queue over ITS OWN reactive arm, which is why every cell runs its own reactive.
#
# DECLARED IN ADVANCE: seeds are 1, 2, 4, 5 -- seed 3's checkpoint deterministically livelocks
# the simulator on cell_s7901 (drainable_objective_v1 Phase C) and whether it does so on a
# fresh topology draw is unknown. It is excluded by name rather than risked, so a livelock
# cannot silently drop a cell to n = 3.
#
#   sbatch --array=0-107%12 scripts_cosim/datalab/scheduler_residence_v1_r3_gate.sbatch
#
#SBATCH --job-name=sr-r3
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/sr-r3-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/sr-r3-%A_%a.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --array=0-107

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
WORKLOAD="$GATE/workloads/drainable_f4000_n50000.json"
MANIFEST=simulation_data/scheduler_residence_v1/r3_cells.json
OUT_DIR="$GATE/results/sr_r3"
mkdir -p "$OUT_DIR"
[[ -f "$WORKLOAD" ]] || { echo "FAIL LOUD: workload missing" >&2; exit 1; }
[[ -f "$MANIFEST" ]] || { echo "FAIL LOUD: R3 cell manifest missing: $MANIFEST" >&2; exit 1; }

# The cells come from the manifest, in its own order -- never re-derived here, so the arms and
# the read cannot disagree about which 12 cells the selection picked.
readarray -t CELLS < <(python3 -c "
import json,sys
m=json.load(open('$MANIFEST'))
sel=m.get('selected') or []
if len(sel)!=12: sys.exit(f'FAIL LOUD: manifest carries {len(sel)} selected cells, expected 12')
print('\n'.join(c['cell'] for c in sel))
")
[[ ${#CELLS[@]} -eq 12 ]] || { echo "FAIL LOUD: read ${#CELLS[@]} cells from manifest" >&2; exit 1; }

SEEDS=(1 2 4 5)     # seed 3 excluded by name; see the header

export HEROSIM_PEER_EXCHANGE=1
export HEROSIM_SERVER_ONLY_REPLICAS=1
export HEROSIM_WARMTH_PHYSICS=node_disk_v2
export PYTHONHASHSEED=0
export HEROSIM_GNN_DEVICE=cpu
export SIM_FORCE_FULL_STATS=1

T=${SLURM_ARRAY_TASK_ID}
C=$(( T / 9 )); I=$(( T % 9 ))
CELL="${CELLS[$C]}"
CFG="$GATE/configs/${CELL}.json"
[[ -f "$CFG" ]] || { echo "FAIL LOUD: cell config missing: $CFG" >&2; exit 1; }

if [[ $I -eq 0 ]]; then
  POLICY=knative_network; ARMNAME=reactive; SEED=0
  ARM="${CELL}__reactive"
else
  POLICY=gnn
  if [[ $I -le 4 ]]; then ARMNAME=gnn; SEED=${SEEDS[$(( I - 1 ))]}
  else ARMNAME=mpoff; SEED=${SEEDS[$(( I - 5 ))]}; fi
  ARM="${CELL}__${ARMNAME}_s${SEED}"
  CK="models/drainable-objective-v1-v1-${ARMNAME}-lr2e3-seed${SEED}.pt"
  [[ -f "$CK" ]] || { echo "FAIL LOUD: missing checkpoint $CK" >&2; exit 1; }
  SC="${CK%.pt}.contract.json"
  [[ -f "$SC" ]] || { echo "FAIL LOUD: no sidecar for $CK" >&2; exit 1; }
  python3 - "$SC" "$ARMNAME" <<'PY'
import json, sys
sc, arm = json.load(open(sys.argv[1])), sys.argv[2]
if sc.get("disable_message_passing") is not (arm == "mpoff"):
    raise SystemExit(f"FAIL LOUD: sidecar disable_message_passing="
                     f"{sc.get('disable_message_passing')!r} for arm {arm}")
PY
  export GNN_MODEL_PATH="$CK"
  export GNN_DECODE_MODE=masked_topo
  export GNN_BATCH_BY_PEER_GROUP=1
  if [[ $ARMNAME == mpoff ]]; then export GNN_DISABLE_MESSAGE_PASSING=1; fi
fi

SUMMARY="$OUT_DIR/${ARM}.summary.json"
echo "[TASK $T] cell=$CELL arm=$ARM policy=$POLICY -> $SUMMARY"
if [[ -f "$SUMMARY" ]]; then echo "[TASK $T] exists, not re-running"; exit 0; fi

RAW="$OUT_DIR/${ARM}.raw.json"
python3 src/executesimulation.py --config "$CFG" --workload "$WORKLOAD" --policy "$POLICY" --output "$RAW"

python3 - "$RAW" "$SUMMARY" "$ARM" "$CELL" "$POLICY" "$ARMNAME" <<'PY'
import json, sys
sys.path.insert(0, ".")
from scripts_cosim.offline_live_transfer_v1_decile_read import decile_queue_summary

raw, summary, arm, cell, policy, armname = sys.argv[1:7]
doc = json.load(open(raw)); st = doc.get("stats") or doc
out = {k: st.get(k) for k in (
    "num_tasks", "total_rtt", "averageElapsedTime", "averageQueueTime", "averageWaitTime",
    "totalPeerExchangeTime", "totalPeerRendezvousWait", "endTime", "schedulerCounters")}
_se = st.get("scaleEvents")
out["scaleEventCount"] = len(_se) if isinstance(_se, list) else _se
out["decile_summary"] = decile_queue_summary(st.get("taskResults") or [])
out["arm"], out["cell"], out["policy_name"], out["arm_kind"] = arm, cell, policy, armname
c = out.get("schedulerCounters") or {}
for k in ("residence_tasks", "residence_batches", "queue_range_records"):
    c.pop(k, None)     # 108 arms do not each need a 50,000-row trace; R0 already has it
out["env"] = {k: v for k, v in (doc.get("run_provenance") or {}).get("env", {}).items() if v}
out["code"] = (doc.get("run_provenance") or {}).get("code")
json.dump(out, open(summary + ".partial", "w"), indent=1)
print(f"[RESULT] {arm}  elapsed={out['averageElapsedTime']:.3f}  "
      f"queue={out['averageQueueTime']:.3f}  wait={out['averageWaitTime']:.3f}")
PY
mv "$SUMMARY.partial" "$SUMMARY"
rm -f "$RAW"
echo "=== task $T done: $ARM ==="
