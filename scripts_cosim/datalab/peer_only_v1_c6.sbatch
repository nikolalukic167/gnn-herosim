#!/bin/bash
# peer_only_v1 C6 (AMENDMENT 11) -- the live decoder's OWN queue-discrimination probe, at R3.
#
# C1 put the bipartite penalty at 80 servers and found nothing at 6; C3 measured the GIN's
# over-smoothing at 6, where the penalty is absent. No cached graphs exist at 80 servers, so
# the offline probe cannot be moved. But `record_queue_feature_discrimination` runs
# unconditionally inside the live decode and records, per task, the queue of the platform the
# arm CHOSE against the shortest it could have chosen -- and the gate deletes it with the raw
# JSON. This keeps it.
#
#   gnn and peeronly (1670) x R3 x 4 cells x 16 checkpoints = 128 arms
#   sbatch --array=0-47%12 ... then 48-95, 96-127 (MaxSubmit = 50 counts every array task)
#
#SBATCH --job-name=po-c6
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/po-c6-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/po-c6-%A_%a.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --array=0-127

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
OUT_DIR="$GATE/results/po_v1_c6"
mkdir -p "$OUT_DIR"
TSEEDS=(9001 9002 9003 9005)
CKSEEDS=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16)

export HEROSIM_PEER_EXCHANGE=1
export HEROSIM_SERVER_ONLY_REPLICAS=1
export HEROSIM_WARMTH_PHYSICS=node_disk_v2
export PYTHONHASHSEED=0
export HEROSIM_GNN_DEVICE=cpu
export SIM_FORCE_FULL_STATS=1
WORKLOAD="$GATE/workloads/drainable_f300_n50000.json"   # R3's workload, held
[[ -f "$WORKLOAD" ]] || { echo "FAIL LOUD: workload missing: $WORKLOAD" >&2; exit 1; }

ARMS=(); CI=(); SEED=()
for a in gnn peeronly; do for c in 0 1 2 3; do
  for s in "${CKSEEDS[@]}"; do ARMS+=("$a"); CI+=("$c"); SEED+=("$s"); done
done; done
NTASKS=${#ARMS[@]}
T=${SLURM_ARRAY_TASK_ID}
if (( T >= NTASKS )); then echo "[TASK $T] beyond the $NTASKS declared arms"; exit 0; fi
ARM=${ARMS[$T]}; C=${CI[$T]}; S=${SEED[$T]}
TSEED=${TSEEDS[$C]}
CELL="cs80s${TSEED}"
CFG="$GATE/configs/${CELL}.json"
[[ -f "$CFG" ]] || { echo "FAIL LOUD: cell config missing: $CFG" >&2; exit 1; }

ARMNAME="${CELL}__R3__1670_${ARM}_s${S}"
SUMMARY="$OUT_DIR/${ARMNAME}.probe.json"
echo "[TASK $T] cell=$CELL arm=$ARM seed=$S -> $SUMMARY"
if [[ -f "$SUMMARY" ]]; then echo "[TASK $T] exists, not re-running"; exit 0; fi
RAW="$OUT_DIR/${ARMNAME}.raw.json"

CK="models/peer-only-v1-1670-${ARM}-lr2e3-seed${S}.pt"
[[ -f "$CK" ]] || { echo "FAIL LOUD: missing checkpoint $CK" >&2; exit 1; }
SC="${CK%.pt}.contract.json"
[[ -f "$SC" ]] || { echo "FAIL LOUD: no sidecar for $CK" >&2; exit 1; }
python3 - "$SC" "$ARM" <<'PY'
import json, sys
sc, arm = json.load(open(sys.argv[1])), sys.argv[2]
want = {"disable_message_passing": False, "mp_peer_edges": True,
        "mp_platform_edges": arm != "peeronly",
        "partial_state_contract": "partial_state_v3", "partial_state_feature_dim": 22}
for k, v in want.items():
    if sc.get(k) != v:
        raise SystemExit(f"FAIL LOUD: sidecar {k}={sc.get(k)!r}, arm {arm} expects {v!r}")
PY
unset PARTIAL_STATE_CONTRACT GNN_MP_PLATFORM_EDGES_OFF || true
export GNN_MODEL_PATH="$CK"
export GNN_DECODE_MODE=masked_topo
export GNN_BATCH_BY_PEER_GROUP=1
# peeronly is served exactly as the gate serves it: the flag is weight-invisible, so it is the
# sidecar that makes this arm peeronly and the env var that makes the SERVED model match it.
if [[ $ARM == peeronly ]]; then export GNN_MP_PLATFORM_EDGES_OFF=1; fi

START=$(date +%s)
python3 src/executesimulation.py --config "$CFG" --workload "$WORKLOAD" --policy gnn --output "$RAW"
WALL=$(( $(date +%s) - START ))

# Keep ONLY the probe block plus the elapsed/queue context, then drop the ~80 MB raw.
python3 - "$RAW" "$SUMMARY" "$ARMNAME" "$CELL" "$ARM" "$S" "$WALL" <<'PY'
import json, os, sys
raw, summary, arm, cell, kind, seed, wall = sys.argv[1:8]
doc = json.load(open(raw)); st = doc.get("stats") or doc


def find_probe(*roots):
    """The probe block, wherever executesimulation put it. Searched, not assumed -- the
    stats layout has moved before and a wrong guess here writes an empty summary."""
    for root in roots:
        if not isinstance(root, dict):
            continue
        if isinstance(root.get("queue_feature_discrimination"), dict):
            return root["queue_feature_discrimination"]
        for key in ("decode_stats", "gnn_decode", "gnnDecodeStats"):
            blk = root.get(key)
            if isinstance(blk, dict) and isinstance(blk.get("queue_feature_discrimination"), dict):
                return blk["queue_feature_discrimination"]
    return None


probe = find_probe(doc, st)
if probe is None:
    # write_run_decode_stats also drops a sidecar beside the output; use it if it is there.
    side = os.path.splitext(raw)[0] + ".decode_stats.json"
    if os.path.isfile(side):
        probe = find_probe(json.load(open(side)))
if probe is None:
    raise SystemExit(
        "FAIL LOUD: no queue_feature_discrimination block in the result or its decode_stats "
        f"sidecar. Keys at top level: {sorted(doc)[:20]}; in stats: {sorted(st)[:20]}. "
        "Do not write an empty summary."
    )
out = {k: st.get(k) for k in ("num_tasks", "averageElapsedTime", "averageQueueTime")}
out.update(arm=arm, cell=cell, arm_kind=kind, checkpoint_seed=int(seed), rung="R3",
           corpus="1670", wallclock_s=int(wall), queue_feature_discrimination=probe,
           code=(doc.get("run_provenance") or {}).get("code"))
json.dump(out, open(summary, "w"), indent=1)
print(f"[probe] {summary}: n_tasks={probe.get('n_tasks')} "
      f"confident_worse={probe.get('confident_worse_queue_rate')}")
PY
rm -f "$RAW"
echo "[TASK $T] done in ${WALL}s"
