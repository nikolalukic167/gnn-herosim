#!/bin/bash
# peer_only_v1 -- the live gate at R0 (6 servers, f4000) and R3 (80 servers, f300), on the
# cells partial_state_v3 P3 minted. docs/lineages/peer_only_v1.md.
#
# Task table (SPECS below is the single source of truth; 4 cells x 4 checkpoint seeds each):
#   A0     tasks   0-3    v3 gnn s1 and v3 mpoff s1 re-served on cs6s9001 and cs80s9001 -- must
#                         reproduce partial_state_v3 P3's mean elapsed to three decimals
#   Phase A tasks  4-35   peeronly (516) x {R0, R3} x 4 cells x seeds 1,2,4,5
#   Phase B tasks 36-131  {gnn, mpoff, peeronly} (1670) x {R0, R3} x 4 cells x 4 seeds
#   B3      tasks 132-227 {peeronly, mpoff} (1670) x R3 only x 4 cells x the OTHER 12 seeds --
#                         AMENDMENT 1: B2's n = 16 is 4 checkpoints x 4 cells, and a claim about
#                         an architecture has the checkpoint as its unit. Bar in the read module.
#   B4      tasks 228-275 mpoff (516, served as corpus tag v3ext) x R3 x 4 cells x those 12 --
#                         AMENDMENT 2: the same fix for clause 3, the comparison against the
#                         best pointwise arm, which was also read on 4 checkpoints.
#   B5b     tasks 532-627 the same two arms at R0 (6 servers) x 4 cells x the other 12 seeds,
#                         so every rung of the B5 ladder rests on 16 checkpoints.
#   C4      tasks 724-787 gnnres (1670) x R3 x 4 cells x 16 -- AMENDMENT 9, the residual
#                         bipartite arm; submit 724-771, 772-787
#   C1      tasks 628-723 gnn (1670) x {R3, R0} x 4 cells x the OTHER 12 seeds -- AMENDMENT 6,
#                         the bipartite arm at power; submit 628-675, 676-723 (%12 each)
#   B5      tasks 276-531 {peeronly, mpoff} (1670) x {R1=12, R2=24 servers} x 4 cells x 16 --
#                         AMENDMENT 3: the middle of the cluster-size curve. Reactive at those
#                         rungs is already in results/psv3_p3, so only the learned arms run.
#
# Reactive, gnn(516) and mpoff(516) arms are REUSED from results/psv3_p3 under A0. Every arm
# name carries cell, rung, corpus, arm and seed. Same physics, window, poll and batch size as
# P3; the R3 rung is saturated and read relative to reactive or paired between arms only.
#
#   sbatch --array=0-35%12   scripts_cosim/datalab/peer_only_v1_gate.sbatch   # A0 + Phase A
#   sbatch --array=36-83%12  scripts_cosim/datalab/peer_only_v1_gate.sbatch   # Phase B, half 1
#   sbatch --array=84-131%12 scripts_cosim/datalab/peer_only_v1_gate.sbatch   # Phase B, half 2
#   sbatch --array=132-179%12 scripts_cosim/datalab/peer_only_v1_gate.sbatch  # B3, half 1
#   sbatch --array=180-227%12 scripts_cosim/datalab/peer_only_v1_gate.sbatch  # B3, half 2
#   sbatch --array=228-275%12 scripts_cosim/datalab/peer_only_v1_gate.sbatch  # B4
#   B5 is 256 tasks: submit 276-323, 324-371, 372-419, 420-467, 468-515, 516-531 (%12 each)
#   B5b is 96 tasks at R0: submit 532-579, 580-627 (%12 each)
#
# Phase B goes in two halves because the account's MaxSubmit is 50 and an array counts every
# task against it: --array=36-131 (96 tasks) is rejected with AssocMaxSubmitJobLimit at
# submission time, after the training array has already drained (peer_only_v1, 2026-09-17).
#
#SBATCH --job-name=po-gate
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/po-gate-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/po-gate-%A_%a.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --array=0-787

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
OUT_DIR="$GATE/results/po_v1"
mkdir -p "$OUT_DIR"
# Indices 0,1 are R0,R3 and MUST NOT MOVE -- tasks 0-275 index into these by position.
# B5 appends R1 (12 servers) and R2 (24) at indices 2,3; same cells and workloads
# partial_state_v3 P3 used, so its reactive baselines at those rungs are directly reusable.
RUNG_TAGS=(R0 R3 R1 R2); RUNG_SERVERS=(6 80 12 24)
RUNG_WORKLOADS=(drainable_f4000_n50000 drainable_f300_n50000 drainable_f2000_n50000 drainable_f1000_n50000)
TSEEDS=(9001 9002 9003 9005)
CKSEEDS=(1 2 4 5)

export HEROSIM_PEER_EXCHANGE=1
export HEROSIM_SERVER_ONLY_REPLICAS=1
export HEROSIM_WARMTH_PHYSICS=node_disk_v2
export PYTHONHASHSEED=0
export HEROSIM_GNN_DEVICE=cpu
export SIM_FORCE_FULL_STATS=1

# --- task table: (corpus, arm, rung index, cell index, checkpoint seed) --------------------
CORP=(); ARMS=(); RI=(); CI=(); SEED=()
# A0: v3 gnn/mpoff seed 1 on cell index 0 of each rung
for r in 0 1; do for a in gnn mpoff; do CORP+=("v3"); ARMS+=("$a"); RI+=("$r"); CI+=(0); SEED+=(1); done; done
# Phase A
for r in 0 1; do for c in 0 1 2 3; do for s in "${CKSEEDS[@]}"; do CORP+=("516"); ARMS+=("peeronly"); RI+=("$r"); CI+=("$c"); SEED+=("$s"); done; done; done
# Phase B
for a in gnn mpoff peeronly; do for r in 0 1; do for c in 0 1 2 3; do for s in "${CKSEEDS[@]}"; do CORP+=("1670"); ARMS+=("$a"); RI+=("$r"); CI+=("$c"); SEED+=("$s"); done; done; done; done
# B3 -- AMENDMENT 1 (2026-09-17), tasks 132-227. The 12 trained checkpoints the gate never
# served, peeronly and mpoff only, at R3 only. APPENDED so indices 0-131 keep their arms and
# their summaries are never re-read; B3_SEEDS in peer_only_v1_read.py is this list plus CKSEEDS.
B3SEEDS=(3 6 7 8 9 10 11 12 13 14 15 16)
for a in peeronly mpoff; do for c in 0 1 2 3; do for s in "${B3SEEDS[@]}"; do CORP+=("1670"); ARMS+=("$a"); RI+=(1); CI+=("$c"); SEED+=("$s"); done; done; done
# B4 -- AMENDMENT 2, tasks 228-275. The same 12 checkpoints for mpoff on the 516 corpus, so
# clause 3 (peeronly_1670 vs the best pointwise arm) stops being a 4-checkpoint read too.
# Corpus tag v3ext: mpoff_516 IS partial_state_v3's mpoff, and A0 proves the re-serve is
# bit-identical, so these extend the SAME arm. The read relabels v3ext_* to 516_*.
for c in 0 1 2 3; do for s in "${B3SEEDS[@]}"; do CORP+=("v3ext"); ARMS+=("mpoff"); RI+=(1); CI+=("$c"); SEED+=("$s"); done; done
# B5 -- AMENDMENT 3, tasks 276-531. The middle of the cluster-size curve: peeronly and mpoff
# on the 1670 corpus at R1 (12 servers) and R2 (24), all 16 checkpoints, 4 cells. Reactive at
# those rungs already exists in results/psv3_p3, which is why only the learned arms are run.
ALLSEEDS=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16)
for a in peeronly mpoff; do for r in 2 3; do for c in 0 1 2 3; do for s in "${ALLSEEDS[@]}"; do CORP+=("1670"); ARMS+=("$a"); RI+=("$r"); CI+=("$c"); SEED+=("$s"); done; done; done; done
# B5b -- tasks 532-627. R0 (6 servers) was left at the original 4 checkpoints when B5 filled in
# R1 and R2, which made the ladder's bottom rung the one that could not be read. Same 12 extra
# checkpoints, rung index 0.
for a in peeronly mpoff; do for c in 0 1 2 3; do for s in "${B3SEEDS[@]}"; do CORP+=("1670"); ARMS+=("$a"); RI+=(0); CI+=("$c"); SEED+=("$s"); done; done; done
# C1 -- AMENDMENT 6 (2026-09-18), tasks 628-723. The gnn arm (PeerConv + the bipartite GIN) was
# never extended past Phase B's 4 checkpoints, so clause 7 -- "the arm that wins is not the
# GNN", peeronly beats gnn by 22.42 % -- is a 4-CHECKPOINT number, the exact shape B3 already
# caught and halved once. Same 12 extra checkpoints, at R3 (rung index 1) and R0 (index 0), so
# the bipartite penalty is read at power on both ends of the server ladder. APPENDED: tasks
# 0-627 keep their arms.
for r in 1 0; do for c in 0 1 2 3; do for s in "${B3SEEDS[@]}"; do CORP+=("1670"); ARMS+=("gnn"); RI+=("$r"); CI+=("$c"); SEED+=("$s"); done; done; done
# C4 -- AMENDMENT 9, tasks 724-787. gnnres (residual bipartite MP) at R3, all 16 checkpoints.
# The LIVE half of C4: R3 is where C1 measured the bipartite penalty at power, so it is where
# the repair has to show up. APPENDED: tasks 0-723 keep their arms.
for c in 0 1 2 3; do for s in "${ALLSEEDS[@]}"; do CORP+=("1670"); ARMS+=("gnnres"); RI+=(1); CI+=("$c"); SEED+=("$s"); done; done
NTASKS=${#ARMS[@]}
T=${SLURM_ARRAY_TASK_ID}
if (( T >= NTASKS )); then echo "[TASK $T] beyond the $NTASKS declared arms"; exit 0; fi
CORPUS=${CORP[$T]}; ARM=${ARMS[$T]}; R=${RI[$T]}; C=${CI[$T]}; S=${SEED[$T]}
RUNG=${RUNG_TAGS[$R]}; NS=${RUNG_SERVERS[$R]}; TSEED=${TSEEDS[$C]}
WORKLOAD="$GATE/workloads/${RUNG_WORKLOADS[$R]}.json"
CELL="cs${NS}s${TSEED}"
CFG="$GATE/configs/${CELL}.json"
[[ -f "$WORKLOAD" ]] || { echo "FAIL LOUD: workload missing: $WORKLOAD" >&2; exit 1; }
[[ -f "$CFG" ]] || { echo "FAIL LOUD: cell config missing: $CFG (mint with partial_state_v3_p3_gate.sbatch task 0)" >&2; exit 1; }
python3 - "$CFG" "$NS" <<'PY'
import json, sys
cfg, want = json.load(open(sys.argv[1])), int(sys.argv[2])
if int(cfg["nodes"]["server_nodes"]["count"]) != want:
    raise SystemExit(f"FAIL LOUD: {sys.argv[1]} declares {cfg['nodes']['server_nodes']['count']} servers, rung expects {want}")
if float(cfg["scheduler"]["batch_timeout"]) != 16.0:
    raise SystemExit("FAIL LOUD: batch_timeout moved -- every policy constant is held fixed")
PY

case "$CORPUS" in
  v3|v3ext) CK="models/partial-state-v3-${ARM}-lr2e3-seed${S}.pt" ;;
  516)  CK="models/peer-only-v1-516-${ARM}-lr2e3-seed${S}.pt" ;;
  1670) CK="models/peer-only-v1-1670-${ARM}-lr2e3-seed${S}.pt" ;;
esac
[[ -f "$CK" ]] || { echo "FAIL LOUD: missing checkpoint $CK" >&2; exit 1; }
SC="${CK%.pt}.contract.json"
[[ -f "$SC" ]] || { echo "FAIL LOUD: no sidecar for $CK" >&2; exit 1; }
python3 - "$SC" "$ARM" <<'PY'
import json, sys
sc, arm = json.load(open(sys.argv[1])), sys.argv[2]
want = {"disable_message_passing": arm == "mpoff", "mp_peer_edges": True,
        "partial_state_contract": "partial_state_v3", "partial_state_feature_dim": 22,
        # C4: gnnres and gnn are otherwise indistinguishable here, and mp_residual is what
        # separates them. Pinned on BOTH so a mislabelled checkpoint cannot be served as the
        # wrong arm -- the flag is weight-visible, but the model is BUILT from this sidecar,
        # so strict loading would not catch it.
        "mp_residual": arm == "gnnres"}
if arm == "peeronly":
    want["mp_platform_edges"] = False
elif "mp_platform_edges" in sc:
    want["mp_platform_edges"] = True
for k, v in want.items():
    if sc.get(k) != v:
        raise SystemExit(f"FAIL LOUD: sidecar {k}={sc.get(k)!r}, arm {arm} expects {v!r}")
PY
unset PARTIAL_STATE_CONTRACT GNN_MP_PLATFORM_EDGES_OFF || true
export GNN_MODEL_PATH="$CK"
export GNN_DECODE_MODE=masked_topo
export GNN_BATCH_BY_PEER_GROUP=1
if [[ $ARM == mpoff ]]; then export GNN_DISABLE_MESSAGE_PASSING=1; fi

ARMNAME="${CELL}__${RUNG}__${CORPUS}_${ARM}_s${S}"
SUMMARY="$OUT_DIR/${ARMNAME}.summary.json"
echo "[TASK $T] rung=$RUNG servers=$NS cell=$CELL corpus=$CORPUS arm=$ARM seed=$S ck=$CK -> $SUMMARY"
if [[ -f "$SUMMARY" ]]; then echo "[TASK $T] exists, not re-running"; exit 0; fi
RAW="$OUT_DIR/${ARMNAME}.raw.json"
START=$(date +%s)
python3 src/executesimulation.py --config "$CFG" --workload "$WORKLOAD" --policy gnn --output "$RAW"
WALL=$(( $(date +%s) - START ))

python3 - "$RAW" "$SUMMARY" "$ARMNAME" "$CELL" "$RUNG" "$NS" "${RUNG_WORKLOADS[$R]}" "$CORPUS" "$ARM" "$S" "$WALL" <<'PY'
import json, sys
sys.path.insert(0, ".")
from scripts_cosim.offline_live_transfer_v1_decile_read import decile_queue_summary
from scripts_cosim.scheduler_residence_v1_read import residence_summary
raw, summary, arm, cell, rung, servers, workload, corpus, kind, seed, wall = sys.argv[1:12]
doc = json.load(open(raw)); st = doc.get("stats") or doc
out = {k: st.get(k) for k in (
    "num_tasks", "total_rtt", "averageElapsedTime", "averageQueueTime", "averageWaitTime",
    "totalPeerExchangeTime", "totalPeerRendezvousWait", "endTime", "schedulerCounters")}
_se = st.get("scaleEvents")
out["scaleEventCount"] = len(_se) if isinstance(_se, list) else _se
out["decile_summary"] = decile_queue_summary(st.get("taskResults") or [])
out.update(arm=arm, cell=cell, rung=rung, servers=int(servers), workload=workload, corpus=corpus,
           arm_kind=kind, checkpoint_seed=int(seed), policy_name="gnn", wallclock_s=int(wall),
           representation="v3")
c = out.get("schedulerCounters") or {}
tasks = c.pop("residence_tasks", None); batches = c.pop("residence_batches", None)
c.pop("queue_range_records", None)
if tasks:
    out["residence_summary"] = residence_summary(tasks, batches or [], out["decile_summary"]["bounds"],
                                                 float(out["averageWaitTime"]), int(c.get("residence_unstamped", 0)))
env = {k: v for k, v in (doc.get("run_provenance") or {}).get("env", {}).items() if v}
out["env"] = env
out["code"] = (doc.get("run_provenance") or {}).get("code")
# The served architecture must be the one the arm claims, read back from provenance.
po = env.get("GNN_MP_PLATFORM_EDGES_OFF")
if kind == "peeronly" and po not in (None, "1", "true", "yes"):
    raise SystemExit(f"FAIL LOUD: peeronly arm served with GNN_MP_PLATFORM_EDGES_OFF={po!r}")
if kind != "peeronly" and po in ("1", "true", "yes"):
    raise SystemExit(f"FAIL LOUD: {kind} arm served with the GIN off")
json.dump(out, open(summary + ".partial", "w"), indent=1)
print(f"[RESULT] {arm} elapsed={out['averageElapsedTime']:.3f} queue={out['averageQueueTime']:.3f} "
      f"peer_x={float(out['totalPeerExchangeTime'] or 0):.1f} rdv={float(out['totalPeerRendezvousWait'] or 0):.1f} wall={wall}s")
PY
mv "$SUMMARY.partial" "$SUMMARY"
rm -f "$RAW"
echo "=== task $T done: $ARMNAME ==="
