#!/bin/bash
# peer_only_v1 B7 -- the CLIENT axis. docs/lineages/peer_only_v1.md.
#
# Every rung of B5 varied servers and held clients at 20. B7 holds servers at 6 and varies the
# client count, which is the axis an edge deployment is actually described by, and the only
# scaling axis that leaves candidates-per-task inside the corpus's support: a task's candidate
# set is the servers its client can reach, so adding clients does not add candidates. Measured
# on the minted cells: 3.85 / 3.83 / 3.70 / 3.58 / 3.60 mean candidates at 5 / 10 / 20 / 40 / 80
# clients, all 0.72-0.77x the corpus maximum of 5. The 80-SERVER rung of B5 was 9.6x out.
#
# The 20-client rung already exists -- it IS cs6s900X, with 16 checkpoints per learned arm and
# reactive, from B5b. Only 5, 10, 40 and 80 clients are minted here.
#
#   Task table (4 client rungs x 4 topology seeds):
#     reactive  tasks   0-15   knative_network, 1 run per cell (deterministic, no ckpt seed)
#     learned   tasks  16-527  {peeronly, mpoff} (1670) x 4 rungs x 4 cells x 16 checkpoints
#     B8        tasks 528-655  mpoff (516, tag v3ext) x {40, 80} clients x 4 cells x 16 --
#                              AMENDMENT 5: clause 3 at the UNSATURATED rungs
#     C2        tasks 656-783  gnn (1670) x {40, 80} clients x 4 cells x 16 -- AMENDMENT 7,
#                              the bipartite arm at the unsaturated rungs, never served there
#
#   sbatch --array=0-15%12 scripts_cosim/datalab/peer_only_v1_client_gate.sbatch   # baselines first
#   then 16-63, 64-111, ... in blocks of <= 48 (MaxSubmit = 50 counts every array task)
#
#SBATCH --job-name=po-cgate
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/po-cgate-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/po-cgate-%A_%a.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --array=0-847

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
OUT_DIR="$GATE/results/po_v1_clients"
MANIFEST="$GATE/configs/po_client_cells.json"
mkdir -p "$OUT_DIR"
CLIENTS=(5 10 40 80)          # the 20-client rung is cs6s900X, already measured
TSEEDS=(9001 9002 9003 9005)
CKSEEDS=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16)
BASECELL="$GATE/configs/cs6s9001.json"

export HEROSIM_PEER_EXCHANGE=1
export HEROSIM_SERVER_ONLY_REPLICAS=1
export HEROSIM_WARMTH_PHYSICS=node_disk_v2
export PYTHONHASHSEED=0
export HEROSIM_GNN_DEVICE=cpu
export SIM_FORCE_FULL_STATS=1
WORKLOAD="$GATE/workloads/drainable_f4000_n50000.json"   # the 6-server rung's workload, held
[[ -f "$WORKLOAD" ]] || { echo "FAIL LOUD: workload missing: $WORKLOAD" >&2; exit 1; }

T=${SLURM_ARRAY_TASK_ID}

# --- cells: task 0 mints, everyone else waits (datalab-pitfalls #7) ------------------------
if [[ $T -eq 0 ]]; then
  if [[ ! -f "$MANIFEST" ]]; then
    echo "[mint] $MANIFEST"
    python3 scripts_cosim/cluster_scale_v1_cells.py --base "$BASECELL" --sim-inputs data/nofs-ids \
      --vary clients --rungs 5,10,40,80 --seeds 9001,9002,9003,9005 \
      --out-dir "$GATE/configs" --manifest "$MANIFEST.partial" --prefix cc
    mv "$MANIFEST.partial" "$MANIFEST"
  fi
else
  for i in $(seq 1 180); do [[ -f "$MANIFEST" ]] && break; sleep 5; done
  [[ -f "$MANIFEST" ]] || { echo "FAIL LOUD: $MANIFEST never appeared (task 0 did not mint)" >&2; exit 1; }
fi

# --- task table ---------------------------------------------------------------------------
NC=(); CI=(); ARMS=(); SEED=()
for nc in "${CLIENTS[@]}"; do for c in 0 1 2 3; do
  NC+=("$nc"); CI+=("$c"); ARMS+=("reactive"); SEED+=(0)
done; done
for a in peeronly mpoff; do for nc in "${CLIENTS[@]}"; do for c in 0 1 2 3; do
  for s in "${CKSEEDS[@]}"; do NC+=("$nc"); CI+=("$c"); ARMS+=("$a"); SEED+=("$s"); done
done; done; done
# B8 -- AMENDMENT 5, tasks 528-655. mpoff on the 516 corpus (partial_state_v3's mpoff, served
# under the tag v3ext and relabelled 516_mpoff by the read) at the two UNSATURATED client rungs,
# so clause 3 stops being a single-operating-point read. APPENDED: tasks 0-527 keep their arms.
B8CLIENTS=(40 80)
for nc in "${B8CLIENTS[@]}"; do for c in 0 1 2 3; do
  for s in "${CKSEEDS[@]}"; do NC+=("$nc"); CI+=("$c"); ARMS+=("v3ext_mpoff"); SEED+=("$s"); done
done; done
# C2 -- AMENDMENT 7 (2026-09-18), tasks 656-783. The full gnn arm -- PeerConv AND the bipartite
# task<->platform GIN -- has never been served on the client axis at all. B7 found the
# programme's first UNSATURATED live win there, so this is the one place where "does a
# bipartite graph work in this environment" can be answered at the operating point that
# matters, rather than at the saturated 80-server rung where every earlier gnn reading was
# taken. All 16 checkpoints, both unsaturated rungs. APPENDED: tasks 0-655 keep their arms.
C2CLIENTS=(40 80)
for nc in "${C2CLIENTS[@]}"; do for c in 0 1 2 3; do
  for s in "${CKSEEDS[@]}"; do NC+=("$nc"); CI+=("$c"); ARMS+=("gnn"); SEED+=("$s"); done
done; done
# C4 -- AMENDMENT 9, tasks 784-847. gnnres at 80 clients, the registered unsaturated rung
# (C4_LIVE_CLIENTS) where peeronly beats reactive on its own. APPENDED: 0-783 keep their arms.
for c in 0 1 2 3; do
  for s in "${CKSEEDS[@]}"; do NC+=(80); CI+=("$c"); ARMS+=("gnnres"); SEED+=("$s"); done
done
NTASKS=${#ARMS[@]}
if (( T >= NTASKS )); then echo "[TASK $T] beyond the $NTASKS declared arms"; exit 0; fi
NCLIENTS=${NC[$T]}; C=${CI[$T]}; ARM=${ARMS[$T]}; S=${SEED[$T]}
TSEED=${TSEEDS[$C]}
CELL="cc${NCLIENTS}s${TSEED}"
CFG="$GATE/configs/${CELL}.json"
[[ -f "$CFG" ]] || { echo "FAIL LOUD: cell config missing: $CFG" >&2; exit 1; }

# The cell must differ from the 6-server base in exactly the client count and the seed.
python3 - "$CFG" "$BASECELL" "$NCLIENTS" <<'PY'
import json, sys
cfg, base, want = json.load(open(sys.argv[1])), json.load(open(sys.argv[2])), int(sys.argv[3])
if int(cfg["nodes"]["client_nodes"]["count"]) != want:
    raise SystemExit(f"FAIL LOUD: {sys.argv[1]} declares "
                     f"{cfg['nodes']['client_nodes']['count']} clients, rung expects {want}")
if int(cfg["nodes"]["server_nodes"]["count"]) != int(base["nodes"]["server_nodes"]["count"]):
    raise SystemExit("FAIL LOUD: the server count moved -- B7 varies CLIENTS only")
if float(cfg["scheduler"]["batch_timeout"]) != 16.0:
    raise SystemExit("FAIL LOUD: batch_timeout moved -- every policy constant is held fixed")
PY

# Resolve kind + corpus BEFORE the arm name is built. v3ext_mpoff is partial_state_v3's mpoff
# (516 corpus) re-served here; every other arm is a 1670 one. The name must carry the corpus or
# the two mpoffs collide on one summary path -- the idempotence-guard defect from 2026-09-16.
if [[ $ARM == v3ext_mpoff ]]; then ARM_KIND=mpoff; CORPUS_TAG=v3ext
else ARM_KIND=$ARM; CORPUS_TAG=1670; fi
ARMNAME="${CELL}__C${NCLIENTS}__${CORPUS_TAG}_${ARM_KIND}_s${S}"
SUMMARY="$OUT_DIR/${ARMNAME}.summary.json"
echo "[TASK $T] clients=$NCLIENTS cell=$CELL arm=$ARM seed=$S -> $SUMMARY"
if [[ -f "$SUMMARY" ]]; then echo "[TASK $T] exists, not re-running"; exit 0; fi
RAW="$OUT_DIR/${ARMNAME}.raw.json"

if [[ $ARM == reactive ]]; then
  POLICY=knative_network
else
  if [[ $CORPUS_TAG == v3ext ]]; then
    CK="models/partial-state-v3-${ARM_KIND}-lr2e3-seed${S}.pt"
  else
    CK="models/peer-only-v1-1670-${ARM_KIND}-lr2e3-seed${S}.pt"
  fi
  [[ -f "$CK" ]] || { echo "FAIL LOUD: missing checkpoint $CK" >&2; exit 1; }
  SC="${CK%.pt}.contract.json"
  [[ -f "$SC" ]] || { echo "FAIL LOUD: no sidecar for $CK" >&2; exit 1; }
  python3 - "$SC" "$ARM_KIND" <<'PY'
import json, sys
sc, arm = json.load(open(sys.argv[1])), sys.argv[2]
want = {"disable_message_passing": arm == "mpoff", "mp_peer_edges": True,
        "partial_state_contract": "partial_state_v3", "partial_state_feature_dim": 22,
        # C4: mp_residual is the only thing separating gnnres from gnn, and the model is BUILT
        # from this sidecar, so strict loading cannot catch a mislabel. Pinned on both.
        "mp_residual": arm == "gnnres"}
# mp_platform_edges postdates the partial_state_v3 checkpoints, so the key is ABSENT from
# their sidecars. Absent means the documented default (GIN on), and for mpoff it is moot --
# disable_message_passing=True already turns all message passing off. Demanding it
# unconditionally failed all 48 B8 arms of job 784522, correctly and loudly. Same rule as
# peer_only_v1_gate.sbatch: peeronly must declare False, everyone else is checked only if
# the key is there.
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
  if [[ $ARM_KIND == mpoff ]]; then export GNN_DISABLE_MESSAGE_PASSING=1; fi
  POLICY=gnn
fi

START=$(date +%s)
python3 src/executesimulation.py --config "$CFG" --workload "$WORKLOAD" --policy "$POLICY" --output "$RAW"
WALL=$(( $(date +%s) - START ))

python3 - "$RAW" "$SUMMARY" "$ARMNAME" "$CELL" "$NCLIENTS" "$ARM_KIND" "$S" "$WALL" "$CORPUS_TAG" <<'PY'
import json, sys
sys.path.insert(0, ".")
from scripts_cosim.offline_live_transfer_v1_decile_read import decile_queue_summary
raw, summary, arm, cell, nclients, kind, seed, wall, corpus = sys.argv[1:10]
doc = json.load(open(raw)); st = doc.get("stats") or doc
out = {k: st.get(k) for k in (
    "num_tasks", "total_rtt", "averageElapsedTime", "averageQueueTime", "averageWaitTime",
    "totalPeerExchangeTime", "totalPeerRendezvousWait", "endTime", "schedulerCounters")}
_se = st.get("scaleEvents")
out["scaleEventCount"] = len(_se) if isinstance(_se, list) else _se
out["decile_summary"] = decile_queue_summary(st.get("taskResults") or [])
out.update(arm=arm, cell=cell, clients=int(nclients), servers=6, arm_kind=kind,
           checkpoint_seed=int(seed), corpus=corpus, rung=f"C{nclients}",
           workload="drainable_f4000_n50000", wallclock_s=int(wall),
           env={k: v for k, v in __import__("os").environ.items()
                if k.startswith(("GNN_", "HEROSIM_", "PARTIAL_STATE", "QUEUE_FEATURE",
                                 "TOPOLOGY_FEATURE", "INFERENCE_FEATURE"))},
           code=(doc.get("run_provenance") or {}).get("code"))
json.dump(out, open(summary, "w"), indent=1)
print(f"[summary] {summary}: elapsed={out['averageElapsedTime']} queue={out['averageQueueTime']}")
PY
rm -f "$RAW"
echo "[TASK $T] done in ${WALL}s"
