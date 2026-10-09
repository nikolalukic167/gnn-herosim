#!/bin/bash
# r1_attribution_v1 PRODUCTION corpus, step 2 of 3 -- submit only on the coordinator's word.
# Live capture of the peer-group snapshots the corpus is built from, WITH HEROSIM_SNAPSHOT_FIDELITY=1 (the corpus build in step 3 needs
# the same flag and refuses a snapshot without the block). One array task per topology; its rungs x windows in parallel.
# Behaviour policy as small_batch_v1_capture.sbatch (the batched rule in the decoder's seat), R1.1 environment.
# Every cell ends with a sentinel <cell>_<rung>_<window>.done (JSON: status ok | hung | failed, snapshots, last trigger task id,
# cap_hit). Resume tests the sentinel, not a non-empty file; a run is killed after CAPTURE_TIMEOUT_S (a starved-client spin hangs
# some cells whatever the load) and recorded as hung. The snapshot cap is 600: at stride 31 a cap of 300 stopped at task ~38,000 of
# 50,000, so the last fifth of the trace was never offered; a cell that still reaches the cap is recorded cap_hit and fails loud.
#   sbatch --array=0-<n_topologies-1> --export=ALL,WT=...,ROOT=...,TOPOS="...",TAGS="light moderate heavy" wf1_corpus_prod_capture.sbatch
#SBATCH --job-name=wf1p-cap
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/wf1p-cap-%A_%a.out
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=06:00:00
set -euo pipefail
WT=${WT:?FAIL LOUD: set WT (pinned worktree)}
ROOT=${ROOT:?FAIL LOUD: set ROOT (the step-1 output root)}
TOPOS=${TOPOS:?FAIL LOUD: set TOPOS}
TAGS=${TAGS:?FAIL LOUD: set TAGS (the rung tags of step 1)}
WINDOWS=${WINDOWS:-"g0 g1 g2 g3"}
T=${SLURM_ARRAY_TASK_ID:?FAIL LOUD: submit as an array}
TOPO_ARR=($TOPOS); SEED=${TOPO_ARR[$T]:?FAIL LOUD: array index beyond TOPOS}
cd "$WT"
# nothing a previous gate driver exported (HEROSIM_KEEP_ALIVE, HEROSIM_INFLIGHT_CAPTURE, ...) may reach the capture
for v in $(compgen -e | grep -E '^(HEROSIM_|GNN_|COSIM_|LIVE_AUDIT_|GATE_|SIM_|QUEUE_FEATURE|MAX_PLACEMENT)'); do unset "$v"; done
if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3 PYTHONPATH="$WT" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=""
echo "HEAD $(git rev-parse HEAD) topology $SEED node $(hostname)"
export HEROSIM_TRANSFER_MODEL=pipelined HEROSIM_REPLICA_RELEASE=1 HEROSIM_SCALEOUT=kpa HEROSIM_SHARED_AUTOSCALER=0 GATE_FIXED_POLICY_TIME_SCALE=1.0
export HEROSIM_PEER_EXCHANGE=1 HEROSIM_SERVER_ONLY_REPLICAS=1 HEROSIM_WARMTH_PHYSICS=node_disk_v2 PYTHONHASHSEED=0 HEROSIM_GNN_DEVICE=cpu
export GNN_DECODE_MODE=masked_topo GNN_BATCH_BY_PEER_GROUP=1
export HEROSIM_SNAPSHOT_FIDELITY=1
unset GNN_MODEL_PATH GNN_DISABLE_MESSAGE_PASSING || true
export LIVE_AUDIT_STRIDE=${LIVE_AUDIT_STRIDE:-31} LIVE_AUDIT_MAX_SNAPSHOTS=${LIVE_AUDIT_MAX_SNAPSHOTS:-600} LIVE_AUDIT_MIN_BATCH_SIZE=2 LIVE_AUDIT_MIN_CANDIDATES=1
export HEROSIM_RAW_DIR=/tmp/herosim_raw_${SLURM_JOB_ID:-$$}; mkdir -p "$HEROSIM_RAW_DIR"; trap 'rm -rf "$HEROSIM_RAW_DIR"' EXIT
# disk guard: fail loudly before writing if the filesystem this task writes to, or the /home quota the job logs live on, has < GUARD_GB free
GUARD_GB=${GUARD_GB:-15}
free_gb() { case "$(readlink -f "$1")" in /home/*) echo $(( (268435456000 - $(getfattr --only-values -n ceph.dir.rbytes "$HOME" 2>/dev/null)) / 1000000000 ));;
  *) df -P -BG "$1" | awk 'NR==2{gsub("G","",$4); print $4}';; esac; }
for d in "$HOME" "$ROOT"; do
  g=$(free_gb "$d"); echo "free on $d: ${g} GB (guard ${GUARD_GB})"
  [[ "$g" -ge "$GUARD_GB" ]] || { echo "FAIL LOUD: only ${g} GB free for $d, below the ${GUARD_GB} GB guard"; exit 1; }
done
mkdir -p "$ROOT/snapshots" "$ROOT/capture_logs"
CAPTURE_TIMEOUT_S=${CAPTURE_TIMEOUT_S:-7200}
pids=()
for R in $TAGS; do
  for W in $WINDOWS; do
    CELL=cc40s${SEED}; SNAP="$ROOT/snapshots/${CELL}_${R}_${W}.jsonl"; DONE="$ROOT/snapshots/${CELL}_${R}_${W}.done"
    [[ -s "$DONE" ]] && { echo "[$CELL $R $W] done already: $(cat "$DONE")"; continue; }
    ( export LIVE_AUDIT_SNAPSHOT_PATH="$SNAP"; rm -f "$SNAP" "$DONE"
      rc=0
      timeout --kill-after=60 "$CAPTURE_TIMEOUT_S" python3 src/executesimulation.py --config "$ROOT/inputs/wf1_$R/cfg/$CELL.json" --workload "$ROOT/inputs/wf1_$R/wl/grounded_${W}_n50000.json" \
        --policy peer_greedy_network_batch --output "$HEROSIM_RAW_DIR/${CELL}_${R}_${W}.raw.json" > "$ROOT/capture_logs/${CELL}_${R}_${W}.log" 2>&1 || rc=$?
      n=0; [[ -f "$SNAP" ]] && n=$(wc -l < "$SNAP")
      status=ok; [[ $rc -eq 124 || $rc -eq 137 ]] && status=hung; [[ $rc -ne 0 && $status == ok ]] && status=failed
      last=$(tail -n1 "$SNAP" 2>/dev/null | python3 -c "import sys,json
l=sys.stdin.readline().strip()
print(json.loads(l)['trigger_task_id'] if l else -1)" 2>/dev/null || echo -1)
      cap=false; [[ $n -ge $LIVE_AUDIT_MAX_SNAPSHOTS ]] && cap=true
      [[ $status == ok && $n -eq 0 ]] && status=failed
      echo "{\"status\": \"$status\", \"rc\": $rc, \"snapshots\": $n, \"last_trigger_task_id\": $last, \"cap_hit\": $cap}" > "$DONE.tmp" && mv "$DONE.tmp" "$DONE"
      echo "[$CELL $R $W] $(cat "$DONE")" ) &
    pids+=($!)
  done
done
for p in "${pids[@]}"; do wait "$p" || true; done
# a hung or failed cell is a sentinel, not an array failure: the build counts them and the volume check fails loud on a shortfall
bad=$(grep -L '"status": "ok"' "$ROOT"/snapshots/cc40s${SEED}_*.done 2>/dev/null | wc -l) || true
caps=$(grep -l '"cap_hit": true' "$ROOT"/snapshots/cc40s${SEED}_*.done 2>/dev/null | wc -l) || true
echo "topology $SEED: cells not ok: $bad, cap_hit: $caps"
[[ $caps -eq 0 ]] || { echo "FAIL LOUD: a cell hit the snapshot cap; raise LIVE_AUDIT_MAX_SNAPSHOTS"; exit 1; }
