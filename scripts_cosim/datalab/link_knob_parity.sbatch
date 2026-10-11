#!/bin/bash
# link_bound_s0_v1 knob parity: one live CD run on the reduced-bandwidth cell, 40 captured decisions replayed with the live plan
# forced (rollout_replay plan "live", H 0), compared per task with the live run (link_knob_parity_read.py).
#   sbatch --export=ALL,WT=<pinned>,CFG=<reduced cfg>,CFG_DEFAULT=<default cfg>,WL=<workload>,BW=62.5,OUT=<dir> link_knob_parity.sbatch
#SBATCH --job-name=lb-parity
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/lb-parity-%j.out
#SBATCH --partition=CPU-amd
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --time=04:00:00
set -euo pipefail
WT=${WT:?}; CFG=${CFG:?}; CFG_DEFAULT=${CFG_DEFAULT:?}; WL=${WL:?}; BW=${BW:?}; OUT=${OUT:?}; N=${N:-40}; mkdir -p $OUT
cd "$WT"; eval "$(micromamba shell hook --shell bash)"; micromamba activate gnn
for v in $(compgen -e | grep -E '^(HEROSIM_|GNN_|COSIM_|LIVE_AUDIT_|GATE_|SIM_|QUEUE_FEATURE|MAX_PLACEMENT|DECLARED_SLATE|KEEP_RAW)'); do unset "$v"; done
echo "HEAD $(git rev-parse HEAD) node $(hostname)"; [[ -z "$(git status --porcelain -- src scripts_cosim)" ]] || { echo "FAIL LOUD: worktree dirty"; exit 1; }
export HEROSIM_PY=python3 PYTHONPATH=$WT PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
export HEROSIM_TRANSFER_MODEL=pipelined HEROSIM_REPLICA_RELEASE=1 HEROSIM_SCALEOUT=kpa HEROSIM_SHARED_AUTOSCALER=0 GATE_FIXED_POLICY_TIME_SCALE=1.0
export HEROSIM_PEER_EXCHANGE=1 HEROSIM_SERVER_ONLY_REPLICAS=1 HEROSIM_WARMTH_PHYSICS=node_disk_v2 PYTHONHASHSEED=0 HEROSIM_GNN_DEVICE=cpu
export GNN_DECODE_MODE=masked_topo GNN_BATCH_BY_PEER_GROUP=1 HEROSIM_SNAPSHOT_FIDELITY=1 SIM_FORCE_FULL_STATS=1
export LIVE_AUDIT_STRIDE=31 LIVE_AUDIT_MAX_SNAPSHOTS=600 LIVE_AUDIT_MIN_BATCH_SIZE=2 LIVE_AUDIT_MIN_CANDIDATES=1
export HEROSIM_RAW_DIR=/tmp/lbp_${SLURM_JOB_ID}; mkdir -p $HEROSIM_RAW_DIR; trap 'rm -rf "$HEROSIM_RAW_DIR"' EXIT
# 1 live capture on the reduced cell
[[ -s $OUT/live.raw.json ]] || LIVE_AUDIT_SNAPSHOT_PATH=$OUT/snap.jsonl python3 src/executesimulation.py --config $CFG --workload $WL --policy peer_greedy_network_batch --output $OUT/live.raw.json > $OUT/live.log 2>&1
wc -l $OUT/snap.jsonl
python3 scripts_cosim/rollout_replay.py --make-table $OUT/live.raw.json $OUT/table.json
# 2 the first N snapshots passing the corpus filters become dry-run datasets; the first is the replay's physics reference
python3 -c "import random,sys;L=open(sys.argv[1]).read().splitlines();random.Random('lbp').shuffle(L);open(sys.argv[2],'w').write('\n'.join(L)+'\n')" $OUT/snap.jsonl $OUT/shuffled.jsonl
python3 scripts_cosim/make_warm_corpus.py --snapshots $OUT/shuffled.jsonl --output-dir $OUT/ds --start-index 0 --limit-batches $N --whole-group none \
  --trace $WL --cell-config $CFG --source-tag lbp --min-time 360 --variable-group-min 2 --require-connected-batch --no-cap-filter --min-choice-fraction 0.2 --fidelity --dry-run --quiet --manifest-per-source --workers 1 > $OUT/build.log 2>&1
# 3 replay jobs
python3 - $OUT <<'PY'
import json, sys, glob
out = sys.argv[1]
ids = [int(json.loads(l)["snapshot_id"]) for l in open(glob.glob(out + "/ds/warm_manifest.*.jsonl")[0]) if json.loads(l).get("status") == "dry-run"]
snaps = {int(json.loads(l)["snapshot_id"]): json.loads(l) for l in open(out + "/snap.jsonl")}
ref = sorted(glob.glob(out + "/ds/ds_*"))[0]
assert json.load(open(ref + "/infrastructure.json"))["live_snapshot_seed"]["fidelity_replay"].get("backbone"), "FAIL LOUD: reference dataset carries no backbone stamp"
jobs = [dict(ds=ref, group={"snapfile": out + "/snap.jsonl", "snapshot_id": i, "time": snaps[i]["time"], "physics_from": ref},
             table=out + "/table.json", workload=None, H=0, plan="live", continuation="live", tag=f"live|{i}") for i in ids]
print(len(ids), "states")
open(out + "/jobs.jsonl", "w").write("".join(json.dumps(j) + "\n" for j in jobs))
PY
python3 - $OUT $WL <<'PY'
import json, sys
out, wl = sys.argv[1], sys.argv[2]
jobs = [json.loads(l) for l in open(out + "/jobs.jsonl")]
for j in jobs: j["workload"] = wl
open(out + "/jobs.jsonl", "w").write("".join(json.dumps(j) + "\n" for j in jobs))
PY
export HEROSIM_LINK_BANDWIDTH_MBPS=   # explicit: the cfg, not the env var, carries the bandwidth
unset HEROSIM_LINK_BANDWIDTH_MBPS
python3 scripts_cosim/rollout_replay.py $OUT/rows.jsonl $OUT/jobs.jsonl --workers 30 > $OUT/replay.log 2>&1
python3 scripts_cosim/link_knob_parity_read.py --rows $OUT/rows.jsonl --table $OUT/table.json --raw $OUT/live.raw.json --snaps $OUT/snap.jsonl \
  --cfg-low $CFG --cfg-default $CFG_DEFAULT --bw $BW --out $OUT/parity.json
