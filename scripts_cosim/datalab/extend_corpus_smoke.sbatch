#!/bin/bash
# extend_corpus_slate smoke: unit tests, a census of the whole source corpus, then 5 datasets copied into a scratch source and extended into a
# scratch out directory, then the replay checks (a)-(c). The production corpus is read only (the 5 datasets are copied out of it).
#   sbatch --export=ALL,WT=<pinned worktree>,EXPECT_HEAD=<sha> extend_corpus_smoke.sbatch
#SBATCH --job-name=ext-smoke
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/ext-smoke-%j.out
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
set -euo pipefail
WT=${WT:?FAIL LOUD: set WT}; EXPECT_HEAD=${EXPECT_HEAD:?FAIL LOUD: set EXPECT_HEAD}
cd "$WT"
for v in $(compgen -e | grep -E '^(HEROSIM_|GNN_|COSIM_|LIVE_AUDIT_|GATE_|SIM_|QUEUE_FEATURE|MAX_PLACEMENT|NEAR_RTT_|PARTIAL_STATE|WF1_)'); do unset "$v"; done
if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"; micromamba activate gnn
HEAD=$(git rev-parse HEAD); echo "HEAD $HEAD node $(hostname) job=${SLURM_JOB_ID:-}"
[[ "$HEAD" == "$EXPECT_HEAD"* ]] || { echo "FAIL LOUD: worktree HEAD $HEAD is not $EXPECT_HEAD"; exit 1; }
export PYTHONPATH="$WT" PYTHONUNBUFFERED=1 EXTEND_CODE_COMMIT=$HEAD
# the physics env generation_provenance.json records for the corpus
export COSIM_SUPPRESS_SIM_PRINTS=1 GNN_CAPTURE_DATASET_STATE=0 HEROSIM_PEER_EXCHANGE=1 HEROSIM_PY=python3 HEROSIM_REPLICA_RELEASE=1 HEROSIM_RETAIN_PEER_STATS=1 \
  HEROSIM_RETAIN_TASK_TIMES=1 HEROSIM_SCALEOUT=kpa HEROSIM_SERVER_ONLY_REPLICAS=1 HEROSIM_SHARED_AUTOSCALER=0 HEROSIM_SNAPSHOT_FIDELITY=1 \
  HEROSIM_TRANSFER_MODEL=pipelined HEROSIM_WARMTH_PHYSICS=node_disk_v2 MAX_PLACEMENT_COMBINATIONS_SKIP=2000000 PYTHONHASHSEED=0
# placement timeouts as the corpus build had them (placement_metadata: 120 s)
export COSIM_PLACEMENT_TIMEOUT_S=120
C=/share/nikola.lukic/workload_fix_v1/accel_replica_corpus/gnn_datasets_wf1_train
O=/share/nikola.lukic/extend_smoke/${SLURM_JOB_ID:-manual}
mkdir -p $O

echo "== 1. unit tests"
python3 -m pytest tests/test_extend_corpus_slate.py -q 2>&1 | tail -6

echo "== 2. census of $C"
python3 scripts_cosim/extend_corpus_slate.py --src $C --out $O/census --census --max-plans 100000 | tail -2
python3 - <<P
import json, collections
rows = [json.loads(l) for l in open("$O/census/extend_census.jsonl")]
c = collections.Counter(r["status"] for r in rows)
ext = [r for r in rows if r["status"] == "would_extend"]
print("statuses:", dict(c), "| plans old slate (sum):", sum(r["plans_old_slate"] for r in ext), "full:", sum(r["full_plans"] for r in ext), "new plans to simulate:", sum(r["new_plans"] for r in ext))
q = sorted(r["new_plans"] for r in ext); n = len(q)
print("new plans per dataset: min %d  p25 %d  median %d  p75 %d  p90 %d  max %d" % (q[0], q[n//4], q[n//2], q[3*n//4], q[int(.9*n)], q[-1]))
pick = [r for r in ext if 300 <= r["new_plans"] <= 3000]
pick = pick[::max(1, len(pick)//5)][:5]
json.dump([r["dataset"] for r in pick], open("$O/picked.json", "w")); print("smoke datasets:", [(r["dataset"], r["plans_old_slate"], r["full_plans"]) for r in pick])
P

echo "== 3. scratch copy of the 5 datasets (dereferenced), hashed"
mkdir -p $O/src
for d in $(python3 -c "import json;print(' '.join(json.load(open('$O/picked.json'))))"); do cp -rL $C/$d $O/src/$d; done
(cd $O/src && find . -type f -print0 | sort -z | xargs -0 sha256sum) > $O/src.sha256

echo "== 4. extend into $O/out"
python3 scripts_cosim/extend_corpus_slate.py --src $O/src --out $O/out --workers 16 --verify 20 --keep-scratch | tail -12
(cd $O/src && sha256sum -c $O/src.sha256 --quiet) && echo "scratch source unchanged"

echo "== 5. replay checks (a)-(c) on the scratch out"
EXTEND_TEST_CORPUS=$O/out EXTEND_TEST_SRC=$O/src python3 -m pytest tests/test_extend_corpus_slate.py -q -k replay_checks 2>&1 | tail -6
python3 - <<P
import json
rows = [json.loads(l) for l in open("$O/out/extend_manifest.jsonl")]
for r in rows: print({k: r.get(k) for k in ("dataset","status","plans_before","plans_added","plans_after","best_before","best_after","new_best_wins","sim_per_second","worker_seconds_per_plan","sweep_seconds")}, r.get("verified"), r.get("error"))
ok = [r for r in rows if r["status"] == "extended"]
if ok:
    w = sum(r["worker_seconds_per_plan"] * r["plans_added"] for r in ok) / sum(r["plans_added"] for r in ok)
    print("worker-seconds per plan (plan-weighted, %d workers): %.4f" % (ok[0]["workers"], w))
P
echo "smoke done: $O"
