#!/bin/bash
# uncut_groups_s0_v1 own-state round, FULL training (from scratch, 100 epochs, lr 1e-3). One job = one (arm, seed); the launcher (uncut_dagger_full_launch.sh) submits the 12 with --nodelist spread over 4 nodes.
# Needs in the environment: WT, EXPECT_HEAD, CFGN (uncut_dagger_full_gnn | uncut_dagger_full_twin), S (1-6), SPLIT_SHA, LABEL_SHA, CACHE_SHA (sha256 of <cache>/metadata.json), EXPECT_CPU (the lscpu model name: all 12 run on one device class). The three shas are the SAME for both arms (one shared corpus).
#SBATCH --job-name=dagger-full
#SBATCH --output=/share/nikola.lukic/uncut_groups/dagger/full-%x-%j.out
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=150G
#SBATCH --time=2-00:00:00
#SBATCH --exclude=os-cpu-slurm-[21,15,13,14,16,10,17]
set -uo pipefail
ulimit -n 65536 2>/dev/null || true
WT=${WT:?FAIL LOUD: set WT}; EXPECT_HEAD=${EXPECT_HEAD:?FAIL LOUD: set EXPECT_HEAD}
CFGN=${CFGN:?FAIL LOUD: set CFGN}; S=${S:?FAIL LOUD: set S}
case "$CFGN" in uncut_dagger_full_gnn|uncut_dagger_full_twin) ;; *) echo "FAIL LOUD: CFGN $CFGN"; exit 1;; esac
[[ "$S" =~ ^[1-6]$ ]] || { echo "FAIL LOUD: seed $S"; exit 1; }
cd "$WT"
for v in $(compgen -e | grep -E '^(HEROSIM_|GNN_|COSIM_|LIVE_AUDIT_|GATE_|SIM_|QUEUE_FEATURE|MAX_PLACEMENT|NEAR_RTT_|PARTIAL_STATE|R1A_|TRAIN_INIT)'); do unset "$v"; done
if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3 PYTHONPATH="$WT:$WT/src/notebooks" PYTHONUNBUFFERED=1 WANDB_ENTITY=nikolalukic167-tu-wien OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6
export CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=""
export HEROSIM_TRANSFER_MODEL=pipelined HEROSIM_REPLICA_RELEASE=1 HEROSIM_SCALEOUT=kpa HEROSIM_SHARED_AUTOSCALER=0 GATE_FIXED_POLICY_TIME_SCALE=1.0
export R1A_SLATE=declared_pruning_v1_uncapped
HEAD=$(git rev-parse HEAD)
echo "cfg $CFGN seed $S | HEAD $HEAD node $(hostname) job ${SLURM_JOB_ID:-} cpus $(nproc) smt $(cat /sys/devices/system/cpu/smt/control 2>/dev/null) ulimit-n $(ulimit -n)"
[[ "$HEAD" == "$EXPECT_HEAD"* ]] || { echo "FAIL LOUD: worktree HEAD $HEAD is not $EXPECT_HEAD"; exit 1; }
CPU=$(lscpu | sed -n "s/^Model name: *//p" | head -1); echo "cpu model: $CPU"
[[ "$CPU" == "${EXPECT_CPU:?FAIL LOUD: set EXPECT_CPU}" ]] || { echo "FAIL LOUD: cpu model $CPU is not $EXPECT_CPU"; exit 1; }
[[ "$(nproc)" -ge 6 ]] || { echo "FAIL LOUD: fewer than 6 cores in the cpuset"; exit 1; }
[[ "$(ulimit -n)" -ge 65536 ]] || { echo "FAIL LOUD: ulimit -n $(ulimit -n) < 65536"; exit 1; }
[[ -z "$(git status --porcelain -- src scripts_cosim experiments run_experiment.py)" ]] || { echo "FAIL LOUD: worktree dirty"; exit 1; }
CFG=experiments/$CFGN.yaml
grep -q TBD "$CFG" && { echo "FAIL LOUD: $CFG still has a TBD"; exit 1; }
# determinism test first (12 s, no GPU); a skipped test is not a pass
DET=$(OMP_NUM_THREADS=1 python3 -m pytest tests/test_trainer_determinism.py -q -rs 2>&1 | tail -15); echo "$DET" | tail -3
echo "$DET" | grep -qE ' passed' && ! echo "$DET" | grep -qE ' (failed|error|skipped)' || { echo "FAIL LOUD: determinism test did not pass cleanly"; exit 1; }
read -r CACHE CONTRACT LO SPLITF INIT < <(python3 -c "import yaml,sys; c=yaml.safe_load(open(sys.argv[1])); e=c['env']; assert c['trainer']=='gnn' and c['args']['epochs']==100 and c['args']['min-epochs']==100 and c['args']['learning-rate']==0.001 and c['args']['patience']==60, 'recipe'; print(c['cache_dir'], e['PARTIAL_STATE_CONTRACT'], e['NEAR_RTT_LABEL_OVERRIDE_JSON'], e['NEAR_RTT_SPLIT_ARTIFACT'], e.get('TRAIN_INIT_CHECKPOINT','-'))" "$CFG") || { echo "FAIL LOUD: config $CFG"; exit 1; }
[[ "$INIT" == "-" ]] || { echo "FAIL LOUD: TRAIN_INIT_CHECKPOINT set; this is from scratch"; exit 1; }
[[ -f "$CFG" && -L models && -f "$SPLITF" && -f "$CACHE/metadata.json" && -f "$LO" ]] || { echo "FAIL LOUD: config, models/ symlink, split ($SPLITF), cache ($CACHE) or labels ($LO) missing"; exit 1; }
[[ "$(sha256sum "$SPLITF" | cut -d' ' -f1)" == "${SPLIT_SHA:?FAIL LOUD: set SPLIT_SHA}" ]] || { echo "FAIL LOUD: split sha256"; exit 1; }
[[ "$(sha256sum "$LO" | cut -d' ' -f1)" == "${LABEL_SHA:?FAIL LOUD: set LABEL_SHA}" ]] || { echo "FAIL LOUD: label override $LO sha256"; exit 1; }
[[ "$(sha256sum "$CACHE/metadata.json" | cut -d' ' -f1)" == "${CACHE_SHA:?FAIL LOUD: set CACHE_SHA}" ]] || { echo "FAIL LOUD: cache metadata sha256"; exit 1; }
mkdir -p logs
RUN=$(python3 -c "import yaml,sys;print(yaml.safe_load(open(sys.argv[1]))['wandb']['run_name'])" "$CFG")
[[ "$RUN" == "uncut-dagger-full-${CFGN##*_}" ]] || { echo "FAIL LOUD: run_name $RUN"; exit 1; }
CK=models/$RUN-seed$S
[[ ! -e "$CK.pt" && ! -e "$CK-final.pt" ]] || { echo "FAIL LOUD: $CK exists; refusing to overwrite"; exit 1; }
python3 -u run_experiment.py "$CFG" --seed "$S" > "logs/dagger-full-$RUN-seed$S.log" 2>&1
rc=$?
echo "run_experiment exit $rc"
fail=0
[[ $rc -eq 0 ]] || { tail -15 "logs/dagger-full-$RUN-seed$S.log" | cut -c1-240; fail=1; }
for f in "$CK-final.pt" "$CK-final.contract.json" "$CK.val.json"; do [[ -f "$f" ]] || { echo "FAIL LOUD: no $f"; fail=$((fail+1)); }; done
echo "label override $LO sha256 $LABEL_SHA; split $SPLITF sha256 $SPLIT_SHA; cache $CACHE metadata sha256 $CACHE_SHA"
tr '\r' '\n' < "logs/dagger-full-$RUN-seed$S.log" | grep -q "train label replaced on 13311 graphs" || { echo "FAIL LOUD: the label override was not applied to all 13311 train graphs"; fail=$((fail+1)); }
grep -qiE "TRAIN_INIT_CHECKPOINT|loaded (init|checkpoint)|warm.?start" "logs/dagger-full-$RUN-seed$S.log" && { echo "FAIL LOUD: log mentions an init checkpoint"; fail=$((fail+1)); }
python3 - "$CK-final.contract.json" "$CONTRACT" "${CFGN##*_}" <<'PY' || fail=$((fail+1))
import json, sys
c = json.load(open(sys.argv[1]))
def find(o, k):
    if isinstance(o, dict):
        if k in o:
            return o[k]
        for v in o.values():
            r = find(v, k)
            if r is not None:
                return r
    return None
bad = []
if find(c, "batched_teacher_forced") is not True:
    bad.append("batched_teacher_forced")
if find(c, "partial_state_contract") != sys.argv[2]:
    bad.append(f"partial_state_contract {find(c, 'partial_state_contract')!r} != {sys.argv[2]}")
print("sidecar check:", "OK" if not bad else "FAIL LOUD: " + "; ".join(bad))
sys.exit(1 if bad else 0)
PY
md5sum "$CK-final.pt" "$CK-final.contract.json" "$CK.pt" 2>/dev/null
echo "=== $RUN seed $S: $fail failure(s)"
exit $fail
