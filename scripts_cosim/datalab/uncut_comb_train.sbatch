#!/bin/bash
# uncut_groups_s0_v1 combined lever training (registered 2026-10-17; F, L, FL gnn on Arm 1b's recipe): array index i = config EXT_CFGS[i // len(EXT_SEEDS)],
# seed EXT_SEEDS[i % len(EXT_SEEDS)]; configs experiments/<cfg>.yaml. CPU, 6 threads on 12 CPUs (SMT off: 6 physical cores; the speed test kept 6 threads,
# 0 workers, and the GPU slower: the record's speed line). A config with NEAR_RTT_LABEL_OVERRIDE_JSON needs LABEL_SHA to match it.
#   sbatch --array=0-1 --export=ALL uncut_comb_train.sbatch   with WT, EXPECT_HEAD, EXT_CFGS, EXT_SEEDS[, LABEL_SHA] set in the shell
#SBATCH --job-name=comb-train
#SBATCH --output=/share/nikola.lukic/uncut_groups/comb/train-%A_%a.out
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=150G
#SBATCH --time=2-00:00:00
set -uo pipefail
ulimit -n 65536 2>/dev/null || true
WT=${WT:?FAIL LOUD: set WT}; EXPECT_HEAD=${EXPECT_HEAD:?FAIL LOUD: set EXPECT_HEAD}
I=${SLURM_ARRAY_TASK_ID:?FAIL LOUD: submit as an array}
read -r CFGN S < <(python3 -c "import sys; c=sys.argv[2].split(','); s=sys.argv[3].split(','); i=int(sys.argv[1]); assert 0 <= i < len(c)*len(s); print(c[i // len(s)], s[i % len(s)])" "$I" "${EXT_CFGS:?FAIL LOUD: set EXT_CFGS}" "${EXT_SEEDS:?FAIL LOUD: set EXT_SEEDS}") || { echo "FAIL LOUD: array index $I"; exit 1; }
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
echo "task $I: cfg $CFGN seed $S | HEAD $HEAD node $(hostname) job ${SLURM_JOB_ID:-} cpus $(nproc) smt $(cat /sys/devices/system/cpu/smt/control 2>/dev/null)"
[[ "$HEAD" == "$EXPECT_HEAD"* ]] || { echo "FAIL LOUD: worktree HEAD $HEAD is not $EXPECT_HEAD"; exit 1; }
[[ "$(nproc)" -ge 6 ]] || { echo "FAIL LOUD: fewer than 6 cores in the cpuset"; exit 1; }
CFG=experiments/$CFGN.yaml; SPLITF=experiments/uncut_arm1_split.json
read -r CACHE CONTRACT LO < <(python3 -c "import yaml,sys; c=yaml.safe_load(open(sys.argv[1])); e=c['env']; print(c['cache_dir'], e.get('PARTIAL_STATE_CONTRACT', '-'), e.get('NEAR_RTT_LABEL_OVERRIDE_JSON', '-'))" "$CFG") || { echo "FAIL LOUD: config $CFG"; exit 1; }
[[ -f "$CFG" && -L models && -f "$SPLITF" && -f "$CACHE/metadata.json" ]] || { echo "FAIL LOUD: config, models/ symlink, split or cache ($CACHE) missing"; exit 1; }
[[ "$(sha256sum "$SPLITF" | cut -d' ' -f1)" == 613c8cbb1cf494b60a215d3fccb4d30e6cfe17943304d7078a7b8f58f6f34808 ]] || { echo "FAIL LOUD: split sha256"; exit 1; }
[[ -z "$(git status --porcelain -- src scripts_cosim experiments run_experiment.py)" ]] || { echo "FAIL LOUD: worktree dirty"; exit 1; }
if [[ "$LO" != "-" ]]; then
  [[ "$(sha256sum "$LO" | cut -d' ' -f1)" == "${LABEL_SHA:?FAIL LOUD: set LABEL_SHA}" ]] || { echo "FAIL LOUD: label override $LO sha256"; exit 1; }
fi
mkdir -p logs
RUN=$(python3 -c "import yaml,sys;print(yaml.safe_load(open(sys.argv[1]))['wandb']['run_name'])" "$CFG")
CK=models/$RUN-seed$S
[[ ! -e "$CK.pt" && ! -e "$CK-final.pt" ]] || { echo "FAIL LOUD: $CK exists; refusing to overwrite"; exit 1; }
python3 -u run_experiment.py "$CFG" --seed "$S" > "logs/comb-train-$RUN-seed$S.log" 2>&1
rc=$?
echo "run_experiment exit $rc"
fail=0
[[ $rc -eq 0 ]] || { tail -15 "logs/comb-train-$RUN-seed$S.log" | cut -c1-240; fail=1; }
for f in "$CK-final.pt" "$CK-final.contract.json" "$CK.val.json"; do [[ -f "$f" ]] || { echo "FAIL LOUD: no $f"; fail=$((fail+1)); }; done
[[ "$LO" == "-" ]] || echo "label override $LO sha256 ${LABEL_SHA:-}"
if [[ "$LO" != "-" ]] && ! grep -q "NEAR_RTT_LABEL_OVERRIDE_JSON\] train label replaced" "logs/comb-train-$RUN-seed$S.log"; then echo "FAIL LOUD: the label override was not applied"; fail=$((fail+1)); fi
python3 - "$CK-final.contract.json" "$CONTRACT" <<'PY' || fail=$((fail+1))
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
if sys.argv[2] != "-" and find(c, "partial_state_contract") != sys.argv[2]:
    bad.append(f"partial_state_contract {find(c, 'partial_state_contract')!r} != {sys.argv[2]}")
# the trainer does not write its label source to the sidecar: the override is checked by its sha256 before the run and by the trainer's log line after
print("sidecar check:", "OK" if not bad else "FAIL LOUD: " + "; ".join(bad))
sys.exit(1 if bad else 0)
PY
md5sum "$CK-final.pt" "$CK-final.contract.json" "$CK.pt" 2>/dev/null
echo "=== task $I ($RUN seed $S): $fail failure(s)"
exit $fail
