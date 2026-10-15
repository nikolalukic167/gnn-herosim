#!/bin/bash
# uncut_groups_s0_v1 expected move-regret lever, wave 1: gnn_eng with lambda x expected move regret (experiments/uncut_arm3_gnn_eng_l{0p5,2p0}.yaml), CPU, 6 threads on 12 CPUs
# (SMT is off on CPU-amd: 12 CPUs hold 6 physical cores; threads never exceed the in-job cores). Array index i: lambda = (0.5, 2.0)[i // 2], seed = i % 2 + 1.
#   sbatch --array=0-3 --export=ALL,WT=<pinned worktree>,EXPECT_HEAD=<sha> uncut_arm3_train.sbatch
# RESUME=1 (the safety net; uncut_arm3_chain.sh submits the links): NEAR_RTT_RESUME=1 makes the trainer save its whole state every epoch to models/<run>-seed<s>.resume.pt and continue from it,
# into the same W&B run, if the file exists at start; a link starts only if the previous one (PREV) ended by the time limit (TIMEOUT), finished (COMPLETED: nothing to do), and stops the chain on
# anything else. The fingerprint a resume must agree with carries lambda and the table's sha256. The wave-1 jobs submitted without RESUME=1 write no resume state.
# SMOKE=1: one epoch of the same config (epochs, min-epochs and patience 1, run name + "-smoke", tag "smoke"), checkpoints under the smoke run name; nothing is gated.
# Not --exclusive. --deadline keeps every wave-1 job from overlapping the hand calibration 871538 (02:00 Sat 2026-10-17).
#SBATCH --job-name=a3-train
#SBATCH --output=/share/nikola.lukic/uncut_groups/lever/train3-%A_%a.out
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=150G
#SBATCH --time=14:00:00
#SBATCH --deadline=2026-10-17T01:00:00
set -uo pipefail
ulimit -n 65536 2>/dev/null || true
WT=${WT:?FAIL LOUD: set WT}; EXPECT_HEAD=${EXPECT_HEAD:?FAIL LOUD: set EXPECT_HEAD}
I=${SLURM_ARRAY_TASK_ID:?FAIL LOUD: submit as --array=0-3}
TAG=$(python3 -c "import sys; print(('l0p5', 'l2p0')[int(sys.argv[1]) // 2])" "$I") || { echo "FAIL LOUD: array index $I"; exit 1; }
S=$(( I % 2 + 1 ))
RESUME=${RESUME:-0}
if [[ "$RESUME" == "1" && -n "${PREV:-}" ]]; then
  PSTATE=$(sacct -j "$PREV" -X -n -o State | awk 'NR==1{print $1}')
  echo "previous link $PREV: $PSTATE"
  case "$PSTATE" in
    TIMEOUT) ;;
    COMPLETED) echo "previous link COMPLETED: nothing to do"; exit 0 ;;
    *) echo "FAIL LOUD: previous link $PREV ended $PSTATE, not TIMEOUT; not continuing the chain"; exit 1 ;;
  esac
fi
cd "$WT"
for v in $(compgen -e | grep -E '^(HEROSIM_|GNN_|COSIM_|LIVE_AUDIT_|GATE_|SIM_|QUEUE_FEATURE|MAX_PLACEMENT|NEAR_RTT_|PARTIAL_STATE|R1A_)'); do unset "$v"; done
if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3 PYTHONPATH="$WT:$WT/src/notebooks" PYTHONUNBUFFERED=1 WANDB_ENTITY=nikolalukic167-tu-wien OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6
export CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=""
export HEROSIM_TRANSFER_MODEL=pipelined HEROSIM_REPLICA_RELEASE=1 HEROSIM_SCALEOUT=kpa HEROSIM_SHARED_AUTOSCALER=0 GATE_FIXED_POLICY_TIME_SCALE=1.0
export R1A_SLATE=declared_pruning_v1_uncapped
[[ "$RESUME" == "1" ]] && export NEAR_RTT_RESUME=1 NEAR_RTT_RESUME_PIN="$EXPECT_HEAD"
HEAD=$(git rev-parse HEAD)
echo "task $I: lambda tag $TAG seed $S | HEAD $HEAD node $(hostname) job ${SLURM_JOB_ID:-} cpus $(nproc) smt $(cat /sys/devices/system/cpu/smt/control 2>/dev/null) smoke ${SMOKE:-0}"
[[ "$HEAD" == "$EXPECT_HEAD"* ]] || { echo "FAIL LOUD: worktree HEAD $HEAD is not $EXPECT_HEAD"; exit 1; }
[[ "$(nproc)" -ge 6 ]] || { echo "FAIL LOUD: fewer than 6 cores in the cpuset"; exit 1; }
CFG=experiments/uncut_arm3_gnn_eng_${TAG}.yaml; SPLITF=experiments/uncut_arm1_split.json
CACHE=$(python3 -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['cache_dir'])" "$CFG")
[[ -f "$CFG" && -L models && -f "$SPLITF" && -f "$CACHE/metadata.json" ]] || { echo "FAIL LOUD: config, models/ symlink, split or cache ($CACHE) missing"; exit 1; }
[[ "$(sha256sum "$SPLITF" | cut -d' ' -f1)" == 613c8cbb1cf494b60a215d3fccb4d30e6cfe17943304d7078a7b8f58f6f34808 ]] || { echo "FAIL LOUD: split sha256"; exit 1; }
[[ -z "$(git status --porcelain -- src scripts_cosim experiments run_experiment.py)" ]] || { echo "FAIL LOUD: worktree dirty"; exit 1; }
D=/share/nikola.lukic/uncut_groups/lever
[[ "$(sha256sum $D/move_regret.pkl | cut -d' ' -f1)" == "${TABLE_SHA:?FAIL LOUD: set TABLE_SHA}" ]] || { echo "FAIL LOUD: move_regret.pkl sha256"; exit 1; }
mkdir -p logs
if [[ "${SMOKE:-0}" == "1" ]]; then
  SC=$(mktemp -d)/smoke_$TAG.yaml
  python3 - "$CFG" "$SC" <<'PY'
import sys, yaml
c = yaml.safe_load(open(sys.argv[1]))
c["args"].update({"epochs": 1, "min-epochs": 1, "patience": 1})
c["wandb"]["run_name"] += "-smoke"
c["wandb"]["tags"] = list(c["wandb"]["tags"]) + ["smoke"]
yaml.safe_dump(c, open(sys.argv[2], "w"), sort_keys=False)
PY
  CFG=$SC
fi
RUN=$(python3 -c "import yaml,sys;print(yaml.safe_load(open(sys.argv[1]))['wandb']['run_name'])" "$CFG")
CK=models/$RUN-seed$S
# a fresh start must not overwrite a checkpoint; a continuation has the trainer's own resume state next to it
if [[ "$RESUME" == "1" && -e "$CK.resume.pt" ]]; then echo "continuing from $CK.resume.pt"; else
  [[ ! -e "$CK.pt" && ! -e "$CK-final.pt" ]] || { echo "FAIL LOUD: $CK exists; refusing to overwrite"; exit 1; }
fi
python3 -u run_experiment.py "$CFG" --seed "$S" >> "logs/a3-train-$RUN-seed$S.log" 2>&1
rc=$?
echo "run_experiment exit $rc"
fail=0
[[ $rc -eq 0 ]] || { tail -15 "logs/a3-train-$RUN-seed$S.log" | cut -c1-240; fail=1; }
for f in "$CK-final.pt" "$CK-final.contract.json" "$CK.val.json"; do [[ -f "$f" ]] || { echo "FAIL LOUD: no $f"; fail=$((fail+1)); }; done
grep -E "NEAR_RTT_REGRET_LAMBDA|\[REGRET\]|Epoch +0|epoch 0|unscored" "logs/a3-train-$RUN-seed$S.log" | head -12 | cut -c1-240
python3 scripts_cosim/joint_burst_v2_sidecheck.py "$CK-final.contract.json" "ra_gnn_eng" "$SPLITF" inf || fail=$((fail+1))
md5sum "$CK-final.pt" "$CK-final.contract.json" "$CK.pt" 2>/dev/null
echo "=== task $I ($RUN seed $S): $fail failure(s)"
exit $fail
