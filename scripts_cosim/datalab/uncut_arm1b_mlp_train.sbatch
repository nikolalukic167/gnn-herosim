#!/bin/bash
# uncut_groups_s0_v1 Arm 1b mlp_same training (amendment of 2026-10-15; batched teacher-forced loss, NEAR_RTT_BATCHED_TF=1): task i = seed i + 1, arm mlp_same. Submit --array=0-1.
# CPU-only on a GPU-partition node (NOGPU=1 hides the GPU; the one-epoch timing was taken that way): --partition=GPU-rtx6000 --gres=gpu:1 --cpus-per-task=32 --export=ALL,WT=..,EXPECT_HEAD=..,NOGPU=1
# One run per node or GPU, nothing else of the program's trainers on it (--exclusive). The device is set at submit time by the partition / --gres of the submit line
# (the one-epoch timings filed in docs/lineages/uncut_groups_s0_v1.md decide), e.g. CPU-amd:
#   sbatch --array=0-3 --partition=CPU-amd --exclusive --export=ALL,WT=<pinned worktree>,EXPECT_HEAD=<sha> uncut_arm1b_train.sbatch
# or one L40S each:
#   sbatch --array=0-3 --partition=GPU-l40s --gres=gpu:1 --export=ALL,WT=<pinned worktree>,EXPECT_HEAD=<sha> uncut_arm1b_train.sbatch
# Checkpoints land in models/ as uncut-arm1b-<arm>-seed<s>{,-final}.pt; the served checkpoint is the -final.pt (epoch 100), as Arm 1.
#SBATCH --job-name=a1bm-train
#SBATCH --output=/share/nikola.lukic/uncut_arm1/train1bm-%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=150G
#SBATCH --time=4-00:00:00
set -uo pipefail
ulimit -n 65536 2>/dev/null || true
WT=${WT:?FAIL LOUD: set WT}; EXPECT_HEAD=${EXPECT_HEAD:?FAIL LOUD: set EXPECT_HEAD}
I=${SLURM_ARRAY_TASK_ID:?FAIL LOUD: submit as --array=0-3}
ARM=mlp_same
[[ "$I" == 0 || "$I" == 1 ]] || { echo "FAIL LOUD: array index $I"; exit 1; }
S=$(( I + 1 ))
cd "$WT"
for v in $(compgen -e | grep -E '^(HEROSIM_|GNN_|COSIM_|LIVE_AUDIT_|GATE_|SIM_|QUEUE_FEATURE|MAX_PLACEMENT|NEAR_RTT_|PARTIAL_STATE|R1A_)'); do unset "$v"; done
if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3 PYTHONPATH="$WT:$WT/src/notebooks" PYTHONUNBUFFERED=1 WANDB_ENTITY=nikolalukic167-tu-wien OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6
export CUBLAS_WORKSPACE_CONFIG=:4096:8
[[ -n "${SLURM_JOB_GPUS:-}${SLURM_GPUS_ON_NODE:-}" ]] || export CUDA_VISIBLE_DEVICES=""
[[ -n "${NOGPU:-}" ]] && export CUDA_VISIBLE_DEVICES=""
export HEROSIM_TRANSFER_MODEL=pipelined HEROSIM_REPLICA_RELEASE=1 HEROSIM_SCALEOUT=kpa HEROSIM_SHARED_AUTOSCALER=0 GATE_FIXED_POLICY_TIME_SCALE=1.0
export R1A_SLATE=declared_pruning_v1_uncapped
HEAD=$(git rev-parse HEAD)
echo "cpu: nproc $(nproc) cpuset $(cat /proc/self/status | grep Cpus_allowed_list | cut -f2) smt $(cat /sys/devices/system/cpu/smt/control 2>/dev/null) threads 6"; lscpu | grep -E "Model name|Thread|Core|Socket"
PHYS=$(lscpu -p=CORE,SOCKET -a | grep -v '^#' | sort -u | wc -l)
taskset -pc $$ >/dev/null; CORES=$(python3 -c "import os,re;a=sorted(os.sched_getaffinity(0));print(len({l for c in a for l in [open(f'/sys/devices/system/cpu/cpu{c}/topology/core_id').read().strip()+'/'+open(f'/sys/devices/system/cpu/cpu{c}/topology/physical_package_id').read().strip()]}))")
echo "physical cores in the allocation: $CORES (host $PHYS)"
[[ "$CORES" -ge 6 ]] || { echo "FAIL LOUD: 6 torch threads > $CORES physical cores in this allocation"; exit 1; }
echo "task $I: arm $ARM seed $S | HEAD $HEAD node $(hostname) job ${SLURM_JOB_ID:-} gpu '${CUDA_VISIBLE_DEVICES-unset}'"
[[ "$HEAD" == "$EXPECT_HEAD"* ]] || { echo "FAIL LOUD: worktree HEAD $HEAD is not $EXPECT_HEAD"; exit 1; }
CFG=experiments/uncut_arm1b_${ARM}.yaml; SPLITF=experiments/uncut_arm1_split.json
CACHE=$(python3 -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['cache_dir'])" "$CFG")
[[ -f "$CFG" && -L models && -f "$SPLITF" && -f "$CACHE/metadata.json" ]] || { echo "FAIL LOUD: config, models/ symlink, split or cache ($CACHE) missing"; exit 1; }
[[ "$(sha256sum "$SPLITF" | cut -d' ' -f1)" == 613c8cbb1cf494b60a215d3fccb4d30e6cfe17943304d7078a7b8f58f6f34808 ]] || { echo "FAIL LOUD: split sha256"; exit 1; }
[[ -z "$(git status --porcelain -- src scripts_cosim experiments run_experiment.py)" ]] || { echo "FAIL LOUD: worktree dirty"; exit 1; }
mkdir -p logs
RUN=$(python3 -c "import yaml,sys;print(yaml.safe_load(open(sys.argv[1]))['wandb']['run_name'])" "$CFG")
CK=models/$RUN-seed$S
[[ ! -e "$CK.pt" && ! -e "$CK-final.pt" ]] || { echo "FAIL LOUD: $CK exists; refusing to overwrite"; exit 1; }
python3 -u run_experiment.py "$CFG" --seed "$S" > "logs/a1bm-train-$RUN-seed$S.log" 2>&1
rc=$?
echo "run_experiment exit $rc"
fail=0
[[ $rc -eq 0 ]] || { tail -15 "logs/a1bm-train-$RUN-seed$S.log" | cut -c1-240; fail=1; }
for f in "$CK-final.pt" "$CK-final.contract.json" "$CK.val.json"; do [[ -f "$f" ]] || { echo "FAIL LOUD: no $f"; fail=$((fail+1)); }; done
python3 scripts_cosim/joint_burst_v2_sidecheck.py "$CK-final.contract.json" "ra_$ARM" "$SPLITF" inf || fail=$((fail+1))
md5sum "$CK-final.pt" "$CK-final.contract.json" "$CK.pt" 2>/dev/null
echo "=== task $I ($RUN seed $S): $fail failure(s)"
exit $fail
