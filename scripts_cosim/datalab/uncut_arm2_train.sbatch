#!/bin/bash
# uncut_groups_s0_v1 Arm 2 (TAG=arm2, random-prefix K=2) and Arm 2-K0 (TAG=arm2k0, K=0) training: GNN and MP-OFF twin x seeds 1, 2.
# One array task = one run: T = 0..3 -> (gnn_eng, seed 1), (twin_eng, seed 1), (gnn_eng, seed 2), (twin_eng, seed 2). CPU-amd, 6 threads, 0 workers
# (the score closure is Python-bound; an A40 was slower, uncut_arm2 step smoke). 100 epochs project to ~5.2 days on one node; MaxTime is 7 days, no resume.
#   sbatch --array=0-3 --export=ALL,TAG=arm2,WT=<pinned worktree>,EXPECT_HEAD=<sha> uncut_arm2_train.sbatch
#SBATCH --job-name=uarm2-train
#SBATCH --output=/share/nikola.lukic/logs/uarm2-train-%A_%a.out
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
set -uo pipefail
ulimit -n 65536 2>/dev/null || true
WT=${WT:?FAIL LOUD: set WT}; EXPECT_HEAD=${EXPECT_HEAD:?FAIL LOUD: set EXPECT_HEAD}; TAG=${TAG:?FAIL LOUD: set TAG=arm2 or arm2k0}
T=${SLURM_ARRAY_TASK_ID:?FAIL LOUD: submit as --array=0-3}
[[ "$TAG" == arm2 || "$TAG" == arm2k0 ]] || { echo "FAIL LOUD: TAG $TAG"; exit 1; }
ARM=$(python3 -c "import sys; print(('gnn_eng', 'twin_eng')[int(sys.argv[1]) % 2])" "$T") || { echo "FAIL LOUD: array index $T"; exit 1; }
S=$(( T / 2 + 1 ))
cd "$WT"
for v in $(compgen -e | grep -E '^(HEROSIM_|GNN_|COSIM_|LIVE_AUDIT_|GATE_|SIM_|QUEUE_FEATURE|MAX_PLACEMENT|NEAR_RTT_|PARTIAL_STATE|R1A_)'); do unset "$v"; done
if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3 PYTHONPATH="$WT:$WT/src/notebooks" PYTHONUNBUFFERED=1 WANDB_ENTITY=nikolalukic167-tu-wien OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 CUDA_VISIBLE_DEVICES=""
export R1A_SLATE=declared_pruning_v1_uncapped  # read by joint_burst_v2_sidecheck.py below
HEAD=$(git rev-parse HEAD)
echo "task $T: tag $TAG arm $ARM seed $S | HEAD $HEAD node $(hostname) job ${SLURM_JOB_ID:-}"
[[ "$HEAD" == "$EXPECT_HEAD"* ]] || { echo "FAIL LOUD: worktree HEAD $HEAD is not $EXPECT_HEAD"; exit 1; }
CFG=experiments/uncut_${TAG}_${ARM}.yaml; SPLITF=experiments/uncut_arm1_split.json
CACHE=$(python3 -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['cache_dir'])" "$CFG")
[[ -f "$CFG" && -L models && -f "$SPLITF" && -f "$CACHE/metadata.json" ]] || { echo "FAIL LOUD: config, models/ symlink, split or cache ($CACHE) missing"; exit 1; }
[[ "$(sha256sum "$SPLITF" | cut -c1-64)" == 613c8cbb1cf494b60a215d3fccb4d30e6cfe17943304d7078a7b8f58f6f34808 ]] || { echo "FAIL LOUD: split sha"; exit 1; }
[[ -z "$(git status --porcelain -- src scripts_cosim experiments run_experiment.py)" ]] || { echo "FAIL LOUD: worktree dirty"; exit 1; }
mkdir -p logs
RUN=$(python3 -c "import yaml,sys;print(yaml.safe_load(open(sys.argv[1]))['wandb']['run_name'])" "$CFG")
CK=models/$RUN-seed$S.pt
[[ ! -e "$CK" ]] || { echo "FAIL LOUD: $CK exists; refusing to overwrite"; exit 1; }
python3 -u run_experiment.py "$CFG" --seed "$S" > "logs/uarm2-train-$RUN-seed$S.log" 2>&1
rc=$?
fail=0; [[ $rc -eq 0 ]] || { echo "run FAILED rc=$rc"; tail -12 "logs/uarm2-train-$RUN-seed$S.log" | cut -c1-240; fail=1; }
for f in "$CK" "${CK%.pt}-final.pt" "${CK%.pt}.contract.json" "${CK%.pt}.val.json"; do [[ -f "$f" ]] || { echo "FAIL LOUD: no $f"; fail=$((fail+1)); }; done
python3 scripts_cosim/joint_burst_v2_sidecheck.py "${CK%.pt}.contract.json" "ra_${ARM}_${TAG}" "$SPLITF" inf || fail=$((fail+1))
md5sum "$CK" "${CK%.pt}-final.pt" "${CK%.pt}.contract.json" 2>/dev/null
echo "=== task $T ($RUN seed $S): $fail failure(s)"
exit $fail
