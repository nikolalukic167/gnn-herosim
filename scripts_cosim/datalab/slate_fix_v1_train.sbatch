#!/bin/bash
# slate_fix_v1 training (docs/lineages/slate_fix_v1.md): the accel_replica_v1 g4 recipe and its MP-OFF twin on the UNCAPPED accel cache, seeds 1 and 2.
# Task 0 = gnn_eng, task 1 = twin_eng; both seeds run in parallel inside the task (as r1a_prod_train.sbatch). CPU-amd, no GPU (r1a: an A40 was ~5x slower).
# Checkpoints land in ~/gnn-herosim/models/ (the worktree's models/ symlink) as slate-fix-v1-<arm>-seed<s>.pt with .contract.json and .val.json.
#   sbatch --array=0-1 --export=ALL,WT=<pinned worktree>,EXPECT_HEAD=<sha>,OLD_CACHE=<accel_replica_v1 cache dir> slate_fix_v1_train.sbatch
#SBATCH --job-name=sf1-train
#SBATCH --output=/share/nikola.lukic/logs/sf1-train-%A_%a.out
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=150G
#SBATCH --time=24:00:00
set -uo pipefail
ulimit -n 65536 2>/dev/null || true
WT=${WT:?FAIL LOUD: set WT}; EXPECT_HEAD=${EXPECT_HEAD:?FAIL LOUD: set EXPECT_HEAD}; OLD_CACHE=${OLD_CACHE:?FAIL LOUD: set OLD_CACHE}
T=${SLURM_ARRAY_TASK_ID:?FAIL LOUD: submit as --array=0-1}; SEEDS=${SEEDS:-"1 2"}
ARM=$(python3 -c "import sys; print(('gnn_eng', 'twin_eng')[int(sys.argv[1])])" "$T") || { echo "FAIL LOUD: array index $T"; exit 1; }
cd "$WT"
for v in $(compgen -e | grep -E '^(HEROSIM_|GNN_|COSIM_|LIVE_AUDIT_|GATE_|SIM_|QUEUE_FEATURE|MAX_PLACEMENT|NEAR_RTT_|PARTIAL_STATE|R1A_)'); do unset "$v"; done
if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3 PYTHONPATH="$WT:$WT/src/notebooks" PYTHONUNBUFFERED=1 WANDB_ENTITY=nikolalukic167-tu-wien OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 CUDA_VISIBLE_DEVICES=""
export HEROSIM_TRANSFER_MODEL=pipelined HEROSIM_REPLICA_RELEASE=1 HEROSIM_SCALEOUT=kpa HEROSIM_SHARED_AUTOSCALER=0 GATE_FIXED_POLICY_TIME_SCALE=1.0
export R1A_SLATE=declared_pruning_v1_uncapped  # read by joint_burst_v2_sidecheck.py below
HEAD=$(git rev-parse HEAD)
echo "task $T: arm $ARM seeds $SEEDS | HEAD $HEAD node $(hostname) job ${SLURM_JOB_ID:-}"
[[ "$HEAD" == "$EXPECT_HEAD"* ]] || { echo "FAIL LOUD: worktree HEAD $HEAD is not $EXPECT_HEAD"; exit 1; }
CFG=experiments/slate_fix_v1_${ARM}.yaml; SPLITF=experiments/accel_replica_v1_split.json
CACHE=$(python3 -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['cache_dir'])" "$CFG")
[[ "$CACHE" != *S7_FILLS_THIS* ]] || { echo "FAIL LOUD: $CFG still names the placeholder cache $CACHE"; exit 1; }
[[ -f "$CFG" && -L models && -f "$SPLITF" && -f "$CACHE/metadata.json" ]] || { echo "FAIL LOUD: config, models/ symlink, split or cache ($CACHE) missing"; exit 1; }
[[ -z "$(git status --porcelain -- src scripts_cosim experiments run_experiment.py | grep -v "$SPLITF")" ]] || { echo "FAIL LOUD: worktree dirty"; exit 1; }
python3 scripts_cosim/slate_fix_v1_cache_check.py --new "$CACHE" --old "$OLD_CACHE" --split "$SPLITF" || exit 1
mkdir -p logs
RUN=$(python3 -c "import yaml,sys;print(yaml.safe_load(open(sys.argv[1]))['wandb']['run_name'])" "$CFG")
pids=(); for S in $SEEDS; do
  CK=models/$RUN-seed$S.pt
  [[ ! -e "$CK" ]] || { echo "FAIL LOUD: $CK exists; refusing to overwrite"; exit 1; }
  python3 -u run_experiment.py "$CFG" --seed "$S" > "logs/sf1-train-$RUN-seed$S.log" 2>&1 &
  pids+=($!)
done
fail=0; i=0
for S in $SEEDS; do
  wait "${pids[$i]}" || { echo "seed $S FAILED"; tail -12 "logs/sf1-train-$RUN-seed$S.log" | cut -c1-240; fail=$((fail+1)); }
  CK=models/$RUN-seed$S.pt
  for f in "$CK" "${CK%.pt}.contract.json" "${CK%.pt}.val.json"; do [[ -f "$f" ]] || { echo "FAIL LOUD: seed $S produced no $f"; fail=$((fail+1)); }; done
  python3 scripts_cosim/joint_burst_v2_sidecheck.py "${CK%.pt}.contract.json" "ra_$ARM" "$SPLITF" inf || fail=$((fail+1))
  md5sum "$CK" "${CK%.pt}.contract.json" 2>/dev/null
  i=$((i+1))
done
echo "=== task $T ($RUN): $fail failure(s)"
exit $fail
