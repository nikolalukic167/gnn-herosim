#!/bin/bash
# uncut_groups_s0_v1 Arm 2 (random-prefix K=2) GNN on ONE L40S: seeds 1 and 2 (T=0, T=2), configs experiments/uncut_arm2_gnn_eng_gpu.yaml (run uncut-arm2-gnn-eng-gpu).
# The CPU variant is uncut_arm2_train.sbatch; the recipe, flags, 0 dataloader workers and OMP/MKL threads (6) are the ones tested bit-equal run to run on an L40S
# (mini-cache, 70 tensors). Each link is a 7-day job (the partition MaxTime); NEAR_RTT_RESUME=1 saves the whole trainer state every epoch; a link killed by the
# time limit continues from it. The L40S nodes are shared with other users' jobs, so the node is not --exclusive (8 GPUs per node); 8 CPUs for the host-bound decode.
#   sbatch --export=ALL,T=0,WT=<pinned worktree>,EXPECT_HEAD=<sha>[,PREV=<previous link job id>] uncut_arm2_gpu_train.sbatch
# Seed extension (2026-10-17): EXT_SEED=<n> replaces the seed T implies (T=0; seeds 11-16).
#SBATCH --job-name=uarm2-gpu-train
#SBATCH --output=/share/nikola.lukic/logs/uarm2-gpu-train-%j.out
#SBATCH --partition=GPU-l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
set -uo pipefail
ulimit -n 65536 2>/dev/null || true
WT=${WT:?FAIL LOUD: set WT}; EXPECT_HEAD=${EXPECT_HEAD:?FAIL LOUD: set EXPECT_HEAD}; T=${T:?FAIL LOUD: set T=0 or 2}
TAG=arm2; ARM=gnn_eng
[[ "$T" == 0 || "$T" == 2 ]] || { echo "FAIL LOUD: T $T (only the K=2 GNN runs, T=0 seed 1 and T=2 seed 2, are on GPU)"; exit 1; }
S=${EXT_SEED:-$(( T / 2 + 1 ))}
[[ "$S" =~ ^[0-9]+$ ]] || { echo "FAIL LOUD: seed $S"; exit 1; }
# A link starts only if the previous one ended by the time limit (more epochs to run) or finished; anything else (a crash, an OOM, a cancel) stops the
# chain here, so a crash loop cannot spend a 7-day slot per link.
if [[ -n "${PREV:-}" ]]; then
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
python3 -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || { echo "FAIL LOUD: no CUDA device on $(hostname)"; exit 1; }
nvidia-smi --query-gpu=name --format=csv,noheader
export HEROSIM_PY=python3 PYTHONPATH="$WT:$WT/src/notebooks" PYTHONUNBUFFERED=1 WANDB_ENTITY=nikolalukic167-tu-wien OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6
export HEROSIM_TRANSFER_MODEL=pipelined HEROSIM_REPLICA_RELEASE=1 HEROSIM_SCALEOUT=kpa HEROSIM_SHARED_AUTOSCALER=0 GATE_FIXED_POLICY_TIME_SCALE=1.0  # the cache's physics_env (metadata.json); the trainer refuses a mismatch
export R1A_SLATE=declared_pruning_v1_uncapped  # read by joint_burst_v2_sidecheck.py below
export NEAR_RTT_RESUME=1 NEAR_RTT_RESUME_PIN="$EXPECT_HEAD"
HEAD=$(git rev-parse HEAD)
echo "task $T: GPU tag $TAG arm $ARM seed $S | HEAD $HEAD node $(hostname) job ${SLURM_JOB_ID:-} prev ${PREV:-none}"
[[ "$HEAD" == "$EXPECT_HEAD"* ]] || { echo "FAIL LOUD: worktree HEAD $HEAD is not $EXPECT_HEAD"; exit 1; }
CFG=experiments/uncut_${TAG}_${ARM}_gpu.yaml; SPLITF=experiments/uncut_arm1_split.json
CACHE=$(python3 -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['cache_dir'])" "$CFG")
[[ -f "$CFG" && -L models && -f "$SPLITF" && -f "$CACHE/metadata.json" ]] || { echo "FAIL LOUD: config, models/ symlink, split or cache ($CACHE) missing"; exit 1; }
[[ "$(sha256sum "$SPLITF" | cut -c1-64)" == 613c8cbb1cf494b60a215d3fccb4d30e6cfe17943304d7078a7b8f58f6f34808 ]] || { echo "FAIL LOUD: split sha"; exit 1; }
[[ -z "$(git status --porcelain -- src scripts_cosim experiments run_experiment.py)" ]] || { echo "FAIL LOUD: worktree dirty"; exit 1; }
mkdir -p logs
RUN=$(python3 -c "import yaml,sys;print(yaml.safe_load(open(sys.argv[1]))['wandb']['run_name'])" "$CFG")
CK=models/$RUN-seed$S.pt
# a fresh start must not overwrite a checkpoint; a continuation has the trainer's own resume state next to it
if [[ ! -e "models/$RUN-seed$S.resume.pt" && -e "$CK" ]]; then echo "FAIL LOUD: $CK exists without a resume state; refusing to overwrite"; exit 1; fi
python3 -u run_experiment.py "$CFG" --seed "$S" >> "logs/uarm2-gpu-train-$RUN-seed$S.log" 2>&1
rc=$?
fail=0; [[ $rc -eq 0 ]] || { echo "run FAILED rc=$rc"; tail -12 "logs/uarm2-gpu-train-$RUN-seed$S.log" | cut -c1-240; exit 1; }
for f in "$CK" "${CK%.pt}-final.pt" "${CK%.pt}.contract.json" "${CK%.pt}.val.json"; do [[ -f "$f" ]] || { echo "FAIL LOUD: no $f"; fail=$((fail+1)); }; done
python3 scripts_cosim/joint_burst_v2_sidecheck.py "${CK%.pt}.contract.json" "ra_${ARM}_${TAG}" "$SPLITF" inf || fail=$((fail+1))
md5sum "$CK" "${CK%.pt}-final.pt" "${CK%.pt}.contract.json" 2>/dev/null
echo "=== task $T ($RUN seed $S): $fail failure(s)"
exit $fail
