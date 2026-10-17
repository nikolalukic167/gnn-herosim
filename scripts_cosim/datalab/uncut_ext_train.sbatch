#!/bin/bash
# uncut_groups_s0_v1 seed extension (B6, registered at 3b1eaa35): one run per job. KIND=arm1b (Arm 1b gnn_eng / twin_eng, pin c94b9459) or arm1 (Arm 1 gnn_eng, ruling E4b, pin 69ed8e53),
# ARM = gnn_eng | twin_eng, SEED = 11..18. 6 threads, 0 workers (the yamls, unchanged), CPU-only, 12 CPUs = 6 physical cores (SMT is off on CPU-amd). Packed 4 runs per idle node by --nodelist at submit;
# not --exclusive. Checkpoints land in /share/nikola.lukic/uncut_arm1/models_ext (the pinned worktrees' models symlink) as <run>-seed<S>{,-final}.pt; the served checkpoint is the -final.pt.
#   sbatch --nodelist=<idle node> --time=<limit> --export=ALL,KIND=arm1b,ARM=gnn_eng,SEED=11 uncut_ext_train.sbatch
#SBATCH --job-name=ext-train
#SBATCH --output=/share/nikola.lukic/uncut_arm1/ext/train-%x-%j.out
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=150G
set -uo pipefail
ulimit -n 65536 2>/dev/null || true
KIND=${KIND:?FAIL LOUD: set KIND (arm1b | arm1)}; ARM=${ARM:?FAIL LOUD: set ARM}; S=${SEED:?FAIL LOUD: set SEED}
case "$KIND:$ARM" in
  arm1b:gnn_eng|arm1b:twin_eng) PIN=c94b9459; CFG=experiments/uncut_arm1b_${ARM}.yaml ;;
  arm1:gnn_eng) PIN=69ed8e53; CFG=experiments/uncut_arm1_${ARM}.yaml ;;
  *) echo "FAIL LOUD: KIND:ARM $KIND:$ARM is not in the registered extension"; exit 1 ;;
esac
(( S >= 11 && S <= 18 )) || { echo "FAIL LOUD: seed $S outside 11..18"; exit 1; }
WT=/home/nikola.lukic/gnn-herosim-wt/uncut_ext_$PIN
cd "$WT"
for v in $(compgen -e | grep -E '^(HEROSIM_|GNN_|COSIM_|LIVE_AUDIT_|GATE_|SIM_|QUEUE_FEATURE|MAX_PLACEMENT|NEAR_RTT_|PARTIAL_STATE|R1A_)'); do unset "$v"; done
if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3 PYTHONPATH="$WT:$WT/src/notebooks" PYTHONUNBUFFERED=1 WANDB_ENTITY=nikolalukic167-tu-wien OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 CUDA_VISIBLE_DEVICES=""
[[ "$KIND" == arm1b ]] && export CUBLAS_WORKSPACE_CONFIG=:4096:8
export HEROSIM_TRANSFER_MODEL=pipelined HEROSIM_REPLICA_RELEASE=1 HEROSIM_SCALEOUT=kpa HEROSIM_SHARED_AUTOSCALER=0 GATE_FIXED_POLICY_TIME_SCALE=1.0
export R1A_SLATE=declared_pruning_v1_uncapped
HEAD=$(git rev-parse HEAD)
echo "ext $KIND $ARM seed $S | HEAD $HEAD node $(hostname) job ${SLURM_JOB_ID:-} cpus ${SLURM_CPUS_ON_NODE:-}"
[[ "$HEAD" == "$PIN"* ]] || { echo "FAIL LOUD: worktree HEAD $HEAD is not $PIN"; exit 1; }
SPLITF=experiments/uncut_arm1_split.json
CACHE=$(python3 -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['cache_dir'])" "$CFG")
[[ -f "$CFG" && -L models && -f "$SPLITF" && -f "$CACHE/metadata.json" ]] || { echo "FAIL LOUD: config, models/ symlink, split or cache ($CACHE) missing"; exit 1; }
[[ "$(sha256sum "$SPLITF" | cut -d' ' -f1)" == 613c8cbb1cf494b60a215d3fccb4d30e6cfe17943304d7078a7b8f58f6f34808 ]] || { echo "FAIL LOUD: split sha256"; exit 1; }
[[ -z "$(git status --porcelain -- src scripts_cosim experiments run_experiment.py)" ]] || { echo "FAIL LOUD: worktree dirty"; exit 1; }
mkdir -p logs
RUN=$(python3 -c "import yaml,sys;print(yaml.safe_load(open(sys.argv[1]))['wandb']['run_name'])" "$CFG")
CK=models/$RUN-seed$S
[[ ! -e "$CK.pt" && ! -e "$CK-final.pt" ]] || { echo "FAIL LOUD: $CK exists; refusing to overwrite"; exit 1; }
python3 -u run_experiment.py "$CFG" --seed "$S" > "logs/ext-train-$RUN-seed$S.log" 2>&1
rc=$?
echo "run_experiment exit $rc"
fail=0
[[ $rc -eq 0 ]] || { tail -15 "logs/ext-train-$RUN-seed$S.log" | cut -c1-240; fail=1; }
for f in "$CK-final.pt" "$CK-final.contract.json" "$CK.val.json"; do [[ -f "$f" ]] || { echo "FAIL LOUD: no $f"; fail=$((fail+1)); }; done
python3 scripts_cosim/joint_burst_v2_sidecheck.py "$CK-final.contract.json" "ra_$ARM" "$SPLITF" inf || fail=$((fail+1))
md5sum "$CK-final.pt" "$CK-final.contract.json" "$CK.pt" 2>/dev/null
echo "=== ext $KIND $ARM seed $S ($RUN): $fail failure(s)"
exit $fail
