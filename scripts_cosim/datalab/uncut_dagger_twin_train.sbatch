#!/bin/bash
# uncut_groups_s0_v1 DAgger round, MP-OFF twin arm: task i = seed i + 1 (amendment A1: lr 1e-4, 40 epochs), each fine-tuned from its own filed -final.pt on old + new labels.
# Asserts the merged override and the combined split by hash, and that the trainer applied the override to EVERY train graph (old and new). Checkpoints: models/uncut-dagger-twin-seed<s>{,-final}.pt.
#   sbatch --array=0-1 --export=ALL,WT=<pinned worktree>,EXPECT_HEAD=<sha> uncut_dagger_twin_train.sbatch
#SBATCH --job-name=dag-twin-train
#SBATCH --output=/share/nikola.lukic/uncut_groups/dagger/%x-%A_%a.out
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=150G
#SBATCH --time=48:00:00
#SBATCH --exclude=os-cpu-slurm-21,os-cpu-slurm-15,os-cpu-slurm-13,os-cpu-slurm-14,os-cpu-slurm-16,os-cpu-slurm-10,os-cpu-slurm-17,os-cpu-slurm-18,os-cpu-slurm-19,os-cpu-slurm-20,os-cpu-slurm-22
set -uo pipefail
ulimit -n 65536 2>/dev/null || true
WT=${WT:?FAIL LOUD: set WT}; EXPECT_HEAD=${EXPECT_HEAD:?FAIL LOUD: set EXPECT_HEAD}
S=$(( ${SLURM_ARRAY_TASK_ID:?FAIL LOUD: submit as --array=0-1} + 1 ))
cd "$WT"
for v in $(compgen -e | grep -E '^(HEROSIM_|GNN_|COSIM_|LIVE_AUDIT_|GATE_|SIM_|QUEUE_FEATURE|MAX_PLACEMENT|NEAR_RTT_|PARTIAL_STATE|R1A_|TRAIN_INIT)'); do unset "$v"; done
if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3 PYTHONPATH="$WT:$WT/src/notebooks" PYTHONUNBUFFERED=1 WANDB_ENTITY=nikolalukic167-tu-wien OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 CUDA_VISIBLE_DEVICES=""
export HEROSIM_TRANSFER_MODEL=pipelined HEROSIM_REPLICA_RELEASE=1 HEROSIM_SCALEOUT=kpa HEROSIM_SHARED_AUTOSCALER=0 GATE_FIXED_POLICY_TIME_SCALE=1.0
export R1A_SLATE=declared_pruning_v1_uncapped
HEAD=$(git rev-parse HEAD)
echo "twin seed $S | HEAD $HEAD node $(hostname) job ${SLURM_JOB_ID:-}"
[[ "$HEAD" == "$EXPECT_HEAD"* ]] || { echo "FAIL LOUD: worktree HEAD $HEAD is not $EXPECT_HEAD"; exit 1; }
CFG=experiments/uncut_dagger_twin_s$S.yaml; SPLITF=experiments/uncut_dagger_twin_split.json
D=/share/nikola.lukic/uncut_groups/dagger/corpus_twin
CACHE=$(python3 -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['cache_dir'])" "$CFG")
[[ -f "$CFG" && -L models && -f "$SPLITF" && -f "$CACHE/metadata.json" ]] || { echo "FAIL LOUD: config, models/ symlink, split or cache ($CACHE) missing"; exit 1; }
[[ "$CACHE" == "$D/cache_v6/cache" ]] || { echo "FAIL LOUD: cache_dir $CACHE"; exit 1; }
[[ "$(sha256sum "$SPLITF" | cut -d' ' -f1)" == d5b4ee9e71a128bc150079512749f480d95514bb7b2af85dd8582e1b7f285f22 ]] || { echo "FAIL LOUD: split sha256"; exit 1; }
[[ "$(md5sum $D/label_override_merged.json | cut -d' ' -f1)" == 7ed6b4eb4df34fdda074f959d116ff5c ]] || { echo "FAIL LOUD: merged override md5"; exit 1; }
grep -q "label_override_merged.json" "$CFG" || { echo "FAIL LOUD: config does not name the merged override"; exit 1; }
INIT=models/uncut-comb-fl-twin-seed$S-final.pt
WANT=$([ $S = 1 ] && echo f7a5d76d || echo c96187ba)
[[ "$(md5sum $INIT | cut -c1-8)" == $WANT ]] || { echo "FAIL LOUD: init checkpoint md5"; exit 1; }
[[ -f "${INIT%.pt}.contract.json" ]] || { echo "FAIL LOUD: init contract sidecar missing"; exit 1; }
[[ -z "$(git status --porcelain -- src scripts_cosim experiments run_experiment.py)" ]] || { echo "FAIL LOUD: worktree dirty"; exit 1; }
mkdir -p logs
RUN=$(python3 -c "import yaml,sys;print(yaml.safe_load(open(sys.argv[1]))['wandb']['run_name'])" "$CFG")
CK=models/$RUN-seed$S
[[ ! -e "$CK.pt" && ! -e "$CK-final.pt" ]] || { echo "FAIL LOUD: $CK exists; refusing to overwrite"; exit 1; }
LOG="logs/dag-twin-train-$RUN-seed$S.log"
python3 -u run_experiment.py "$CFG" --seed "$S" > "$LOG" 2>&1
rc=$?
echo "run_experiment exit $rc"
fail=0
[[ $rc -eq 0 ]] || { tail -15 "$LOG" | cut -c1-240; fail=1; }
tr '\r' '\n' < "$LOG" | grep -E "NEAR_RTT_LABEL_OVERRIDE_JSON\] train label replaced|Valid labels|wandb: .*(Syncing|View run)" | head -5
[[ "$(tr '\r' '\n' < "$LOG" | grep -c "train label replaced on 11521 graphs")" -ge 1 ]] || { echo "FAIL LOUD: the override was not applied to all 11521 train graphs"; fail=$((fail+1)); }
for f in "$CK-final.pt" "$CK-final.contract.json" "$CK.val.json"; do [[ -f "$f" ]] || { echo "FAIL LOUD: no $f"; fail=$((fail+1)); }; done
R1A_SLATE=declared_pruning_v1_uncapped python3 scripts_cosim/joint_burst_v2_sidecheck.py "$CK-final.contract.json" "ra_twin_eng_v6" "$SPLITF" inf || fail=$((fail+1))
md5sum "$CK-final.pt" "$CK-final.contract.json" "$CK.pt" 2>/dev/null
echo "=== $RUN seed $S: $fail failure(s)"
exit $fail
