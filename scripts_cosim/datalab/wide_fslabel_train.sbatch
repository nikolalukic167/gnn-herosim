#!/bin/bash
# WIDE MLP on the fair-share labels (user, 2026-10-23; capacity control for the fslabel result): experiments/uncut_comb_fl_mlp_wide_fslabel.yaml (hidden 675 = 500,176 trainable parameters, 20 epochs, the fair-share label
# override asserted by sha256 below). One run per array task = one seed: --array=1-3. Checkpoints land in the pin's models/ symlink as uncut-comb-fl-mlp-wide-fslabel-seed<S>{,-final}.pt; the served one is -final.pt.
#   sbatch --array=1-3 --export=ALL,WT=<pinned worktree>,EXPECT_HEAD=<sha> wide_fslabel_train.sbatch
#SBATCH --job-name=wide-fslabel-train
#SBATCH --output=/share/nikola.lukic/uncut_groups/dagger/%x-%j.out
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=150G
#SBATCH --time=12:00:00
#SBATCH --exclude=os-cpu-slurm-21,os-cpu-slurm-15,os-cpu-slurm-13,os-cpu-slurm-14,os-cpu-slurm-16,os-cpu-slurm-10,os-cpu-slurm-17
set -uo pipefail
ulimit -n 65536 2>/dev/null || true
WT=${WT:?FAIL LOUD: set WT}; EXPECT_HEAD=${EXPECT_HEAD:?FAIL LOUD: set EXPECT_HEAD}; S=${SLURM_ARRAY_TASK_ID:?FAIL LOUD: submit as an array, one seed per task}
LABEL_SHA256=7865ed329079bbc40b0e3ed91301ad3b84e721497c1aa832192c6f01dcfe2e46
[[ "$S" =~ ^[123]$ ]] || { echo "FAIL LOUD: seed $S not in 1-3"; exit 1; }
cd "$WT"
for v in $(compgen -e | grep -E '^(HEROSIM_|GNN_|COSIM_|LIVE_AUDIT_|GATE_|SIM_|QUEUE_FEATURE|MAX_PLACEMENT|NEAR_RTT_|PARTIAL_STATE|R1A_)'); do unset "$v"; done
if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3 PYTHONPATH="$WT:$WT/src/notebooks" PYTHONUNBUFFERED=1 WANDB_ENTITY=nikolalukic167-tu-wien OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 CUDA_VISIBLE_DEVICES=""
export HEROSIM_TRANSFER_MODEL=pipelined HEROSIM_REPLICA_RELEASE=1 HEROSIM_SCALEOUT=kpa HEROSIM_SHARED_AUTOSCALER=0 GATE_FIXED_POLICY_TIME_SCALE=1.0
export R1A_SLATE=declared_pruning_v1_uncapped
HEAD=$(git rev-parse HEAD)
echo "wide-fslabel seed $S | HEAD $HEAD node $(hostname) job ${SLURM_JOB_ID:-}"
[[ "$HEAD" == "$EXPECT_HEAD"* ]] || { echo "FAIL LOUD: worktree HEAD $HEAD is not $EXPECT_HEAD"; exit 1; }
KIND=ra_mlp_same_v6
CFG=experiments/uncut_comb_fl_mlp_wide_fslabel.yaml; SPLITF=experiments/uncut_arm1_split.json
CACHE=$(python3 -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['cache_dir'])" "$CFG")
[[ -f "$CFG" && -L models && -f "$SPLITF" && -f "$CACHE/metadata.json" ]] || { echo "FAIL LOUD: config, models/ symlink, split or cache ($CACHE) missing"; exit 1; }
[[ "$(sha256sum "$SPLITF" | cut -d' ' -f1)" == 613c8cbb1cf494b60a215d3fccb4d30e6cfe17943304d7078a7b8f58f6f34808 ]] || { echo "FAIL LOUD: split sha256"; exit 1; }
[[ -z "$(git status --porcelain -- src scripts_cosim experiments run_experiment.py)" ]] || { echo "FAIL LOUD: worktree dirty"; exit 1; }
LABELF=$(python3 -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['env']['NEAR_RTT_LABEL_OVERRIDE_JSON'])" "$CFG")
[[ -f "$LABELF" && "$(sha256sum "$LABELF" | cut -d' ' -f1)" == "$LABEL_SHA256" ]] || { echo "FAIL LOUD: label override $LABELF missing or its sha256 is not $LABEL_SHA256"; exit 1; }
echo "label override $LABELF sha256 $LABEL_SHA256 md5 $(md5sum "$LABELF" | cut -d' ' -f1)"
mkdir -p logs
RUN=$(python3 -c "import yaml,sys;print(yaml.safe_load(open(sys.argv[1]))['wandb']['run_name'])" "$CFG")
CK=models/$RUN-seed$S
[[ ! -e "$CK.pt" && ! -e "$CK-final.pt" ]] || { echo "FAIL LOUD: $CK exists; refusing to overwrite"; exit 1; }
python3 -u run_experiment.py "$CFG" --seed "$S" > "logs/wide-fslabel-train-$RUN-seed$S.log" 2>&1
rc=$?
echo "run_experiment exit $rc"
fail=0
[[ $rc -eq 0 ]] || { tail -15 "logs/wide-fslabel-train-$RUN-seed$S.log" | cut -c1-240; fail=1; }
for f in "$CK-final.pt" "$CK-final.contract.json" "$CK.val.json"; do [[ -f "$f" ]] || { echo "FAIL LOUD: no $f"; fail=$((fail+1)); }; done
python3 scripts_cosim/joint_burst_v2_sidecheck.py "$CK-final.contract.json" "$KIND" "$SPLITF" inf || fail=$((fail+1))
python3 - "$CK-final.pt" <<'PY'
import sys, torch
sd = torch.load(sys.argv[1], map_location="cpu"); sd = sd.get("model_state_dict", sd)
print("trainable parameters in the checkpoint:", sum(v.numel() for v in sd.values() if hasattr(v, "numel")))
PY
md5sum "$CK-final.pt" "$CK-final.contract.json" "$CK.pt" 2>/dev/null
echo "=== $RUN seed $S: $fail failure(s)"
exit $fail
