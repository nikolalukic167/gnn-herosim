#!/bin/bash
# uncut_groups_s0_v1 expected move-regret lever, wave 1: gnn_eng with lambda x expected move regret (experiments/uncut_arm3_gnn_eng_l{0p5,2p0}.yaml), CPU, 6 threads on 12 CPUs
# (SMT is off on CPU-amd: 12 CPUs hold 6 physical cores; threads never exceed the in-job cores). Array index i: lambda = (0.5, 2.0)[i // 2], seed = i % 2 + 1.
#   sbatch --array=0-3 --export=ALL,WT=<pinned worktree>,EXPECT_HEAD=<sha> uncut_arm3_train.sbatch
# SMOKE=1: one epoch of the same config (epochs, min-epochs and patience 1, run name + "-smoke", tag "smoke"), checkpoints under the smoke run name; nothing is gated.
# Not --exclusive. (Wave 1 carried --deadline=2026-10-17T01:00:00 to keep clear of the hand calibration 871538; dropped for the seed extension.)
# Seed extension (2026-10-17): EXT_ARMS=gnn_eng,twin_eng EXT_SEEDS=11,...,18 maps array index i to arm EXT_ARMS[i // len(EXT_SEEDS)], seed EXT_SEEDS[i % len],
# lambda 2.0, config experiments/uncut_arm3_<arm>_l2p0.yaml; unset, the wave-1 mapping above is unchanged.
# Combined lever FAST arms (2026-10-17): EXT_ARMS gnn_eng_sim (from scratch on the simulator labels) and gnn_eng_simft (fine-tune of the filed
# Arm 3 -final.pt of the same seed, md5 from experiments/uncut_arm3_ckpt_md5.json; TRAIN_INIT_CHECKPOINT). Their configs name the label override
# (LABEL_SHA must match it) and the recomputed move-regret table (TABLE_SHA must match the config's NEAR_RTT_MOVE_REGRET_PATH).
#SBATCH --job-name=a3-train
#SBATCH --output=/share/nikola.lukic/uncut_groups/lever/train3-%A_%a.out
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=150G
#SBATCH --time=14:00:00
set -uo pipefail
ulimit -n 65536 2>/dev/null || true
WT=${WT:?FAIL LOUD: set WT}; EXPECT_HEAD=${EXPECT_HEAD:?FAIL LOUD: set EXPECT_HEAD}
I=${SLURM_ARRAY_TASK_ID:?FAIL LOUD: submit as --array=0-3}
if [[ -n "${EXT_SEEDS:-}" ]]; then
  read -r ARM S < <(python3 -c "import sys; a=sys.argv[2].split(','); s=sys.argv[3].split(','); i=int(sys.argv[1]); assert 0 <= i < len(a)*len(s); print(a[i // len(s)], s[i % len(s)])" "$I" "${EXT_ARMS:?FAIL LOUD: set EXT_ARMS}" "$EXT_SEEDS") || { echo "FAIL LOUD: array index $I for EXT_ARMS/EXT_SEEDS"; exit 1; }
  [[ "$ARM" == gnn_eng || "$ARM" == twin_eng || "$ARM" == gnn_eng_sim || "$ARM" == gnn_eng_simft ]] || { echo "FAIL LOUD: EXT arm $ARM"; exit 1; }
  TAG=l2p0
else
  TAG=$(python3 -c "import sys; print(('l0p5', 'l2p0')[int(sys.argv[1]) // 2])" "$I") || { echo "FAIL LOUD: array index $I"; exit 1; }
  S=$(( I % 2 + 1 )); ARM=gnn_eng
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
HEAD=$(git rev-parse HEAD)
echo "task $I: arm $ARM lambda tag $TAG seed $S | HEAD $HEAD node $(hostname) job ${SLURM_JOB_ID:-} cpus $(nproc) smt $(cat /sys/devices/system/cpu/smt/control 2>/dev/null) smoke ${SMOKE:-0}"
[[ "$HEAD" == "$EXPECT_HEAD"* ]] || { echo "FAIL LOUD: worktree HEAD $HEAD is not $EXPECT_HEAD"; exit 1; }
[[ "$(nproc)" -ge 6 ]] || { echo "FAIL LOUD: fewer than 6 cores in the cpuset"; exit 1; }
CFG=experiments/uncut_arm3_${ARM}_${TAG}.yaml; SPLITF=experiments/uncut_arm1_split.json
CACHE=$(python3 -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['cache_dir'])" "$CFG")
[[ -f "$CFG" && -L models && -f "$SPLITF" && -f "$CACHE/metadata.json" ]] || { echo "FAIL LOUD: config, models/ symlink, split or cache ($CACHE) missing"; exit 1; }
[[ "$(sha256sum "$SPLITF" | cut -d' ' -f1)" == 613c8cbb1cf494b60a215d3fccb4d30e6cfe17943304d7078a7b8f58f6f34808 ]] || { echo "FAIL LOUD: split sha256"; exit 1; }
[[ -z "$(git status --porcelain -- src scripts_cosim experiments run_experiment.py)" ]] || { echo "FAIL LOUD: worktree dirty"; exit 1; }
D=/share/nikola.lukic/uncut_groups/lever
MR=$(python3 -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['env']['NEAR_RTT_MOVE_REGRET_PATH'])" "$CFG")
[[ "$(sha256sum "$MR" | cut -d' ' -f1)" == "${TABLE_SHA:?FAIL LOUD: set TABLE_SHA}" ]] || { echo "FAIL LOUD: $MR sha256"; exit 1; }
LO=$(python3 -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['env'].get('NEAR_RTT_LABEL_OVERRIDE_JSON', ''))" "$CFG")
if [[ "$ARM" == gnn_eng_sim* ]]; then
  [[ -n "$LO" && "$(sha256sum "$LO" | cut -d' ' -f1)" == "${LABEL_SHA:?FAIL LOUD: set LABEL_SHA}" ]] || { echo "FAIL LOUD: label override $LO sha256"; exit 1; }
else
  [[ -z "$LO" ]] || { echo "FAIL LOUD: $ARM must not carry a label override"; exit 1; }
fi
unset TRAIN_INIT_CHECKPOINT
if [[ "$ARM" == gnn_eng_simft ]]; then
  INIT=models/uncut-arm3-gnn-eng-l2p0-seed$S-final.pt
  WANT=$(python3 -c "import json,sys; print(json.load(open('experiments/uncut_arm3_ckpt_md5.json'))['gnn_eng|s' + sys.argv[1]])" "$S") || { echo "FAIL LOUD: no filed md5 for seed $S"; exit 1; }
  [[ -f "$INIT" && "$(md5sum < "$INIT" | cut -c1-32)" == "$WANT" ]] || { echo "FAIL LOUD: init checkpoint $INIT md5 is not the filed $WANT"; exit 1; }
  export TRAIN_INIT_CHECKPOINT="$PWD/$INIT"; echo "fine-tune from $INIT md5 $WANT"
fi
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
[[ ! -e "$CK.pt" && ! -e "$CK-final.pt" ]] || { echo "FAIL LOUD: $CK exists; refusing to overwrite"; exit 1; }
python3 -u run_experiment.py "$CFG" --seed "$S" > "logs/a3-train-$RUN-seed$S.log" 2>&1
rc=$?
if [[ "$ARM" == gnn_eng_simft ]] && ! grep -q "Loaded init checkpoint: $TRAIN_INIT_CHECKPOINT" "logs/a3-train-$RUN-seed$S.log"; then echo "FAIL LOUD: the trainer did not load the init checkpoint"; rc=1; fi
if [[ "$ARM" == gnn_eng_sim* ]] && ! grep -q "NEAR_RTT_LABEL_OVERRIDE_JSON\] train label replaced" "logs/a3-train-$RUN-seed$S.log"; then echo "FAIL LOUD: the trainer did not apply the label override"; rc=1; fi
echo "run_experiment exit $rc"
fail=0
[[ $rc -eq 0 ]] || { tail -15 "logs/a3-train-$RUN-seed$S.log" | cut -c1-240; fail=1; }
for f in "$CK-final.pt" "$CK-final.contract.json" "$CK.val.json"; do [[ -f "$f" ]] || { echo "FAIL LOUD: no $f"; fail=$((fail+1)); }; done
grep -E "NEAR_RTT_REGRET_LAMBDA|\[REGRET\]|Epoch +0|epoch 0|unscored" "logs/a3-train-$RUN-seed$S.log" | head -12 | cut -c1-240
python3 scripts_cosim/joint_burst_v2_sidecheck.py "$CK-final.contract.json" "ra_${ARM%%_sim*}" "$SPLITF" inf || fail=$((fail+1))
md5sum "$CK-final.pt" "$CK-final.contract.json" "$CK.pt" 2>/dev/null
echo "=== task $I ($RUN seed $S): $fail failure(s)"
exit $fail
