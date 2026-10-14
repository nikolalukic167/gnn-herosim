#!/bin/bash
# Arm 1b: one epoch on the real Arm 1 cache, for the device table in docs/lineages/uncut_groups_s0_v1.md. Submit per device, e.g.
#   sbatch --job-name=a1b-T-cpu6 --partition=CPU-amd --exclusive --export=ALL,TAG=cpu6,THREADS=6,WT=<worktree> uncut_arm1b_timing.sbatch
#   sbatch --job-name=a1b-T-l40s --partition=GPU-l40s --gres=gpu:1 --export=ALL,TAG=l40s,THREADS=6,WT=<worktree> uncut_arm1b_timing.sbatch
#SBATCH --output=/share/nikola.lukic/uncut_groups/arm1b/T-%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=150G
#SBATCH --time=04:00:00
set -uo pipefail
ulimit -n 65536 2>/dev/null || true
D=/share/nikola.lukic/uncut_groups/arm1b
W=${WT:?FAIL LOUD: set WT}; TAG=${TAG:?FAIL LOUD: set TAG}; THREADS=${THREADS:?FAIL LOUD: set THREADS}
eval "$(micromamba shell hook --shell bash)"; micromamba activate gnn
cd "$W"
for v in $(compgen -e | grep -E '^(HEROSIM_|GNN_|COSIM_|LIVE_AUDIT_|GATE_|SIM_|QUEUE_FEATURE|MAX_PLACEMENT|NEAR_RTT_|PARTIAL_STATE|R1A_)'); do unset "$v"; done
export HEROSIM_PY=python3 PYTHONUNBUFFERED=1 WANDB_MODE=offline WANDB_DIR=$D/wandb OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS PYTHONPATH="$W:$W/src/notebooks" CUBLAS_WORKSPACE_CONFIG=:4096:8
[[ -n "${SLURM_JOB_GPUS:-}${SLURM_GPUS_ON_NODE:-}" ]] || export CUDA_VISIBLE_DEVICES=""
export HEROSIM_TRANSFER_MODEL=pipelined HEROSIM_REPLICA_RELEASE=1 HEROSIM_SCALEOUT=kpa HEROSIM_SHARED_AUTOSCALER=0 GATE_FIXED_POLICY_TIME_SCALE=1.0
export DAG_PRIMARY_ALPHA_KEY=inf PARTIAL_STATE_CONTRACT=partial_state_v5 NEAR_RTT_LABEL_BACKLOG_CLOCK=seeded R1A_SLATE=declared_pruning_v1_uncapped
echo "HEAD $(git rev-parse --short HEAD) node $(hostname) tag $TAG threads $THREADS gpus '${CUDA_VISIBLE_DEVICES-unset}' $(nvidia-smi -L 2>/dev/null | head -1)"
python3 - "$D" "$TAG" "$THREADS" <<'PY'
import sys, yaml
d, tag, thr = sys.argv[1:4]
c = yaml.safe_load(open("experiments/uncut_arm1b_gnn_eng.yaml"))
c["args"].update({"epochs": 1, "min-epochs": 1, "patience": 1, "torch-threads": int(thr), "num-dataloader-workers": 0})
c["wandb"]["run_name"] = f"uncut-arm1b-timing-{tag}"
c["wandb"]["tags"] = c["wandb"]["tags"] + ["timing"]
yaml.safe_dump(c, open(f"{d}/timing_{tag}.yaml", "w"))
PY
START=$(date +%s)
python3 -u run_experiment.py "$D/timing_$TAG.yaml" --seed 1 > "$D/timing_$TAG.log" 2>&1; echo "rc=$? wall=$(( $(date +%s) - START ))s"
grep -a "Epoch " "$D/timing_$TAG.log" | tail -3 | sed -E 's/Rank.*Val greedy/ Val greedy/' | cut -c1-260
rm -f models/uncut-arm1b-timing-"$TAG"*
