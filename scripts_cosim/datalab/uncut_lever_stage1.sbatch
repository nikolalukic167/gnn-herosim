#!/bin/bash
# Lever stage 1, offline (uncut_groups_s0_v1 Record). One file, four phases: PHASE=prep | decode | replay | analyze (submit with --array=0-11 for decode and replay, STRIDE=12).
# Usage: sbatch --export=ALL,PHASE=decode,WT=<pin worktree>,EXPECT_HEAD=<sha> [--array=0-11] --exclude=os-cpu-slurm-6 scripts_cosim/datalab/uncut_lever_stage1.sbatch
#SBATCH --job-name=l1-stage1
#SBATCH --output=/share/nikola.lukic/uncut_groups/lever/stage1/%x-%A_%a.out
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=48G
#SBATCH --time=06:00:00
set -euo pipefail
ulimit -n 65536 2>/dev/null || true
W=${WT:?FAIL LOUD: set WT}; PHASE=${PHASE:?FAIL LOUD: set PHASE}; EXPECT_HEAD=${EXPECT_HEAD:?FAIL LOUD: set EXPECT_HEAD}
D=/share/nikola.lukic/uncut_groups/lever/stage1
MASK=/share/nikola.lukic/uncut_groups/lever/unscored_mask_val.pkl
eval "$(micromamba shell hook --shell bash)"; micromamba activate gnn
cd "$W"
for v in $(compgen -e | grep -E "^(HEROSIM_|GNN_|COSIM_|LIVE_AUDIT_|GATE_|SIM_|QUEUE_FEATURE|MAX_PLACEMENT|NEAR_RTT_|PARTIAL_STATE|R1A_)"); do unset "$v"; done
export HEROSIM_PY=python3 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONPATH="$W:$W/src/notebooks" DAG_PRIMARY_ALPHA_KEY=inf
HEAD=$(git rev-parse HEAD)
echo "phase $PHASE node $(hostname) job ${SLURM_JOB_ID} task ${SLURM_ARRAY_TASK_ID:-} HEAD $HEAD nproc $(nproc) smt $(cat /sys/devices/system/cpu/smt/control 2>/dev/null)"
[[ "$HEAD" == "$EXPECT_HEAD"* ]] || { echo "FAIL LOUD: worktree HEAD $HEAD is not $EXPECT_HEAD"; exit 1; }
[[ -z "$(git status --porcelain -- src scripts_cosim)" ]] || { echo "FAIL LOUD: worktree dirty"; exit 1; }
mkdir -p "$D"
DRAW=docs/lineages/uncut_groups_s0_v1/replay_diag_draw.json
STRIDE=${STRIDE:-12}
case "$PHASE" in
  prep) python3 scripts_cosim/uncut_lever_stage1.py prep --out "$D/shards" ;;
  decode)
    NS=$(ls "$D"/shards/g_*.pkl | wc -l); mkdir -p "$D/decode"
    for K in $(seq "$SLURM_ARRAY_TASK_ID" "$STRIDE" $((NS - 1))); do
      T=$(printf "%03d" "$K")
      python3 scripts_cosim/uncut_lever_stage1.py decode --shard "$D/shards/g_$T.pkl" --mask "$MASK" --out "$D/decode/d_${T}_stage1.json"
    done ;;
  replay)
    NS=$(ls "$D"/shards/g_*.pkl | wc -l); mkdir -p "$D/replay"
    python3 scripts_cosim/uncut_lever_stage1.py replay --shards "$D/shards" --n-shards "$NS" --task "$SLURM_ARRAY_TASK_ID" --stride "$STRIDE" --decode-dir "$D/decode" --draw "$DRAW" --out "$D/replay/r_$(printf "%02d" "$SLURM_ARRAY_TASK_ID").json" ;;
  analyze) python3 scripts_cosim/uncut_lever_stage1.py analyze --replay-dir "$D/replay" --decode-dir "$D/decode" --out "$D/${OUT:-stage1}.json" --md "$D/${OUT:-stage1}.md" ;;
  *) echo "FAIL LOUD: PHASE $PHASE"; exit 1 ;;
esac
