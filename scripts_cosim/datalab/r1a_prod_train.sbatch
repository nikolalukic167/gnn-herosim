#!/bin/bash
# r1_attribution_v1 PRODUCTION training: 7 arms x 6 configs x 3 seeds = 126 runs as a 42-task array; task T = arm (T // 6), config (T % 6), and its 3
# seeds run in parallel inside the task. run_experiment.py, W&B, the 100-epoch cap (run_experiment clamps it). CPU-amd: on this workload an A40 was
# measured ~5x SLOWER per epoch than the CPU (tiny graphs), so no GPU. Checkpoints land in ~/gnn-herosim/models/ as
# r1-attribution-v1-<arm>-g<k>-seed<s>.pt with .contract.json and .val.json (scripts_cosim/r1a_select_checkpoints.py reads them).
#   sbatch --dependency=afterok:<cache job> --array=0-41%12 --export=ALL,WT=<pinned worktree> r1a_prod_train.sbatch
#SBATCH --job-name=r1a-train
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/r1a-train-%A_%a.out
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=150G
#SBATCH --time=48:00:00
set -uo pipefail
ulimit -n 65536 2>/dev/null || true
WT=${WT:?}
# Array index T -> (arm, config), longest first so the slowest runs bound the wall time from the start: cost = measured CPU epoch seconds
# of the arm (r1a dry run, 859 train datasets) x 1.8 for the width-128 configs g3-g5 (assumed, unmeasured).
T=${SLURM_ARRAY_TASK_ID:?FAIL LOUD: submit as an array 0-41}; SEEDS=${SEEDS:-"1 2 3"}
read -r ARM G < <(python3 - "$T" <<'PY'
import sys
EPOCH_S = {"gnn_eng": 10.6, "twin_eng": 8.9, "mlp_same": 9.5, "gnn_raw": 42.0, "twin_raw": 13.5, "gnn_eng_physmp": 10.8, "set_transformer": 16.5}
order = sorted(((a, g) for a in EPOCH_S for g in range(6)), key=lambda ag: (-EPOCH_S[ag[0]] * (1.8 if ag[1] >= 3 else 1.0), list(EPOCH_S).index(ag[0]), ag[1]))
t = int(sys.argv[1]); assert 0 <= t < len(order), f"array index {t} beyond {len(order) - 1}"
print(*order[t])
PY
) || { echo "FAIL LOUD: could not map array index $T"; exit 1; }
cd "$WT"
for v in $(compgen -e | grep -E '^(HEROSIM_|GNN_|COSIM_|LIVE_AUDIT_|GATE_|SIM_|QUEUE_FEATURE|MAX_PLACEMENT|NEAR_RTT_|PARTIAL_STATE)'); do unset "$v"; done
if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3 PYTHONPATH="$WT:$WT/src/notebooks" PYTHONUNBUFFERED=1 WANDB_ENTITY=nikolalukic167-tu-wien OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 CUDA_VISIBLE_DEVICES=""
export HEROSIM_TRANSFER_MODEL=pipelined HEROSIM_REPLICA_RELEASE=1 HEROSIM_SCALEOUT=kpa HEROSIM_SHARED_AUTOSCALER=0 GATE_FIXED_POLICY_TIME_SCALE=1.0
mkdir -p logs
CFG=experiments/r1_attribution_v1_${ARM}_g${G}.yaml
echo "task $T: arm $ARM config g$G seeds $SEEDS | HEAD $(git rev-parse HEAD) node $(hostname)"
[[ -f "$CFG" && -L models ]] || { echo "FAIL LOUD: $CFG missing or models/ is not the symlink"; exit 1; }
[[ -z "$(git status --porcelain -- src scripts_cosim experiments run_experiment.py | grep -v 'experiments/r1_attribution_v1_split.json')" ]] || { echo "FAIL LOUD: worktree dirty"; exit 1; }
[[ -f experiments/r1_attribution_v1_split.json && -f simulation_data/graphs_cache_r1_attribution_v1_psv5_inf/metadata.json ]] || { echo "FAIL LOUD: the production cache or split is missing"; exit 1; }
[[ -z "${SPLIT_SHA:-}" || "$(sha256sum experiments/r1_attribution_v1_split.json | cut -d' ' -f1)" == "$SPLIT_SHA" ]] || { echo "FAIL LOUD: split sha mismatch"; exit 1; }
RUN=$(python3 -c "import yaml,sys;print(yaml.safe_load(open(sys.argv[1]))['wandb']['run_name'])" "$CFG")
pids=(); for S in $SEEDS; do
  CK=models/$RUN-seed$S.pt
  [[ ! -e "$CK" ]] || { echo "FAIL LOUD: $CK exists; refusing to overwrite"; exit 1; }
  python3 -u run_experiment.py "$CFG" --seed "$S" > "logs/r1a-train-$RUN-seed$S.log" 2>&1 &
  pids+=($!)
done
fail=0; i=0
for S in $SEEDS; do
  wait "${pids[$i]}" || { echo "seed $S FAILED"; tail -12 "logs/r1a-train-$RUN-seed$S.log" | cut -c1-240; fail=$((fail+1)); }
  CK=models/$RUN-seed$S.pt
  for f in "$CK" "${CK%.pt}.contract.json" "${CK%.pt}.val.json"; do [[ -f "$f" ]] || { echo "FAIL LOUD: seed $S produced no $f"; fail=$((fail+1)); }; done
  python3 scripts_cosim/joint_burst_v2_sidecheck.py "${CK%.pt}.contract.json" "ra_$ARM" experiments/r1_attribution_v1_split.json inf || fail=$((fail+1))
  i=$((i+1))
done
echo "=== task $T ($RUN): $fail failure(s)"
exit $fail
