#!/bin/bash
# accel_replica_v1 per-dataset served-regret dump (validation split only, no training, no test read).
# For each of gnn_eng g4 / gnn_eng_physmp g3 x seeds 1-3 x {best, -final}: NEAR_RTT_EVAL_CKPT run through run_experiment.py with the production config.
#   sbatch --export=ALL,WT=<pinned worktree>,OUT=<dir> accel_eval_perdataset.sbatch
#SBATCH --job-name=accel-evalds
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/accel-evalds-%j.out
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --time=04:00:00
set -uo pipefail
ulimit -n 65536 2>/dev/null || true
WT=${WT:?}; OUT=${OUT:?}; MODELS=${MODELS:-$HOME/gnn-herosim/models}
cd "$WT"
for v in $(compgen -e | grep -E '^(HEROSIM_|GNN_|COSIM_|LIVE_AUDIT_|GATE_|SIM_|QUEUE_FEATURE|MAX_PLACEMENT|NEAR_RTT_|PARTIAL_STATE)'); do unset "$v"; done
if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3 PYTHONPATH="$WT:$WT/src/notebooks" PYTHONUNBUFFERED=1 WANDB_MODE=disabled OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 CUDA_VISIBLE_DEVICES=""
export HEROSIM_TRANSFER_MODEL=pipelined HEROSIM_REPLICA_RELEASE=1 HEROSIM_SCALEOUT=kpa HEROSIM_SHARED_AUTOSCALER=0 GATE_FIXED_POLICY_TIME_SCALE=1.0
mkdir -p "$OUT" logs
[[ -z "$(git status --porcelain -- src scripts_cosim experiments run_experiment.py | grep -v accel_replica_v1_split)" ]] || { echo "FAIL LOUD: worktree dirty"; exit 1; }
one() { # arm cfgname seed
  local CFG=experiments/accel_replica_v1_$1.yaml RUN S=$3 K
  RUN=$(python3 -c "import yaml,sys;print(yaml.safe_load(open(sys.argv[1]))['wandb']['run_name'])" "$CFG")
  for K in ${KINDS:-"" "-final"}; do
    local CK=$MODELS/$RUN-seed$S$K.pt
    [[ -f "$CK" ]] || { echo "FAIL LOUD: $CK missing"; return 1; }
    NEAR_RTT_EVAL_CKPT=$CK NEAR_RTT_EVAL_OUT=$OUT/$RUN-seed$S${K:-"-best"}.json \
      python3 -u run_experiment.py "$CFG" --seed "$S" > "$OUT/$RUN-seed$S${K:-"-best"}.log" 2>&1 || { echo "FAIL LOUD: eval $CK"; tail -5 "$OUT/$RUN-seed$S${K:-"-best"}.log"; return 1; }
  done
}
pids=(); for S in 1 2 3; do ( one gnn_eng_g4 x $S && one gnn_eng_physmp_g3 x $S ) & pids+=($!); done
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=$((fail+1)); done
echo "=== $fail failure(s)"; ls "$OUT"/*.json | wc -l
exit $fail
