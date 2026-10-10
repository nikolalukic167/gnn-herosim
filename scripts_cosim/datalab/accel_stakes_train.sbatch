#!/bin/bash
# accel_replica_v1 loss test: physmp g3 with NEAR_RTT_CE_GRAPH_WEIGHT=stakes, seeds 1-3 in parallel, run_experiment.py, W&B on, 100-epoch cap (clamped by run_experiment).
#   sbatch --export=ALL,WT=<pinned worktree> accel_stakes_train.sbatch     (WT needs experiments/accel_replica_v1_split.json, the cache symlink and a real models/ dir)
#SBATCH --job-name=accel-stakes-train
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/accel-stakes-train-%j.out
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=150G
#SBATCH --time=24:00:00
set -uo pipefail
ulimit -n 65536 2>/dev/null || true
WT=${WT:?}; cd "$WT"
for v in $(compgen -e | grep -E '^(HEROSIM_|GNN_|COSIM_|LIVE_AUDIT_|GATE_|SIM_|QUEUE_FEATURE|MAX_PLACEMENT|NEAR_RTT_|PARTIAL_STATE)'); do unset "$v"; done
if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"; micromamba activate gnn
export HEROSIM_PY=python3 PYTHONPATH="$WT:$WT/src/notebooks" PYTHONUNBUFFERED=1 WANDB_ENTITY=nikolalukic167-tu-wien OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 CUDA_VISIBLE_DEVICES=""
export HEROSIM_TRANSFER_MODEL=pipelined HEROSIM_REPLICA_RELEASE=1 HEROSIM_SCALEOUT=kpa HEROSIM_SHARED_AUTOSCALER=0 GATE_FIXED_POLICY_TIME_SCALE=1.0
mkdir -p logs; CFG=experiments/accel_replica_v1_stakes_gnn_eng_physmp_g3.yaml; SPLITF=experiments/accel_replica_v1_split.json
RUN=$(python3 -c "import yaml,sys;print(yaml.safe_load(open(sys.argv[1]))['wandb']['run_name'])" "$CFG")
echo "HEAD $(git rev-parse HEAD) node $(hostname) run $RUN"
[[ -f "$CFG" && -f "$SPLITF" && -f simulation_data/graphs_cache_accel_replica_v1_psv5_inf/metadata.json && ! -L models ]] || { echo "FAIL LOUD: config/split/cache missing or models/ is a symlink"; exit 1; }
[[ -z "$(git status --porcelain -- src scripts_cosim experiments run_experiment.py | grep -v accel_replica_v1_split)" ]] || { echo "FAIL LOUD: worktree dirty"; exit 1; }
pids=(); for S in 1 2 3; do
  [[ ! -e "models/$RUN-seed$S.pt" ]] || { echo "FAIL LOUD: models/$RUN-seed$S.pt exists"; exit 1; }
  python3 -u run_experiment.py "$CFG" --seed "$S" > "logs/accel-stakes-train-seed$S.log" 2>&1 & pids+=($!)
done
fail=0; i=0; for S in 1 2 3; do
  wait "${pids[$i]}" || { echo "seed $S FAILED"; tail -10 "logs/accel-stakes-train-seed$S.log" | cut -c1-240; fail=$((fail+1)); }
  for f in "models/$RUN-seed$S.pt" "models/$RUN-seed$S.contract.json" "models/$RUN-seed$S.val.json"; do [[ -f "$f" ]] || { echo "FAIL LOUD: no $f"; fail=$((fail+1)); }; done
  i=$((i+1))
done
echo "=== $RUN: $fail failure(s)"; exit $fail
