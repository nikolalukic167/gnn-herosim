#!/bin/bash
# slate_fix_v1 offline screen (ORDERING ONLY, docs/lineages/archive/slate_fix_v1.md): each checkpoint's val regret on the UNCAPPED cache's held-out tables
# (the 438 wf1_heldout parents of experiments/accel_replica_v1_split.json), through train_near_rtt's evaluate() (NEAR_RTT_EVAL_ONLY_JSON; no
# gradient step, no checkpoint written). Task T in 0-7: (new | old) x (gnn_eng | twin_eng) x seed (1 | 2). New = slate-fix-v1-<arm>-seed<s>.pt;
# old = the capped g4 checkpoints the gates served: accel-replica-v1-gnn-eng-seed<s>.pt (OLD_GNN_MODELS, md5 0f915a36 / 446f99fd) and the twin
# accel-replica-v1-twin-eng-seed<s>.pt (OLD_TWIN_MODELS, md5 bc44f9c2 / 7798e128).
# Task 8 is the IDENTITY check, run first and alone: the old gnn_eng seed-1 checkpoint through this eval path on its OWN capped cache (the g4
# config, simulation_data/graphs_cache_accel_replica_v1_psv5_inf in the worktree) must reproduce the best_val regret_masked_topo its training
# recorded (accel-replica-v1-gnn-eng-g4-seed1.val.json: 0.7807019727098458) within 1e-6; otherwise the screen is not read.
#   sbatch --array=8 ...   then   sbatch --array=0-7 --export=ALL,WT=<pinned worktree>,EXPECT_HEAD=<sha>,OUT=<dir> slate_fix_v1_eval.sbatch
#SBATCH --job-name=sf1-eval
#SBATCH --output=/share/nikola.lukic/logs/sf1-eval-%A_%a.out
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
set -euo pipefail
ulimit -n 65536 2>/dev/null || true
WT=${WT:?FAIL LOUD: set WT}; EXPECT_HEAD=${EXPECT_HEAD:?}; OUT=${OUT:?}
OLD_GNN_MODELS=${OLD_GNN_MODELS:-/share/nikola.lukic/accel_prod/gate_inputs/models}
OLD_TWIN_MODELS=${OLD_TWIN_MODELS:-/share/nikola.lukic/selfsearch/twin_inputs_8c8ebc7f/models}
T=${SLURM_ARRAY_TASK_ID:?FAIL LOUD: submit as --array=0-8}
read -r WHICH ARM S < <(python3 -c "import sys; t=int(sys.argv[1]); print(*(('identity','gnn_eng',1) if t == 8 else (('new','old')[t//4], ('gnn_eng','twin_eng')[(t//2)%2], 1+t%2)))" "$T")
cd "$WT"
for v in $(compgen -e | grep -E '^(HEROSIM_|GNN_|COSIM_|LIVE_AUDIT_|GATE_|SIM_|QUEUE_FEATURE|MAX_PLACEMENT|NEAR_RTT_|PARTIAL_STATE|R1A_)'); do unset "$v"; done
if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3 PYTHONPATH="$WT:$WT/src/notebooks" PYTHONUNBUFFERED=1 WANDB_ENTITY=nikolalukic167-tu-wien OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=""
export HEROSIM_TRANSFER_MODEL=pipelined HEROSIM_REPLICA_RELEASE=1 HEROSIM_SCALEOUT=kpa HEROSIM_SHARED_AUTOSCALER=0 GATE_FIXED_POLICY_TIME_SCALE=1.0
HEAD=$(git rev-parse HEAD)
echo "task $T: $WHICH $ARM seed $S | HEAD $HEAD node $(hostname)"
[[ "$HEAD" == "$EXPECT_HEAD"* ]] || { echo "FAIL LOUD: worktree HEAD $HEAD is not $EXPECT_HEAD"; exit 1; }
CFG=experiments/slate_fix_v1_${ARM}.yaml; SPLITF=experiments/accel_replica_v1_split.json
[[ $WHICH == identity ]] && CFG=experiments/accel_replica_v1_gnn_eng_g4.yaml
if [[ $WHICH == new ]]; then CK=models/slate-fix-v1-${ARM//_/-}-seed$S.pt; SLATE=declared_pruning_v1_uncapped
elif [[ $WHICH == identity ]]; then CK=$OLD_GNN_MODELS/accel-replica-v1-gnn-eng-seed1.pt; SLATE=declared_pruning_v1
else OLDD=$OLD_GNN_MODELS; [[ $ARM == twin_eng ]] && OLDD=$OLD_TWIN_MODELS; CK=$OLDD/accel-replica-v1-${ARM//_/-}-seed$S.pt; SLATE=declared_pruning_v1; fi
[[ -f "$CK" && -f "${CK%.pt}.contract.json" ]] || { echo "FAIL LOUD: $CK or its sidecar missing"; exit 1; }
# the recipe check (MP flags, contract, split sha) against this arm, with the slate the checkpoint was trained on
R1A_SLATE=$SLATE python3 scripts_cosim/joint_burst_v2_sidecheck.py "${CK%.pt}.contract.json" "ra_$ARM" "$SPLITF" inf
mkdir -p "$OUT"; REPORT=$OUT/val_${WHICH}_${ARM}_s$S.json
[[ ! -e "$REPORT" ]] || { echo "FAIL LOUD: $REPORT exists; refusing to overwrite"; exit 1; }
NEAR_RTT_EVAL_ONLY_CHECKPOINT=$CK NEAR_RTT_EVAL_ONLY_JSON=$REPORT python3 -u run_experiment.py "$CFG" --seed "$S"
python3 -c "import json,sys; d=json.load(open(sys.argv[1])); m=d['metrics']; print('[val]', sys.argv[2], 'regret_masked_topo %.4f s median %.4f plan_exact %.3f n %d' % (m['regret_masked_topo'], m['regret_masked_topo_median'], m['mt_plan_exact'], d['val_datasets']))" "$REPORT" "$WHICH/$ARM/s$S"
if [[ $WHICH == identity ]]; then
  python3 -c "import json,sys; m=json.load(open(sys.argv[1]))['metrics']['regret_masked_topo']; want=0.7807019727098458; ok=abs(m-want)<=1e-6; print('[identity]', 'PASS' if ok else 'FAIL LOUD', m, 'recorded', want); sys.exit(0 if ok else 1)" "$REPORT"
fi
