#!/bin/bash
# r1_attribution_v1 PRODUCTION corpus, step 4 of 4 -- the end-of-build check. Submit afterany on the build array.
# The build array is chained aftercorr on the capture; scancel any build task left DependencyNeverSatisfied before this check can run.
#   sbatch --dependency=afterany:<build> --export=ALL,WT=...,ROOT=... wf1_corpus_prod_check.sbatch
# 1. wf1_corpus_volume_check.py: batches vs the 5,000 target, filter and discard rates, capture-not-ok cells.
# 2. wf1_fidelity_corpus_check.py on each split: sweep complete, best.json == min(rows), independent re-simulation of sampled plans.
#SBATCH --job-name=wf1p-chk
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/wf1p-chk-%j.out
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
set -euo pipefail
WT=${WT:?FAIL LOUD: set WT}
ROOT=${ROOT:?FAIL LOUD: set ROOT}
SAMPLE=${SAMPLE:-5}
cd "$WT"
for v in $(compgen -e | grep -E '^(HEROSIM_|GNN_|COSIM_|LIVE_AUDIT_|GATE_|SIM_|QUEUE_FEATURE|MAX_PLACEMENT)'); do unset "$v"; done
if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3 PYTHONPATH="$WT" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
export HEROSIM_TRANSFER_MODEL=pipelined HEROSIM_REPLICA_RELEASE=1 HEROSIM_SCALEOUT=kpa HEROSIM_SHARED_AUTOSCALER=0 GATE_FIXED_POLICY_TIME_SCALE=1.0
export HEROSIM_PEER_EXCHANGE=1 HEROSIM_SERVER_ONLY_REPLICAS=1 HEROSIM_WARMTH_PHYSICS=node_disk_v2 PYTHONHASHSEED=0 HEROSIM_SNAPSHOT_FIDELITY=1
export SIM_FORCE_FULL_STATS=1
rc=0
python3 scripts_cosim/wf1_corpus_volume_check.py --root "$ROOT" --target "${BATCHES_TARGET:-5000}" || rc=1
for split in train heldout; do
  d="$ROOT/gnn_datasets_wf1_$split"
  [[ -d "$d" ]] || continue
  python3 scripts_cosim/wf1_fidelity_corpus_check.py --datasets "$d" --sample "$SAMPLE" || rc=1
done
exit $rc
