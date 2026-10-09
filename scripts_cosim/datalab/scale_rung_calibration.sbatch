#!/bin/bash
# scale_160_v1 rung calibration (load_recalibration_v1 protocol on the scaled cell): CD-steered log bisection on the effective queue
# share, calibration topologies x windows in parallel per step, guards per cell, Knative as context at the chosen rung.
#   (accel_replica_v1: add REPLICA_RULE=fastest_compatible)
#   sbatch --export=ALL,WT=...,OUT=...,SEEDS="9905 9906 9907 9908",TAG=heavy,BAND="0.40 0.50",LO=2.9,HI=46.46 scale_rung_calibration.sbatch
#SBATCH --job-name=sp-calib
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/sp-calib-%j.out
#SBATCH --partition=CPU-amd
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=08:00:00
set -euo pipefail
WT=${WT:?}; OUT=${OUT:?}; SEEDS=${SEEDS:?}; TAG=${TAG:?}; BAND=${BAND:?}; LO=${LO:?}; HI=${HI:?}
NC=${NC:-160}; NS=${NS:-24}; P=${P:-0.6}; WINDOWS=${WINDOWS:-"g0 g1"}; STEPS=${STEPS:-8}
cd "$WT"; eval "$(micromamba shell hook --shell bash)"; micromamba activate gnn
for v in $(compgen -e | grep -E '^(HEROSIM_|GNN_|COSIM_|LIVE_AUDIT_|GATE_|SIM_|QUEUE_FEATURE|MAX_PLACEMENT|DECLARED_SLATE)'); do unset "$v"; done
export HEROSIM_PY=python3 PYTHONPATH=$WT PYTHONUNBUFFERED=1 HEROSIM_RAW_DIR=/tmp/sp_calib_$SLURM_JOB_ID; mkdir -p "$HEROSIM_RAW_DIR" "$OUT"; trap 'rm -rf "$HEROSIM_RAW_DIR"' EXIT
echo "HEAD $(git rev-parse HEAD) tag $TAG band $BAND bracket [$LO, $HI] seeds $SEEDS windows $WINDOWS node $(hostname) $(date -Is)"
python3 scripts_cosim/scale_rung_bisect.py "$OUT" --wt "$WT" --seeds $SEEDS --windows $WINDOWS --clients $NC --servers $NS --p $P \
  --band $BAND --lo "$LO" --hi "$HI" --steps "$STEPS" --tag "$TAG" --parallel 8 ${REPLICA_RULE:+--replica-rule $REPLICA_RULE} ${KNATIVE:+--knative} ${EXTRA:+--extra-policies $EXTRA}
echo "done $(date -Is)"
