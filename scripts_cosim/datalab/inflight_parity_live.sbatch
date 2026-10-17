#!/bin/bash
# partial_state_v6 live parity: one gate cell (cd_ib0, k = 0, 19537 g0, both rungs) with the parity hook (scripts_cosim/inflight_parity_hook/usercustomize.py) active. At sampled batch decisions the hook
# compares the cache-side in-flight numbers (fidelity capture -> state_from_ghosts) with the live ones (live_state) and the live builder's graph columns, and writes the tally to $OUT/parity_<rung>.json.
#   sbatch --export=ALL,WT=<pin>,EXPECT_HEAD=<sha>,OUT=<dir>[,EVERY=200] inflight_parity_live.sbatch
#SBATCH --job-name=if-parity
#SBATCH --output=/share/nikola.lukic/uncut_groups/inflight/logs/%x-%j.out
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
set -euo pipefail
WT=${WT:?}; OUT=${OUT:?}; EXPECT_HEAD=${EXPECT_HEAD:?}; EVERY=${EVERY:-200}
U=/share/nikola.lukic/uncut_groups
eval "$(micromamba shell hook --shell bash)"; micromamba activate gnn
PYV=$(python3 -c "import sys; print('%d.%d' % sys.version_info[:2])")
UB=$(mktemp -d); SP="$UB/lib/python$PYV/site-packages"; mkdir -p "$SP" "$OUT"
cp "$WT/scripts_cosim/inflight_parity_hook/usercustomize.py" "$SP/usercustomize.py"
python3 - <<PY
import site, sys
print("user site enabled:", site.ENABLE_USER_SITE)
PY
mkdir -p "$OUT/cells"
# one cell per rung: the hook writes its tally next to the cell (INFLIGHT_PARITY_OUT is per process, so run the rungs one after the other)
for RUNG in moderate heavy; do
  PYTHONUSERBASE="$UB" INFLIGHT_PARITY_OUT="$OUT/parity_$RUNG.json" INFLIGHT_PARITY_EVERY="$EVERY" INFLIGHT_PARITY_REPO="$WT" \
  PARTIAL_STATE_CONTRACT=partial_state_v6 INFERENCE_FEATURE_LAYOUT=dim22 \
  WT="$WT" EXPECT_HEAD="$EXPECT_HEAD" IN=$U/final/in_arm3 OUT="$OUT/cells_$RUNG" SELECT=$U/arm1_selected.json R1A_TOPOS=19537 R1A_WINDOWS=g0 R1A_ARMS=cd_ib0 R1A_SEEDS=0 \
  WF1_RUNGS=$RUNG PAR=1 CELL_MEM=64G LIMIT=14400 bash "$WT/scripts_cosim/datalab/r1a_gate.sbatch" || echo "cell rc $?"
done
echo "=== parity tallies in $OUT"; ls "$OUT"
