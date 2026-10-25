#!/bin/bash
#SBATCH -J v9-label-probe
#SBATCH -p CPU-amd
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 06:00:00
# outbound_link_features_v1 label probe: ONE counted run of this frozen script and this sbatch, after the node merges. Replay pin 95293af0 (worktree below).
# State set: the union-corpus bar snapshots (flowcap/collect_union), every topology with a cell config, windows g0-g3, moderate rung, batches of 3-24 tasks,
# up to 120 evenly spaced whole-batch decisions per cell, up to 3 tasks per decision. The script must match PROBE_MD5 (the md5 filed in the node).
set -euo pipefail
eval "$(micromamba shell hook --shell bash)" && micromamba activate gnn
export HEROSIM_PY=python3
U=/share/nikola.lukic/uncut_groups
WT=$HOME/gnn-herosim-wt/s7_probe_95293af0
PROBE=${PROBE_SCRIPT:?set PROBE_SCRIPT to the filed label_probe.py}
[ "$(md5sum "$PROBE" | cut -d' ' -f1)" = "${PROBE_MD5:?set PROBE_MD5 to the md5 filed in the node}" ] || { echo "FAIL LOUD: label_probe.py md5 differs from the filed one"; exit 2; }
[ "$(git -C $WT rev-parse --short=8 HEAD)" = 95293af0 ] || { echo "FAIL LOUD: probe worktree is not at 95293af0"; exit 2; }
export PROBE_REPO=$WT PYTHONPATH=$WT OMP_NUM_THREADS=1 U WT PROBE
OUT=$U/flowcap/v9_label_probe_counted; [ -e $OUT ] && { echo "FAIL LOUD: $OUT exists"; exit 2; }; mkdir -p $OUT; export OUT
one() {
  t=$1; w=$2
  stem=cc40s${t}__${w}moderate__cd_ib0_inflightfull_k100_s0
  python3 $PROBE --cell-dir $U/flowcap/collect_union --stem $stem --cell-config $U/flowcap/in_flowcap/wf1_moderate/cfg/cc40s${t}.json \
    --sim-input $WT/data/nofs-ids --out $OUT/rows_${t}_$w.jsonl --n-states 120 --max-batch 24 --per-state 3 > $OUT/log_${t}_$w.txt 2>&1 || echo "FAIL $t $w" >> $OUT/fails.txt
}
export -f one
for f in $U/flowcap/collect_union/cc40s*__g?moderate__*.summary.json; do
  b=$(basename $f); t=${b:5:5}; w=${b:12:2}
  [ -f $U/flowcap/in_flowcap/wf1_moderate/cfg/cc40s${t}.json ] && echo "$t $w" || true
done | xargs -P 8 -L 1 bash -c 'one $0 $1'
echo done
