#!/bin/bash
# knative_stable_v1 calibration cells: reactive s0 (kind reactive), pin 0ec7cdcb, option D flags as cells_react, the sealed 12 ids, windows g0+g1, rung tagged f<tag> (= the minted moderate-tag inputs of m<tag>, linked as wf1_f<tag>) at fraction <tag> of the moderate rate; one job per multiplier = 4 jobs, 24 cells each.
# usage: kstab_cells.sh <tag: 0.5|0.25|0.125|0.0625> "<12 sealed ids>" <sha256>        requires the inputs of kstab_mint_<tag>.sh
set -euo pipefail
TAG=$1; IDS=$2; WANT=$3; PIN=0ec7cdcb
U=/share/nikola.lukic/uncut_groups; X=$U/llf_cd_conf/kstab_calib/m$TAG; SEL=$U/llf_conf/selected.json; WL=$HOME/gnn-herosim-wt/s5_llf_${PIN}
GOT=$(echo -n $IDS | tr ' ' ',' | sha256sum | cut -c1-64); [ "$GOT" = "$WANT" ] || { echo "FAIL LOUD: id list sha256"; exit 2; }
[ -d $X/in_cap200/wf1_moderate ] || { echo "FAIL LOUD: inputs of m$TAG not built"; exit 2; }
(cd $X/gate_inputs_cap200/inputs_group && md5sum -c --quiet MD5SUMS) || { echo "FAIL LOUD: md5sum -c"; exit 2; }
[[ $(git -C $WL rev-parse HEAD) == $PIN* && $(git -C $WL status --porcelain | wc -l) == 0 ]] || { echo "FAIL LOUD: worktree not clean at $PIN"; exit 2; }
IN=$X/in_f$TAG
if [ ! -e $IN ]; then mkdir -p $IN; cp -p $X/in_cap200/uncut_arm1_split.json $IN/; ln -s $X/in_cap200/models $IN/models; ln -s $X/gate_inputs_cap200/inputs_group/wf1_moderate $IN/wf1_f$TAG; fi
[ ! -e $X/cells ] || { echo "FAIL LOUD: $X/cells exists"; exit 2; }
[ $(( $(squeue -u nikola.lukic -h | wc -l) + 1 )) -le 44 ] || { echo "FAIL LOUD: queue would exceed 44"; exit 2; }
T=$(echo $IDS | tr ' ' ',')
env IN=$IN SELECT=$SEL R1A_TOPOS=$T R1A_WINDOWS=g0,g1 WF1_RUNGS=f$TAG CELL_MEM=16G LIMIT=28800 R1A_SPLIT=uncut_arm1_split.json R1A_SLATE=declared_pruning_v1_uncapped R1A_EXCHANGE_TCP=1 R1A_EXCHANGE_FAIRSHARE=1 \
  PAR=12 WT=$WL EXPECT_HEAD=$PIN OUT=$X/cells R1A_ARMS=reactive R1A_SEEDS=1,2 \
  sbatch --parsable --export=ALL -p CPU-amd -c 24 --mem=200G -t 1-00:00:00 -J s5-kstab-$TAG -o $X/logs/%x-%j.out $WL/scripts_cosim/datalab/r1a_gate.sbatch
