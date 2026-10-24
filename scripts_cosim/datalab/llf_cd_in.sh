#!/bin/bash
# live_link_features_v1: builds an IN dir for llf_cd_launch.sh on a freshly minted id set (S4). Links the minted cap200 gate inputs (wf1_heavy, wf1_moderate), both splits, and the served checkpoints:
# the static e40 -final.pt (gate names, from the primary secondary-wave dir xs/in_llf_conf_final: gnn, twin, mlp, wide x seeds 1-3, already md5-listed in xs/llf_conf_models_final.md5).
#   bash llf_cd_in.sh <IN dir to create> <gate_inputs_cap200/inputs_group dir of the mint>
set -euo pipefail
U=/share/nikola.lukic/uncut_groups; D=${1:?IN dir}; G=${2:?inputs_group dir}; FIN=$U/xs/in_llf_conf_final; SPL=$U/xs/in_linkstatic_live
[[ -e $D ]] && { echo "FAIL LOUD: $D exists"; exit 2; }
[[ -d $G/wf1_heavy && -d $G/wf1_moderate && -d $FIN/models ]] || { echo "FAIL LOUD: $G or $FIN incomplete"; exit 2; }
mkdir -p $D/models
for w in wf1_heavy wf1_moderate; do ln -s $(readlink -f $G/$w) $D/$w; done
for f in uncut_arm1_split.json uncut_link_split.json; do ln -s $(readlink -f $SPL/$f) $D/$f; done
for f in $FIN/models/*; do ln -s $(readlink -f $f) $D/models/$(basename $f); done
if ls $D/models | grep -E "^uncut-link(conv)?-.*e40"; then echo "FAIL LOUD: a non-static e40 checkpoint in $D"; rm -rf $D; exit 5; fi
ls $D $D/models | head -40; ls $D/models | wc -l
