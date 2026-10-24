#!/bin/bash
# live_link_features_v1 confirmation: builds the two input dirs of llf_confirm_launch.sh (xs/in_llf_conf for ids 30285-30296, xs/in_llf_conf_r1 for 30297) and the md5 manifest of every served checkpoint.
# Served: the 12 static e40 checkpoints (best-validation files, A6: <stem>-e40-seed<N>.pt with its .contract.json; gnn, twin, mlp, wide x seeds 1-3) copied from SRC and md5-verified, and the filed v6 fslabel
# references (gnn, twin, mlp, wide; seeds 1-3) linked from xs/in_fsl3. Refuses if a checkpoint is missing or a run has not finished (its -final.pt is the end-of-run marker). Never overwrites.
#   SRC=~/gnn-herosim/models bash llf_confirm_in.sh
set -euo pipefail
U=/share/nikola.lukic/uncut_groups; SRC=${SRC:-$HOME/gnn-herosim/models}; REF=$U/xs/in_fsl3; SPL=$U/xs/in_linkstatic_live
MAN=$U/xs/llf_conf_models.md5
declare -A STEM=([gnn]=uncut-linkstatic-gnn-eng-e40 [twin]=uncut-linkstatic-twin-e40 [mlp]=uncut-linkstatic-mlp-same-e40 [wide]=uncut-linkstatic-mlp-wide-e40)
for arm in gnn twin mlp wide; do for s in 1 2 3; do for e in pt contract.json; do
  [[ -f $SRC/${STEM[$arm]}-seed$s.$e && -f $SRC/${STEM[$arm]}-seed$s-final.pt ]] || { echo "FAIL LOUD: ${STEM[$arm]}-seed$s ($e or its -final.pt) missing in $SRC"; exit 3; }
done; done; done
[[ -e $MAN ]] && { echo "FAIL LOUD: $MAN exists"; exit 2; }
: > "$MAN.tmp"
for D in $U/xs/in_llf_conf $U/xs/in_llf_conf_r1; do
  [[ -e $D ]] && { echo "FAIL LOUD: $D exists"; exit 2; }
done
mkdir -p $U/xs/in_llf_conf/models $U/xs/in_llf_conf_r1/models
for arm in gnn twin mlp wide; do for s in 1 2 3; do for e in pt contract.json; do
  f=${STEM[$arm]}-seed$s.$e
  cp $SRC/$f $U/xs/in_llf_conf/models/$f
  a=$(md5sum < $SRC/$f | cut -d' ' -f1); b=$(md5sum < $U/xs/in_llf_conf/models/$f | cut -d' ' -f1); [[ $a == $b ]] || { echo "FAIL LOUD: md5 $f"; exit 4; }
  echo "$a  $f" >> "$MAN.tmp"; ln -s ../../in_llf_conf/models/$f $U/xs/in_llf_conf_r1/models/$f
done; done; done
for f in $REF/models/uncut-comb-fl-*-fslabel-seed[123].*; do b=$(basename $f); ln -s $(readlink -f $f) $U/xs/in_llf_conf/models/$b; ln -s $(readlink -f $f) $U/xs/in_llf_conf_r1/models/$b; echo "$(md5sum < $f | cut -d' ' -f1)  $b" >> "$MAN.tmp"; done
for D in in_llf_conf:llf_conf in_llf_conf_r1:llf_conf_r1; do d=${D%%:*}; r=${D##*:}
  for f in uncut_arm1_split.json uncut_link_split.json; do ln -s $(readlink -f $SPL/$f) $U/xs/$d/$f; done
  for w in wf1_heavy wf1_moderate; do ln -s $(readlink -f $U/$r/gate_inputs_cap200/inputs_group/$w) $U/xs/$d/$w; done
done
mv "$MAN.tmp" "$MAN"; wc -l "$MAN"; ls $U/xs/in_llf_conf $U/xs/in_llf_conf_r1
