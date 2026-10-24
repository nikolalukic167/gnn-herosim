#!/bin/bash
# live_link_features_v1 confirmation: builds the two input dirs of llf_confirm_launch.sh (xs/in_llf_conf for ids 30285-30296, xs/in_llf_conf_r1 for 30297) and the md5 manifests of every served checkpoint.
# Two stages, so the references can run before the static e40 runs have finished:
#   MODE=refs    creates both dirs (refuses if they exist): wf1_* inputs, both splits, and the filed v6 fslabel references (gnn, twin, mlp, wide; seeds 1-3) linked from xs/in_fsl3;
#                manifest xs/llf_conf_models_refs.md5.
#   MODE=static  adds the 12 static e40 checkpoints (best-validation files, A6: <stem>-e40-seed<N>.pt with its .contract.json; gnn, twin, mlp, wide x seeds 1-3), copied from SRC and
#                md5-verified, to the existing dirs; refuses unless each run's -final.pt exists (the end-of-run marker); manifest xs/llf_conf_models_static.md5.
#   MODE=all     (default) both, in one go: the dirs must not exist, and the static checkpoints must be final.
#   MODE=final   a SECONDARY, separate pair of dirs xs/in_llf_conf_final and xs/in_llf_conf_r1_final holding the same 12 static runs' epoch-40 -final.pt (and -final.contract.json) copied under the gate name
#                <stem>-seed<N>.pt, md5-verified; manifest xs/llf_conf_models_final.md5. No references (the primary's refs/bar cells are the same). The primary dirs are never touched.
# After either stage a hard refusal removes nothing it did not create and exits 5 if a non-static e40 checkpoint (as-built v7 uncut-link-*-e40, (b) uncut-linkconv-*-e40, GPU twin) is in an IN dir.
#   SRC=~/gnn-herosim/models MODE=refs|static|all bash llf_confirm_in.sh        (U=<uncut_groups root> only for tests)
set -euo pipefail
U=${U:-/share/nikola.lukic/uncut_groups}; SRC=${SRC:-$HOME/gnn-herosim/models}; REF=$U/xs/in_fsl3; SPL=$U/xs/in_linkstatic_live; MODE=${MODE:-all}
[[ $MODE == refs || $MODE == static || $MODE == all || $MODE == final ]] || { echo "FAIL LOUD: MODE=$MODE"; exit 2; }
MANR=$U/xs/llf_conf_models_refs.md5; MANS=$U/xs/llf_conf_models_static.md5
declare -A STEM=([gnn]=uncut-linkstatic-gnn-eng-e40 [twin]=uncut-linkstatic-twin-e40 [mlp]=uncut-linkstatic-mlp-same-e40 [wide]=uncut-linkstatic-mlp-wide-e40)
DIRS=($U/xs/in_llf_conf $U/xs/in_llf_conf_r1)
[[ $MODE == final ]] && DIRS=($U/xs/in_llf_conf_final $U/xs/in_llf_conf_r1_final)
MANF=$U/xs/llf_conf_models_final.md5
guard() {
  if ls "${DIRS[0]}/models" "${DIRS[1]}/models" | grep -E "^uncut-link(conv)?-.*e40"; then echo "FAIL LOUD: a non-static e40 checkpoint is in a confirmation IN dir"; exit 5; fi
}
if [[ $MODE == refs || $MODE == all ]]; then
  for D in "${DIRS[@]}"; do [[ -e $D ]] && { echo "FAIL LOUD: $D exists"; exit 2; }; done
  [[ -e $MANR ]] && { echo "FAIL LOUD: $MANR exists"; exit 2; }
fi
if [[ $MODE == final ]]; then
  for D in "${DIRS[@]}"; do [[ -e $D ]] && { echo "FAIL LOUD: $D exists"; exit 2; }; done
  [[ -e $MANF ]] && { echo "FAIL LOUD: $MANF exists"; exit 2; }
  for arm in gnn twin mlp wide; do for s in 1 2 3; do for e in pt contract.json; do
    [[ -f $SRC/${STEM[$arm]}-seed$s-final.$e ]] || { echo "FAIL LOUD: ${STEM[$arm]}-seed$s-final.$e missing in $SRC"; exit 3; }
  done; done; done
  : > "$MANF.tmp"; mkdir -p "${DIRS[0]}/models" "${DIRS[1]}/models"
  for arm in gnn twin mlp wide; do for s in 1 2 3; do for e in pt contract.json; do
    f=${STEM[$arm]}-seed$s.$e; src=$SRC/${STEM[$arm]}-seed$s-final.$e
    cp $src ${DIRS[0]}/models/$f
    a=$(md5sum < $src | cut -d' ' -f1); b=$(md5sum < ${DIRS[0]}/models/$f | cut -d' ' -f1); [[ $a == $b ]] || { echo "FAIL LOUD: md5 $f"; exit 4; }
    echo "$a  ${STEM[$arm]}-seed$s-final.$e -> $f" >> "$MANF.tmp"; ln -s ../../in_llf_conf_final/models/$f ${DIRS[1]}/models/$f
  done; done; done
  for D in in_llf_conf_final:llf_conf in_llf_conf_r1_final:llf_conf_r1; do d=${D%%:*}; r=${D##*:}
    for f in uncut_arm1_split.json uncut_link_split.json; do ln -s $(readlink -f $SPL/$f) $U/xs/$d/$f; done
    for w in wf1_heavy wf1_moderate; do ln -s $(readlink -f $U/$r/gate_inputs_cap200/inputs_group/$w) $U/xs/$d/$w; done
  done
  guard; mv "$MANF.tmp" "$MANF"; wc -l "$MANF"; ls "${DIRS[0]}" "${DIRS[1]}"; exit 0
fi
if [[ $MODE == static || $MODE == all ]]; then
  for arm in gnn twin mlp wide; do for s in 1 2 3; do for e in pt contract.json; do
    [[ -f $SRC/${STEM[$arm]}-seed$s.$e && -f $SRC/${STEM[$arm]}-seed$s-final.pt ]] || { echo "FAIL LOUD: ${STEM[$arm]}-seed$s ($e or its -final.pt) missing in $SRC"; exit 3; }
  done; done; done
  [[ -e $MANS ]] && { echo "FAIL LOUD: $MANS exists"; exit 2; }
  if [[ $MODE == static ]]; then for D in "${DIRS[@]}"; do [[ -d $D/models ]] || { echo "FAIL LOUD: $D missing: run MODE=refs first"; exit 2; }; done; fi
fi
if [[ $MODE == refs || $MODE == all ]]; then
  : > "$MANR.tmp"
  mkdir -p "${DIRS[0]}/models" "${DIRS[1]}/models"
  for f in $REF/models/uncut-comb-fl-*-fslabel-seed[123].*; do b=$(basename $f); for D in "${DIRS[@]}"; do ln -s $(readlink -f $f) $D/models/$b; done; echo "$(md5sum < $f | cut -d' ' -f1)  $b" >> "$MANR.tmp"; done
  for D in in_llf_conf:llf_conf in_llf_conf_r1:llf_conf_r1; do d=${D%%:*}; r=${D##*:}
    for f in uncut_arm1_split.json uncut_link_split.json; do ln -s $(readlink -f $SPL/$f) $U/xs/$d/$f; done
    for w in wf1_heavy wf1_moderate; do ln -s $(readlink -f $U/$r/gate_inputs_cap200/inputs_group/$w) $U/xs/$d/$w; done
  done
  guard; mv "$MANR.tmp" "$MANR"; wc -l "$MANR"
fi
if [[ $MODE == static || $MODE == all ]]; then
  : > "$MANS.tmp"
  for arm in gnn twin mlp wide; do for s in 1 2 3; do for e in pt contract.json; do
    f=${STEM[$arm]}-seed$s.$e
    cp $SRC/$f ${DIRS[0]}/models/$f
    a=$(md5sum < $SRC/$f | cut -d' ' -f1); b=$(md5sum < ${DIRS[0]}/models/$f | cut -d' ' -f1); [[ $a == $b ]] || { echo "FAIL LOUD: md5 $f"; exit 4; }
    echo "$a  $f" >> "$MANS.tmp"; ln -s ../../in_llf_conf/models/$f ${DIRS[1]}/models/$f
  done; done; done
  guard; mv "$MANS.tmp" "$MANS"; wc -l "$MANS"
fi
ls "${DIRS[0]}" "${DIRS[1]}"
