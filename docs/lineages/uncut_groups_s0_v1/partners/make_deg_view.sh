#!/bin/bash
# Read-only symlink view of S5's partners tree in the frozen P11 reader's --deg-root layout. Creates links and one empty dir per level.
set -euo pipefail
P=/share/nikola.lukic/uncut_groups/partners; V=$P/deg_view
[[ ! -e $V ]] || { echo "FAIL LOUD: $V exists"; exit 1; }
for L in 1 2 4 8; do
  d=$V/k$L; mkdir -p $d/cells_bar
  ln -s ../../cells_react_k$L $d/cells_reactive
  ln -s ../../cells_k$L $d/cells_learned
  ln -s ../../gate_inputs_partners_k$L/inputs_group $d/inputs_group
done
mkdir -p $P/reads
cd $V && ls -lR --time-style=+ . | md5sum
