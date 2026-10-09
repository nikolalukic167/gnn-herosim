#!/bin/bash
# r1_attribution_v1 live gate, learned and seeded-CD arms, per arm (docs/lineages/r1_attribution_v1.md), at the pinned worktree.
# Submits one array over the registered 1,368-cell layout restricted to the given arms. The kind list goes through the environment
# (sbatch --export splits a value at its commas).
#   R1A_KINDS=ra_gnn_eng           MODELS=<dir with r1-attribution-v1-<arm>-seed<N>.pt + .contract.json> SPLIT=<r1_attribution_v1_split.json> \
#     r1a_gate_arm.sh               (the 7 ra_* arms; the seeded-CD arms ra_gnn_eng_cdapply / ra_twin_eng_cdapply need their source arm's models)
#   R1A_KINDS=cd_random_seed        r1a_gate_arm.sh      (needs no checkpoint)
#   scale_160_v1: R1A_STEM=<checkpoint prefix> R1A_SPLIT=<split file name>, IN_DIR/OUT_DIR for its own directories
#   optional: SBATCH_FILE (job script path), TASKS (array size, default 4), PAR (cells per task, default 56), SEEDS (default 1,2), WT/EXPECT_HEAD (default 07b0acba), DRY=1 (print only)
set -euo pipefail
KINDS=${R1A_KINDS:?FAIL LOUD: set R1A_KINDS}
D=/home/nikola.lukic/gnn-herosim/simulation_data/r1a_gate
S=${EXPECT_HEAD:-07b0acba}
W=${WT:-$HOME/gnn-herosim-wt/r1a_gate_$S}
SB=${SBATCH_FILE:-$W/scripts_cosim/datalab/r1a_gate.sbatch}  # the job script may come from a newer, script-only commit; WT (the code that runs) stays pinned
IN=${IN_DIR:-$D/inputs_learned_$S}
OUT=${OUT_DIR:-$D/learned_$S}  # OUT_DIR: a separate result directory (descriptive arms outside the families)
[[ "$(git -C "$W" rev-parse HEAD)" == "$S"* ]] || { echo "FAIL LOUD: $W is not at $S"; exit 1; }
mkdir -p "$IN"
for r in light moderate heavy; do [[ -e "$IN/wf1_$r" ]] || ln -s "$D/inputs_bb9fecf6/inputs/wf1_$r" "$IN/wf1_$r"; done
if [[ ",$KINDS" == *,ra_* ]]; then
  : "${MODELS:?FAIL LOUD: ra_* arms need MODELS}" "${SPLIT:?FAIL LOUD: ra_* arms need SPLIT}"
  [[ -d "$MODELS" && -f "$SPLIT" ]] || { echo "FAIL LOUD: $MODELS or $SPLIT missing"; exit 1; }
  [[ -e "$IN/models" ]] || ln -s "$MODELS" "$IN/models"
  [[ "$(readlink -f "$IN/models")" == "$(readlink -f "$MODELS")" ]] || { echo "FAIL LOUD: $IN/models points elsewhere"; exit 1; }
  SN=${R1A_SPLIT:-r1_attribution_v1_split.json}  # scale_160_v1 names its own split file; the driver reads the same R1A_SPLIT
  if [[ -f "$IN/$SN" ]]; then cmp -s "$SPLIT" "$IN/$SN" || { echo "FAIL LOUD: a different split file is already in $IN"; exit 1; }
  else cp "$SPLIT" "$IN/$SN"; fi
  # every ra_* kind named must have its checkpoints and sidecar for every seed
  for k in ${KINDS//,/ }; do
    [[ "$k" == ra_* ]] || continue
    base=${k#ra_}; base=${base%_cdapply}; stem=${R1A_STEM:-r1-attribution-v1}-${base//_/-}
    for q in ${SEEDS:-1,2}; do q=${q//,/ }; for sd in $q; do
      [[ -f "$MODELS/$stem-seed$sd.pt" && -f "$MODELS/$stem-seed$sd.contract.json" ]] || { echo "FAIL LOUD: $MODELS/$stem-seed$sd.pt (+ .contract.json) missing for $k"; exit 1; }
    done; done
  done
fi
export R1A_ARMS=$KINDS R1A_SEEDS=${SEEDS:-1,2}
CMD=(sbatch --array=0-$(( ${TASKS:-4} - 1 )) --export=ALL,WT=$W,EXPECT_HEAD=$S,IN=$IN,OUT=$OUT,PAR=${PAR:-56} "$SB")
if [[ "${DRY:-0}" == 1 ]]; then echo "R1A_ARMS=$R1A_ARMS R1A_SEEDS=$R1A_SEEDS ${CMD[*]}"; else "${CMD[@]}"; fi
