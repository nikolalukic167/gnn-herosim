#!/bin/bash
# slate_fix_v1 (docs/lineages/slate_fix_v1.md): wire the checkpoints into the gate and smoke inputs. Symlinks, md5 and sidechecks only (login-node safe).
#   GATE_IN    the s160_gate_inputs.sbatch output's inputs/ (wf1_moderate, wf1_heavy, accel_replica_v1_split.json) for topologies 19420-19431
#   SMOKE_IN   created here: wf1_* and the split of selfsearch_confirm_v1's inputs (topology 16490 lives there) plus the same models/
#   NEW_MODELS the directory the training wrote slate-fix-v1-{gnn,twin}-eng-seed{1,2}.pt (+ .contract.json) to
#   bash slate_fix_v1_gate_models.sh <worktree> <GATE_IN> <SMOKE_IN> <NEW_MODELS>
set -euo pipefail
WT=${1:?worktree}; GATE_IN=${2:?GATE_IN}; SMOKE_IN=${3:?SMOKE_IN}; NEW_MODELS=${4:?NEW_MODELS}
OLD_GNN=/share/nikola.lukic/accel_prod/gate_inputs/models
SS_IN=/share/nikola.lukic/selfsearch_confirm/gate/inputs
SPLIT_SHA=f5bc4f51106d71a1830febec8c06dca66121a81336acd5662cf24b7997cf7e53
declare -A WANT=([accel-replica-v1-gnn-eng-seed1.pt]=0f915a365f13fd4399175c498ce8acf9 [accel-replica-v1-gnn-eng-seed2.pt]=446f99fdda30e4531e0d5a7fabc01c95)
for d in "$GATE_IN/wf1_moderate" "$GATE_IN/wf1_heavy" "$SS_IN/wf1_moderate" "$SS_IN/wf1_heavy"; do [[ -d $d ]] || { echo "FAIL LOUD: $d missing"; exit 1; }; done
for f in "$GATE_IN/accel_replica_v1_split.json" "$SS_IN/accel_replica_v1_split.json"; do
  [[ "$(sha256sum "$f" | cut -d' ' -f1)" == "$SPLIT_SHA" ]] || { echo "FAIL LOUD: $f is not the registered split"; exit 1; }
done
[[ ! -e "$GATE_IN/models" ]] || { echo "FAIL LOUD: $GATE_IN/models exists; refusing to rewire it"; exit 1; }
mkdir -p "$GATE_IN/models"
for s in 1 2; do
  for stem in slate-fix-v1-gnn-eng slate-fix-v1-twin-eng; do
    for ext in pt contract.json; do ln -s "$NEW_MODELS/$stem-seed$s.$ext" "$GATE_IN/models/$stem-seed$s.$ext"; done
  done
  for ext in pt contract.json; do ln -s "$OLD_GNN/accel-replica-v1-gnn-eng-seed$s.$ext" "$GATE_IN/models/accel-replica-v1-gnn-eng-seed$s.$ext"; done
done
for f in "${!WANT[@]}"; do
  [[ "$(md5sum "$GATE_IN/models/$f" | cut -d' ' -f1)" == "${WANT[$f]}" ]] || { echo "FAIL LOUD: $f is not the checkpoint the earlier gates served"; exit 1; }
done
cd "$WT"
for s in 1 2; do
  for arm in gnn_eng twin_eng; do
    R1A_SLATE=declared_pruning_v1_uncapped python3 scripts_cosim/joint_burst_v2_sidecheck.py "$GATE_IN/models/slate-fix-v1-${arm//_/-}-seed$s.contract.json" "ra_$arm" "$GATE_IN/accel_replica_v1_split.json" inf
  done
  R1A_SLATE=declared_pruning_v1 python3 scripts_cosim/joint_burst_v2_sidecheck.py "$GATE_IN/models/accel-replica-v1-gnn-eng-seed$s.contract.json" ra_gnn_eng "$GATE_IN/accel_replica_v1_split.json" inf
done
mkdir -p "$SMOKE_IN"
for x in wf1_moderate wf1_heavy accel_replica_v1_split.json manifest_moderate.json manifest_heavy.json; do [[ -e "$SMOKE_IN/$x" ]] || ln -s "$SS_IN/$x" "$SMOKE_IN/$x"; done
[[ -e "$SMOKE_IN/models" ]] || ln -s "$GATE_IN/models" "$SMOKE_IN/models"
md5sum "$GATE_IN"/models/*.pt
echo "wired: $GATE_IN/models ($(ls "$GATE_IN/models" | wc -l) links), $SMOKE_IN"
