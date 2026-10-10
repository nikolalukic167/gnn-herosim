#!/bin/bash
# scale_160_v1 (and accel_replica_v1: LINEAGE=accel_replica_v1, stem accel-replica-v1): submit cache -> training array (gnn_eng + gnn_eng_physmp, 6 configs, 3 seeds per task = 12 tasks) -> per-arm validation-only selection,
# on the datalab login node, as ONE chain (14 jobs). The r1a recipe unchanged: same 6-config grid, 100-epoch cap, NEAR_RTT_SKIP_FINAL_TEST=1, val.json
# sidecars. twin_eng is prepared (experiments/scale_160_v1_twin_eng_g*.yaml via make_r1a_arm_configs.py --lineage scale_160_v1 --arms twin_eng) but is
# NOT part of this chain: it trains only if a GNN wins.
#
#   CORPUS=<dir with gnn_datasets_wf1_{train,heldout}_<SUF>>  SUF=<suffix>  CORPUS_SPLIT=<split.json of the corpus: {"train":[...],"heldout":[...]}>
#   OUT=<new run dir>  WT=<pinned worktree>  EXPECT_HEAD=<sha>  [INPUTS=<gate inputs dir for the staged checkpoints>]  [TRAIN_MEM=60G] [LIMIT=44]
#   bash scale160_submit.sh [dry]
# Fails loud when the chain would not fit under LIMIT, and refuses a dirty worktree or a moved HEAD. The held-out topologies of CORPUS_SPLIT become the
# VALIDATION split (selection happens on them); the loader's non-empty test is an unscored placeholder train topology (make_r1a_split.py --heldout-as-val).
set -euo pipefail
CORPUS=${CORPUS:?FAIL LOUD: set CORPUS}; SUF=${SUF:?FAIL LOUD: set SUF}; CORPUS_SPLIT=${CORPUS_SPLIT:?FAIL LOUD: set CORPUS_SPLIT}
OUT=${OUT:?FAIL LOUD: set OUT}; WT=${WT:?FAIL LOUD: set WT (pinned detached worktree)}; EXPECT_HEAD=${EXPECT_HEAD:?FAIL LOUD: set EXPECT_HEAD}
LIMIT=${LIMIT:-44}; TRAIN_MEM=${TRAIN_MEM:-60G}; LINEAGE=${LINEAGE:-scale_160_v1}; STEM=${STEM:-$(echo "${LINEAGE:-scale_160_v1}" | tr _ -)}; ARMS="gnn_eng gnn_eng_physmp"; N_TASKS=12
INPUTS=${INPUTS:-$OUT/gate_inputs}
cd "$WT"
HEAD=$(git rev-parse HEAD)
[[ "$HEAD" == "$EXPECT_HEAD"* ]] || { echo "FAIL LOUD: worktree HEAD $HEAD is not $EXPECT_HEAD"; exit 1; }
[[ -z "$(git status --porcelain -- src scripts_cosim experiments run_experiment.py)" ]] || { echo "FAIL LOUD: worktree dirty"; exit 1; }
[[ -L models ]] || { echo "FAIL LOUD: models/ must be the symlink into ~/gnn-herosim/models"; exit 1; }
[[ ! -e "$OUT" ]] || { echo "FAIL LOUD: $OUT exists"; exit 1; }
SFX=_$SUF; [[ "$SUF" == none ]] && SFX=""   # SUF=none: no suffix on the corpus directories
for s in train heldout; do [[ -d "$CORPUS/gnn_datasets_wf1_${s}$SFX" ]] || { echo "FAIL LOUD: no $CORPUS/gnn_datasets_wf1_${s}$SFX"; exit 1; }; done
[[ -f "$CORPUS_SPLIT" ]] || { echo "FAIL LOUD: no $CORPUS_SPLIT"; exit 1; }
for a in $ARMS; do for g in 0 1 2 3 4 5; do [[ -f experiments/${LINEAGE}_${a}_g$g.yaml ]] || { echo "FAIL LOUD: experiments/${LINEAGE}_${a}_g$g.yaml missing"; exit 1; }; done; done
queued=$(squeue -u "$USER" -r -h | wc -l)
need=$((1 + N_TASKS + 1))   # cache + 12 array elements + selection
(( queued + need <= LIMIT )) || { echo "FAIL LOUD: $queued jobs queued + $need > $LIMIT; not submitting a partial chain"; exit 1; }
echo "worktree $HEAD; queued $queued; the chain adds $need; outputs $OUT"
[[ "${1:-}" == dry ]] && { echo "dry: nothing submitted"; exit 0; }
mkdir -p "$OUT"
export LINEAGE ARMS CORPUS CORPUS_SPLIT SUF HELDOUT_AS_VAL=1
CACHE_JOB=$(sbatch --parsable --export=ALL,WT="$WT",OUT="$OUT/cache_run" scripts_cosim/datalab/r1a_prod_cache.sbatch)
TRAIN_JOB=$(sbatch --parsable --dependency=afterok:$CACHE_JOB --array=0-$((N_TASKS - 1)) --mem="$TRAIN_MEM" --export=ALL,WT="$WT" scripts_cosim/datalab/r1a_prod_train.sbatch)
SEL_JOB=$(sbatch --parsable --dependency=afterok:$TRAIN_JOB --export=ALL,WT="$WT",INPUTS="$INPUTS",OUT="$OUT/selection.json",SPLIT="experiments/${LINEAGE}_split.json",PREFIX=$STEM scripts_cosim/datalab/r1a_prod_select.sbatch)
for j in "$CACHE_JOB" "$TRAIN_JOB" "$SEL_JOB"; do [[ "$j" =~ ^[0-9]+$ ]] || { echo "FAIL LOUD: a submission was refused (got '$j'); cancel the others: scancel $CACHE_JOB $TRAIN_JOB $SEL_JOB"; exit 1; }; done
echo "cache $CACHE_JOB -> train array 0-$((N_TASKS - 1)) $TRAIN_JOB (mem $TRAIN_MEM) -> selection $SEL_JOB -> $OUT/selection.json ; staged checkpoints under $INPUTS/models/$STEM-<arm>-seed<N>.pt"
