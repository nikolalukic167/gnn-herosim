#!/bin/bash
#SBATCH --job-name=llf-relabel
#SBATCH --output=/share/nikola.lukic/uncut_groups/llf_corpus/%x-%A_%a.out
#SBATCH --partition=CPU-amd
#SBATCH --cpus-per-task=16
#SBATCH --mem=150G
#SBATCH --time=12:00:00
# Union corpus (live_link_features_v1) stage 4: the simulator relabel under option D with the stored flows. One array task = one shard of SPLIT.
#   usage: sbatch --array=0-23 --export=ALL,WT=<worktree>,EXPECT_HEAD=<sha8>,SPLIT=<split.json>,OUTD=<dir>,N=24 llf_relabel.sbatch      (shard K of N of SPLIT -> OUTD/relabel_K_of_N.jsonl)
#   or, for a list of splits (one task per split file split_rem_K.json): SPLIT_PATTERN='.../split_rem_%d.json' with N=1.
# The job exits NON-ZERO unless the relabel exited 0, no state failed, and the shard file holds one row per state of the shard (a state timeout used to leave the job COMPLETED with a short file).
set -uo pipefail
for v in $(compgen -e | grep -E '^(HEROSIM_|GNN_)'); do unset "$v"; done
eval "$(micromamba shell hook --shell bash)" && micromamba activate gnn
export HEROSIM_PY=python3 HEROSIM_EXCHANGE_FAIRSHARE=1 HEROSIM_EXCHANGE_TCP=1
WT=${WT:?}; EXPECT_HEAD=${EXPECT_HEAD:?}; OUTD=${OUTD:?}; N=${N:-24}; K=${SLURM_ARRAY_TASK_ID:?}; WORKERS=${WORKERS:-14}; STATE_TIMEOUT=${STATE_TIMEOUT:-3600}
CORPUS=${CORPUS:-/share/nikola.lukic/uncut_groups/llf_corpus/corpus}
cd "$WT"; export PYTHONPATH=$PWD
[[ "$(git rev-parse HEAD)" == ${EXPECT_HEAD}* ]] || { echo "FAIL LOUD: HEAD $(git rev-parse HEAD) is not $EXPECT_HEAD"; exit 1; }
if [[ -n "${SPLIT_PATTERN:-}" ]]; then SPLITF=$(printf "$SPLIT_PATTERN" "$K"); SHARD=0/1; OUT=$OUTD/relabel_rem_$K.jsonl; else SPLITF=${SPLIT:?}; SHARD=$K/$N; OUT=$OUTD/relabel_${K}_of_$N.jsonl; fi
mkdir -p "$OUTD"
LOG=$(mktemp)
python3 -u scripts_cosim/uncut_label_screen.py relabel --split "$SPLITF" --corpus "$CORPUS" --shard $SHARD --out "$OUT" --workers "$WORKERS" --state-timeout "$STATE_TIMEOUT" --stored-flows --record-pair --union-sfs 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
EXPECTED=$(python3 - "$SPLITF" "$SHARD" <<'PY'
import json, sys
sp = json.load(open(sys.argv[1])); i, n = (int(x) for x in sys.argv[2].split("/"))
print(len(range(i, len(sp["train"]) + len(sp["val"]) + len(sp["test"]), n)))
PY
)
ROWS=$(grep -c . "$OUT" 2>/dev/null || echo 0)
echo "relabel rc=$rc rows=$ROWS expected=$EXPECTED"
if [[ $rc -ne 0 || "$ROWS" -ne "$EXPECTED" ]] || grep -q "FAIL LOUD" "$LOG"; then echo "FAIL LOUD: shard $SHARD of $SPLITF incomplete (rc $rc, $ROWS of $EXPECTED rows)"; exit 1; fi
echo "shard complete"
