#!/bin/bash
# prune_g8_s0_v1 offline S screen (scripts_cosim/prune_g8_screen.py run): the cd_exactS decision replay of each big_groups G=8 state with over-cap batches pruned.
#   sbatch --export=ALL,WT=<pinned worktree>,EXPECT_HEAD=<sha>,ORDER=cost|gnn,OUT=<rows.jsonl>[,LIMIT=<n>,CKPT=<gnn_eng .pt>] prune_g8_screen.sbatch
#SBATCH --job-name=pg8-screen
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/pg8-screen-%j.out
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=06:00:00
set -euo pipefail
WT=${WT:?FAIL LOUD: set WT}; EXPECT_HEAD=${EXPECT_HEAD:?}; ORDER=${ORDER:?}; OUT=${OUT:?}
STATES=${STATES:-/share/nikola.lukic/big_groups/states/top5}
cd "$WT"
for v in $(compgen -e | grep -E '^(HEROSIM_|GNN_|COSIM_|LIVE_AUDIT_|GATE_|SIM_|QUEUE_FEATURE|MAX_PLACEMENT|NEAR_RTT_|PARTIAL_STATE|WF1_)'); do unset "$v"; done
if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3 PYTHONPATH="$WT" PYTHONUNBUFFERED=1
HEAD=$(git rev-parse HEAD)
echo "HEAD $HEAD node $(hostname) order=$ORDER out=$OUT"
[[ "$HEAD" == "$EXPECT_HEAD"* ]] || { echo "FAIL LOUD: worktree HEAD $HEAD is not $EXPECT_HEAD"; exit 1; }
[[ ! -e "$OUT" ]] || { echo "FAIL LOUD: $OUT exists"; exit 1; }
EXTRA=""
[[ -z "${LIMIT:-}" ]] || EXTRA="$EXTRA --limit $LIMIT"
[[ -z "${CKPT:-}" ]] || EXTRA="$EXTRA --ckpt $CKPT"
python3 scripts_cosim/prune_g8_screen.py run "$OUT" --states "$STATES" --order "$ORDER" --workers "${WORKERS:-14}" $EXTRA
