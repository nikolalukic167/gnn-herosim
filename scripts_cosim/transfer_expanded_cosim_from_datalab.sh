#!/usr/bin/env bash
# Pull expanded co-sim corpora (non-unique warmth/sparse + contention_v2 900) from datalab.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SSH_KEY="${SSH_KEY:-$HOME/.ssh/nikolalukic167}"
REMOTE="${REMOTE:-nikola.lukic@cluster.datalab.tuwien.ac.at}"
REPO="${REPO:-/home/nikola.lukic/gnn-herosim}"

CORPORA=(
  gnn_datasets_4tasks_1060_warmth_v2
  gnn_datasets_4tasks_sparse_warmth_v2
  gnn_datasets_4tasks_contention_v2
)

mkdir -p logs simulation_data

log() { echo "[$(date -Is)] $*" | tee -a logs/transfer_expanded_cosim.log; }

log "=== Pull expanded co-sim from datalab ==="

for sub in "${CORPORA[@]}"; do
  mkdir -p "simulation_data/${sub}"
  log "Rsync ${sub}..."
  rsync -avP --partial -e "ssh -i ${SSH_KEY}" \
    "${REMOTE}:${REPO}/simulation_data/${sub}/" \
    "simulation_data/${sub}/" 2>&1 | tee -a logs/transfer_expanded_cosim.log
done

log "=== Post-pull counts ==="
for sub in "${CORPORA[@]}"; do
  jsonl=$(find "simulation_data/${sub}" -name placements.jsonl 2>/dev/null | wc -l)
  best=$(find "simulation_data/${sub}" -name best.json 2>/dev/null | wc -l)
  log "${sub}: jsonl=${jsonl} best=${best}"
done

nu_lines=$(pipenv run python3 -c "
from pathlib import Path
p = Path('simulation_data/gnn_datasets_4tasks_1060_warmth_v2/ds_00000/placements/placements.jsonl')
print(sum(1 for _ in open(p)) if p.exists() else 0)
" 2>/dev/null || echo 0)
cont_jsonl=$(find simulation_data/gnn_datasets_4tasks_contention_v2 -name placements.jsonl 2>/dev/null | wc -l)
if [[ "$nu_lines" -lt 2000 || "$cont_jsonl" -lt 850 ]]; then
  log "ERROR: post-pull verification failed (ds_00000_lines=${nu_lines} cont_jsonl=${cont_jsonl})" >&2
  exit 1
fi

mkdir -p logs/expanded_cosim_pipeline
touch logs/expanded_cosim_pipeline/phase_rsync.done
log "Verified + marked phase_rsync.done"
