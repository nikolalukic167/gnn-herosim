#!/usr/bin/env bash
# Attempt auto-remediation for stalled/failed datalab co-sim jobs.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
STATE_FILE="${ROOT}/logs/cosim_monitor_state.env"
PRE="/tmp/cosim_health_pre_remediate.txt"

echo "[$(date -u +%H:%M:%S)] === cosim remediate ==="

set +e
bash scripts_cosim/datalab/cosim_health_report.sh 2>&1 | tee "$PRE"
health_rc=$?
set -e

if [[ "$health_rc" -eq 0 ]]; then
  echo "Nothing to remediate — all complete"
  exit 0
fi
if [[ "$health_rc" -eq 1 ]]; then
  echo "In progress and healthy — no remediation"
  exit 0
fi
if [[ "$health_rc" -eq 3 ]]; then
  echo "ERROR: datalab unreachable" >&2
  exit 3
fi

REMEDIATE_LOG="${ROOT}/logs/cosim_remediate_$(date +%Y%m%d_%H%M%S).log"

if grep -q "warmth backfill stalled" "$PRE"; then
  echo "Re-submitting warmth non-unique backfill..."
  bash scripts_cosim/datalab/submit_warmth_non_unique_datalab.sh 2>&1 | tee "$REMEDIATE_LOG"
  warmth_job=$(grep 'Submitted warmth non-unique array:' "$REMEDIATE_LOG" | sed -n 's/.*array: \([0-9]*\).*/\1/p' | tail -1)
  sparse_job=$(grep 'Submitted sparse non-unique array:' "$REMEDIATE_LOG" | sed -n 's/.*array: \([0-9]*\).*/\1/p' | tail -1)
  cat > "$STATE_FILE" <<EOF
# Updated by cosim_health_remediate.sh $(date -u +%Y-%m-%dT%H:%M:%SZ)
WARMTH_NU_JOB=${warmth_job:-482755}
SPARSE_NU_JOB=${sparse_job:-482759}
CONT_V2_JOB=${CONT_V2_JOB:-482647}
EOF
  echo "Updated ${STATE_FILE}"
fi

if grep -q "sparse backfill stalled" "$PRE"; then
  nu=$(echo "$PRE" | sed -n 's/^sparse:.* non_unique_meta=\([0-9]*\).*/\1/p' | head -1)
  if [[ "${nu:-0}" -lt 348 ]]; then
    echo "Re-submitting sparse non-unique backfill (non_unique_meta=${nu})..."
    ssh -o BatchMode=yes datalab "cd /home/nikola.lukic/gnn-herosim && sbatch scripts_cosim/datalab/warmth_non_unique_sparse.sbatch" | awk '{print "sparse job:", $NF}'
  else
    echo "Sparse effectively complete (348+ ds) — skip resubmit"
  fi
fi

cont_running=$(ssh -o BatchMode=yes datalab 'squeue -u $(whoami) -h 2>/dev/null | grep -c cont-v2-f || true')
cont_jsonl=$(ssh -o BatchMode=yes datalab 'find /home/nikola.lukic/gnn-herosim/simulation_data/gnn_datasets_4tasks_contention_v2 -name placements.jsonl 2>/dev/null | wc -l')
if [[ "${cont_running:-0}" -eq 0 ]] && [[ "${cont_jsonl:-0}" -lt 900 ]]; then
  echo "Re-submitting contention_v2 finisher..."
  new_cont=$(ssh -o BatchMode=yes datalab 'cd /home/nikola.lukic/gnn-herosim && sbatch scripts_cosim/datalab/contention_v2_finish.sbatch' | awk '{print $NF}')
  echo "contention_v2_finish job: ${new_cont}"
fi

exit 0
