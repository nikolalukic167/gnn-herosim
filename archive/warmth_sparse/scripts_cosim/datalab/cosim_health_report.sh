#!/usr/bin/env bash
# Datalab co-sim health snapshot: non-unique backfill (warmth/sparse) + contention_v2 finisher.
# Exit codes: 0=all done, 1=in-progress healthy, 2=unhealthy (stalled/failed), 3=ssh/unreachable
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STATE_FILE="${ROOT}/logs/cosim_monitor_state.env"
REMOTE_HOST="${REMOTE_HOST:-datalab}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/nikola.lukic/gnn-herosim}"

# Job IDs from last submit (override via state file or env)
WARMTH_NU_JOB="${WARMTH_NU_JOB:-482755}"
SPARSE_NU_JOB="${SPARSE_NU_JOB:-482759}"
CONT_V2_JOB="${CONT_V2_JOB:-482647}"

if [[ -f "$STATE_FILE" ]]; then
  # shellcheck source=/dev/null
  source "$STATE_FILE"
fi

WARMTH_TARGET="${WARMTH_TARGET:-500}"
SPARSE_TARGET="${SPARSE_TARGET:-351}"
CONT_V2_TARGET="${CONT_V2_TARGET:-900}"

timestamp() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }

if ! ssh -o ConnectTimeout=15 -o BatchMode=yes "${REMOTE_HOST}" "test -d ${REMOTE_ROOT}" 2>/dev/null; then
  echo "[$(timestamp)] ERROR: cannot reach ${REMOTE_HOST}:${REMOTE_ROOT}"
  exit 3
fi

report="$(ssh -o ConnectTimeout=30 -o BatchMode=yes "${REMOTE_HOST}" bash -s <<REMOTE
set -euo pipefail
cd "${REMOTE_ROOT}"

progress_tail() {
  local f="\$1" n="\${2:-3}"
  if [[ -f "\$f" ]]; then tail -"\$n" "\$f"; else echo "(no log)"; fi
}

warmth_log="logs/non_unique_progress_gnn_datasets_4tasks_1060_warmth_v2.txt"
sparse_log="logs/non_unique_progress_gnn_datasets_4tasks_sparse_warmth_v2.txt"

warmth_succ=0; warmth_fail=0; warmth_skip=0
sparse_succ=0; sparse_fail=0; sparse_skip=0
for pair in "warmth:\$warmth_log" "sparse:\$sparse_log"; do
  name="\${pair%%:*}"; f="\${pair#*:}"
  if [[ -f "\$f" ]]; then
    s=\$(grep -c ' SUCCESS ' "\$f" 2>/dev/null || true)
    fl=\$(grep -c ' FAILED ' "\$f" 2>/dev/null || true)
    sk=\$(grep -c ' SKIPPED ' "\$f" 2>/dev/null || true)
  else s=0; fl=0; sk=0; fi
  case "\$name" in
    warmth) warmth_succ=\$s; warmth_fail=\$fl; warmth_skip=\$sk ;;
    sparse) sparse_succ=\$s; sparse_fail=\$fl; sparse_skip=\$sk ;;
  esac
done

warmth_jsonl=\$(find simulation_data/gnn_datasets_4tasks_1060_warmth_v2 -name placements.jsonl 2>/dev/null | wc -l)
sparse_jsonl=\$(find simulation_data/gnn_datasets_4tasks_sparse_warmth_v2 -name placements.jsonl 2>/dev/null | wc -l)
cont_jsonl=\$(find simulation_data/gnn_datasets_4tasks_contention_v2 -name placements.jsonl 2>/dev/null | wc -l)

warmth_nu_meta=\$(python3 - <<'PY'
import json
from pathlib import Path
n=0
for m in Path("simulation_data/gnn_datasets_4tasks_1060_warmth_v2").glob("ds_*/placement_metadata.json"):
    d=json.loads(m.read_text())
    if d.get("non_unique_placements",0)>0: n+=1
print(n)
PY
)
sparse_nu_meta=\$(python3 - <<'PY'
import json
from pathlib import Path
n=0
for m in Path("simulation_data/gnn_datasets_4tasks_sparse_warmth_v2").glob("ds_*/placement_metadata.json"):
    d=json.loads(m.read_text())
    if d.get("non_unique_placements",0)>0: n+=1
print(n)
PY
)
echo "=== COSIM HEALTH @ \$(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo ""
echo "## SLURM (co-sim jobs)"
squeue -u "\$(whoami)" -o "%.10i %.9j %.8T %.10M %R" 2>/dev/null | grep -E 'nu-warmth|nu-sparse|cont-v2|JOBID' || echo "(no matching jobs in queue)"
echo ""
echo "## Non-unique backfill"
echo "warmth:  SUCCESS=\${warmth_succ} FAILED=\${warmth_fail} SKIPPED=\${warmth_skip}  jsonl=\${warmth_jsonl}  non_unique_meta=\${warmth_nu_meta}  target=${WARMTH_TARGET}"
echo "sparse:  SUCCESS=\${sparse_succ} FAILED=\${sparse_fail} SKIPPED=\${sparse_skip}  jsonl=\${sparse_jsonl}  non_unique_meta=\${sparse_nu_meta}  target=${SPARSE_TARGET}"
echo ""
echo "## contention_v2 finisher"
echo "jsonl=\${cont_jsonl}  target=${CONT_V2_TARGET}"
squeue -u "\$(whoami)" -h 2>/dev/null | grep -c cont-v2-fin || true | xargs -I{} echo "cont-v2-fin_running={}"
echo ""
echo "## Latest warmth SUCCESS"
progress_tail "\$warmth_log" 2
echo "## Latest sparse SUCCESS"
progress_tail "\$sparse_log" 2
echo ""
echo "## Shard job states (sacct)"
for j in ${WARMTH_NU_JOB} ${SPARSE_NU_JOB} ${CONT_V2_JOB}; do
  sacct -j "\${j}" --format=JobID,State,Elapsed,ExitCode -P 2>/dev/null | head -6 || true
  echo "---"
done
echo ""
echo "## Recent FAILED (if any)"
grep ' FAILED ' "\$warmth_log" "\$sparse_log" 2>/dev/null | tail -5 || echo "(none)"
REMOTE
)"

echo "$report"

# Parse for exit status
warmth_succ=$(echo "$report" | sed -n 's/^warmth:  SUCCESS=\([0-9]*\).*/\1/p' | head -1)
sparse_succ=$(echo "$report" | sed -n 's/^sparse:  SUCCESS=\([0-9]*\).*/\1/p' | head -1)
warmth_fail=$(echo "$report" | sed -n 's/^warmth:  SUCCESS=[0-9]* FAILED=\([0-9]*\).*/\1/p' | head -1)
sparse_fail=$(echo "$report" | sed -n 's/^sparse:  SUCCESS=[0-9]* FAILED=\([0-9]*\).*/\1/p' | head -1)
warmth_skip=$(echo "$report" | sed -n 's/^warmth:  SUCCESS=[0-9]* FAILED=[0-9]* SKIPPED=\([0-9]*\).*/\1/p' | head -1)
sparse_skip=$(echo "$report" | sed -n 's/^sparse:  SUCCESS=[0-9]* FAILED=[0-9]* SKIPPED=\([0-9]*\).*/\1/p' | head -1)
cont_jsonl=$(echo "$report" | sed -n 's/^jsonl=\([0-9]*\).*/\1/p' | head -1)
nu_warmth_running=$(echo "$report" | grep -c 'nu-warmth.*RUNNING' || true)
nu_sparse_running=$(echo "$report" | grep -c 'nu-sparse.*RUNNING' || true)
cont_running=$(echo "$report" | grep -c 'cont-v2.*RUNNING' || true)

warmth_succ=${warmth_succ:-0}
sparse_succ=${sparse_succ:-0}
warmth_fail=${warmth_fail:-0}
sparse_fail=${sparse_fail:-0}
warmth_skip=${warmth_skip:-0}
sparse_skip=${sparse_skip:-0}
cont_jsonl=${cont_jsonl:-0}
warmth_processed=$((warmth_succ + warmth_skip))
sparse_processed=$((sparse_succ + sparse_skip))

warmth_jsonl=$(echo "$report" | sed -n 's/^warmth:  SUCCESS=[0-9]* FAILED=[0-9]* SKIPPED=[0-9]*  jsonl=\([0-9]*\).*/\1/p' | head -1)
warmth_jsonl=${warmth_jsonl:-0}

warmth_done=0
sparse_done=0
cont_done=0

# Done when all datasets with placements.jsonl are SUCCESS or SKIP (already_complete)
if [[ "$warmth_processed" -ge "$warmth_jsonl" ]] && [[ "$warmth_jsonl" -ge 1 ]]; then warmth_done=1; fi
sparse_jsonl=$(echo "$report" | sed -n 's/^sparse:.* jsonl=\([0-9]*\).*/\1/p' | head -1)
sparse_nu_meta_count=$(echo "$report" | sed -n 's/^sparse:.* non_unique_meta=\([0-9]*\).*/\1/p' | head -1)
sparse_jsonl=${sparse_jsonl:-0}
sparse_nu_meta_count=${sparse_nu_meta_count:-0}
if [[ "$sparse_jsonl" -ge 351 ]] && [[ "$sparse_nu_meta_count" -ge 348 ]]; then sparse_done=1; fi
# contention finisher: no running shards and jsonl at target
if [[ "$cont_running" -eq 0 ]] && [[ "$cont_jsonl" -ge 900 ]]; then cont_done=1; fi

if [[ "$warmth_fail" -gt 0 ]] || [[ "$sparse_fail" -gt 0 ]]; then
  echo ""
  echo "STATUS: UNHEALTHY (FAILED datasets in progress logs)"
  exit 2
fi

# Stalled: no queue jobs but work incomplete
if [[ "$warmth_done" -eq 0 ]] && [[ "$nu_warmth_running" -eq 0 ]]; then
  echo ""
  echo "STATUS: UNHEALTHY (warmth backfill stalled: ${warmth_succ} done, no nu-warmth jobs)"
  exit 2
fi
if [[ "$sparse_done" -eq 0 ]] && [[ "$nu_sparse_running" -eq 0 ]]; then
  echo ""
  echo "STATUS: UNHEALTHY (sparse backfill stalled: ${sparse_succ} done, no nu-sparse jobs)"
  exit 2
fi

if [[ "$warmth_done" -eq 1 ]] && [[ "$sparse_done" -eq 1 ]] && [[ "$cont_done" -eq 1 ]]; then
  echo ""
  echo "STATUS: ALL_COSIM_COMPLETE"
  exit 0
fi

echo ""
echo "STATUS: IN_PROGRESS (warmth=${warmth_processed}/${warmth_jsonl} sparse=${sparse_processed}/351 cont_jsonl=${cont_jsonl})"
exit 1
