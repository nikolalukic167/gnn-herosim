#!/usr/bin/env bash
# Wait for the running coupled-trio gate, then start the sealed-holdout re-baseline.
#
# Both sweeps are CPU-bound on this host, so they must not overlap: a re-baseline
# running next to the trio would contend for the same cores and pollute both
# wall-clock reads. The trio is tracked by its own script cmdline rather than by a
# caller-supplied pid, because the pid printed by a backgrounded launcher is the
# launching shell, which can exit long before the sweep does.
#
# Aborts without starting the re-baseline if the trio never reaches its marker.
#
# Usage: chain_coupled_trio_then_rebaseline.sh <trio_log>
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

TRIO_LOG="${1:?trio log path}"
TRIO_PATTERN='^bash scripts_cosim/important/run_contention_v2_873_coupled_trio\.sh'
MARKER="coupled trio gate complete"
POLL="${POLL:-60}"

[[ -f "$TRIO_LOG" ]] || { echo "[chain] ERROR: no such trio log: $TRIO_LOG" >&2; exit 1; }

trio_running() { pgrep -f "$TRIO_PATTERN" >/dev/null 2>&1; }

if ! trio_running; then
  echo "[chain] ERROR: no trio process matches ${TRIO_PATTERN} — refusing to guess" >&2
  exit 1
fi

echo "[chain] $(date -Is) waiting for trio (log=${TRIO_LOG})"
while trio_running; do
  sleep "$POLL"
done
echo "[chain] $(date -Is) trio process gone"

if ! grep -q "$MARKER" "$TRIO_LOG"; then
  echo "[chain] ERROR: trio never logged '${MARKER}' — not starting re-baseline" >&2
  tail -20 "$TRIO_LOG" >&2
  exit 1
fi

echo "[chain] $(date -Is) trio complete; starting sealed-holdout re-baseline"
exec bash scripts_cosim/important/run_contention_v2_873_sealed_holdout_rebaseline.sh
