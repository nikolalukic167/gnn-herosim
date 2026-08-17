#!/usr/bin/env bash
# Re-run the sealed holdout on current HEAD, into a NEW sweep dir.
#
# Why: the original sweep (contention_v2_873_v5.5_sealed_holdout_20260806) ran on
# 2026-08-06, six days before 25732cf moved the image-pull timeout inside the node
# FilterStore hold in KnativeAutoscaler.initialize_replica. Before that commit N
# co-located cold pulls ran in parallel; after it they serialize per node.
# Measured effect on an identical cell (sparse_p25 / knative / seed 42 /
# node_disk_v2 / workload-125-225): total_rtt 6.90M -> 17.21M (2.50x), carried
# entirely by averageQueueTime (12.23s -> 30.57s), not by per-task pull time.
#
# The historical results stay untouched as the pre-serialization record; this
# produces the post-serialization record so RQ3 and the coupled trio can be read
# against the same physics.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

ORIGINAL="simulation_data/normal_sim_sweeps/contention_v2_873_v5.5_sealed_holdout_20260806"
export SWEEP_DIR="${SWEEP_DIR:-simulation_data/normal_sim_sweeps/contention_v2_873_v5.5_sealed_holdout_rebaseline_20260813}"
export HEROSIM_WARMTH_PHYSICS="${HEROSIM_WARMTH_PHYSICS:-node_disk_v2}"
export HEROSIM_REQUIRE_EXPLICIT_PHYSICS=1
export CPU_PARALLEL="${CPU_PARALLEL:-4}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-6}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-6}"

if [[ "$SWEEP_DIR" == "$ORIGINAL" ]]; then
  echo "ERROR: refusing to overwrite the pre-serialization record at $ORIGINAL" >&2
  exit 1
fi

mkdir -p "${SWEEP_DIR}/configs" "${SWEEP_DIR}/results"
for cfg in 01_balanced_40_40_p50 02_balanced_50_50_p60 03_client_heavy_50_35_p50 04_server_heavy_35_50_p50; do
  src="${ORIGINAL}/configs/${cfg}.json"
  [[ -f "$src" ]] || { echo "ERROR: config missing: $src" >&2; exit 1; }
  cp -f "$src" "${SWEEP_DIR}/configs/${cfg}.json"
done

export MANIFEST_KIND="sealed_multi_seed_live_holdout_rebaseline"
export MANIFEST_NOTE="Post-25732cf re-run of ${ORIGINAL##*/} on identical configs, seeds, and checkpoints. 25732cf moved the image-pull timeout inside the node FilterStore hold in KnativeAutoscaler.initialize_replica, so N co-located cold pulls serialize per node instead of running in parallel."
export MANIFEST_EXTRA_JSON="$(
  cat <<JSON
{
  "supersedes_for_current_physics": "${ORIGINAL##*/}",
  "incomparable_with": {
    "sweep": "${ORIGINAL##*/}",
    "reason": "pre-serialization pull physics; never present both in one table"
  },
  "require_explicit_physics": true,
  "runner": "scripts_cosim/important/run_contention_v2_873_sealed_holdout_rebaseline.sh"
}
JSON
)"

echo "=== sealed holdout re-baseline on current HEAD ==="
echo "  sweep=${SWEEP_DIR}"
echo "  physics=${HEROSIM_WARMTH_PHYSICS} (strict)"
echo "  head=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"

exec bash scripts_cosim/important/run_contention_v2_873_sealed_holdout.sh
