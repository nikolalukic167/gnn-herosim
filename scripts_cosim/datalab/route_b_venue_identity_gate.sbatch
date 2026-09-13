#!/bin/bash
#SBATCH --job-name=rb-venue-gate
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/route-b-venue-gate-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/route-b-venue-gate-%A_%a.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --array=0-0

# VENUE IDENTITY GATE — prerequisite for reading the 8-task probe against the 4-task
# baseline. Not a probe, not a lineage artifact.
#
# The route_b_v1 4-task Arm S corpus was generated LOCALLY (LINEAGES.md:4021). The 8-task
# probe is generated HERE. PARITY.md rule 3 says never compare across venues without
# running the checks, and "unknown is not a pass". Rather than lean on the generic parity
# scripts (which measure GNN logits, not simulator sweeps), this measures the thing that
# actually matters for this comparison: does the co-sim pipeline produce the SAME corpus
# on this machine as it does locally?
#
# Method: regenerate the first FOUR 4-task Arm S datasets (ds_00000..ds_00003) with the
# exact recorded Arm S recipe, into a throwaway subdir. Then diff against the frozen local
# corpus. The same test passed locally on 2026-08-25 (Step 1 of the item-4 plan):
# best.json byte-identical, placements.jsonl identical AS A SET, infrastructure.json
# differing only in metadata/generation_time and metadata/config_file.
#
# PASS  = same result here => venue is not a variable => 4-vs-8 comparison is clean.
# FAIL  = the 8-task corpus must be generated locally instead, or the gap explained first.
#
# Row ORDER in placements.jsonl is expected to differ (parallel completion order) and is
# not a failure — compare sorted. Worker count differs from the local run for the same
# reason and is likewise not a variable; Step 1 already showed an 8-worker local run
# reproducing a corpus built at a different worker count.

set -euo pipefail
PROJECT_ROOT="/home/nikola.lukic/gnn-herosim"
cd "$PROJECT_ROOT"

export OUTPUT_SUBDIR=gnn_datasets_route_b_venue_gate
export GRID=route_b_pilot_v1
export WARMTH_PHYSICS=node_disk_v2
export REMAINING_START=0
export TOTAL_DATASETS=4
export NUM_SHARDS=1
export ONLY_MISSING_JSONL=0
export ALLOW_NON_UNIQUE=0
export GNN_CAPTURE_DATASET_STATE=0
export COSIM_SUPPRESS_SIM_PRINTS=1

# The Arm S recipe, exactly as recorded at LINEAGES.md:4022-4023. All four matter:
# the keep-alive and retain-task-times pair is what job 713615 was missing, and the
# data-locality / output-size pair is Arm S's physics.
export HEROSIM_COSIM_KEEP_ALIVE=1000000
export HEROSIM_RETAIN_TASK_TIMES=1
export HEROSIM_DATA_LOCALITY=1
export HEROSIM_OUTPUT_SIZE_BYTES=800000000

# 4-task sweeps are 256-1248 combinations; the 250k default is nowhere near binding here.
# Left at the default deliberately, so this gate exercises the same code path the frozen
# local corpus was built under rather than a raised-threshold variant.

echo "=== venue identity gate: interpreter actually serving ==="
bash -c 'eval "$(micromamba shell hook --shell bash)" && micromamba activate gnn && \
  python3 -c "import sys, numpy; print(sys.executable); print(\"numpy\", numpy.__version__)"'

bash scripts_cosim/datalab/run_contention_regen_shard.sh
