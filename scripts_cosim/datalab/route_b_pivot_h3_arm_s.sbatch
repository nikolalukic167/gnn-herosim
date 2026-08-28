#!/bin/bash
#SBATCH --job-name=rbh3-s
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/route-b-pivot-h3-s-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/route-b-pivot-h3-s-%A_%a.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --array=0-15

# route_b_env_pivot_v1 rung H3, Arm S (primary physics) — GENERATION ONLY.
#
# Licensed by AMENDMENT 3, signed off 2026-08-28
# (docs/lineages/route_b_env_pivot_v1/screen-amendment-3.md §9). The sign-off licenses
# generating the amended H2 and H3 plus their paired separable controls and reading S0-S4
# on them. It does NOT license re-reading H0/H1, moving a bar, or treating the 2026-08-27
# probe as a rung verdict. THIS JOB GENERATES; it reads no bar.
#
# 204 datasets (2 conn_probs x 2 replica_configs x 3 queue_dists x 17 seeds), 16 shards of
# ~13. Seeds are the registered fresh block 3301-3317; the grid preset carries them, do not
# override. The paired separable control is route_b_pivot_h3_ctrl.sbatch — S0 is a VOID
# gate and no S1-S4 number on this corpus may be read until that control passes
# r_exact.frac_gt_1pct <= 0.02.
#
# COST: AMENDMENT 3 moves H3 to per_server 4/5, giving pools of 8 and 9 => 8P8 = 40,320 and
# 9P8 = 362,880 sweep rows per dataset (the pool for per_server=5 is 9, not 10 — one hosting
# node carries fewer suitable platforms; measured, not derived). As registered H3 generated
# 0/204, uniqueness-exhausted on both arms (route_b_pivot_h3_genprobe_registered). The 12h
# walltime is margin against the 362,880-row arm; check the measured rate on the first
# finished dataset rather than trusting this estimate.
#
# Reuses run_contention_regen_shard.sh unchanged. That script does
# `micromamba shell hook --shell bash` and calls a plain `python3` — never `pipenv run`
# (datalab-pitfalls #1 and #8). Do not reintroduce either spelling here.

set -euo pipefail
PROJECT_ROOT="/home/nikola.lukic/gnn-herosim"
cd "$PROJECT_ROOT"

export OUTPUT_SUBDIR=gnn_datasets_route_b_pivot_h3
export GRID=route_b_pivot_h3
export WARMTH_PHYSICS=node_disk_v2
export REMAINING_START=0
export TOTAL_DATASETS=204
export NUM_SHARDS=16
export ONLY_MISSING_JSONL=0
export GNN_CAPTURE_DATASET_STATE=0
export COSIM_SUPPRESS_SIM_PRINTS=1

# NOT the runner's default of 1. route_b's decode mask is "no replica reuse"
# (docs/lineages/route_b_v1/stage2-preregistration.md §4), and the globally-distinct-replica
# constraint is exactly what defines H3's pool law (pool = per_server x n_hosting_nodes must
# seat all 8 tasks). Allowing reuse would silently change what this rung measures.
export ALLOW_NON_UNIQUE=0

# AMENDMENT 3's re-derived bound. The threshold is tested against the PRE-uniqueness product
# (herosim-cosim-skip-threshold-is-pre-uniqueness: sizing it from observed sweep sizes is the
# documented defect) and the comparison is a strict `>`. Bound: max(per_server)=5 x
# server_node_counts=4 = 20 candidates/type => 20^8 = 25,600,000,000. The old 16,777,216 was
# computed from max(per_server)=2; it admitted per_server=4 with ZERO headroom and skipped
# per_server=5 outright.
export MAX_PLACEMENT_COMBINATIONS_SKIP=30000000000

# Arm S physics. STORAGE_NEUTRAL is deliberately NOT exported here — it is a
# control-arm-only lever (screen-amendment-1.md §4) and setting it on Arm S voids the rung.
export HEROSIM_DATA_LOCALITY=1
export HEROSIM_OUTPUT_SIZE_BYTES=800000000

# The other half of the five-variable recipe, and not optional. Omitting these produced two
# VOID-GENERATION corpora that reported 204/204 SUCCESS while carrying truncated sweeps, and
# nondeterministically so (now a GATE TOOLS row). At the 30s KEEP_ALIVE default a DAG stage
# stretched past it by route B's 800MB transfers gets the CHILD's forced replica scaled down
# before the child dispatches, killing the episode. 8-task episodes do not survive it.
# HEROSIM_RETAIN_TASK_TIMES is what makes rows carry task_times, without which
# score_route_b_contention.py refuses the makespan objective.
export HEROSIM_COSIM_KEEP_ALIVE=1000000
export HEROSIM_RETAIN_TASK_TIMES=1

bash scripts_cosim/datalab/run_contention_regen_shard.sh
