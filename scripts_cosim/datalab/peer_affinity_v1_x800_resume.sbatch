#!/bin/bash
# peer_affinity_v1 -- resume the four x800 corpora after the 2026-09-11 /home quota death
# (jobs 756580-756583, every task FAILED at write). Runs after the archive job frees space
# (submit with --dependency=afterok:<archive job>). Two steps:
#   1. scrub every dataset dir the quota truncated: any 0-byte json among the required files
#      means the dir is not evidence (--resume keys on best.json + placements.jsonl EXISTING,
#      so a 0-byte best.json next to a complete jsonl would be skipped as done).
#   2. submit the same generate sbatch with --resume for all four corpora.
#
#SBATCH --job-name=pa-x800-resume
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/pa-x800-resume-%j.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/pa-x800-resume-%j.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=01:00:00

set -euo pipefail
cd /home/nikola.lukic/gnn-herosim
getfattr -n ceph.dir.rbytes /home/nikola.lukic 2>/dev/null | tail -1
[[ -L simulation_data/normal_sim_sweeps ]] || { echo "FAIL LOUD: normal_sim_sweeps is not yet a symlink -- archive did not finish"; exit 1; }

for OUT in gnn_datasets_peer_affinity_v1_c3_x800_p3_train gnn_datasets_peer_affinity_v1_c3_x800_p3_r2 \
           gnn_datasets_peer_affinity_v1_c3_x800_p2_train gnn_datasets_peer_affinity_v1_c3_x800_p2_r2; do
  D=simulation_data/$OUT
  mkdir -p "$D"
  total=$(find "$D" -maxdepth 1 -mindepth 1 -type d -name 'ds_*' | wc -l)
  scrubbed=0
  for ds in "$D"/ds_*/; do
    [[ -d "$ds" ]] || continue
    bad=0
    for f in best.json infrastructure.json workload.json optimal_result.json placement_metadata.json placements/placements.jsonl; do
      if [[ -e "$ds/$f" && ! -s "$ds/$f" ]]; then bad=1; fi
    done
    # a dir with best.json must also carry a complete sweep marker
    if [[ -s "$ds/best.json" ]] && ! grep -q '"sweep_complete": *true' "$ds/placement_metadata.json" 2>/dev/null; then bad=1; fi
    if (( bad )); then echo "[SCRUB] $ds"; rm -rf "$ds"; scrubbed=$((scrubbed+1)); fi
  done
  echo "[$OUT] dirs=$total scrubbed=$scrubbed kept=$((total-scrubbed))"
done

S=scripts_cosim/datalab/peer_affinity_v1_x800_generate.sbatch
sbatch --export=ALL,GRID=peer_affinity_screen_c3_x800_p3_train,OUT=gnn_datasets_peer_affinity_v1_c3_x800_p3_train,TOTAL=500,SHARDS=20 --array=0-19 $S
sbatch --export=ALL,GRID=peer_affinity_screen_c3_x800_p3_r2,OUT=gnn_datasets_peer_affinity_v1_c3_x800_p3_r2,TOTAL=34,SHARDS=2 --array=0-1 $S
sbatch --export=ALL,GRID=peer_affinity_screen_c3_x800_p2_train,OUT=gnn_datasets_peer_affinity_v1_c3_x800_p2_train,TOTAL=500,SHARDS=20 --array=0-19 $S
sbatch --export=ALL,GRID=peer_affinity_screen_c3_x800_p2_r2,OUT=gnn_datasets_peer_affinity_v1_c3_x800_p2_r2,TOTAL=34,SHARDS=2 --array=0-1 $S
echo "ALL_SUBMITTED $(date)"
