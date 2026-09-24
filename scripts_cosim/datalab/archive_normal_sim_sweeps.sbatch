#!/bin/bash
# Archive simulation_data/normal_sim_sweeps (173 GB of old live-gate result JSONs, every one
# already scored and recorded in its lineage node) from the 250 GB /home quota to
# /share/nikola.lukic/herosim_archive/, leaving a symlink so every recorded path still
# resolves. Same protocol as the 2026-08-26 offload (/share/nikola.lukic/offload_20260826.log):
# copy, verify (rsync --checksum dry run must transfer nothing), then remove the source.
# Runs under sbatch because a 173 GB copy is exactly the kind of login-node job the watchdog kills.
#
#SBATCH --job-name=archive-sweeps
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/archive-sweeps-%j.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/archive-sweeps-%j.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=12:00:00

set -euo pipefail
cd /home/nikola.lukic/gnn-herosim
SRC=simulation_data/normal_sim_sweeps
DST=/share/nikola.lukic/herosim_archive/normal_sim_sweeps
LOG=/share/nikola.lukic/herosim_archive/offload_normal_sim_sweeps_$(date +%Y%m%d).log
mkdir -p "$(dirname "$DST")"
[[ -d "$SRC" && ! -L "$SRC" ]] || { echo "nothing to archive: $SRC is not a real directory"; exit 0; }

echo "COPY_START $(date)" | tee -a "$LOG"
du -sh "$SRC" | tee -a "$LOG"
rsync -a --info=stats1 "$SRC/" "$DST/" | tail -15 | tee -a "$LOG"
echo "COPY_DONE $(date)" | tee -a "$LOG"

# verify: a checksum dry run against the copy must have nothing left to transfer
PENDING=$(rsync -a --checksum --dry-run --itemize-changes "$SRC/" "$DST/" | grep -c '^>f' || true)
echo "VERIFY_DONE pending_files=$PENDING" | tee -a "$LOG"
[[ "$PENDING" == "0" ]] || { echo "FAIL LOUD: $PENDING files differ after copy; source kept" | tee -a "$LOG"; exit 1; }

rm -rf "$SRC"
ln -s "$DST" "$SRC"
echo "SOURCE_REMOVED symlink $SRC -> $DST $(date)" | tee -a "$LOG"
getfattr -n ceph.dir.rbytes /home/nikola.lukic 2>/dev/null | tail -1 | tee -a "$LOG"
echo "ALL_DONE $(date)" | tee -a "$LOG"
