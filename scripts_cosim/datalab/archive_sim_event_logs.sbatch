#!/bin/bash
# Archive the giant per-event SLURM .err logs off the 250 GiB /home quota.
#
# Why they exist at all (fixed in-tree 2026-09-13, src/placement/simulation.py): `basicConfig`
# is a no-op once the root logger has a handler, and `prepare_graphs_cache` calls it at module
# IMPORT -- which loading a prefix-conditioned checkpoint triggers. The simulator's intended
# "ERROR only, on stdout" config was therefore discarded in silence and every per-event
# logging.info went to stderr: ~390 MB per 450,729-task arm. The H5 gate (102 arms) wrote
# ~40 GB on top of ~130 GB already there, hit the quota, and 128 arms died with exit 1 and no
# traceback -- the traceback could not be written either.
#
# These files are pure per-event simulation trace: every gate's evidence is its .summary.json
# and its .out banner, both tiny and untouched here. Archived rather than deleted, same
# protocol as the 2026-09-11 offload: copy, verify with a checksum dry run, then remove.
#
#SBATCH --job-name=archive-simlogs
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/archive-simlogs-%j.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/archive-simlogs-%j.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=12:00:00

set -euo pipefail
cd /home/nikola.lukic/gnn-herosim
SRC=logs
DST=/share/nikola.lukic/herosim_archive/sim_event_logs
LOG=/share/nikola.lukic/herosim_archive/offload_sim_event_logs_$(date +%Y%m%d).log
MIN_MB=${MIN_MB:-64}
mkdir -p "$DST"

echo "QUOTA_BEFORE $(getfattr -n ceph.dir.rbytes /home/nikola.lukic 2>/dev/null | tail -1)" | tee -a "$LOG"
du -sh "$SRC" | tee -a "$LOG"

# Only files that are actually big, and never the log this job is writing.
mapfile -t BIG < <(find "$SRC" -maxdepth 1 -type f -name '*.err' -size +${MIN_MB}M \
                     ! -name "archive-simlogs-${SLURM_JOB_ID}.err" | sort)
echo "CANDIDATES ${#BIG[@]} files over ${MIN_MB}MB" | tee -a "$LOG"
[[ ${#BIG[@]} -gt 0 ]] || { echo "nothing to archive"; exit 0; }

printf '%s\n' "${BIG[@]}" > /tmp/archive_simlogs_$$.list
echo "COPY_START $(date)" | tee -a "$LOG"
rsync -a --files-from=/tmp/archive_simlogs_$$.list --info=stats1 . "$DST/" | tail -12 | tee -a "$LOG"
echo "COPY_DONE $(date)" | tee -a "$LOG"

PENDING=$(rsync -a --checksum --dry-run --itemize-changes --files-from=/tmp/archive_simlogs_$$.list . "$DST/" | grep -c '^>f' || true)
echo "VERIFY_DONE pending_files=$PENDING" | tee -a "$LOG"
[[ "$PENDING" == "0" ]] || { echo "FAIL LOUD: $PENDING files differ after copy; source kept" | tee -a "$LOG"; exit 1; }

xargs -a /tmp/archive_simlogs_$$.list rm -f
rm -f /tmp/archive_simlogs_$$.list
echo "SOURCE_REMOVED $(date)" | tee -a "$LOG"
du -sh "$SRC" | tee -a "$LOG"
echo "QUOTA_AFTER $(getfattr -n ceph.dir.rbytes /home/nikola.lukic 2>/dev/null | tail -1)" | tee -a "$LOG"
echo "ALL_DONE $(date)" | tee -a "$LOG"
