#!/bin/bash
# Move cold data off the 250 GiB /home quota to /share/nikola.lukic: copy, verify file count and bytes, then replace
# the home copy with a symlink. Nothing is deleted unless the copy matches.
#   sbatch scripts_cosim/datalab/offload_to_share.sbatch simulation_data/<dir> [more dirs, relative to ~/gnn-herosim]
#   LOGS=1 sbatch ...   also move ~/gnn-herosim/logs files older than a day to /share/nikola.lukic/logs_archive/gnn-herosim
#SBATCH --job-name=offload-share
#SBATCH --output=/share/nikola.lukic/offload_%j.log
#SBATCH --partition=CPU-amd
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=12:00:00
set -euo pipefail
H=/home/nikola.lukic/gnn-herosim
S=/share/nikola.lukic
stats() { echo "$(find "$1" -type f | wc -l) $(find "$1" -type f -printf '%s\n' | awk '{s+=$1} END {printf "%d", s}')"; }
echo "start $(date)  home bytes=$(getfattr --only-values -n ceph.dir.rbytes /home/nikola.lukic)"
for rel in "$@"; do
  src=$H/$rel; dst=$S/$rel
  if [[ -L $src ]]; then echo "[skip] $rel already a symlink"; continue; fi
  [[ -d $src ]] || { echo "FAIL LOUD: $src is not a directory"; exit 1; }
  [[ -e $dst ]] && { echo "FAIL LOUD: $dst exists"; exit 1; }
  mkdir -p "$(dirname "$dst")"
  rsync -a "$src/" "$dst.partial/"
  a=$(stats "$src"); b=$(stats "$dst.partial")
  [[ "$a" == "$b" ]] || { echo "FAIL LOUD: $rel mismatch home=[$a] share=[$b]"; exit 1; }
  mv "$dst.partial" "$dst"
  rm -rf "$src"; ln -s "$dst" "$src"
  echo "[moved] $rel files/bytes=[$a]"
done
if [[ ${LOGS:-0} == 1 ]]; then
  L=$S/logs_archive/gnn-herosim; mkdir -p "$L"; cd $H/logs
  list=$(mktemp); find . -maxdepth 1 -type f -mtime +1 -printf '%P\n' | sort > "$list"
  rsync -a --files-from="$list" . "$L/"
  while read -r f; do
    [[ -f "$L/$f" && $(stat -c %s "$f") == $(stat -c %s "$L/$f") ]] || { echo "FAIL LOUD: log copy mismatch $f; nothing deleted"; exit 1; }
  done < "$list"
  xargs -d '\n' rm -f < "$list"
  echo "[moved] $(wc -l < "$list") old logs -> $L"
fi
echo "done $(date)  home bytes=$(getfattr --only-values -n ceph.dir.rbytes /home/nikola.lukic)"
