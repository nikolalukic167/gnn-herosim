#!/bin/bash
# Submits the xs queue (xs/queue.txt) one job at a time under a node throttle; resumable (xs/submitted.txt). Rules: >= RESERVE idle eligible nodes must remain after each submission,
# never os-cpu-slurm-10/13-17/21, queue <= 40 jobs of mine, short exclusive jobs with --nice so fl3 jobs go first; it holds while any fl3 job is queued, and 00:45-03:00 while the fl3 learned cells have not started.
set -uo pipefail
U=/share/nikola.lukic/uncut_groups; X=$U/xs; Q=$X/queue.txt; S=$X/submitted.txt; touch $S
RESERVE=${RESERVE:-4}; EXC=os-cpu-slurm-10,os-cpu-slurm-13,os-cpu-slurm-14,os-cpu-slurm-15,os-cpu-slurm-16,os-cpu-slurm-17,os-cpu-slurm-21
idle() { n=0; for x in $(sinfo -p CPU-amd -h -N -t idle -o "%N"); do case ",$EXC," in *",$x,"*) continue;; esac; [ "$(squeue -w $x -h | wc -l)" = 0 ] && n=$((n+1)); done; echo $n; }
fl3busy() { [ -n "$(squeue -u nikola.lukic -h -o %j | grep -E '^s5-fl3|^s5-s64-twin|^s6-fl3')" ]; }
while true; do
  next=""
  while IFS= read -r ln; do nm=${ln#NAME=}; nm=${nm%% *}; grep -q "^$nm|" $S || { next="$nm|$ln"; break; }; done < $Q
  [ -n "$next" ] || { echo "queue drained"; exit 0; }
  hm=$(date +%H%M)
  if fl3busy || { [ $((10#$hm)) -ge 45 ] && [ $((10#$hm)) -lt 300 ] && [ ! -e $U/fl3/cells_learned ]; }; then sleep 120; continue; fi
  [ $(squeue -u nikola.lukic -h | wc -l) -le 40 ] || { sleep 60; continue; }
  [ $(idle) -gt $RESERVE ] || { sleep 60; continue; }
  name=${next%%|*}; spec=${next#*|}
  declare -A K=(); for kv in $spec; do K[${kv%%=*}]=${kv#*=}; done
  [[ $(git -C ${K[WT]} rev-parse HEAD) == ${K[HEAD]}* ]] || { echo "FAIL LOUD: ${K[WT]} not at ${K[HEAD]}"; exit 2; }
  j=$(env IN=${K[IN]} SELECT=${K[SELECT]} R1A_TOPOS=${K[TOPOS]} R1A_WINDOWS=${K[WINDOWS]} WF1_RUNGS=${K[RUNGS]} PAR=${K[PAR]} CELL_MEM=16G LIMIT=14400 R1A_SPLIT=uncut_arm1_split.json R1A_SLATE=declared_pruning_v1_uncapped \
      R1A_SEEDS=${K[SEEDS]} WT=${K[WT]} EXPECT_HEAD=${K[HEAD]} OUT=${K[OUT]} R1A_ARMS=${K[ARMS]} R1A_SHARD=${K[SHARD]} \
      sbatch --parsable --export=ALL -p CPU-amd --exclude=$EXC --exclusive --nodes=1 --mem=0 --nice=1000 -t ${K[WALL]} -J xs-$name -o $X/logs/%x-%j.out ${K[WT]}/scripts_cosim/datalab/r1a_gate.sbatch)
  echo "$name|$j|$(date +%H:%M:%S)" >> $S; echo "$(date +%H:%M:%S) submitted $name job $j (idle eligible before: ok)"
  sleep 20
done
