#!/bin/bash
#SBATCH --job-name=twin-read-primary
#SBATCH --partition=CPU-amd
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --exclude=os-cpu-slurm-21,os-cpu-slurm-15
#SBATCH --output=/share/nikola.lukic/uncut_groups/fl3/reads/twin-read-primary-%j.out
# FL MP-OFF twin, the PRIMARY read (frozen P9: s6/reader-twin a3c5304e, blob 04e368af, frozen at reference-physics b12d9f3f).
# After FL + lambda's primary (md5 e6cbb39d, Record b74ec353) and secondary are filed. Opens cells_twin (384) and FL gnn / the bars
# in cells_learned / cells_bar. Written once; its md5 is filed before the twin's secondary opens anything.
set -uo pipefail
eval "$(micromamba shell hook --shell bash)" && micromamba activate gnn
U=/share/nikola.lukic/uncut_groups; F=$U/fl3; REPO=$HOME/gnn-herosim
RD=$HOME/gnn-herosim-wt/s6_reader_a3c5304e; cd $RD
[[ $(git rev-parse HEAD) == a3c5304e* && -z "$(git status --porcelain)" ]] || { echo "FAIL LOUD: reader HEAD"; exit 1; }
[[ $(git rev-parse HEAD:scripts_cosim/uncut_arm_gate_read.py) == 04e368af* ]] || { echo "FAIL LOUD: reader blob"; exit 1; }
[[ ! -e $F/reads/read_twin_primary.json ]] || { echo "FAIL LOUD: a twin primary read exists; one read only"; exit 1; }
[[ $(md5sum < $F/reads/read_fll_primary.json | cut -c1-32) == e6cbb39db912e4705a27852bb28d844e ]] || { echo "FAIL LOUD: FL + lambda primary md5"; exit 1; }
[[ $(md5sum < $F/md5_twin.json | cut -c1-32) == d294f97dde3c46a4a75fbf8ede440f3e ]] || { echo "FAIL LOUD: md5_twin.json"; exit 1; }
git -C $REPO fetch -q origin || { echo "FAIL LOUD: fetch"; exit 1; }
FLLREC=b74ec353ba28a533892af0e5bd03cd290a5bded6
git -C $REPO merge-base --is-ancestor 73d9708fa335bcf00eb7d9fa1f221b4f511b29e0 origin/reference-physics || { echo "FAIL LOUD: FL + lambda secondary not merged yet"; exit 1; }
T=$(seq -s, 30149 30160); H=f6c5af0a8f74a8caa83ba45a97903ec47798ae8898e8e226b3ff22291bb59379
echo "cells: twin $(ls $F/cells_twin/*.summary.json | wc -l) summaries; failed $(ls $F/cells_twin/*.failed.json 2>/dev/null | wc -l)"
python3 scripts_cosim/uncut_arm_gate_read.py --gate $F/cells_learned,$F/cells_bar --twin-gate $F/cells_twin --topos $T --topos-sha256 $H --repo $REPO \
  --pin a459dd98 --gate-change 095948ea,f6818d09,74ddf70e,b8909f58,05ec45c1 --w-pins 91269ec0,f8a55ff0 \
  --bar cd_ib0_inflightfull --cotenancy required --primary ra_gnn_eng_combfl_nosplit --twin ra_twin_eng_combfl_nosplit \
  --checkpoint-md5 $U/comb/md5_comball.json,$F/md5_twin.json --seeds 1,2 --suffix _k100 \
  --fll-primary-read $F/reads/read_fll_primary.json --fll-record $FLLREC \
  --confirm-mp-twin --stage primary \
  --out $F/reads/read_twin_primary.json > $F/reads/read_twin_primary.stdout 2>&1
rc=$?
if [[ $rc != 0 || ! -s $F/reads/read_twin_primary.json ]]; then
  echo "PRIMARY READ FAILED rc=$rc"; tail -5 $F/reads/read_twin_primary.stdout; exit 1
fi
echo "[primary] read_twin_primary.json md5 $(md5sum < $F/reads/read_twin_primary.json | cut -c1-32)"
echo "[primary] stdout md5 $(md5sum < $F/reads/read_twin_primary.stdout | cut -c1-32)"
cat $F/reads/read_twin_primary.stdout
