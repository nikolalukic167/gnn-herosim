#!/bin/bash
#SBATCH --job-name=twin-read-secondary
#SBATCH --partition=CPU-amd
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --exclude=os-cpu-slurm-21,os-cpu-slurm-15
#SBATCH --output=/share/nikola.lukic/uncut_groups/fl3/reads/twin-read-secondary-%j.out
# FL MP-OFF twin, the descriptive SECONDARY read (frozen P9: s6/reader-twin a3c5304e, blob 04e368af). After the twin's primary
# (md5 ea2e91ba, Record 4fd4d6ba) is on origin/reference-physics. Opens fl3/cells_desc's FL MLP s1-2, size64/cells_twin (192) and
# size-64's FL gnn s1-2 _k100 and bar (288). Written once.
set -uo pipefail
eval "$(micromamba shell hook --shell bash)" && micromamba activate gnn
U=/share/nikola.lukic/uncut_groups; F=$U/fl3; REPO=$HOME/gnn-herosim
RD=$HOME/gnn-herosim-wt/s6_reader_a3c5304e; cd $RD
[[ $(git rev-parse HEAD) == a3c5304e* && -z "$(git status --porcelain)" ]] || { echo "FAIL LOUD: reader HEAD"; exit 1; }
[[ $(git rev-parse HEAD:scripts_cosim/uncut_arm_gate_read.py) == 04e368af* ]] || { echo "FAIL LOUD: reader blob"; exit 1; }
[[ ! -e $F/reads/read_twin_secondary.json ]] || { echo "FAIL LOUD: a twin secondary read exists; one read only"; exit 1; }
[[ $(md5sum < $F/reads/read_twin_primary.json | cut -c1-32) == ea2e91ba00b043c2955584e27635ccdc ]] || { echo "FAIL LOUD: twin primary md5"; exit 1; }
[[ $(md5sum < $F/reads/read_fll_primary.json | cut -c1-32) == e6cbb39db912e4705a27852bb28d844e ]] || { echo "FAIL LOUD: FL + lambda primary md5"; exit 1; }
[[ $(md5sum < $F/md5_twin.json | cut -c1-32) == d294f97dde3c46a4a75fbf8ede440f3e ]] || { echo "FAIL LOUD: md5_twin.json"; exit 1; }
git -C $REPO fetch -q origin || { echo "FAIL LOUD: fetch"; exit 1; }
FLLREC=b74ec353ba28a533892af0e5bd03cd290a5bded6
TWREC=4fd4d6ba934e9ace19671e327e45017029bbe8e9
git -C $REPO merge-base --is-ancestor $TWREC origin/reference-physics || { echo "FAIL LOUD: the twin primary Record line is not merged yet"; exit 1; }
S64=$U/size64
T=$(seq -s, 30149 30160); H=f6c5af0a8f74a8caa83ba45a97903ec47798ae8898e8e226b3ff22291bb59379
echo "cells: size64 twin $(ls $S64/cells_twin/*.summary.json | wc -l) summaries; failed $(ls $S64/cells_twin/*.failed.json 2>/dev/null | wc -l)"
python3 scripts_cosim/uncut_arm_gate_read.py --gate $F/cells_learned,$F/cells_bar --twin-gate $F/cells_twin --topos $T --topos-sha256 $H --repo $REPO \
  --pin a459dd98 --gate-change 095948ea,f6818d09,74ddf70e,b8909f58,05ec45c1 --w-pins 91269ec0,f8a55ff0 \
  --bar cd_ib0_inflightfull --cotenancy required --primary ra_gnn_eng_combfl_nosplit --twin ra_twin_eng_combfl_nosplit \
  --checkpoint-md5 $U/comb/md5_comball.json,$F/md5_twin.json --seeds 1,2 --suffix _k100 \
  --fll-primary-read $F/reads/read_fll_primary.json --fll-record $FLLREC \
  --confirm-mp-twin --stage secondary --mlp ra_mlp_same_combfl_nosplit --desc-gate $F/cells_desc \
  --desc64-gate $S64/cells_twin --s64-gate $S64/cells_bar,$S64/cells_learned \
  --primary-read $F/reads/read_twin_primary.json --record $TWREC \
  --out $F/reads/read_twin_secondary.json > $F/reads/read_twin_secondary.stdout 2>&1
rc=$?
if [[ $rc != 0 || ! -s $F/reads/read_twin_secondary.json ]]; then
  echo "SECONDARY READ FAILED rc=$rc"; tail -5 $F/reads/read_twin_secondary.stdout; exit 1
fi
echo "[secondary] read_twin_secondary.json md5 $(md5sum < $F/reads/read_twin_secondary.json | cut -c1-32)"
echo "[secondary] stdout md5 $(md5sum < $F/reads/read_twin_secondary.stdout | cut -c1-32)"
cat $F/reads/read_twin_secondary.stdout
