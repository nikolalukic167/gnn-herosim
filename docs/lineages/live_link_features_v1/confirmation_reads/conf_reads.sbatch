#!/bin/bash
#SBATCH -J s5-llf-conf-reads
#SBATCH -p CPU-amd
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 00:30:00
#SBATCH -o /share/nikola.lukic/uncut_groups/anomaly/fairshare/d_conf_reads/log-%j.out
# live_link_features_v1 CONFIRMATION reads, ONCE EACH. Reader llf_confirm_read.py at reference-physics f16c359a (blob bdce5ed3), rulings confirm_rulings.json from the same checkout. Never pooled.
set -uo pipefail
eval "$(micromamba shell hook --shell bash)" && micromamba activate gnn
RW=$HOME/gnn-herosim-wt/s5_reader_f16c359a; F=/share/nikola.lukic/uncut_groups/anomaly/fairshare; O=$F/d_conf_reads
[[ $(git -C $RW rev-parse HEAD) == f16c359a* && $(git -C $RW status --porcelain | wc -l) == 0 ]] || { echo "FAIL LOUD: reader worktree"; exit 2; }
[[ $(git -C $RW rev-parse HEAD:scripts_cosim/llf_confirm_read.py) == bdce5ed3* ]] || { echo "FAIL LOUD: reader blob"; exit 2; }
R=$RW/docs/lineages/live_link_features_v1/confirm_rulings.json; md5sum $R
echo "=== PRIMARY (epoch-40, option A)"; python3 $RW/scripts_cosim/llf_confirm_read.py --cells $F/d_conf_primaryA --pin static=02668ecf,refs=0ec7cdcb,bar=0ec7cdcb --rulings $R --out $O/read_primary_A_epoch40.json; echo "rc=$?"
echo "=== SECONDARY (best-val)"; python3 $RW/scripts_cosim/llf_confirm_read.py --cells $F/d_conf --pin 0ec7cdcb --rulings $R --out $O/read_secondary_bestval.json; echo "rc=$?"
md5sum $O/*.json
