#!/bin/bash
# usage: determinism_cell.sh <topo> <rung> <outdir>   (I8: two runs plus a PYTHONHASHSEED=1 run of one cell; wraps run_determinism.sh with run_cell.sh)
# env as run_cell.sh. Run over det_cells.txt: xargs -P 6 -L1 bash -c './determinism_cell.sh $0 $1 OUT' < det_cells.txt; then check_invariants.py determinism --pair <a> <b> ...
D=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); mkdir -p "$3"
bash "$D/run_determinism.sh" "$D/run_cell.sh" $1 $2 12000 "$(realpath "$3")"
