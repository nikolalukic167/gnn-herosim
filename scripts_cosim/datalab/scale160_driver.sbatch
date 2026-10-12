#!/bin/bash
# Runs scale160_corpus_submit.sh as a 1-CPU job (datalab-pitfalls 16: never a nohup watcher on the login node). It occupies one queue
# slot, which the driver's own squeue count includes. Pass the driver's environment (PHASE, WT, ROOT, ...) with export + --export=ALL.
#   export PHASE=capture WT=... ROOT=... M_MODERATE=... M_HEAVY=... [INPUTS_DONE=1 START_OFFSET=30]; sbatch --export=ALL scale160_driver.sbatch
#SBATCH --job-name=s160-drv
#SBATCH --partition=CPU-amd
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/s160-drv-%j.out
set -euo pipefail
WT=${WT:?FAIL LOUD: set WT}
exec bash "$WT/scripts_cosim/datalab/scale160_corpus_submit.sh"
