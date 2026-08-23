#!/bin/bash
# Does the interpreter change move end-to-end total_rtt?
#
# Jobs 710315/710341 ran their simulations in the rogue pipenv venv (torch 2.12.0+cu130)
# because run_simulation.py spawned `pipenv run python` regardless of HEROSIM_PY. Since
# that was fixed (2026-08-23) later gates run in the declared `gnn` env (torch
# 2.5.1+cu121). PARITY.md measured the library axis at exactly 0.0 |delta| on LOGITS, but
# never on end-to-end `total_rtt` across a full 301k-task trace, where a single flipped
# decision could compound.
#
# This re-runs ONE cell of the completed drawgate backbone/deployed arm under the fixed
# interpreter, into a separate sweep dir, so the two numbers can be diffed directly. If
# they agree inside the 0.1-0.4% noise floor, results produced before and after the fix
# are comparable and nothing needs re-running.
#SBATCH --job-name=interp-delta
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/interp-delta-%j.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/interp-delta-%j.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00

set -euo pipefail
cd /home/nikola.lukic/gnn-herosim

if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TORCH_NUM_THREADS=4

CELL=cell01_p25_s9001
export SWEEP_DIR="simulation_data/normal_sim_sweeps/interpdelta_backbone_deployed"
export GNN_MODEL="models/near-rtt-v2-full-corpus-siv1-dim14-ce-only.pt"
export WORKLOAD="data/nofs-ids/traces/workload-150-100.json"
export INFERENCE_FEATURE_LAYOUT=dim22
export PARITY_EXTRA_ARGS="--allow-backbone-latency-divergence"
SRC_SWEEP="simulation_data/normal_sim_sweeps/a1_backbone_bw1p5"

mkdir -p "${SWEEP_DIR}/results" "${SWEEP_DIR}/cell_infrastructure" "${SWEEP_DIR}/configs" logs
for d in cell_infrastructure configs; do
  rsync -a "${SRC_SWEEP}/${d}/" "${SWEEP_DIR}/${d}/"
done

bash scripts_cosim/important/run_full_corpus_siv1_live_gate.sh gnn "$CELL"

python3 - <<'PY'
import json
old = json.load(open("simulation_data/normal_sim_sweeps/drawgate_backbone_deployed/results/cell01_p25_s9001_s0_gnn.json"))
new = json.load(open("simulation_data/normal_sim_sweeps/interpdelta_backbone_deployed/results/cell01_p25_s9001_s0_gnn.json"))
o, n = old["total_rtt"], new["total_rtt"]
oe = old["run_provenance"]["python_env"]
ne = new["run_provenance"]["python_env"]
print(f"old torch={oe.get('torch')} total_rtt={o:,.1f}")
print(f"new torch={ne.get('torch')} total_rtt={n:,.1f}")
d = 100.0 * (n - o) / o
print(f"delta = {d:+.4f}%  (noise floor 0.1-0.4%)")
print("VERDICT:", "COMPARABLE — inside the noise floor" if abs(d) <= 0.4
      else "NOT COMPARABLE — the interpreter moves end-to-end total_rtt")
PY
