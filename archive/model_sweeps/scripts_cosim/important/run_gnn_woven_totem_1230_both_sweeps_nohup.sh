#!/usr/bin/env bash
# Run woven-totem-1230 argmax (resume) then seqblend m2 — sequential on one GPU.
set -euo pipefail

ROOT="/root/projects/my-herosim"
cd "$ROOT"

sed -i 's/\r$//' scripts_cosim/important/run_gnn_woven_totem_1230_all_configs_nohup.sh
sed -i 's/\r$//' scripts_cosim/important/run_gnn_woven_totem_1230_seqblend_m2_all_configs_nohup.sh

echo "[$(date -Is)] === woven-totem-1230 sweep 1/2: argmax (resume) ==="
bash scripts_cosim/important/run_gnn_woven_totem_1230_all_configs_nohup.sh \
  simulation_data/normal_sim_sweeps/gnn_woven_totem_1230_20260608

echo "[$(date -Is)] === woven-totem-1230 sweep 2/2: seqblend m2 ==="
bash scripts_cosim/important/run_gnn_woven_totem_1230_seqblend_m2_all_configs_nohup.sh \
  simulation_data/normal_sim_sweeps/gnn_woven_totem_1230_seqblend_m2_20260608

echo "[$(date -Is)] === woven-totem-1230 both sweeps complete ==="
