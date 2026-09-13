#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/projects/my-herosim"
OUT_ROOT="${1:-simulation_data/normal_sim_sweeps/gnn_near_rtt_v2_soft_combo_conc_5cfg_frozen_topk_20260608}"

"${ROOT}/scripts_cosim/important/run_gnn_near_rtt_v2_5cfg_sweep_common.sh" \
  "$OUT_ROOT" \
  "models/near-rtt-v2-clean-1230-soft-combo-conc.pt" \
  "gnn_near_rtt_v2_soft_combo_conc" \
  "frozen_topk" \
  "10"
