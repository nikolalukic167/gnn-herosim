#!/usr/bin/env bash
# Diagnostic A1: temperature probe on dim14-full, default config only.
# Tests whether logit sharpness alone explains the ranking model's hot-spotting.
#
# Run A: GNN_LOGIT_TEMPERATURE=3.0   (reduces 8-unit margins to ~2.7)
# Run B: GNN_LQB_LAMBDA=1.5          (log1p blend — model-agnostic fix)
# Run C: baseline argmax T=1.0       (reproduced for clean comparison)
#
# Expected: if T=3.0 recovers to ~4.1-4.5M, sharpness is the primary cause.
# If LQB recovers similarly, it validates the decode-time fix independently.

set -euo pipefail
cd "$(dirname "$0")/../.."

MODEL=models/near-rtt-v2-dim14-1060.pt
CONFIG=simulation_data/space_with_network.json
WORKLOAD=data/nofs-ids/traces/workload-100-100.json
OUTDIR=simulation_data/normal_sim_sweeps/gnn_dim14_full_temp_probe_$(date +%Y%m%d_%H%M%S)/results
SEED=42

mkdir -p "$OUTDIR" logs

echo "[probe] dim14-full temperature / LQB probe — $(date)" | tee logs/dim14_full_temp_probe.log

echo "[A] T=1.0 baseline (argmax)" | tee -a logs/dim14_full_temp_probe.log
GNN_MODEL_PATH=$MODEL GNN_DECODE_MODE=argmax GNN_LOGIT_TEMPERATURE=1.0 \
  pipenv run python3 scripts_cosim/run_simulation.py \
    --gnn --config "$CONFIG" --workload "$WORKLOAD" \
    --output "$OUTDIR/dim14_full_T1_argmax.json" --seed $SEED --timeout 1200 \
  2>&1 | tee -a logs/dim14_full_temp_probe.log

echo "[B] T=3.0 (sharpness reduction)" | tee -a logs/dim14_full_temp_probe.log
GNN_MODEL_PATH=$MODEL GNN_DECODE_MODE=argmax GNN_LOGIT_TEMPERATURE=3.0 \
  pipenv run python3 scripts_cosim/run_simulation.py \
    --gnn --config "$CONFIG" --workload "$WORKLOAD" \
    --output "$OUTDIR/dim14_full_T3_argmax.json" --seed $SEED --timeout 1200 \
  2>&1 | tee -a logs/dim14_full_temp_probe.log

echo "[C] LQB lambda=1.5 (log1p blend)" | tee -a logs/dim14_full_temp_probe.log
GNN_MODEL_PATH=$MODEL GNN_DECODE_MODE=argmax GNN_LQB_LAMBDA=1.5 \
  pipenv run python3 scripts_cosim/run_simulation.py \
    --gnn --config "$CONFIG" --workload "$WORKLOAD" \
    --output "$OUTDIR/dim14_full_LQB15_argmax.json" --seed $SEED --timeout 1200 \
  2>&1 | tee -a logs/dim14_full_temp_probe.log

echo "[probe] Extracting RTT summary..." | tee -a logs/dim14_full_temp_probe.log
pipenv run python3 - <<'PY'
import json
from pathlib import Path
import glob

outdir = sorted(Path("simulation_data/normal_sim_sweeps").glob("gnn_dim14_full_temp_probe_*"))[-1] / "results"
ref_ce = json.load(open("simulation_data/normal_sim_sweeps/gnn_near_rtt_v2_dim14_ce_only_20260609/results/default_20_20_p50.json"))["total_rtt"]

print(f"\nReference dim14-ce-only: {ref_ce:,.0f}")
print(f"{'Run':<30} {'RTT':>14} {'vs dim14-ce':>12} {'qvm_p95':>10}")
for f in sorted(outdir.glob("*.json")):
    if "decode_stats" in f.name: continue
    rtt = json.load(open(f))["total_rtt"]
    ds = outdir / f.name.replace(".json", ".decode_stats.json")
    p95 = json.load(open(ds)).get("chosen_queue_vs_min", {}).get("p95", "?") if ds.exists() else "?"
    print(f"{f.stem:<30} {rtt:>14,.0f} {(rtt/ref_ce-1)*100:>+11.1f}% {str(p95):>10}")
PY

echo "[probe] DONE — $(date)" | tee -a logs/dim14_full_temp_probe.log
