#!/usr/bin/env bash
# Alert watcher for dim14-full LQB sweep.
SWEEP_DIR="simulation_data/normal_sim_sweeps/gnn_near_rtt_v2_dim14_1060_lqb15_20260609_100843/results"
CE_DIR="simulation_data/normal_sim_sweeps/gnn_near_rtt_v2_dim14_ce_only_20260609/results"
FULL_DIR="simulation_data/normal_sim_sweeps/gnn_near_rtt_v2_dim14_1060_20260608/results"
ALERT_LOG="logs/dim14_lqb_sweep_alerts.log"
SEEN="logs/dim14_lqb_sweep_seen.txt"
touch "$SEEN"
mkdir -p logs
echo "[watcher] LQB sweep watcher started $(date -Is)" | tee -a "$ALERT_LOG"
while true; do
  for f in "$SWEEP_DIR"/*.json; do
    [[ -f "$f" ]] || continue
    [[ "$f" == *decode_stats* ]] && continue
    name="$(basename "$f" .json)"
    grep -qxF "$name" "$SEEN" && continue
    echo "$name" >> "$SEEN"
    lqb=$(python3 -c "import json; print(f\"{json.load(open('$f'))['total_rtt']/1e6:.3f}\")" 2>/dev/null || echo "?")
    ce_f="$CE_DIR/${name}.json"
    full_f="$FULL_DIR/${name}.json"
    ce=$(python3 -c "import json; print(f\"{json.load(open('$ce_f'))['total_rtt']/1e6:.3f}\")" 2>/dev/null || echo "?")
    full=$(python3 -c "import json; print(f\"{json.load(open('$full_f'))['total_rtt']/1e6:.3f}\")" 2>/dev/null || echo "?")
    done_count=$(grep -c . "$SEEN")
    echo "" | tee -a "$ALERT_LOG"
    echo "## ALERT: LQB config=${name} (${done_count}/7)" | tee -a "$ALERT_LOG"
    echo "| model | RTT (M) |" | tee -a "$ALERT_LOG"
    echo "|-------|---------|" | tee -a "$ALERT_LOG"
    echo "| dim14-full LQB λ=1.5 | **${lqb}** |" | tee -a "$ALERT_LOG"
    echo "| dim14-ce-only argmax  | ${ce} |" | tee -a "$ALERT_LOG"
    echo "| dim14-full argmax     | ${full} |" | tee -a "$ALERT_LOG"
  done
  sleep 30
done
