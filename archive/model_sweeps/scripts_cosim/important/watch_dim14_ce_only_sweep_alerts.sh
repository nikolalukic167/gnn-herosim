#!/usr/bin/env bash
# Poll dim14-ce-only sweep results; log ALERT + 4-way comparison when a new config JSON appears.
set -euo pipefail

ROOT="/root/projects/my-herosim"
cd "$ROOT"

SWEEP_RES="${1:-simulation_data/normal_sim_sweeps/gnn_near_rtt_v2_dim14_ce_only_20260609/results}"
STATE_FILE="logs/dim14_ce_only_sweep_seen_configs.txt"
ALERT_LOG="logs/dim14_ce_only_sweep_alerts.log"
POLL_SEC="${POLL_SEC:-30}"

BASELINES=(
  "dim13 CE-only|simulation_data/normal_sim_sweeps/gnn_near_rtt_v2_clean_1230_ce_only_20260608/results"
  "dim13 clean|simulation_data/normal_sim_sweeps/gnn_near_rtt_v2_clean_1230_20260608/results"
  "dim14 full|simulation_data/normal_sim_sweeps/gnn_near_rtt_v2_dim14_1060_20260608/results"
)

mkdir -p logs
touch "$STATE_FILE" "$ALERT_LOG"

log() {
  echo "[$(date -Is)] $*" | tee -a "$ALERT_LOG"
}

compare_config() {
  local cfg="$1"
  pipenv run python3 - <<PY
import json
from pathlib import Path

cfg = "$cfg"
new_p = Path("$SWEEP_RES") / f"{cfg}.json"
new_d = Path("$SWEEP_RES") / f"{cfg}.decode_stats.json"

def rtt(p):
    return json.load(open(p))["total_rtt"]

def decode(p):
    if not p.exists():
        return {}
    q = json.load(open(p)).get("chosen_queue_vs_min", {})
    col = json.load(open(p)).get("intra_batch_platform_collisions", {}).get("collision_batch_rate")
    return {"p95": q.get("p95"), "median": q.get("median"), "mean": q.get("mean"), "collision_pct": (col * 100 if col is not None else None)}

baselines = [
    ("dim14-ce-only", new_p),
$(for entry in "${BASELINES[@]}"; do
  IFS='|' read -r name dir <<< "$entry"
  echo "    (\"$name\", Path(\"$dir\") / f\"{cfg}.json\"),"
done)
]

rows = []
for name, p in baselines:
    if not p.exists():
        rows.append((name, None, {}))
        continue
    dec_p = p.with_suffix(".decode_stats.json")
    rows.append((name, rtt(p), decode(dec_p)))

valid = [(n, v, d) for n, v, d in rows if v is not None]
if not valid:
    print("ERROR: no RTT data for", cfg)
    raise SystemExit(1)

winner = min(valid, key=lambda t: t[1])[0]
new_rtt = next(v for n, v, _ in rows if n == "dim14-ce-only")

print("=" * 72)
print(f"ALERT: config finished — {cfg}")
print(f"dim14-ce RTT: {new_rtt:,.0f}  |  winner: {winner}")
print()
print(f"{'Model':<16} {'total_rtt':>14} {'vs dim14-ce':>12} {'qvm p95':>10} {'qvm med':>8}")
for name, v, d in rows:
    if v is None:
        print(f"{name:<16} {'NA':>14} {'':>12} {'—':>10} {'—':>8}")
        continue
    delta = (v / new_rtt - 1) * 100 if name != "dim14-ce-only" else 0.0
    delta_s = "—" if name == "dim14-ce-only" else f"{delta:+.1f}%"
    p95 = d.get("p95")
    med = d.get("median")
    p95s = f"{p95:,.0f}" if isinstance(p95, (int, float)) else "—"
    meds = f"{med:,.0f}" if isinstance(med, (int, float)) else "—"
    mark = " ◀" if name == winner else ""
    print(f"{name:<16} {v:>14,.0f} {delta_s:>12} {p95s:>10} {meds:>8}{mark}")

print()
wins = sum(1 for n, v, _ in valid if n != "dim14-ce-only" and new_rtt < v)
print(f"dim14-ce-only beats {wins}/{len(valid)-1} baselines on this config")
print("=" * 72)
PY
}

log "Watcher start — sweep=${SWEEP_RES} poll=${POLL_SEC}s"
log "State: ${STATE_FILE}  Alerts: ${ALERT_LOG}"

while true; do
  if [[ ! -d "$SWEEP_RES" ]]; then
    sleep "$POLL_SEC"
    continue
  fi

  for f in "$SWEEP_RES"/*.json; do
    [[ -f "$f" ]] || continue
    [[ "$f" == *.decode_stats.json ]] && continue
    cfg="$(basename "$f" .json)"
    if grep -qxF "$cfg" "$STATE_FILE" 2>/dev/null; then
      continue
    fi
    # wait until file size stable (sim still writing)
    sz1=$(stat -c%s "$f")
    sleep 2
    sz2=$(stat -c%s "$f")
    if [[ "$sz1" != "$sz2" ]]; then
      continue
    fi
    echo "$cfg" >> "$STATE_FILE"
    log "NEW CONFIG DETECTED: ${cfg}"
    compare_config "$cfg" | tee -a "$ALERT_LOG"
  done

  # stop when all 7 standard configs seen
  n_seen=$(wc -l < "$STATE_FILE" | tr -d ' ')
  if [[ "$n_seen" -ge 7 ]]; then
    log "All 7 configs seen — watcher exiting"
    exit 0
  fi

  sleep "$POLL_SEC"
done
