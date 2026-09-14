#!/bin/bash
# drainable_regime_v1 -- S0, step 1: build the load ladder (docs/lineages/drainable_regime_v1.md).
# Four rescaled copies of the gate trace. Only the timestamp scale and the event count move:
# same events, types, peer groups, 200 MB payloads, demand scales, seed.
#   sbatch scripts_cosim/datalab/drainable_regime_v1_s0_traces.sbatch
#
#SBATCH --job-name=drain-s0-traces
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/drain-s0-traces-%j.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/drain-s0-traces-%j.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=02:00:00

set -euo pipefail
cd /home/nikola.lukic/gnn-herosim
if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3 PYTHONPATH="$PWD" PYTHONUNBUFFERED=1
echo "[CODE] $(git describe --dirty --always) on $(git rev-parse --abbrev-ref HEAD)"

GATE=simulation_data/peer_affinity_live_gate
SRC=$GATE/workloads/workload-150-150-peer_p2_x200.json
N=50000
FACTORS="${DRAIN_FACTORS:-300 1000 2000 4000}"
[[ -f "$SRC" ]] || { echo "FAIL LOUD: source trace missing: $SRC" >&2; exit 1; }
echo "[INPUTS] source $(md5sum "$SRC" | cut -c1-12)"

for F in $FACTORS; do
  OUT=$GATE/workloads/drainable_f${F}_n${N}.json
  if [[ -f "$OUT" ]]; then echo "[skip] exists: $OUT"; continue; fi
  python3 scripts_cosim/rescale_workload_arrivals.py \
    --workload "$SRC" --factor "$F" --max-events "$N" --group-size 10 --output "$OUT.partial"
  mv "$OUT.partial" "$OUT"
done

for F in $FACTORS; do
  OUT=$GATE/workloads/drainable_f${F}_n${N}.json
  [[ -s "$OUT" ]] || { echo "FAIL LOUD: trace not built: $OUT" >&2; exit 1; }
  python3 - "$OUT" <<'PY'
import json, sys
d = json.loads(open(sys.argv[1]).read())
r = d["arrival_rescale"]
t = r["truncation"]
assert len(d["events"]) == t["max_events"], "FAIL LOUD: event count does not match the cut"
assert max(max(p[0], p[1]) for p in d["peer_exchange"]) < len(d["events"]), \
    "FAIL LOUD: a peer pair reaches past the cut"
print(f"[trace] {sys.argv[1]}: {len(d['events'])} events, "
      f"{r['rate_after_per_s']:.3f} arrivals/s (x{r['factor']:.0f}), "
      f"span {r['span_after_s']:.0f} s, peer pairs {t['peer_pairs_after']}")
PY
done
echo "=== S0 traces done ==="
