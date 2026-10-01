#!/bin/bash
# raw_plan_v2 Phase C (docs/lineages/raw_plan_v2.md): mint the unseen confirmation topologies LO-HI (9473-9520; Amendment
# A1 extension 9521-9568), screen them
# under fresh_topo_burst_v1's admission rule, and select the 12 lowest admitted ids.
# - Mint: cluster_scale_v1_cells.py, base cs6s9001.json, 40 clients, seed = id, as joint_burst_v1_mint.sbatch did.
#   Generator identity is witnessed by re-minting 9409 and 9470 and comparing them byte-for-byte with the
#   fresh_topo_burst_v1 cells.
# - Unused: no file anywhere under simulation_data names an id in the range, and none is a training or development id.
# - Screen: reactive + batched on burst windows w0-w3 at x1 (phase rp2screen), 600 s timeout as Amendment A1.
# - Select: fresh_topo_burst_v1_read.py select --pool rp2c. Fewer than 12 admitted -> DESIGN-SHORT, exit 1.
#   sbatch --export=ALL,WT=<grounded worktree>[,LO=9521,HI=9568] raw_plan_v2_mint_screen.sbatch
# Selection is over the whole Phase C pool (RP2C_POOL); finished screen runs are never re-run.
#SBATCH --job-name=rp2-screen
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/rp2-screen-%j.out
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=12:00:00
set -euo pipefail
WT=${WT:-/home/nikola.lukic/gnn-herosim-wt/grounded}
BASE=/home/nikola.lukic/gnn-herosim/simulation_data/peer_affinity_live_gate/configs/cs6s9001.json
FRESH=/home/nikola.lukic/gnn-herosim/simulation_data/cd_gap_v1/gates/fresh
C=/home/nikola.lukic/gnn-herosim/simulation_data/raw_plan_v2/confirm
P=${PAR:-$(( ${SLURM_CPUS_PER_TASK:-64} - 4 ))}
LO=${LO:-9473}; HI=${HI:-9520}
[[ ( $LO == 9473 && $HI == 9520 ) || ( $LO == 9521 && $HI == 9568 ) ]] || { echo "FAIL LOUD: LO-HI=$LO-$HI is not a registered range"; exit 1; }
CELLS="$C/cells_$LO.json"; [[ $LO == 9473 ]] && CELLS="$C/cells.json"
cd "$WT"
if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3 PYTHONPATH="$WT"
python3 -c "import sys,torch;print('[INTERP]',sys.executable,torch.__version__)"
echo "HEAD $(git rev-parse HEAD)  node $(hostname)  P=$P"
[[ -z "$(git status --porcelain -- src scripts_cosim experiments run_experiment.py)" ]] || { echo "FAIL LOUD: worktree dirty"; exit 1; }

HITS=""
for t in $(seq $LO $HI); do
  [[ -f "$CELLS" ]] && break
  H=$(find /home/nikola.lukic/gnn-herosim/simulation_data -path "$C" -prune -o -type f -name "*s$t[_.]*" -print -quit)
  [[ -z "$H" ]] || HITS="$HITS $H"
done
[[ -z "$HITS" ]] || { echo "FAIL LOUD: ids $LO-$HI already in use:$HITS"; exit 1; }
python3 - "$LO" "$HI" <<'PY'
import json, sys
lo, hi = int(sys.argv[1]), int(sys.argv[2])
split = json.load(open("experiments/backlog_corpus_v1_split.json"))
dev = json.load(open("/home/nikola.lukic/gnn-herosim/simulation_data/cd_gap_v1/gates/fresh/selected.json"))["topologies"]
used = set(dev) | set(range(9201, 9225))
txt = json.dumps(split)
for t in range(lo, hi + 1):
    assert t not in used and f"s{t}" not in txt, t
print(f"[unused] {lo}-{hi} disjoint from training (9201-9224) and development", sorted(dev))
PY

mkdir -p "$C/cfg" "$C/witness"
[[ -e "$C/wl" ]] || ln -s "$FRESH/wl" "$C/wl"
if [[ ! -f "$CELLS" ]]; then
  python3 scripts_cosim/cluster_scale_v1_cells.py --base "$BASE" --sim-inputs data/nofs-ids --vary clients --rungs 40 \
    --seeds 9409,9470 --out-dir "$C/witness" --manifest "$C/witness/cells_$LO.json" --prefix cc
  for t in 9409 9470; do
    cmp "$C/witness/cc40s$t.json" "$FRESH/cfg/cc40s$t.json" || { echo "FAIL LOUD: re-mint of $t differs from the fresh pool"; exit 1; }
  done
  echo "[mint] witness: 9409 and 9470 re-mint byte-identical"
  python3 scripts_cosim/cluster_scale_v1_cells.py --base "$BASE" --sim-inputs data/nofs-ids --vary clients --rungs 40 \
    --seeds "$(seq -s, $LO $HI)" --out-dir "$C/cfg" --manifest "$CELLS.partial" --prefix cc
  mv "$CELLS.partial" "$CELLS"
fi
python3 - "$C/cfg" "$BASE" "$LO" "$HI" <<'PY'
import json, sys
d, base = sys.argv[1], json.load(open(sys.argv[2]))
for s in range(int(sys.argv[3]), int(sys.argv[4]) + 1):
    c = json.load(open(f"{d}/cc40s{s}.json"))
    assert int(c["nodes"]["server_nodes"]["count"]) == 6 and int(c["nodes"]["client_nodes"]["count"]) == 40, s
    assert c["scheduler"] == base["scheduler"], s
    assert float(c["network"]["topology"]["connection_probability"]) == 0.6 and float(c["network"]["backbone"]["bandwidth_mbps"]) == 1000.0
    assert int(c["network"]["topology"]["seed"]) == s
print(f"[mint] {sys.argv[3]}-{sys.argv[4]} verified")
PY

python3 scripts_cosim/fresh_topo_burst_v1_gate.py rp2screen --inputs "$C" --out "$C/screen" \
  --parallel "$P" --mem 4G --timeout 600 --no-scope
[[ -f "$C/selected.json" ]] && cp -n "$C/selected.json" "$C/selected_before_$LO.json"
python3 scripts_cosim/fresh_topo_burst_v1_read.py select --screen "$C/screen" --pool rp2c --out "$C/selected.json"
python3 - "$C/selected.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
n = len(r.get("topologies") or [])
if r.get("verdict") != "DESIGN-READY" or n < 12:
    raise SystemExit(f"FAIL LOUD: DESIGN-SHORT for raw_plan_v2 Phase C: {n} admitted of {r['n_candidates']} (need 12)")
print("[select] Phase C study:", r["topologies"])
PY
echo "=== rp2 mint + screen + select done ==="
