#!/bin/bash
# joint_burst_v1 -- mint the extended burst topology pool. docs/lineages/joint_burst_v1.md.
#
# 16 more study candidates (9109-9124) and 24 training topologies (9201-9224), 6 servers / 40
# clients, from the same base every 6-server cell of this programme was minted from. The burst
# workloads already exist (env_lever_v1_mint.py burst). One task, minutes.
#
#SBATCH --job-name=jb1-mint
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/jb1-mint-%j.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/jb1-mint-%j.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00

set -euo pipefail
cd /home/nikola.lukic/gnn-herosim
if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3
export PYTHONPATH="$PWD"
export PYTHONUNBUFFERED=1
echo "[CODE] $(git describe --dirty --always) on $(git rev-parse --abbrev-ref HEAD)"

GATE=simulation_data/peer_affinity_live_gate
BASE="$GATE/configs/cs6s9001.json"
JB=simulation_data/joint_burst_v1
MANIFEST=$JB/cells.json
mkdir -p "$JB" "$GATE/results/jb_v1/screen"
[[ -f "$BASE" ]] || { echo "FAIL LOUD: base cell missing: $BASE" >&2; exit 1; }
for w in w0 w1 w2 w3; do
  if [[ $w == w0 ]]; then f=burst_drainable_f4000_n50000; else f=burst_drainable_${w}_n50000; fi
  [[ -f "$GATE/workloads/$f.json" ]] || { echo "FAIL LOUD: burst workload missing: $f (env_lever_v1_mint.py burst)" >&2; exit 1; }
done

STUDY_SEEDS=$(seq -s, 9109 9124)
TRAIN_SEEDS=$(seq -s, 9201 9224)
if [[ -f "$MANIFEST" ]]; then
  echo "[mint] $MANIFEST exists, not re-minting"
else
  python3 scripts_cosim/cluster_scale_v1_cells.py --base "$BASE" --sim-inputs data/nofs-ids \
    --vary clients --rungs 40 --seeds "$STUDY_SEEDS,$TRAIN_SEEDS" --out-dir "$GATE/configs" \
    --manifest "$MANIFEST.partial" --prefix cc
  mv "$MANIFEST.partial" "$MANIFEST"
fi
python3 - "$GATE/configs" "$BASE" <<'PY'
import json, sys
d, base = sys.argv[1], json.load(open(sys.argv[2]))
seeds = list(range(9109, 9125)) + list(range(9201, 9225))
for s in seeds:
    c = json.load(open(f"{d}/cc40s{s}.json"))
    assert int(c["nodes"]["server_nodes"]["count"]) == 6 and int(c["nodes"]["client_nodes"]["count"]) == 40, s
    assert c["scheduler"] == base["scheduler"], (s, c["scheduler"])
    assert float(c["network"]["topology"]["connection_probability"]) == 0.6 and float(c["network"]["backbone"]["bandwidth_mbps"]) == 1000.0
    assert int(c["network"]["topology"]["seed"]) == s
print(f"[mint] {len(seeds)} cells verified")
PY
# the 12 original candidates' reactive burst screens already exist (burst_groups_v1 L0): reuse
# them verbatim so the pool's screen is one directory
for f in "$GATE"/results/el_v1/burst/cc40s*__w*__reactive_s0.summary.json; do
  b=$(basename "$f"); [[ -e "$GATE/results/jb_v1/screen/$b" ]] || ln -s "$(realpath "$f")" "$GATE/results/jb_v1/screen/$b"
done
echo "[mint] reused $(ls "$GATE"/results/jb_v1/screen/*reactive_s0.summary.json | wc -l) reactive burst screens"
echo "=== mint done ==="
