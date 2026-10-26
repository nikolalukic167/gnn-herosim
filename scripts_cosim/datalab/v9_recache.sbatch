#!/bin/bash
# outbound_link_features_v1: the partial_state_v9 cache over the SAME recaptured, relabelled union corpus as cache_v7 and cache_v8 (the SSC mirror and the labelled view are reused as they are; nothing is regenerated).
# The graphs differ from the v8 ones only in the contract (partial_state_v9, 37-wide partial-state block: R1-R3 and U1 are computed from each graph's link_counts and link_routes at training and serving time).
#   sbatch --export=ALL,PIN=<sha8>,EXPECT_N=10720 v9_recache.sbatch
#SBATCH --job-name=v9-recache
#SBATCH --output=/share/nikola.lukic/uncut_groups/flowcap/link_corpus/logs/%x-%j.out
#SBATCH --partition=CPU-amd
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=06:00:00
#SBATCH --exclude=os-cpu-slurm-[21,15]
set -euo pipefail
PIN=${PIN:?}; EXPECT_N=${EXPECT_N:?}
WT=$HOME/gnn-herosim-wt/v9_${PIN}; cd "$WT"
for v in $(compgen -e | grep -E '^(HEROSIM_|GNN_|COSIM_|LIVE_AUDIT_|GATE_|SIM_|QUEUE_FEATURE|MAX_PLACEMENT|NEAR_RTT_|PARTIAL_STATE)'); do unset "$v"; done
eval "$(micromamba shell hook --shell bash)"; micromamba activate gnn
export HEROSIM_PY=python3 PYTHONPATH="$WT:$WT/src/notebooks" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
[[ "$(git rev-parse HEAD)" == ${PIN}* ]] || { echo "FAIL LOUD: HEAD $(git rev-parse HEAD)"; exit 1; }
L=/share/nikola.lukic/uncut_groups/flowcap/link_corpus; VS=$L/view_ssc; OUT=$L/cache_v9
[[ -d "$VS" && -d "$L/cache_v8/cache" ]] || { echo "FAIL LOUD: $VS or cache_v8 missing"; exit 1; }
[[ ! -e "$OUT" ]] || { echo "FAIL LOUD: $OUT exists"; exit 1; }
[[ "$(ls "$L"/mirror/*/ds_*/system_state_captured_unique.json | wc -l)" == "$EXPECT_N" ]] || { echo "FAIL LOUD: SSC files != $EXPECT_N"; exit 1; }
mkdir -p "$OUT"
export HEROSIM_TRANSFER_MODEL=pipelined HEROSIM_REPLICA_RELEASE=1 HEROSIM_SCALEOUT=kpa HEROSIM_SHARED_AUTOSCALER=0 GATE_FIXED_POLICY_TIME_SCALE=1.0
export DAG_PRIMARY_ALPHA_KEY=inf PARTIAL_STATE_CONTRACT=partial_state_v9 NEAR_RTT_LABEL_BACKLOG_CLOCK=seeded
T0=$(date +%s)
python3 -u src/notebooks/prepare_graphs_cache.py --base-dirs "$VS"/*/ --cache-dir "$OUT/cache" --dag-partial-state --platform-feature-dim 14 > "$OUT/cache.log" 2>&1 || { tail -30 "$OUT/cache.log"; echo "FAIL LOUD: cache build failed"; exit 1; }
echo "cache build wall seconds: $(( $(date +%s) - T0 ))"
python3 - "$OUT/cache" "$EXPECT_N" <<'PY'
import json, pickle, sys
from pathlib import Path
c = Path(sys.argv[1]); n = int(sys.argv[2])
ids = pickle.load(open(c / "dataset_ids.pkl", "rb")); g = pickle.load(open(c / "graphs.pkl", "rb")); md = json.load(open(c / "metadata.json"))
assert len(ids) == len(g) == n, (len(ids), len(g), n)
assert {int(x.platform_features.size(-1)) for x in g} == {21}, "v9 keeps the v7 platform block (width 21)"
assert md.get("partial_state_contract") == "partial_state_v9", md.get("partial_state_contract")
assert all(tuple(x.link_static_attr.shape) == (int(x.edge_index.size(1)), 3) for x in g), "a graph lacks link_static_attr [E, 3]"
assert all("link_counts" in x.partial_state_ctx and "link_routes" in x.partial_state_ctx for x in g)
print(f"cache_v9: {len(g)} graphs, contract {md['partial_state_contract']}, every graph carries link_static_attr [E, 3], link_counts and link_routes")
PY
sha256sum "$OUT/cache/metadata.json"; md5sum "$OUT/cache/metadata.json"
echo "=== done; cache in $OUT/cache"
