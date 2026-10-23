#!/bin/bash
# live_link_features_v1: the partial_state_v7 cache over the recaptured, relabelled union corpus (S6 capture, S7 labels), the recipe of datalab/uncut_arm1_cache_v6.sbatch with the contract changed.
# The new datasets carry no system_state_captured_unique.json (job 899571 failed on that): as uncut_arm1_cache.sbatch does, build a PRIVATE mirror (every file a symlink, optimal_result.json a copy) and rewrite the SSC
# files in the mirror with refresh_optimal_full_stats.py --rewrite-ssc (no simulation); the corpus is only read. Every dataset must carry fidelity.flows (a state without it is refused by the cache builder).
#   sbatch --export=ALL,PIN=<sha8>,CORPUS=<dir holding one subdir per arm>,ARMS="a1 a2",EXPECT_N=<datasets> v7_recache.sbatch
#SBATCH --job-name=link-recache
#SBATCH --output=/share/nikola.lukic/uncut_groups/flowcap/link_corpus/logs/%x-%j.out
#SBATCH --partition=CPU-amd
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=06:00:00
#SBATCH --exclude=os-cpu-slurm-[21,15]
set -euo pipefail
PIN=${PIN:?}; CORPUS=${CORPUS:?}; ARMS=${ARMS:?}; EXPECT_N=${EXPECT_N:?}
WT=$HOME/gnn-herosim-wt/v7corpus_${PIN}; cd "$WT"
for v in $(compgen -e | grep -E '^(HEROSIM_|GNN_|COSIM_|LIVE_AUDIT_|GATE_|SIM_|QUEUE_FEATURE|MAX_PLACEMENT|NEAR_RTT_|PARTIAL_STATE)'); do unset "$v"; done
eval "$(micromamba shell hook --shell bash)"; micromamba activate gnn
export HEROSIM_PY=python3 PYTHONPATH="$WT:$WT/src/notebooks" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
[[ "$(git rev-parse HEAD)" == ${PIN}* ]] || { echo "FAIL LOUD: HEAD $(git rev-parse HEAD)"; exit 1; }
L=/share/nikola.lukic/uncut_groups/flowcap/link_corpus; V=$L/view; M=$L/mirror; VS=$L/view_ssc; OUT=$L/cache_v7
[[ ! -e "$OUT" && ! -e "$M" && ! -e "$VS" ]] || { echo "FAIL LOUD: $OUT, $M or $VS exists"; exit 1; }
[[ -e "$V" ]] || python3 scripts_cosim/v7_corpus.py view --corpus "$CORPUS" --arms $ARMS --out "$V"
python3 - "$L" <<'PY2'
import os, shutil, sys
from pathlib import Path
L = Path(sys.argv[1]); mir = L / "mirror"; view = L / "view_ssc"; mir.mkdir(); view.mkdir()
for b in sorted((L / "view").iterdir()):
    dst = mir / b.name; dst.mkdir(); n = 0
    for d in sorted(Path(os.path.realpath(b)).glob("ds_*")):
        t = dst / d.name; t.mkdir()
        for f in d.iterdir():
            if f.name == "optimal_result.json":
                shutil.copyfile(f, t / f.name)
            else:
                os.symlink(f.resolve(), t / f.name)
        n += 1
    os.symlink(dst, view / b.name)
    print(f"mirror {b.name}: {n} datasets")
PY2
for B in "$M"/*/; do python3 scripts_cosim/refresh_optimal_full_stats.py --base-dir "$B" --rewrite-ssc | tail -1; done
[[ "$(ls "$M"/*/ds_*/system_state_captured_unique.json | wc -l)" == "$EXPECT_N" ]] || { echo "FAIL LOUD: SSC files != $EXPECT_N"; exit 1; }
mkdir -p "$OUT"
export HEROSIM_TRANSFER_MODEL=pipelined HEROSIM_REPLICA_RELEASE=1 HEROSIM_SCALEOUT=kpa HEROSIM_SHARED_AUTOSCALER=0 GATE_FIXED_POLICY_TIME_SCALE=1.0
export DAG_PRIMARY_ALPHA_KEY=inf PARTIAL_STATE_CONTRACT=partial_state_v7 NEAR_RTT_LABEL_BACKLOG_CLOCK=seeded
T0=$(date +%s)
python3 -u src/notebooks/prepare_graphs_cache.py --base-dirs "$VS"/*/ --cache-dir "$OUT/cache" --dag-partial-state --platform-feature-dim 14 > "$OUT/cache.log" 2>&1 || { tail -30 "$OUT/cache.log"; echo "FAIL LOUD: cache build failed"; exit 1; }
echo "cache build wall seconds: $(( $(date +%s) - T0 ))"
python3 - "$OUT/cache" "$EXPECT_N" <<'PY3'
import json, pickle, sys
from pathlib import Path
c = Path(sys.argv[1]); n = int(sys.argv[2])
ids = pickle.load(open(c / "dataset_ids.pkl", "rb")); g = pickle.load(open(c / "graphs.pkl", "rb")); md = json.load(open(c / "metadata.json"))
assert len(ids) == len(g) == n, (len(ids), len(g), n)
w = {int(x.platform_features.size(-1)) for x in g}
assert w == {21}, f"platform width {w}, v7 wants 14 + 2 four-type + 2 in-flight + 3 link = 21"
assert md.get("partial_state_contract") == "partial_state_v7", md.get("partial_state_contract")
assert all("link_counts" in x.partial_state_ctx and "link_routes" in x.partial_state_ctx for x in g)
print(f"cache: {len(g)} graphs, platform width 21, contract {md['partial_state_contract']}, every graph carries link_counts and link_routes")
PY3
sha256sum "$OUT/cache/metadata.json"; md5sum "$OUT/cache/metadata.json"
echo "label-override check: run uncut_relabel_cache_check.py $OUT/cache <override.json> once the override exists"
echo "=== done; cache in $OUT/cache"
