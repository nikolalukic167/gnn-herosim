#!/bin/bash
# r1_attribution_v1 PRODUCTION cache: the v5 declared-pruning cache of the whole production corpus, the topology split artifact, and the checks.
# Reads corpus_prod through a private view (real directories of symlinks; optimal_result.json is a copy because the SSC rewrite rewrites it, and the
# production build does not write system_state_captured_unique.json). Submit with --dependency=afterok:<the corpus check> once it passes.
#   sbatch [--dependency=afterok:853822] --export=ALL,WT=<pinned worktree>,OUT=<new dir> r1a_prod_cache.sbatch
# Writes: $OUT/{view,cache,split.json,cache.log,checks.json}; links $WT/simulation_data/graphs_cache_r1_attribution_v1_psv5_inf -> $OUT/cache and
# copies the split to $WT/experiments/r1_attribution_v1_split.json (untracked; commit it with the results).
#SBATCH --job-name=r1a-pcache
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/r1a-pcache-%j.out
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=12:00:00
set -euo pipefail
WT=${WT:?}; OUT=${OUT:?}
CP=${CORPUS:-/share/nikola.lukic/workload_fix_v1/corpus_prod}
SUF=${SUF:-E8894a08}
cd "$WT"
for v in $(compgen -e | grep -E '^(HEROSIM_|GNN_|COSIM_|LIVE_AUDIT_|GATE_|SIM_|QUEUE_FEATURE|MAX_PLACEMENT|NEAR_RTT_|PARTIAL_STATE)'); do unset "$v"; done
if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3 PYTHONPATH="$WT:$WT/src/notebooks" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
echo "HEAD $(git rev-parse HEAD) node $(hostname)"
[[ -z "$(git status --porcelain -- src scripts_cosim experiments run_experiment.py | grep -v 'experiments/r1_attribution_v1_split.json')" ]] || { echo "FAIL LOUD: worktree dirty"; exit 1; }
[[ ! -e "$OUT" ]] || { echo "FAIL LOUD: $OUT exists"; exit 1; }
LINK=simulation_data/graphs_cache_r1_attribution_v1_psv5_inf; SPLITF=experiments/r1_attribution_v1_split.json
[[ ! -e "$LINK" && ! -L "$LINK" && ! -e "$SPLITF" ]] || { echo "FAIL LOUD: $LINK or $SPLITF already exists in the worktree"; exit 1; }
for s in train heldout; do [[ -d "$CP/gnn_datasets_wf1_${s}_$SUF" ]] || { echo "FAIL LOUD: no $CP/gnn_datasets_wf1_${s}_$SUF"; exit 1; }; done
# the held-out directory must hold datasets when this job runs: no empty-test carve-out in production
NH=$(find "$CP/gnn_datasets_wf1_heldout_$SUF" -maxdepth 1 -name 'ds_*' | wc -l)
[[ "$NH" -gt 0 ]] || { echo "FAIL LOUD: $CP/gnn_datasets_wf1_heldout_$SUF holds no datasets; the production split needs held-out topologies"; exit 1; }
echo "held-out datasets present: $NH"
mkdir -p "$OUT/view"
export HEROSIM_TRANSFER_MODEL=pipelined HEROSIM_REPLICA_RELEASE=1 HEROSIM_SCALEOUT=kpa HEROSIM_SHARED_AUTOSCALER=0 GATE_FIXED_POLICY_TIME_SCALE=1.0
export DAG_PRIMARY_ALPHA_KEY=inf PARTIAL_STATE_CONTRACT=partial_state_v5 NEAR_RTT_LABEL_BACKLOG_CLOCK=seeded
T0=$(date +%s)
python3 - "$CP" "$SUF" "$OUT/view" <<'PY'
import os, shutil, sys
from pathlib import Path
cp, suf, view = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
for split in ("train", "heldout"):
    src = cp / f"gnn_datasets_wf1_{split}_{suf}"; dst = view / src.name; dst.mkdir()
    n = 0
    for d in sorted(src.glob("ds_*")):
        t = dst / d.name; t.mkdir()
        for f in d.iterdir():
            if f.name == "optimal_result.json":
                shutil.copyfile(f, t / f.name)
            else:
                os.symlink(f, t / f.name)
        n += 1
    print(f"view {dst.name}: {n} datasets")
PY
for B in "$OUT"/view/*/; do python3 scripts_cosim/refresh_optimal_full_stats.py --base-dir "$B" --rewrite-ssc | tail -1; done
T1=$(date +%s); echo "view + ssc rewrite wall seconds: $((T1-T0))"
N=$(find "$OUT"/view -mindepth 2 -maxdepth 2 | wc -l)
python3 -u src/notebooks/prepare_graphs_cache.py --base-dirs "$OUT"/view/*/ --cache-dir "$OUT/cache" --dag-partial-state --platform-feature-dim 14 > "$OUT/cache.log" 2>&1 || { tail -30 "$OUT/cache.log"; echo "FAIL LOUD: cache build failed"; exit 1; }
T2=$(date +%s); echo "cache build wall seconds: $((T2-T1)) for $N linked datasets"
grep -vE "it/s\]|^\s*$" "$OUT/cache.log" | tail -22 | cut -c1-220
python3 scripts_cosim/make_r1a_split.py --cache-dir "$OUT/cache" --corpus-split "$CP/split.json" --out "$OUT/split.json"
python3 - "$OUT" "$CP" "$N" <<'PY'
import json, pickle, re, sys
from pathlib import Path
import torch
out, cp, n_linked = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
cache = out / "cache"; m = json.loads((cache / "metadata.json").read_text()); g = pickle.load(open(cache / "graphs.pkl", "rb"))
log = (out / "cache.log").read_text()
skipped = re.search(r"incomplete sweeps skipped: (\d+); per-dataset load errors: (\d+)", log)
sp = json.loads((out / "split.json").read_text()); corpus = json.loads((cp / "split.json").read_text())
checks = {
    "linked_datasets": n_linked, "graphs": len(g), "incomplete_sweeps_skipped": int(skipped.group(1)) if skipped else None,
    "load_errors": int(skipped.group(2)) if skipped else None, "physics_env": m.get("physics_env"), "candidate_slate": m.get("candidate_slate"),
    "partial_state_contract": m.get("partial_state_contract"), "platform_feature_dim": m.get("platform_feature_dim"),
    "split": {k: len(sp[k]) for k in ("train", "val", "test")}, "topologies": {k: len(v) for k, v in sp["topologies"].items()},
    "heldout_topologies_in_corpus_split": len(corpus["heldout"]), "dry_run": sp.get("dry_run"),
    "exposure_flags_binary": all(set(torch.unique(h.platform_features[:, -2:]).tolist()) <= {0.0, 1.0} for h in g[:2000]),
}
missing = sorted((set(corpus["train"]) | set(corpus["heldout"])) - {t for k in ("train", "val", "test") for t in sp["topologies"][k]})
checks["topologies_in_corpus_split_without_datasets"] = missing
print(json.dumps(checks, indent=1)); (out / "checks.json").write_text(json.dumps(checks, indent=1))
assert checks["physics_env"].get("inflight_capture") == "legacy" and checks["candidate_slate"] == "declared_pruning_v1"
assert checks["partial_state_contract"] == "partial_state_v5" and checks["platform_feature_dim"] == 14
assert checks["load_errors"] == 0, "FAIL LOUD: per-dataset load errors"
assert checks["incomplete_sweeps_skipped"] == 0, "FAIL LOUD: incomplete sweeps were skipped; the corpus check should have caught them"
assert not checks["dry_run"] and checks["split"]["test"] > 0 and checks["exposure_flags_binary"]
held = set(corpus["heldout"])
assert set(sp["topologies"]["test"]) <= held and not (set(sp["topologies"]["train"]) | set(sp["topologies"]["val"])) & held, \
    "FAIL LOUD: held-out topologies leaked into train or val"
PY
ln -s "$OUT/cache" "$LINK"; cp "$OUT/split.json" "$SPLITF"
echo "split sha256 $(sha256sum "$SPLITF" | cut -d' ' -f1)"
echo "=== done; outputs in $OUT; $LINK -> $OUT/cache"
