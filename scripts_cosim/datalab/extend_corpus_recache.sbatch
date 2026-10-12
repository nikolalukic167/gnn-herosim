#!/bin/bash
# Recache of the slate-extended accel corpus (rp/corpus-uncap) into a NEW cache directory; the production cache is read only (split and parent ids are compared with it).
# Same recipe as r1a_prod_cache.sbatch (private view, optimal_result.json copied, SSC rewrite, v5 cache, dim 14), except: no new split (the production split is reused, by sha),
# no links into the worktree, and the candidate_slate assertion is the mixed corpus's (declared_pruning_v1_uncapped, candidate_slate_rules both).
#   sbatch --export=ALL,WT=<pinned worktree>,EXPECT_HEAD=<sha>,UNCAP=/share/nikola.lukic/accel_uncap/<commit>,OUT=<new dir> extend_corpus_recache.sbatch
#SBATCH --job-name=ext-recache
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/ext-recache-%j.out
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=12:00:00
set -euo pipefail
WT=${WT:?}; OUT=${OUT:?}; UNCAP=${UNCAP:?}; EXPECT_HEAD=${EXPECT_HEAD:?}
OLD=/share/nikola.lukic/accel_prod/20261010_061132/cache_run
cd "$WT"
for v in $(compgen -e | grep -E '^(HEROSIM_|GNN_|COSIM_|LIVE_AUDIT_|GATE_|SIM_|QUEUE_FEATURE|MAX_PLACEMENT|NEAR_RTT_|PARTIAL_STATE)'); do unset "$v"; done
if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"; micromamba activate gnn
export HEROSIM_PY=python3 PYTHONPATH="$WT:$WT/src/notebooks" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
HEAD=$(git rev-parse HEAD); echo "HEAD $HEAD node $(hostname)"
[[ "$HEAD" == "$EXPECT_HEAD"* ]] || { echo "FAIL LOUD: worktree HEAD $HEAD is not $EXPECT_HEAD"; exit 1; }
[[ -z "$(git status --porcelain -- src scripts_cosim experiments run_experiment.py)" ]] || { echo "FAIL LOUD: worktree dirty"; exit 1; }
[[ ! -e "$OUT" ]] || { echo "FAIL LOUD: $OUT exists"; exit 1; }
for s in train heldout; do [[ -d "$UNCAP/$s" ]] || { echo "FAIL LOUD: no $UNCAP/$s"; exit 1; }; done
mkdir -p "$OUT/view"
export HEROSIM_TRANSFER_MODEL=pipelined HEROSIM_REPLICA_RELEASE=1 HEROSIM_SCALEOUT=kpa HEROSIM_SHARED_AUTOSCALER=0 GATE_FIXED_POLICY_TIME_SCALE=1.0
export DAG_PRIMARY_ALPHA_KEY=inf PARTIAL_STATE_CONTRACT=partial_state_v5 NEAR_RTT_LABEL_BACKLOG_CLOCK=seeded
T0=$(date +%s)
python3 - "$UNCAP" "$OUT/view" <<'PY'
import os, shutil, sys
from pathlib import Path
cp, view = Path(sys.argv[1]), Path(sys.argv[2])
for split in ("train", "heldout"):
    src = cp / split; dst = view / f"gnn_datasets_wf1_{split}"; dst.mkdir()
    n = 0
    for d in sorted(src.glob("ds_*")):
        t = dst / d.name; t.mkdir()
        for f in d.iterdir():
            if f.name == "optimal_result.json":
                shutil.copyfile(f, t / f.name)
            else:
                os.symlink(f.resolve(), t / f.name)
        n += 1
    print(f"view {dst.name}: {n} datasets")
PY
for B in "$OUT"/view/*/; do python3 scripts_cosim/refresh_optimal_full_stats.py --base-dir "$B" --rewrite-ssc | tail -1; done
T1=$(date +%s); echo "view + ssc rewrite wall seconds: $((T1-T0))"
N=$(find "$OUT"/view -mindepth 2 -maxdepth 2 | wc -l)
python3 -u src/notebooks/prepare_graphs_cache.py --base-dirs "$OUT"/view/*/ --cache-dir "$OUT/cache" --dag-partial-state --platform-feature-dim 14 > "$OUT/cache.log" 2>&1 || { tail -30 "$OUT/cache.log"; echo "FAIL LOUD: cache build failed"; exit 1; }
T2=$(date +%s); echo "cache build wall seconds: $((T2-T1)) for $N linked datasets"
grep -vE "it/s\]|^\s*$" "$OUT/cache.log" | tail -22 | cut -c1-220
cp "$OLD/split.json" "$OUT/split.json"
python3 - "$OUT" "$OLD" "$N" <<'PY'
import hashlib, json, pickle, re, sys
from pathlib import Path
out, old, n_linked = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
m = json.loads((out / "cache/metadata.json").read_text()); mo = json.loads((old / "cache/metadata.json").read_text())
log = (out / "cache.log").read_text()
skipped = re.search(r"incomplete sweeps skipped: (\d+); per-dataset load errors: (\d+)", log)
same = ["partial_state_contract", "physics_env", "dag_primary_alpha_key", "inference_feature_layout", "queue_feature_contract", "topology_feature_contract", "platform_feature_dim",
        "dag_task_type_vocab", "label_objective", "peer_exchange_block", "dag_partial_state", "queue_norm_mode", "training_contract", "num_graphs", "num_datasets"]
diff = {k: [m.get(k), mo.get(k)] for k in same if m.get(k) != mo.get(k)}
checks = {"linked_datasets": n_linked, "graphs": m["num_graphs"], "incomplete_sweeps_skipped": int(skipped.group(1)) if skipped else None,
          "load_errors": int(skipped.group(2)) if skipped else None, "candidate_slate": m.get("candidate_slate"), "candidate_slate_rules": m.get("candidate_slate_rules"),
          "keys_differing_from_production_cache": diff, "parent_ids_equal": m["parent_dataset_ids"] == mo["parent_dataset_ids"],
          "split_sha256": hashlib.sha256((out / "split.json").read_bytes()).hexdigest(), "rtt_rows_new": m.get("num_rtt_combo_rows"), "rtt_rows_old": mo.get("num_rtt_combo_rows")}
print(json.dumps(checks, indent=1)); (out / "checks.json").write_text(json.dumps(checks, indent=1))
assert checks["candidate_slate"] == "declared_pruning_v1_uncapped" and checks["candidate_slate_rules"] == ["declared_pruning_v1", "declared_pruning_v1_uncapped"], "FAIL LOUD: slate rule"
assert not diff, f"FAIL LOUD: contract keys differ from the production cache: {diff}"
assert checks["parent_ids_equal"] and checks["load_errors"] == 0 and checks["incomplete_sweeps_skipped"] == 0
assert checks["split_sha256"].startswith("f5bc4f51")
PY
echo "=== done; cache in $OUT/cache"
