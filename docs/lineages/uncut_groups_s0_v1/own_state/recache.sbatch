#!/bin/bash
# Own-state corpus stage 4: the partial_state_v6 cache over the COMBINED view (old Arm 1 corpus + the new datasets), the same recipe as datalab/uncut_arm1_cache_v6.sbatch; then
#   (i) the label-override json (old sim labels + the new relabel) against the new cache (uncut_relabel_cache_check.py: same ids, every label plan a candidate), and
#   (ii) the old graphs of the new cache equal the filed v6 cache's, graph for graph (dagger_corpus.py check-cache).
#   usage: sbatch --export=ALL,ARM=twin,PIN=<sha8> recache.sbatch
#SBATCH --job-name=dag-recache
#SBATCH --output=/share/nikola.lukic/uncut_groups/dagger/%x-%j.out
#SBATCH --partition=CPU-amd
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=05:00:00
#SBATCH --exclude=os-cpu-slurm-21,os-cpu-slurm-15,os-cpu-slurm-13,os-cpu-slurm-14,os-cpu-slurm-16,os-cpu-slurm-10,os-cpu-slurm-17
set -euo pipefail
ARM=${ARM:?}; PIN=${PIN:?}
WT=$HOME/gnn-herosim-wt/dagc_${PIN}; cd "$WT"
for v in $(compgen -e | grep -E '^(HEROSIM_|GNN_|COSIM_|LIVE_AUDIT_|GATE_|SIM_|QUEUE_FEATURE|MAX_PLACEMENT|NEAR_RTT_|PARTIAL_STATE)'); do unset "$v"; done
eval "$(micromamba shell hook --shell bash)"; micromamba activate gnn
export HEROSIM_PY=python3 PYTHONPATH="$WT:$WT/src/notebooks" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
[[ "$(git rev-parse HEAD)" == ${PIN}* ]] || { echo "FAIL LOUD: HEAD $(git rev-parse HEAD)"; exit 1; }
C=/share/nikola.lukic/uncut_groups/dagger/corpus_$ARM; V=$C/view; OUT=$C/cache_v6
[[ ! -e "$OUT" ]] || { echo "FAIL LOUD: $OUT exists"; exit 1; }
mkdir -p "$OUT"
export HEROSIM_TRANSFER_MODEL=pipelined HEROSIM_REPLICA_RELEASE=1 HEROSIM_SCALEOUT=kpa HEROSIM_SHARED_AUTOSCALER=0 GATE_FIXED_POLICY_TIME_SCALE=1.0
export DAG_PRIMARY_ALPHA_KEY=inf PARTIAL_STATE_CONTRACT=partial_state_v6 NEAR_RTT_LABEL_BACKLOG_CLOCK=seeded
T0=$(date +%s)
python3 -u src/notebooks/prepare_graphs_cache.py --base-dirs "$V"/*/ --cache-dir "$OUT/cache" --dag-partial-state --platform-feature-dim 14 > "$OUT/cache.log" 2>&1 || { tail -30 "$OUT/cache.log"; echo "FAIL LOUD: cache build failed"; exit 1; }
echo "cache build wall seconds: $(( $(date +%s) - T0 ))"
python3 scripts_cosim/uncut_relabel_cache_check.py "$OUT/cache" "$C/label_override_merged.json"
python3 scripts_cosim/dagger_corpus.py check-cache --old /share/nikola.lukic/uncut_groups/inflight/cache_v6_a/cache --new "$OUT/cache"
echo "=== done; cache in $OUT/cache"
