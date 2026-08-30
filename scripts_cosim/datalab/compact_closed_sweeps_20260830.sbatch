#!/bin/bash
# Maintenance (2026-08-30): gzip-compact closed-lineage raw gate results in place, to keep
# the per-account /home quota alive for the link_mp_v1 gate (~90G of new results).
#
# Why this is safe: extract_gate_stats_summary.py reads .json.gz transparently (added the
# same day, tested raw==gz), so every compacted sweep stays re-extractable and re-scorable.
# Only results/*.json are touched — cell_infrastructure/ and configs/ are live INPUTS for
# future gate runs and stay raw. Nothing is deleted; gzip -9 keeps the exact bytes.
#
# The 2026-08-28 quota exhaustion killed 191 gate tasks twice with 0-byte SLURM logs and a
# misleading df; this is the standing fix, not an emergency one.
#SBATCH --job-name=compact-sweeps
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/compact-sweeps-%j.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/compact-sweeps-%j.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=8G
#SBATCH --time=12:00:00

set -euo pipefail
cd /home/nikola.lukic/gnn-herosim/simulation_data/normal_sim_sweeps

# Closed/superseded lineage sweeps (June-August), plus the frozen 30-cell gate arms whose
# statistics already live in gate_stats_summary.json. Explicit list, no wildcard-everything:
# anything not named here stays raw.
DIRS=(
  contention_v2_873_v5.5_sealed_holdout_rebaseline_20260813
  sweep_bipartite_coordination_v1
  warmth_sparse_merged_ce_reduced_hub9_20260611
  contention_v2_873_v5.7_siv1_coupled_trio_20260813
  full_corpus_siv1_coupled_trio_20260815
  contention_v2_873_v5.5_coupled_trio_20260813
  warmth_sparse_skew_merged_ce_reduced_hub9_20260612
  contention_v2_873_v5.5_sealed_holdout_20260806
  tiered_hub_gnn_mlp_125225_20260610
  bipartite_v2_skew_merged_20260614
  mega_compare_all7_20260614
  full_corpus_siv1_live_gate_20260821_synced
  full_corpus_siv1_live_gate_20260820
  siv1_tempfix_gate_125-225
  tiered_hub_gnn_mlp_20260610
  merged_contention_weighted_live_gate_20260616_105500
  skew4_new_models_20260614
  dim14_old_models_skew4_125225_20260610
  siv1_tempfix_gate_175-100
  bipartite_v2_mlp_dim22_fix_20260614
  strategic_merge_contention_live_gate_20260616
  contention_v3_live_gate_20260620
  siv1_tempfix_gate_150-100
  strategic_merge_wss_live_gate_20260616
  bipartite_v2_mlp_dim22_batchcache_20260614
  a1_backbone_bw1p5
  bb_core8_bw1p5
  bb_core4_bw0p5
)
# The frozen 30-cell gate arm sweeps (objective_pivot_v1 + mp_ablation_v1 + draw studies).
# Pattern-scoped to the three gate prefixes; their cell_infrastructure/configs stay raw.
for d in drawgate_* promo175_* bbrob_* live_audit_*; do
  [[ -d "$d" ]] && DIRS+=("$d")
done

before=$(df -B1G --output=used /home 2>/dev/null | tail -1 || echo "?")
total_files=0
for d in "${DIRS[@]}"; do
  [[ -d "$d" ]] || { echo "(absent) $d"; continue; }
  n=$(find "$d" -name '*.json' -size +1M -not -path '*/cell_infrastructure/*' \
        -not -path '*/configs/*' 2>/dev/null | wc -l)
  [[ "$n" -gt 0 ]] || continue
  echo "compacting $d ($n files >1M)"
  find "$d" -name '*.json' -size +1M -not -path '*/cell_infrastructure/*' \
       -not -path '*/configs/*' -print0 \
    | xargs -0 -r -P "${SLURM_CPUS_PER_TASK:-8}" -n 1 gzip -9
  total_files=$((total_files + n))
done
echo "compacted ${total_files} files"
du -sh /home/nikola.lukic/gnn-herosim/simulation_data/normal_sim_sweeps
echo "=== compact complete ==="
