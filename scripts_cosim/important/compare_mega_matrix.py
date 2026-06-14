#!/usr/bin/env python3
"""
compare_mega_matrix.py — Analyze all 4 experiment groups from the mega experiment matrix.

Reads total_rtt from result JSONs; prints per-group tables and win counts.

Usage:
    pipenv run python3 scripts_cosim/important/compare_mega_matrix.py [--sweep-base DIR]

Output:
    - Per-group RTT table (config x policy, lower is better)
    - Win counts (GNN vs MLP vs Knative)
    - Summary sums per group
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).parents[2]


def read_rtt(path: Path) -> Optional[float]:
    """Fast extraction of total_rtt from potentially large result JSONs (may be 40-50 MB)."""
    if not path.exists():
        return None
    try:
        # Stream the file line by line; total_rtt appears in the first ~20 lines.
        with path.open(errors="replace") as f:
            for i, line in enumerate(f):
                if i > 40:
                    break
                if '"total_rtt"' in line:
                    # Extract numeric value after the colon
                    parts = line.split(":", 1)
                    if len(parts) == 2:
                        val_str = parts[1].strip().rstrip(",").rstrip()
                        try:
                            rtt = float(val_str)
                            if rtt > 0:
                                return rtt
                        except ValueError:
                            pass
        # Fallback: full parse (slower but correct)
        d = json.loads(path.read_bytes())
        rtt = d.get("total_rtt")
        if rtt and float(rtt) > 0:
            return float(rtt)
    except Exception:
        pass
    return None


def fmt_m(rtt: Optional[float]) -> str:
    if rtt is None:
        return "—"
    return f"{rtt / 1e6:.2f}M"


def delta_pct(policy_rtt: Optional[float], ref_rtt: Optional[float]) -> str:
    if policy_rtt is None or ref_rtt is None or ref_rtt == 0:
        return "—"
    return f"{(policy_rtt - ref_rtt) / ref_rtt * 100:+.1f}%"


def winner(values: dict) -> str:
    valid = {k: v for k, v in values.items() if v is not None}
    if not valid:
        return "—"
    best_key = min(valid, key=lambda k: valid[k])
    # tie check
    best_val = valid[best_key]
    tied = [k for k, v in valid.items() if v is not None and abs(v - best_val) / best_val < 0.005]
    return "tie" if len(tied) > 1 else best_key


def print_table(title: str, configs: list, policies: list, results: dict, ref_policy: str = None):
    """results[config][policy] -> rtt or None"""
    col_width = 12
    header_cols = ["config"] + policies + (["Δ%GNN/Kn", "Δ%GNN/MLP", "best"] if ref_policy else ["best"])
    header = "  ".join(f"{c:<{col_width}}" for c in header_cols)
    print(f"\n{'='*len(header)}")
    print(f"  {title}")
    print('='*len(header))
    print(header)
    print('-'*len(header))

    wins = {p: 0 for p in policies}
    sums = {p: 0.0 for p in policies}
    sum_count = {p: 0 for p in policies}

    gnn_col = next((p for p in policies if "gnn" in p.lower()), None)
    kn_col = next((p for p in policies if "knative" in p.lower() or p == "kn"), None)
    mlp_col = next((p for p in policies if "mlp" in p.lower()), None)

    for cfg in configs:
        vals = results.get(cfg, {})
        row_parts = [f"{cfg:<{col_width}}"]
        for p in policies:
            rtt = vals.get(p)
            row_parts.append(f"{fmt_m(rtt):<{col_width}}")
            if rtt is not None:
                sums[p] += rtt
                sum_count[p] += 1

        best = winner(vals)
        if best in wins:
            wins[best] += 1

        if ref_policy:
            gnn_rtt = vals.get(gnn_col) if gnn_col else None
            kn_rtt = vals.get(kn_col) if kn_col else None
            mlp_rtt = vals.get(mlp_col) if mlp_col else None
            row_parts.append(f"{delta_pct(gnn_rtt, kn_rtt):<{col_width}}")
            row_parts.append(f"{delta_pct(gnn_rtt, mlp_rtt):<{col_width}}")

        row_parts.append(f"{best:<{col_width}}")
        print("  ".join(row_parts))

    print('-'*len(header))

    # Sums row
    sum_parts = [f"{'SUM':<{col_width}}"]
    for p in policies:
        s = sums[p] if sum_count[p] > 0 else None
        sum_parts.append(f"{fmt_m(s):<{col_width}}")
    if ref_policy:
        gnn_sum = sums.get(gnn_col, 0) if gnn_col else 0
        kn_sum = sums.get(kn_col, 0) if kn_col else 0
        mlp_sum = sums.get(mlp_col, 0) if mlp_col else 0
        sum_parts.append(f"{delta_pct(gnn_sum or None, kn_sum or None):<{col_width}}")
        sum_parts.append(f"{delta_pct(gnn_sum or None, mlp_sum or None):<{col_width}}")
        sum_parts.append("")
    print("  ".join(sum_parts))

    print("\nWin counts:", "  |  ".join(f"{p}: {wins[p]}/{len(configs)}" for p in policies))
    print()


# ─── Group 1: mega compare all7 ───────────────────────────────────────────────

G1_CONFIGS = [
    "default_20_20_p50",
    "00_balanced_30_30_p35",
    "01_balanced_40_40_p50",
    "02_balanced_50_50_p60",
    "03_client_heavy_50_35_p50",
    "04_server_heavy_35_50_p50",
    "05_sparse_40_40_p25",
]

G1_POLICIES = {
    # existing anchors (from separate sweep dirs)
    "gnn_dim14ce": ("gnn_near_rtt_v2_dim14_ce_only_20260609/results", "{cfg}.json"),
    "mlp_dim22":   ("reviewer_triangle_all7_20260609/results",        "{cfg}_mlp_batch.json"),
    # knative: no default in knative sweep dir (lives in baseline_default); try both patterns
    "knative":     ("knative_network_20260606_192413/results",         "{cfg}.json"),
    # new models (from mega_compare sweep dir)
    "gnn_warmth":  ("mega_compare_all7_20260614/results", "{cfg}_gnn_warmth.json"),
    "gnn_wsm":     ("mega_compare_all7_20260614/results", "{cfg}_gnn_wsm.json"),
    "gnn_wssm":    ("mega_compare_all7_20260614/results", "{cfg}_gnn_wssm.json"),
    "mlp_warmth":  ("mega_compare_all7_20260614/results", "{cfg}_mlp_warmth.json"),
    "mlp_wsm":     ("mega_compare_all7_20260614/results", "{cfg}_mlp_wsm.json"),
    "mlp_wssm":    ("mega_compare_all7_20260614/results", "{cfg}_mlp_wssm.json"),
}

# ─── Group 2: bipartite v2 skew-merged ────────────────────────────────────────

G2_CONFIGS = [
    "hub_k4_seek35", "hub_k4_seek50", "hub_k4_seek65",
    "hub_k6_seek35", "hub_k6_seek50", "hub_k6_seek65",
    "hub_k8_seek35", "hub_k8_seek50", "hub_k8_seek65",
]

G2_POLICIES = {
    # legacy physics (from bipartite v1, already done) — batch-loop regime, comparable to new models
    "gnn_v1(dim14ce)":   ("sweep_bipartite_coordination_v1/results", "{cfg}_gnn_dim22.json"),
    "mlp_v1(dim22)":     ("sweep_bipartite_coordination_v1/results", "{cfg}_mlp_dim22.json"),
    "knative_v1":        ("sweep_bipartite_coordination_v1/results", "{cfg}_knative.json"),
    # new models with v2 physics (node_disk_v2, batch-loop regime, directly comparable)
    "gnn_wssm_v2":       ("bipartite_v2_skew_merged_20260614/results", "{cfg}_gnn_wssm.json"),
    "mlp_wssm_v2":       ("bipartite_v2_skew_merged_20260614/results", "{cfg}_mlp_wssm.json"),
    # NOTE: regime_b_hub9 Knative uses per-arrival loop — total_rtt is NOT comparable to batch-loop
    # results above. Excluded from this table to avoid misleading comparisons.
}

# ─── Group 3: skew3 full gate ─────────────────────────────────────────────────

G3_CONFIGS = [
    "default_20_20_p50",
    "default_20_20_degree_skew",
    "05_sparse_40_40_p25_degree_skew",
]

G3_POLICIES = {
    "gnn_wssm":  ("skew3_full_gate_20260614/results", "{cfg}_gnn_wssm.json"),
    "mlp_wssm":  ("skew3_full_gate_20260614/results", "{cfg}_mlp_wssm.json"),
    "knative":   ("skew3_full_gate_20260614/results", "{cfg}_knative.json"),
}

# ─── Group 4: skew4 new models ────────────────────────────────────────────────

G4_CONFIGS = [
    "default_20_20_p50",
    "05_sparse_40_40_p25",
    "default_20_20_degree_skew",
    "05_sparse_40_40_p25_degree_skew",
]

G4_POLICIES = {
    # legacy physics for reference
    "gnn_v1(legacy)":   ("dim14_old_models_skew4_125225_20260610/results", "{cfg}_gnn_dim14.json"),
    "mlp_v1(legacy)":   ("dim14_old_models_skew4_125225_20260610/results", "{cfg}_mlp_dim22.json"),
    "kn_v1(legacy)":    ("dim14_old_models_skew4_125225_20260610/results", "{cfg}_knative_network.json"),
    # new models + v2 physics
    "gnn_wssm":         ("skew4_new_models_20260614/results", "{cfg}_gnn_wssm.json"),
    "mlp_wssm":         ("skew4_new_models_20260614/results", "{cfg}_mlp_wssm.json"),
    "knative_v2":       ("skew4_new_models_20260614/results", "{cfg}_knative.json"),
}


def load_group(base: Path, configs: list, policy_map: dict,
               extra_fallbacks: dict = None) -> dict:
    """Returns results[cfg][policy_tag] -> rtt or None.

    extra_fallbacks: {tag: [(sweep_subdir, fname_template), ...]} for when the primary
    path fails (e.g. Knative default in a separate baseline dir).
    """
    results = {}
    for cfg in configs:
        results[cfg] = {}
        for tag, (sweep_subdir, fname_template) in policy_map.items():
            fname = fname_template.format(cfg=cfg)
            path = base / sweep_subdir / fname
            # Try alternate naming for default config (some sweeps omit suffix)
            if not path.exists() and cfg == "default_20_20_p50":
                alt = base / sweep_subdir / fname_template.format(cfg="default")
                if alt.exists():
                    path = alt
            rtt = read_rtt(path)
            # Try extra fallbacks when primary is missing
            if rtt is None and extra_fallbacks and tag in extra_fallbacks:
                for fb_sweep, fb_fname in extra_fallbacks[tag]:
                    fb_path = base / fb_sweep / fb_fname.format(cfg=cfg)
                    rtt = read_rtt(fb_path)
                    if rtt is not None:
                        break
            results[cfg][tag] = rtt
    return results


def main():
    parser = argparse.ArgumentParser(description="Analyze mega experiment matrix results.")
    parser.add_argument("--sweep-base", type=Path,
                        default=ROOT / "simulation_data" / "normal_sim_sweeps",
                        help="Base directory containing all sweep subdirs")
    parser.add_argument("--group", choices=["1", "2", "3", "4", "all"], default="all",
                        help="Which group(s) to print (default: all)")
    args = parser.parse_args()

    base = args.sweep_base
    groups = ["1", "2", "3", "4"] if args.group == "all" else [args.group]

    if "1" in groups:
        # Knative default_20_20_p50 lives in a separate baseline dir
        g1_fallbacks = {
            "knative": [("baseline_default_100100/results", "knative_{cfg}.json")],
        }
        r1 = load_group(base, G1_CONFIGS, G1_POLICIES, extra_fallbacks=g1_fallbacks)
        print_table(
            "Group 1 — Mega Compare All7  (workload-100-100, legacy physics, seed 42)",
            G1_CONFIGS, list(G1_POLICIES.keys()), r1,
            ref_policy="gnn_dim14ce",
        )

    if "2" in groups:
        r2 = load_group(base, G2_CONFIGS, G2_POLICIES, extra_fallbacks={})
        print_table(
            "Group 2 — Bipartite v2 (workload-125-225, node_disk_v2, seed 42)",
            G2_CONFIGS, list(G2_POLICIES.keys()), r2,
            ref_policy="knative_v1",
        )

    if "3" in groups:
        r3 = load_group(base, G3_CONFIGS, G3_POLICIES, extra_fallbacks={})
        print_table(
            "Group 3 — Skew3 Full Gate  (workload-100-100, node_disk_v2, seed 42)",
            G3_CONFIGS, list(G3_POLICIES.keys()), r3,
            ref_policy="knative",
        )

    if "4" in groups:
        r4 = load_group(base, G4_CONFIGS, G4_POLICIES, extra_fallbacks={})
        print_table(
            "Group 4 — Skew4 New Models  (workload-125-225, node_disk_v2, seed 42)",
            G4_CONFIGS, list(G4_POLICIES.keys()), r4,
            ref_policy="knative_v2",
        )

    print("Legend: all RTT in millions of seconds (lower is better).")
    print("Δ%GNN/Kn = (GNN − Knative) / Knative × 100  (negative = GNN better).")
    print("Δ%GNN/MLP = (GNN − MLP) / MLP × 100          (negative = GNN better).")
    print("— = result not yet available on this machine.")


if __name__ == "__main__":
    main()
