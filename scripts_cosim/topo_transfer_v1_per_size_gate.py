#!/usr/bin/env python3
"""Post-hoc per-size breakout of the topology_transfer_v1 Phase 4 gate.

The frozen gate reports (simulation_data/topo_transfer_v1_phase4_seed{42..46}.json)
pool held-out sizes 60+80 into one `paired_comparisons` entry per model. That pooling
makes it impossible to tell whether gnn_base's small observed loss is uniform noise
matching the drift-anchor control (scripts_cosim/drift_anchor_check.py) at both sizes,
or whether it diverges from the anchor at one size specifically. No retraining needed --
`results[model]['per_ds']` already has per-dataset regret for every held-out dataset;
this just re-buckets by server_node_count (recovered from infrastructure.json) and
re-runs paired_regret_comparison + pool_seed_comparisons per size.

Usage:
  pipenv run python3 scripts_cosim/topo_transfer_v1_per_size_gate.py \
      simulation_data/topo_transfer_v1_phase4_seed*.json --corpus-root simulation_data
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Dict, List

from scripts_cosim.gate_statistics import paired_regret_comparison, pool_seed_comparisons


def server_count(corpus_root: Path, rel_ds_path: str) -> int | None:
    infra_path = corpus_root / rel_ds_path / "infrastructure.json"
    if not infra_path.exists():
        return None
    infra = json.loads(infra_path.read_text())
    names = infra.get("network_maps", {})
    return sum(1 for n in names if not str(n).startswith("client_node"))


def bucket_per_ds(per_ds: Dict[str, float], corpus_root: Path,
                   size_cache: Dict[str, int]) -> Dict[int, Dict[str, float]]:
    buckets: Dict[int, Dict[str, float]] = {}
    for rel_ds, regret in per_ds.items():
        if rel_ds not in size_cache:
            size_cache[rel_ds] = server_count(corpus_root, rel_ds)
        size = size_cache[rel_ds]
        if size is None:
            continue
        buckets.setdefault(size, {})[rel_ds] = regret
    return buckets


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("reports", nargs="+")
    ap.add_argument("--corpus-root", type=Path, default=Path("simulation_data"))
    ap.add_argument("--models", nargs="+", default=["gnn_base", "gnn_node"])
    args = ap.parse_args()

    report_paths: List[Path] = []
    for pattern in args.reports:
        report_paths.extend(Path(p) for p in sorted(glob.glob(pattern)))

    size_cache: Dict[str, int] = {}
    # model -> size -> list of per-seed paired_regret_comparison dicts
    per_model_size_seed: Dict[str, Dict[int, List[dict]]] = {m: {} for m in args.models}

    for path in report_paths:
        data = json.loads(path.read_text())
        results = data["results"]
        ref_per_ds = results["pointwise"]["per_ds"]
        ref_buckets = bucket_per_ds(ref_per_ds, args.corpus_root, size_cache)

        for model in args.models:
            if model not in results:
                continue
            model_per_ds = results[model]["per_ds"]
            model_buckets = bucket_per_ds(model_per_ds, args.corpus_root, size_cache)
            for size, model_ds in model_buckets.items():
                ref_ds = ref_buckets.get(size, {})
                cmp = paired_regret_comparison(model_ds, ref_ds)
                per_model_size_seed[model].setdefault(size, []).append(cmp)

    print(f"Loaded {len(report_paths)} seed reports.\n")

    for model in args.models:
        print(f"=== {model} vs pointwise, per held-out size ===")
        for size in sorted(per_model_size_seed[model]):
            seed_cmps = per_model_size_seed[model][size]
            pooled = pool_seed_comparisons(seed_cmps)
            per_seed_wr = [c.get("win_rate") for c in seed_cmps if c.get("n_paired")]
            print(f"  size={size}: n_seeds={pooled.get('n_seeds')} "
                  f"n_paired/seed(min)={pooled.get('n_paired')} "
                  f"win_rate={pooled.get('win_rate'):.4f} "
                  f"ci95={[round(v, 4) for v in pooled.get('win_rate_ci95', [])]} "
                  f"per_seed_win_rate={[round(v, 4) for v in per_seed_wr]}")
        sizes = sorted(per_model_size_seed[model])
        if len(sizes) >= 2:
            wr_by_size = [pool_seed_comparisons(per_model_size_seed[model][s])["win_rate"]
                          for s in sizes]
            print(f"  win_rate trend {sizes[0]}->{sizes[-1]}: "
                  f"{wr_by_size[0]:.4f} -> {wr_by_size[-1]:.4f} "
                  f"(delta={wr_by_size[-1] - wr_by_size[0]:+.4f})")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
