#!/usr/bin/env python3
"""Does the live-magnitude queue column break the graph arm more than its pointwise twin?

Context (bug audit, 2026-09-12). The peer_affinity_v1 corpora were cached and trained under
`legacy_v0`, whose dim-7 divisor is `min(max(1, p90_over_all_platforms), 100)`. In every one
of the 516 training datasets that p90 is **0** (only 3.7 % of the 134 platforms carry any
queue, because replicas are server-only), so the divisor is 1.0 in 100 % of them and the
"normalized" queue column is the RAW depth, 0-42. Live on the production trace the same
column is the raw depth again (the divisor collapses to 1.0 on most batches for the same
reason) but the depths are 10^3-10^4 -- and on the batches where enough platforms are busy
the divisor jumps to its p90, so the column's UNITS change by up to 100x inside one run.
docs/adr/0002 already declares this fatal for legacy_v0 and says new training uses
`scale_invariant_v1`; this corpus did not.

This probe isolates the effect with no simulator: take the held-out co-sim datasets, present
the SAME state with the queue columns multiplied by k (which is exactly what the live venue
does to them, since the divisor is 1.0 in both venues), and re-measure each arm's decode
regret against the brute-force optimum. The label never moves -- only what the model is
shown. If the graph arm's regret degrades faster in k than its MP-OFF twin's, the offline/live
sign reversal has a mechanism: message passing carries the out-of-range platform embedding
into every task embedding, and PeerConv then mixes it across peers.

Columns scaled: 7 (normalized queue depth) and 13 (usage ratio) -- the two the contract
defines as queue-derived (src/placement/queue_features.py).
"""
from __future__ import annotations

import argparse
import json
import pickle
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "src" / "notebooks")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402

from scripts_cosim.eval_route_b_stage2_arm import (  # noqa: E402
    evaluate_dataset,
    load_arm,
    split_membership,
)
from non_unique_lib.training_contract import canonical_parent_id  # noqa: E402

QUEUE_COLUMNS = (7, 13)


def scaled_graph(graph: Any, k: float) -> Any:
    """A shallow clone whose queue columns are multiplied by k. Everything else -- the
    candidate lists, the caps, the peer edges, the partial-state context -- is shared, so
    the only difference the model can see is the queue magnitude."""
    if k == 1.0:
        return graph
    clone = graph.clone() if hasattr(graph, "clone") else graph
    pf = clone.platform_features.clone()
    for c in QUEUE_COLUMNS:
        pf[:, c] = pf[:, c] * float(k)
    clone.platform_features = pf
    # `clone()` on a PyG Data drops nothing we use, but be explicit about the non-tensor
    # attributes the evaluator reads, so a future PyG version cannot silently lose them.
    for attr in ("partial_state_ctx", "task_logit_to_placement", "dag_parents",
                 "node_caps_by_alpha", "dag_primary_alpha_key", "dag_task_type_vocab",
                 "queue_key_to_platform_meta", "queue_snapshot", "dataset_id"):
        if hasattr(graph, attr):
            setattr(clone, attr, getattr(graph, attr))
    return clone


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--split-artifact", type=Path, required=True)
    ap.add_argument("--alpha-key", default="2.5")
    ap.add_argument("--split", default="test")
    ap.add_argument("--models-dir", type=Path, default=Path("models"))
    ap.add_argument("--tag", required=True, help="checkpoint prefix, e.g. peer-affinity-v1-t1b")
    ap.add_argument("--lr", default="lr2e3")
    ap.add_argument("--arms", nargs="+", default=["gnn", "mpoff"])
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(1, 17)))
    ap.add_argument("--scales", type=float, nargs="+", default=[1.0, 10.0, 100.0, 300.0])
    ap.add_argument("--task-types", default="data/nofs-ids/task-types.json")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    with open(args.cache_dir / "graphs.pkl", "rb") as fh:
        graphs = pickle.load(fh)
    with open(args.cache_dir / "dataset_ids.pkl", "rb") as fh:
        dataset_ids = pickle.load(fh)
    membership, split_sha = split_membership(args.split_artifact)
    selected = [
        (g, did) for g, did in zip(graphs, dataset_ids)
        if membership.get(canonical_parent_id(did)) == args.split
    ]
    if not selected:
        raise SystemExit(f"no datasets in split {args.split!r}")
    print(f"[probe] {len(selected)} datasets in split {args.split}", flush=True)

    task_types_db = json.loads(Path(args.task_types).read_text())
    sim_root = REPO_ROOT / "simulation_data"

    rows: Dict[str, Dict[str, Any]] = {}
    for arm_name in args.arms:
        for seed in args.seeds:
            ck = args.models_dir / f"{args.tag}-{arm_name}-{args.lr}-seed{seed}.pt"
            if not ck.is_file():
                raise SystemExit(f"missing checkpoint: {ck}")
            arm = load_arm(ck)
            for k in args.scales:
                regrets: List[float] = []
                infeasible = 0
                for graph, dataset_id in selected:
                    res = evaluate_dataset(
                        scaled_graph(graph, k), dataset_id, arm=arm,
                        alpha_key=args.alpha_key, simulation_data_root=sim_root,
                        task_types_db=task_types_db,
                    )
                    if res["infeasible"]:
                        infeasible += 1
                    else:
                        regrets.append(res["decode_regret_pct"]["registered"])
                key = f"{arm_name}|seed{seed}|k{k:g}"
                rows[key] = {
                    "arm": arm_name, "seed": seed, "scale": k,
                    "median_regret_pct": statistics.median(regrets) if regrets else None,
                    "mean_regret_pct": statistics.fmean(regrets) if regrets else None,
                    "n": len(regrets), "infeasible": infeasible,
                }
                print(f"[probe] {key}: median={rows[key]['median_regret_pct']}", flush=True)

    summary: Dict[str, Any] = {}
    for k in args.scales:
        per_arm = {
            a: {r["seed"]: r["median_regret_pct"]
                for r in rows.values() if r["arm"] == a and r["scale"] == k}
            for a in args.arms
        }
        entry: Dict[str, Any] = {
            a: statistics.median([v for v in per_arm[a].values() if v is not None])
            for a in args.arms
        }
        if len(args.arms) == 2:
            a, b = args.arms
            seeds = sorted(set(per_arm[a]) & set(per_arm[b]))
            diffs = [per_arm[b][s] - per_arm[a][s] for s in seeds
                     if per_arm[a][s] is not None and per_arm[b][s] is not None]
            entry["n_pairs"] = len(diffs)
            entry[f"{b}_minus_{a}_pp_median"] = statistics.median(diffs) if diffs else None
            entry["a_better_seeds"] = sum(1 for d in diffs if d > 0)
            if len(diffs) > 1:
                from scipy.stats import wilcoxon
                entry["p_exact_wilcoxon"] = float(
                    wilcoxon(diffs, alternative="two-sided", mode="exact").pvalue)
        summary[f"k{k:g}"] = entry

    out = {
        "probe": "queue_scale_sensitivity",
        "cache_dir": str(args.cache_dir), "tag": args.tag, "lr": args.lr,
        "alpha_key": args.alpha_key, "split": args.split,
        "split_artifact": {"path": str(args.split_artifact), "sha256": split_sha},
        "n_datasets": len(selected), "queue_columns_scaled": list(QUEUE_COLUMNS),
        "per_checkpoint": rows, "summary": summary,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
