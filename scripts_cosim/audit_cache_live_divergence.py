#!/usr/bin/env python3
"""Audit: where do the cache and live feature builders actually disagree?

Two independent divergences, measured separately because only one of them matters.

1. `platform_order` — BENIGN, and universal
--------------------------------------------
The cache enumerates platforms from `stats.nodeResults`; live enumerates from
`config.infrastructure.nodes` (`src/placement/simulation.py:112-163` assigns `platform_id`
from a global counter walking config order, and `_collect_platforms_info` walks nodes in
that same order). These orders differ on essentially every corpus in the repo.

It looks alarming -- graph position *is* how a platform is addressed, since the scorer reads
`platform_emb[edge_index[1] - n_tasks]` with nothing carrying the platform's id into the
lookup. But `TaskPlacementGNN` has **no per-position parameter**: the platform encoder is
applied row-wise and edges are relabelled consistently with the rows. A different order is a
relabelling, not a mismatch. Measured on `netc_multihop_v1_core4/ds_00000` (208 platforms,
74 rows moved): matching platforms by `(node_name, platform_id)` makes bipartite edges,
candidate sets, `node_edge_index`, `task_features` and `edge_attr` **identical**, and with
the dim 9-11 estimates equalized per-candidate logits agree to **3e-8**.

So this check exists to *document* the reordering, not to condemn it -- and to stop the next
person reading a wall of positional diffs as a train/serve bug. `verify_cache_live_feature_parity.py`
compares by identity for exactly this reason.

2. `temporal_estimate` — was REAL, FIXED 2026-08-19
---------------------------------------------------
Dims 9-11 (`current_task_remaining`, `cold_start_remaining`, `comm_remaining`) are estimated
from queue depth when no remainder was recorded. Two bugs lived in that estimate:

* the three cache builders gated it **per snapshot** (`if temporal_state: ... else:
  <estimate>`) while live gated it **per platform**, so a queued platform with no recorded
  remainder trained on 0.0 and served an estimate;
* live averaged execution time over **every** key of `task-types.json`, pulling in `rf` and
  `cnn` which no corpus dispatches — and `cnn`'s 3.09s on `rpiCpu` inflated the estimate
  9.5x (0.0815 served where 0.0086 is correct).

Both are fixed: the formula now lives once in `src/placement/temporal_features.py` and all
four call sites use it. **What this check now measures is the blast radius of that fix** —
the datasets and platforms where the estimator fires at all, i.e. where a cache built before
2026-08-19 differs from one built after. It is no longer a divergence between the two paths.
Any collection listed here needs a recache before a model trained on it is trusted.

Both checks are cheap and need no cache and no training.

Usage:
    pipenv run python3 scripts_cosim/audit_cache_live_divergence.py
    pipenv run python3 scripts_cosim/audit_cache_live_divergence.py --all-datasets
    pipenv run python3 scripts_cosim/audit_cache_live_divergence.py --collection netc_multihop_v1_core4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import orjson

    def _load(path: Path) -> Any:
        return orjson.loads(path.read_bytes())
except ImportError:  # pragma: no cover - orjson is in the Pipfile
    def _load(path: Path) -> Any:
        return json.loads(path.read_text())


SIM_DATA = PROJECT_ROOT / "simulation_data"
REGISTRY = SIM_DATA / "REGISTRY.json"


def cache_platform_order(result: Dict[str, Any]) -> List[Tuple[int, int]]:
    """`(node_id, platform_id)` in the order the cache builder enumerates them."""
    order: List[Tuple[int, int]] = []
    for node_result in result.get("stats", {}).get("nodeResults", []):
        node_id = node_result.get("nodeId")
        if node_id is None:
            continue
        for plat_result in node_result.get("platformResults", []):
            plat_id = plat_result.get("platformId")
            if plat_id is None:
                continue
            order.append((int(node_id), int(plat_id)))
    return order


def audit_dataset(path: Path) -> Optional[Dict[str, Any]]:
    """None when the dataset carries no platform results to compare."""
    result = _load(path)
    cache_order = cache_platform_order(result)
    if not cache_order:
        return None
    live_order = sorted(cache_order)
    if cache_order == live_order:
        return {"dataset": path.parent.name, "diverges": False, "n_platforms": len(cache_order)}

    first = next(
        i for i, (c, l) in enumerate(zip(cache_order, live_order)) if c != l
    )
    node_ids = [n for n, _ in cache_order]
    return {
        "dataset": path.parent.name,
        "diverges": True,
        "n_platforms": len(cache_order),
        "first_divergent_position": first,
        "cache_at_first": list(cache_order[first]),
        "live_at_first": list(live_order[first]),
        "node_ids_sorted": node_ids == sorted(node_ids),
        # How many platforms sit at a position that maps to a different platform. This is
        # the count of logits that would score the wrong platform.
        "misplaced_platforms": sum(1 for c, l in zip(cache_order, live_order) if c != l),
    }


def audit_temporal_estimate(ds_dir: Path) -> Optional[Dict[str, Any]]:
    """Platforms where the dims 9-11 estimator fires, i.e. where a pre-fix cache is stale.

    Needs `system_state_captured_unique.json`; returns None without it.

    The condition is the one both paths now share (`temporal_features.temporal_remainders`):
    the snapshot carries some recorded temporal data, this platform has queue > 0, and its
    recorded `current_task_remaining` is absent or zero. Before 2026-08-19 the cache wrote
    0.0 for exactly this set while live estimated; now both estimate, so a cache built
    before that date differs from one built after on precisely these platforms.
    """
    ssc = ds_dir / "system_state_captured_unique.json"
    if not ssc.is_file():
        return None
    payload = _load(ssc)
    placements = payload.get("task_placements") or []
    if not placements:
        return None
    queue = placements[0].get("full_queue_snapshot") or {}
    temporal = placements[0].get("full_temporal_state_at_scheduling") or {}
    if not queue:
        return None
    # Cache only takes the recorded branch when the snapshot has temporal data at all.
    # With no temporal data both paths estimate and there is nothing to diverge.
    if not temporal:
        return {"dataset": ds_dir.name, "diverges": False, "n_divergent": 0, "n_queued": 0}

    n_queued = 0
    divergent = 0
    for key, depth in queue.items():
        try:
            depth_i = int(depth if not isinstance(depth, dict) else depth.get("length", 0))
        except (TypeError, ValueError):
            continue
        if depth_i <= 0:
            continue
        n_queued += 1
        recorded = (temporal.get(key) or {}).get("current_task_remaining", 0.0)
        try:
            recorded_f = float(recorded)
        except (TypeError, ValueError):
            recorded_f = 0.0
        if recorded_f == 0.0:
            divergent += 1

    return {
        "dataset": ds_dir.name,
        "diverges": divergent > 0,
        "n_divergent": divergent,
        "n_queued": n_queued,
    }


def audit_collection(base: Path, *, limit: Optional[int]) -> Dict[str, Any]:
    ds_dirs = sorted(d for d in base.glob("ds_*") if d.is_dir())
    if limit:
        ds_dirs = ds_dirs[:limit]

    checked: List[Dict[str, Any]] = []
    no_platform_results = 0
    unreadable = 0
    temporal_rows: List[Dict[str, Any]] = []
    for ds in ds_dirs:
        opt = ds / "optimal_result.json"
        if not opt.is_file():
            continue
        try:
            row = audit_dataset(opt)
            t_row = audit_temporal_estimate(ds)
        except Exception as exc:  # noqa: BLE001 - report, never skip silently
            unreadable += 1
            checked.append({"dataset": ds.name, "error": str(exc)})
            continue
        if t_row is not None:
            temporal_rows.append(t_row)
        if row is None:
            no_platform_results += 1
            continue
        checked.append(row)

    diverging = [r for r in checked if r.get("diverges")]
    temporal_bad = [r for r in temporal_rows if r["diverges"]]
    return {
        "collection": base.name,
        "datasets_checked": len([r for r in checked if "error" not in r]),
        "datasets_unreadable": unreadable,
        "datasets_without_platform_results": no_platform_results,
        # 1. Ordering — expected, benign, documented in the module docstring.
        "platform_order_differs": len(diverging),
        "worst_misplaced_platforms": (
            max(r["misplaced_platforms"] for r in diverging) if diverging else 0
        ),
        # 2. Temporal estimate — the real one.
        "temporal_datasets_with_ssc": len(temporal_rows),
        "temporal_datasets_diverging": len(temporal_bad),
        "temporal_worst_platforms": (
            max(r["n_divergent"] for r in temporal_bad) if temporal_bad else 0
        ),
        "verdict": (
            "STALE_CACHE" if temporal_bad
            else "order-only" if diverging
            else "clean" if checked else "no data"
        ),
        "examples": diverging[:2],
        "temporal_examples": temporal_bad[:2],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--collection",
        action="append",
        help="Collection dir name (repeatable). Default: every collection in REGISTRY.json.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Datasets per collection (default 5). Ordering is a property of the generator, "
        "so a handful is representative; use --all-datasets to be sure.",
    )
    parser.add_argument("--all-datasets", action="store_true", help="Check every dataset.")
    parser.add_argument("--out", type=Path, help="Write the full report as JSON.")
    args = parser.parse_args()

    if args.collection:
        names = args.collection
    else:
        registry = _load(REGISTRY)
        names = sorted(registry.get("collections", {}))

    limit = None if args.all_datasets else args.limit
    reports = []
    for name in names:
        base = SIM_DATA / name
        if not base.is_dir():
            print(f"  ?  {name}: directory not found", flush=True)
            continue
        report = audit_collection(base, limit=limit)
        reports.append(report)
        mark = {"STALE_CACHE": "!!", "order-only": "ok", "clean": "ok", "no data": " -"}[
            report["verdict"]
        ]
        print(
            f"  {mark} {name}: order {report['platform_order_differs']}"
            f"/{report['datasets_checked']}"
            f" (max {report['worst_misplaced_platforms']} rows moved)"
            f" | est-fires {report['temporal_datasets_diverging']}"
            f"/{report['temporal_datasets_with_ssc']}"
            + (
                f" (max {report['temporal_worst_platforms']} platforms)"
                if report["temporal_datasets_diverging"]
                else ""
            ),
            flush=True,
        )

    bad = [r for r in reports if r["verdict"] == "STALE_CACHE"]
    order_only = [r for r in reports if r["platform_order_differs"]]
    print("\n=== SUMMARY ===")
    print(f"collections checked                        : {len(reports)}")
    print(f"collections with reordered platforms (benign): {len(order_only)}")
    print(f"collections needing a recache (dims 9-11)    : {len(bad)}")
    for r in bad:
        print(
            f"  - {r['collection']}: "
            f"{r['temporal_datasets_diverging']}/{r['temporal_datasets_with_ssc']} datasets, "
            f"up to {r['temporal_worst_platforms']} platforms"
        )
    if not bad:
        print("  none — the dims 9-11 estimator fires nowhere SSC is available")

    if args.out:
        args.out.write_text(json.dumps({"collections": reports}, indent=2) + "\n")
        print(f"\nwrote {args.out}")

    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
