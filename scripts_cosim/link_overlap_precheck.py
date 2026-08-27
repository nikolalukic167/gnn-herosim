#!/usr/bin/env python3
"""P0 pre-check for link_contention_v1 — does route overlap even exist? No simulation.

Every coupling mechanism before this one was built first and measured second, and three of
them died at the gate after a corpus had been generated. The Hall's-condition free-spreading
check (theta*) caught `netc_scarce_v1` and `netc_funnel_v1` *before* they cost a corpus by
measuring the structural precondition directly. This is the same instinct for links: if two
tasks' routes almost never share a link, no capacity value can make link congestion couple
anything, and the whole lineage should stop here.

It overlays a candidate backbone on an existing corpus's topology and replays that corpus's
real enumerated plans (`placements/placements.jsonl`) through the resulting route table. No
simulator is invoked and no RTT is recomputed -- the question is purely which links a plan
loads.

THE DECISION STATISTIC is `node_blind_share_frac`: among task pairs that contend on a shared
link, the fraction whose two tasks landed on *different destination nodes*. Those are exactly
the pairs the one-integer control cannot express, because node-occupancy excess only ever
counts co-location at a destination. A high `pair_share_frac` with a low
`node_blind_share_frac` means the sharing is just co-location wearing a link costume, and the
mechanism will collapse under `--gate-one-integer-repair` like the four before it.

`core_link_load_p90` and `distinct_loaded_core_links` guard the other failure mode: if all the
load lands on one segment, "load on the busiest link" becomes a single scalar that repairs
everything and the degeneracy has just moved up a level.

Note on determinism: the backbone is drawn from `random.Random(seed)` fresh rather than
resumed from the generator's RNG state mid-stream, so the attachment draw here will not be
byte-identical to what `generate_infrastructure.py` would produce for the same seed. That is
fine and deliberate -- this measures the *structure* a backbone of these parameters induces,
which is what tuning `n_core` and `attach_degree` needs. It is not a corpus.

Run:
  pipenv run python scripts_cosim/link_overlap_precheck.py \
      simulation_data/gnn_datasets_4tasks_shallow_v1 --limit 30
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.generate_infrastructure import build_core_backbone  # noqa: E402
from src.placement.network_fabric import is_core_link, route_links  # noqa: E402


def node_index_to_name(index: int, n_clients: int) -> str:
    """Plans store node indices; the generator builds clients first, then servers."""
    return f"client_node{index}" if index < n_clients else f"node{index - n_clients}"


def load_plans(ds_dir: Path, limit: int) -> List[Dict[str, Any]]:
    plans: List[Dict[str, Any]] = []
    jsonl = ds_dir / "placements" / "placements.jsonl"
    if not jsonl.exists():
        raise FileNotFoundError(
            f"{ds_dir.name}: missing placements.jsonl. The full placement sweep is a hard "
            f"requirement (docs/notes/placements_jsonl_required.md); it is not optional here "
            f"either -- the pre-check needs real candidate plans, not synthetic ones."
        )
    with jsonl.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            plans.append(json.loads(line))
            if limit and len(plans) >= limit:
                break
    if not plans:
        raise ValueError(f"{ds_dir.name}: placements.jsonl is empty")
    return plans


def analyse_dataset(
    ds_dir: Path,
    backbone_config: Dict[str, Any],
    max_plans: int,
) -> Optional[Dict[str, Any]]:
    infrastructure = json.loads((ds_dir / "infrastructure.json").read_text())
    space = json.loads((ds_dir / "space_with_network.json").read_text())
    workload = json.loads((ds_dir / "workload.json").read_text())

    n_clients = int(space["nodes"]["client_nodes"]["count"])
    n_servers = int(space["nodes"]["server_nodes"]["count"])
    seed = int(infrastructure.get("metadata", {}).get("seed", 0))

    nodes = (
        [{"node_name": f"client_node{i}", "type": "rpi"} for i in range(n_clients)]
        + [{"node_name": f"node{i}", "type": "rpi"} for i in range(n_servers)]
    )
    network_maps = {name: dict(edges) for name, edges in infrastructure["network_maps"].items()}

    link_topology = build_core_backbone(
        network_maps, nodes, {"network": {"backbone": backbone_config}}, random.Random(seed),
        seed=seed,
    )
    if link_topology is None:
        raise ValueError("backbone config produced no link topology")
    routes = link_topology["routes"]

    sources = [event["node_name"] for event in workload["events"]]
    plans = load_plans(ds_dir, max_plans)

    pair_total = 0
    pair_shared = 0
    pair_shared_core = 0
    pair_shared_core_distinct_dest = 0
    core_loads: List[int] = []
    distinct_loaded: List[int] = []

    for plan in plans:
        placement = plan["placement_plan"]
        task_links: Dict[str, set] = {}
        task_dest: Dict[str, str] = {}
        for task_id, (node_index, _platform) in placement.items():
            dest = node_index_to_name(int(node_index), n_clients)
            src = sources[int(task_id)]
            task_dest[task_id] = dest
            # Local execution never traverses the network, exactly as the simulator's
            # `task.node_name != self.node.node_name` guard has it.
            task_links[task_id] = set() if dest == src else set(route_links(routes, src, dest))

        load = Counter()
        for links in task_links.values():
            for key in links:
                load[key] += 1
        core_contended = [key for key, count in load.items() if count > 1 and is_core_link(key)]
        distinct_loaded.append(len(core_contended))
        core_loads.append(max((load[key] for key in load if is_core_link(key)), default=0))

        for a, b in combinations(sorted(task_links), 2):
            pair_total += 1
            shared = task_links[a] & task_links[b]
            if not shared:
                continue
            pair_shared += 1
            if any(is_core_link(key) for key in shared):
                pair_shared_core += 1
                # THE statistic: contention between tasks on *different* destination
                # nodes is invisible to a node-occupancy-excess repair column.
                if task_dest[a] != task_dest[b]:
                    pair_shared_core_distinct_dest += 1

    return {
        "dataset_id": ds_dir.name,
        "n_plans": len(plans),
        "n_links": len(link_topology["links"]),
        "n_core_links": sum(1 for key in link_topology["links"] if is_core_link(key)),
        "pair_share_frac": pair_shared / pair_total if pair_total else 0.0,
        "core_pair_share_frac": pair_shared_core / pair_total if pair_total else 0.0,
        "node_blind_share_frac": (
            pair_shared_core_distinct_dest / pair_shared_core if pair_shared_core else None
        ),
        "mean_distinct_loaded_core_links": statistics.fmean(distinct_loaded) if distinct_loaded else 0.0,
        "max_core_link_load": max(core_loads) if core_loads else 0,
        "mean_core_link_load": statistics.fmean(core_loads) if core_loads else 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus_dir", type=Path)
    ap.add_argument("--limit", type=int, default=30, help="datasets to analyse")
    ap.add_argument("--max-plans", type=int, default=400, help="plans per dataset")
    ap.add_argument("--n-core", type=int, default=6)
    ap.add_argument("--attach-degree", type=int, default=2)
    ap.add_argument("--chord-count", type=int, default=None)
    ap.add_argument("--core-link-latency-ms", type=float, default=4.0)
    ap.add_argument("--access-link-latency-ms", type=float, default=20.0)
    ap.add_argument("--bandwidth-mbps", type=float, default=1.5)
    ap.add_argument("--output", type=Path)
    ap.add_argument(
        "--gate-node-blind-share",
        type=float,
        default=None,
        help=(
            "Fail (exit 1) if the mean fraction of core-link-sharing task pairs with "
            "DIFFERENT destination nodes is below this. Those pairs are the only ones a "
            "node-occupancy-excess repair column cannot express, so they are the entire "
            "reason to expect this mechanism to survive --gate-one-integer-repair. "
            "Suggested 0.5."
        ),
    )
    args = ap.parse_args()

    backbone_config: Dict[str, Any] = {
        "n_core": args.n_core,
        "attach_degree": args.attach_degree,
        "core_link_latency_ms": args.core_link_latency_ms,
        "access_link_latency_ms": args.access_link_latency_ms,
        "bandwidth_mbps": args.bandwidth_mbps,
    }
    if args.chord_count is not None:
        backbone_config["chord_count"] = args.chord_count

    ds_dirs = sorted(d for d in args.corpus_dir.glob("ds_*") if d.is_dir())
    if args.limit:
        ds_dirs = ds_dirs[: args.limit]
    if not ds_dirs:
        raise SystemExit(f"no ds_* directories under {args.corpus_dir}")

    rows = [analyse_dataset(d, backbone_config, args.max_plans) for d in ds_dirs]

    def mean_of(key: str) -> float:
        values = [r[key] for r in rows if r[key] is not None]
        return statistics.fmean(values) if values else 0.0

    node_blind_coverage = sum(1 for r in rows if r["node_blind_share_frac"] is not None) / len(rows)

    summary = {
        "datasets_analysed": len(rows),
        "backbone": backbone_config,
        "pair_share_frac": mean_of("pair_share_frac"),
        "core_pair_share_frac": mean_of("core_pair_share_frac"),
        "node_blind_share_frac": mean_of("node_blind_share_frac"),
        "node_blind_share_frac_coverage": node_blind_coverage,
        "mean_distinct_loaded_core_links": mean_of("mean_distinct_loaded_core_links"),
        "mean_core_link_load": mean_of("mean_core_link_load"),
        "max_core_link_load": max(r["max_core_link_load"] for r in rows),
        "n_core_links": rows[0]["n_core_links"],
    }

    print(f"\nlink_contention_v1 P0 overlap pre-check — {args.corpus_dir}")
    print(f"  backbone: n_core={args.n_core} attach_degree={args.attach_degree} "
          f"core_links={summary['n_core_links']}")
    print(f"  datasets={summary['datasets_analysed']}")
    print(f"  task pairs sharing ANY link          : {summary['pair_share_frac']:.3f}")
    print(f"  task pairs sharing a CORE link       : {summary['core_pair_share_frac']:.3f}")
    print(f"  ...of those, DIFFERENT destinations  : {summary['node_blind_share_frac']:.3f}  "
          f"<- node-occupancy-blind share")
    print(f"  distinct contended core links / plan : {summary['mean_distinct_loaded_core_links']:.2f}")
    print(f"  core link load  mean / max           : {summary['mean_core_link_load']:.2f} / "
          f"{summary['max_core_link_load']}")

    if args.output:
        args.output.write_text(json.dumps({"summary": summary, "datasets": rows}, indent=2))
        print(f"  wrote {args.output}")

    if args.gate_node_blind_share is not None:
        # node_blind_share_frac is conditioned on datasets that had ANY core-link overlap
        # (see module docstring: it's a fraction *of those pairs*). If most datasets had no
        # overlap at all, the mean is computed over a small, unrepresentative subset and can
        # PASS even though route overlap essentially doesn't exist -- the P0 question this
        # script exists to answer. core_pair_share_frac is the unconditional signal for that.
        if node_blind_coverage < 0.5:
            print(
                f"\n[gate] FAIL LOUD: node_blind_share_frac is backed by only "
                f"{node_blind_coverage:.1%} of datasets (core_pair_share_frac="
                f"{summary['core_pair_share_frac']:.4f} overall) -- route overlap barely "
                "exists, so the conditional statistic is not a meaningful gate here."
            )
            return 1
        ok = summary["node_blind_share_frac"] >= args.gate_node_blind_share
        verdict = "PASS" if ok else "FAIL"
        print(
            f"\n[gate] node_blind_share_frac {summary['node_blind_share_frac']:.3f} "
            f"(must be >= {args.gate_node_blind_share:.3f}, coverage="
            f"{node_blind_coverage:.1%}) -> {verdict}"
        )
        if not ok:
            print(
                "  Sharing that only happens between tasks on the SAME destination node is "
                "co-location wearing a link costume; one node-occupancy integer will repair "
                "it, as it did for the four mechanisms before this one."
            )
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
