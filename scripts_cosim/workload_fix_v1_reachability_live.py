#!/usr/bin/env python3
"""workload_fix_v1 W4, live path -- static reachability per (topology, window) on the generator a gate run uses.

`workload_fix_v1_reachability_check.py` checks `generate_deterministic_infrastructure` (the co-sim path, with replica
placements and the reachability repair). A gate run does not use it: `src/executesimulation.py` calls
`prepare_infrastructure_for_real_simulation` -> `generate_network_topology_deterministic`, which plans no replicas and
has no repair; the autoscaler creates replicas from zero. This check builds the topology through that function, with the
config as the cell reads it, and asks of each (window, client, task type) the question `create_first_replica` /
`scale_up` ask at run time (src/policy/gnn/autoscaler.py, src/placement/autoscaler.py):

  * the client must reach a server (the server's `network_map` holds it) that is routed to it on the link fabric;
  * that server needs a platform of a type the task can run on (`task-types.json`), allowed by
    HEROSIM_REPLICA_PLATFORM_TYPES, on a node whose memory holds the task;
  * with HEROSIM_SERVER_ONLY_REPLICAS=1 (every gate cell without `client_local_v1`) clients host no replica;
  * a platform holds one replica and a node's memory is shared, so the types one client sends in the window need
    DISTINCT reachable platforms whose memory fits together (exact, by search).

Only the clients a window sends from and the task types it sends are checked. The check is per client and necessary:
it cannot see two clients competing for the same platforms, so a pass is not a proof that the run completes.

  workload_fix_v1_reachability_live.py --cfg-dir <cfg> --topologies 9601 9602 --wl-dir <wl> --out live.json
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

WINDOWS = tuple(f"grounded_g{i}_n50000.json" for i in range(4))


def window_usage(path: Path) -> Dict[str, Set[str]]:
    """client -> task types its events send."""
    used: Dict[str, Set[str]] = {}
    for e in json.loads(path.read_text())["events"]:
        used.setdefault(e["node_name"], set()).update(e["application"]["dag"])
    return used


def reachable_platforms(infra: Dict[str, Any], task_types: Dict[str, Any], client: str, ttype: str,
                        server_only: bool, allowed: Optional[Set[str]]) -> Tuple[List[Tuple[str, int]], str]:
    """(server, platform index) pairs `ttype` could take for `client`, and the first reason the list is empty."""
    spec = task_types[ttype]
    routes = (infra.get("link_topology") or {}).get("routes") or {}
    has_fabric = bool(infra.get("link_topology"))
    out: List[Tuple[str, int]] = []
    reached = routed = typed = False
    for node in infra["nodes"]:
        name = node["node_name"]
        is_client = name.startswith("client_node")
        if is_client and (server_only or name != client):
            continue
        if not is_client and client not in node.get("network_map", {}):
            continue
        reached = True
        if has_fabric and not is_client and name not in routes.get(client, {}) and client not in routes.get(name, {}):
            continue
        routed = True
        for i, p in enumerate(node["platforms"]):
            if p not in spec["platforms"] or (allowed is not None and p not in allowed):
                continue
            typed = True
            if node["memory"] >= spec["memoryRequirements"][p]:
                out.append((name, i))
    if out:
        return out, ""
    return out, ("no-server-reaches-client" if not reached else "no-route" if not routed
                 else "no-compatible-platform" if not typed else "memory")


def distinct_fit(infra: Dict[str, Any], task_types: Dict[str, Any], options: Dict[str, List[Tuple[str, int]]]) -> bool:
    """One platform per replica, each node's memory shared: can every type in ``options`` get a replica at once?"""
    node_by = {n["node_name"]: n for n in infra["nodes"]}
    types = sorted(options, key=lambda t: len(options[t]))

    def go(k: int, taken: Set[Tuple[str, int]], mem: Dict[str, float]) -> bool:
        if k == len(types):
            return True
        t = types[k]
        for (name, i) in options[t]:
            if (name, i) in taken:
                continue
            node = node_by[name]
            need = task_types[t]["memoryRequirements"][node["platforms"][i]]
            if mem.get(name, 0.0) + need > node["memory"]:
                continue
            if go(k + 1, taken | {(name, i)}, {**mem, name: mem.get(name, 0.0) + need}):
                return True
        return False

    return go(0, set(), {})


def check_window(infra: Dict[str, Any], task_types: Dict[str, Any], used: Dict[str, Set[str]],
                 server_only: bool, allowed: Optional[Set[str]]) -> Dict[str, Any]:
    bad: Dict[str, Dict[str, str]] = {}
    slack = None
    for client in sorted(used):
        opts: Dict[str, List[Tuple[str, int]]] = {}
        for t in sorted(used[client]):
            o, why = reachable_platforms(infra, task_types, client, t, server_only, allowed)
            if not o:
                bad.setdefault(client, {})[t] = why
            opts[t] = o
        if client in bad:
            continue
        if not distinct_fit(infra, task_types, opts):
            bad[client] = {"*": "no-distinct-platforms"}
            continue
        n = len({p for o in opts.values() for p in o})
        slack = n if slack is None else min(slack, n)
    return {"ok": not bad, "n_clients": len(used), "bad_clients": bad, "min_reachable_platforms": slack}


def build_live_infrastructure(cfg: Dict[str, Any], sim_input: Path) -> Dict[str, Any]:
    from src.executesimulation import prepare_infrastructure_for_real_simulation

    with contextlib.redirect_stdout(io.StringIO()):
        return prepare_infrastructure_for_real_simulation(cfg, seed=None, sim_input_path=sim_input)


def run_topology(cfg_path: Path, sim_input: Path, usage: Dict[str, Dict[str, Set[str]]]) -> Dict[str, Any]:
    from src.placement.autoscaler import replica_platform_types_allowed

    cfg = json.loads(cfg_path.read_text())
    task_types = json.loads((sim_input / "task-types.json").read_text())
    infra = build_live_infrastructure(cfg, sim_input)
    server_only = not cfg.get("client_local_v1") and os.environ.get("HEROSIM_SERVER_ONLY_REPLICAS", "1") == "1"
    allowed = replica_platform_types_allowed()
    return {"topology": cfg_path.stem, "server_only": server_only,
            "windows": {w: check_window(infra, task_types, u, server_only, allowed) for w, u in usage.items()}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cfg-dir", type=Path, required=True)
    ap.add_argument("--topologies", type=int, nargs="+", required=True)
    ap.add_argument("--wl-dir", type=Path, required=True, help="a built rung's wl dir (clients and types do not depend on the rung)")
    ap.add_argument("--sim-input", type=Path, default=ROOT / "data" / "nofs-ids")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    usage = {w.replace("grounded_", "").replace("_n50000.json", ""): window_usage(a.wl_dir / w) for w in WINDOWS}
    rows = []
    for t in a.topologies:
        hits = sorted(a.cfg_dir.glob(f"*s{t}.json"))
        if len(hits) != 1:
            raise SystemExit(f"FAIL LOUD: seed {t}: expected one config in {a.cfg_dir}, found {len(hits)}")
        rows.append(run_topology(hits[0], a.sim_input, usage))
    bad = [(r["topology"], w, c) for r in rows for w, c in r["windows"].items() if not c["ok"]]
    for r in rows:
        cells = " ".join(f"{w}:{'ok' if c['ok'] else 'BAD(' + str(len(c['bad_clients'])) + ')'}/{c['min_reachable_platforms']}"
                         for w, c in r["windows"].items())
        print(f"{r['topology']} server_only={r['server_only']} {cells}")
    print(f"{4 * len(rows) - len(bad)}/{4 * len(rows)} cells pass; infeasible: {[(t, w) for t, w, _ in bad]}")
    if a.out:
        a.out.write_text(json.dumps({"windows_used": {w: {c: sorted(v) for c, v in u.items()} for w, u in usage.items()},
                                     "rows": rows}, indent=1))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
