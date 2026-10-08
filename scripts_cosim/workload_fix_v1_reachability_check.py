#!/usr/bin/env python3
"""workload_fix_v1 W4 precondition -- static reachability check over topology configs.

For every config (cfg dir x topology ids) the infrastructure is generated exactly as a cell would be, and
each task type is checked from each client: at least one server hosting a replica of that type must be
a logical neighbour of the client (``network_maps``) and, when the topology has a link fabric, routed to it.
Exit status 1 if any (topology, client, type) is unreachable, or any type has no server replica at all.

  workload_fix_v1_reachability_check.py --cfg-dir simulation_data/small_batch_confirm_v1/inputs/cfg \\
      --selected simulation_data/small_batch_confirm_v1/inputs/selected.json \\
      --calibration-seeds 9501 9502 9503 9504 --out reach.json

``--repair`` sets ``network.reachability_repair = {"task_types": "all"}`` in memory before generating, to
check the fix; without it the config is generated as written. Topologies are named by their config stem
``cc40s<seed>``; ``--all`` checks every config in the directory.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.generate_infrastructure import generate_deterministic_infrastructure  # noqa: E402


def check_infrastructure(infra: Dict[str, Any], task_types: List[str]) -> Dict[str, Any]:
    """Pure check over a generated infrastructure dict."""
    network_maps = infra["network_maps"]
    placements = infra["replica_placements"]
    routes = ((infra.get("link_topology") or {}).get("routes")) or {}
    has_fabric = bool(infra.get("link_topology"))
    clients = sorted(n for n in network_maps if n.startswith("client_node"))

    def routed(a: str, b: str) -> bool:
        return not has_fabric or b in routes.get(a, {}) or a in routes.get(b, {})

    no_server_replica: List[str] = []
    unreachable: Dict[str, List[str]] = {}
    for task_type in task_types:
        servers = sorted({p["node_name"] for p in placements.get(task_type, []) if not p["node_name"].startswith("client_")})
        if not servers:
            no_server_replica.append(task_type)
            continue
        for client in clients:
            ok = [s for s in servers if s in network_maps[client] and routed(client, s)]
            if not ok:
                unreachable.setdefault(task_type, []).append(client)
    return {
        "ok": not no_server_replica and not unreachable,
        "no_server_replica": no_server_replica,
        "unreachable_clients": {t: c for t, c in sorted(unreachable.items())},
        "n_clients": len(clients),
    }


def run_topology(cfg_path: Path, sim_input: Path, repair: bool, task_types: Optional[List[str]]) -> Dict[str, Any]:
    cfg = json.loads(cfg_path.read_text())
    seed = cfg["network"]["topology"]["seed"]
    if repair:
        cfg["network"]["reachability_repair"] = {"task_types": "all"}
    all_types = list(json.loads((sim_input / "task-types.json").read_text()))
    types = task_types or all_types
    with tempfile.TemporaryDirectory() as tmp:
        space = Path(tmp) / "space_with_network.json"
        space.write_text(json.dumps(cfg))
        out = Path(tmp) / "infrastructure.json"
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                infra = generate_deterministic_infrastructure(str(space), sim_input, str(out), int(seed))
        except Exception as exc:  # a generator refusal is a result, not a crash of the sweep
            return {"topology": cfg_path.stem, "seed": seed, "ok": False, "generator_error": f"{type(exc).__name__}: {exc}"}
    res = check_infrastructure(infra, types)
    return {"topology": cfg_path.stem, "seed": seed, **res}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cfg-dir", type=Path, required=True)
    ap.add_argument("--sim-input", type=Path, default=ROOT / "data" / "nofs-ids")
    ap.add_argument("--selected", type=Path, help="selected.json whose `topologies` are the test seeds")
    ap.add_argument("--calibration-seeds", type=int, nargs="*", default=[])
    ap.add_argument("--all", action="store_true", help="every config in --cfg-dir")
    ap.add_argument("--task-types", nargs="*", help="default: every type in task-types.json")
    ap.add_argument("--repair", action="store_true")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()

    if a.all:
        cfgs = sorted(a.cfg_dir.glob("*.json"))
    else:
        if a.selected is None:
            raise SystemExit("FAIL LOUD: pass --selected (test topologies) and/or --all")
        seeds = list(json.loads(a.selected.read_text())["topologies"]) + list(a.calibration_seeds)
        cfgs = []
        for seed in seeds:
            hits = sorted(a.cfg_dir.glob(f"*s{seed}.json"))
            if len(hits) != 1:
                raise SystemExit(f"FAIL LOUD: seed {seed}: expected one config in {a.cfg_dir}, found {len(hits)}")
            cfgs.append(hits[0])
    rows = [run_topology(c, a.sim_input, a.repair, a.task_types) for c in cfgs]
    bad = [r for r in rows if not r["ok"]]
    for r in rows:
        status = "ok " if r["ok"] else "BAD"
        detail = r.get("generator_error") or (
            f"no-server-replica={r['no_server_replica']} unreachable={ {t: len(c) for t, c in r['unreachable_clients'].items()} }"
            if not r["ok"] else ""
        )
        print(f"{status} {r['topology']} {detail}")
    print(f"{len(rows) - len(bad)}/{len(rows)} topologies pass (repair={a.repair})")
    if a.out:
        a.out.write_text(json.dumps({"repair": a.repair, "n": len(rows), "n_bad": len(bad), "rows": rows}, indent=1))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
