#!/usr/bin/env python3
"""workload_fix_v1 W3 / W4 topology configs for the 19 test topologies, and the W4 static checks.

  workload_fix_v1_stage_topologies.py --cfg-dir <96-config pool> --selected <selected.json> --out <dir>

writes <dir>/cfg_w3 (pool config + network.backbone.access_classes, mix 40/40/20) and <dir>/cfg_w4 (the same plus
network.reachability_repair = {task_types: all}) for each selected topology, then

  1. runs the reachability check over all of cfg_w4 (every task type reachable from every client), and
  2. generates each topology under W3 and under W4 and diffs the two infrastructure dicts. The repair adds
     client<->server edges to ``network_maps`` (it adds no replicas; replicas come from the placement step before it),
     so the diff lists every added edge and every other key that differs.

Writes <dir>/reach_w4.json and <dir>/w4_vs_w3.json. Exit 1 if any topology fails the check.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts_cosim"))

from src.generate_infrastructure import generate_deterministic_infrastructure  # noqa: E402
from src.placement.network_fabric import DEFAULT_ACCESS_MIX  # noqa: E402
from workload_fix_v1_reachability_check import check_infrastructure  # noqa: E402

REPAIR = {"task_types": "all"}


def w3_config(cfg: dict) -> dict:
    out = json.loads(json.dumps(cfg))
    backbone = out["network"].get("backbone")
    if backbone is None:
        raise SystemExit("FAIL LOUD: config has no network.backbone; access classes need the link fabric")
    if "access_classes" in backbone or "reachability_repair" in out["network"]:
        raise SystemExit("FAIL LOUD: pool config already carries W3/W4 keys")
    backbone["access_classes"] = {"mix": dict(DEFAULT_ACCESS_MIX)}
    return out


def w4_config(cfg: dict) -> dict:
    out = json.loads(json.dumps(cfg))
    out["network"]["reachability_repair"] = dict(REPAIR)
    return out


def generate(cfg: dict, sim_input: Path) -> dict:
    seed = int(cfg["network"]["topology"]["seed"])
    with tempfile.TemporaryDirectory() as tmp:
        space = Path(tmp) / "space_with_network.json"
        space.write_text(json.dumps(cfg))
        with contextlib.redirect_stdout(io.StringIO()):
            return generate_deterministic_infrastructure(str(space), sim_input, str(Path(tmp) / "infrastructure.json"), seed)


def diff_infra(w3: dict, w4: dict) -> dict:
    added_edges = []
    for a, row in w4["network_maps"].items():
        for b in row:
            if b not in w3["network_maps"].get(a, {}) and a < b:
                added_edges.append([a, b])
    differing = sorted(k for k in set(w3) | set(w4) if json.dumps(w3.get(k), sort_keys=True) != json.dumps(w4.get(k), sort_keys=True))
    replica_diff = json.dumps(w3["replica_placements"], sort_keys=True) != json.dumps(w4["replica_placements"], sort_keys=True)
    access_same = json.dumps((w3.get("link_topology") or {}).get("access_classes"), sort_keys=True) == \
        json.dumps((w4.get("link_topology") or {}).get("access_classes"), sort_keys=True)
    return {"added_edges": sorted(added_edges), "keys_differing": differing, "replica_placements_differ": replica_diff,
            "access_classes_identical": access_same}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cfg-dir", type=Path, required=True)
    ap.add_argument("--selected", type=Path, required=True)
    ap.add_argument("--sim-input", type=Path, default=ROOT / "data" / "nofs-ids")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    seeds = list(json.loads(a.selected.read_text())["topologies"])
    task_types = list(json.loads((a.sim_input / "task-types.json").read_text()))
    for sub in ("cfg_w3", "cfg_w4"):
        (a.out / sub).mkdir(parents=True, exist_ok=True)
    reach_rows, diff_rows = [], []
    for seed in seeds:
        hits = sorted(a.cfg_dir.glob(f"*s{seed}.json"))
        if len(hits) != 1:
            raise SystemExit(f"FAIL LOUD: seed {seed}: expected one config in {a.cfg_dir}, found {len(hits)}")
        base = json.loads(hits[0].read_text())
        c3, c4 = w3_config(base), w4_config(w3_config(base))
        (a.out / "cfg_w3" / hits[0].name).write_text(json.dumps(c3, indent=1))
        (a.out / "cfg_w4" / hits[0].name).write_text(json.dumps(c4, indent=1))
        i3, i4 = generate(c3, a.sim_input), generate(c4, a.sim_input)
        res = check_infrastructure(i4, task_types)
        reach_rows.append({"topology": hits[0].stem, "seed": seed, **res})
        diff_rows.append({"topology": hits[0].stem, **diff_infra(i3, i4)})
    bad = [r for r in reach_rows if not r["ok"]]
    for r in reach_rows:
        print(("ok  " if r["ok"] else "BAD ") + r["topology"], "" if r["ok"] else {k: r[k] for k in ("no_server_replica", "unreachable_clients")})
    print(f"{len(reach_rows) - len(bad)}/{len(reach_rows)} W4 topologies reachable (all {len(task_types)} types, every client)")
    n_changed = sum(1 for d in diff_rows if d["added_edges"] or d["keys_differing"])
    for d in diff_rows:
        print(f"{d['topology']}: added edges {len(d['added_edges'])}, keys differing {d['keys_differing']}, "
              f"replicas differ {d['replica_placements_differ']}, access classes identical {d['access_classes_identical']}")
    print(f"{len(diff_rows) - n_changed}/{len(diff_rows)} W4 topologies are identical to W3; {n_changed} differ")
    (a.out / "reach_w4.json").write_text(json.dumps({"n": len(reach_rows), "n_bad": len(bad), "rows": reach_rows}, indent=1))
    (a.out / "w4_vs_w3.json").write_text(json.dumps({"rows": diff_rows}, indent=1))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
