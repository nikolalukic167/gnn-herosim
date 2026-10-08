#!/usr/bin/env python3
"""workload_fix_v1 W2+W3 -- build the W3 gate inputs on top of built W2 rungs, and prove the only change is the
access-link bandwidth fields.

For each rung tag the built W2 rung (<w2-inputs>/wf1_<tag>/{cfg,wl}) is copied: workloads byte for byte (sha256
checked against <w2-inputs>/manifest_<tag>.json), configs with `network.backbone.access_classes` added and nothing
else. Then for every topology the infrastructure is generated from the W2 config and from the W3 config by both
paths a cell uses -- `prepare_infrastructure_for_real_simulation` (the live path) and
`generate_deterministic_infrastructure` (nodes, replica placements, queues) -- and compared field by field.
Anything that differs outside the allowed set (access links' bandwidth fields, `access_classes`,
`params.access_classes`) fails loud.

  workload_fix_v1_w3_build.py --w2-inputs <dir> --tags lo hi --out <dir>/inputs --diff-out <dir>/diff.json
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import io
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.executecosimulation import load_simulation_inputs  # noqa: E402
from src.executesimulation import prepare_infrastructure_for_real_simulation  # noqa: E402
from src.generate_infrastructure import generate_deterministic_infrastructure  # noqa: E402
from src.placement.network_fabric import DEFAULT_ACCESS_MIX  # noqa: E402

ALLOWED_LINK_FIELDS = {"bandwidth_mbps", "access_node", "bandwidth_out_mbps", "bandwidth_in_mbps"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def diff(a: Any, b: Any, path: str, out: List[Dict[str, Any]]) -> None:
    """Field-level diff; each entry is {path, w2, w3} (or w2/w3 absent for a key present on one side only)."""
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b), key=str):
            p = f"{path}.{k}"
            if k not in a:
                out.append({"path": p, "w3": b[k]})
            elif k not in b:
                out.append({"path": p, "w2": a[k]})
            else:
                diff(a[k], b[k], p, out)
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b)):
            diff(x, y, f"{path}[{i}]", out)
    elif a != b:
        out.append({"path": path, "w2": a, "w3": b})


def allowed(entry: Dict[str, Any]) -> bool:
    p = entry["path"]
    if ".access_classes" in p:
        return True
    parts = p.split(".")
    return ".links." in p and parts[-1] in ALLOWED_LINK_FIELDS


def live_infra(cfg: Dict[str, Any], sim_input: Path) -> Dict[str, Any]:
    with contextlib.redirect_stdout(io.StringIO()):
        return prepare_infrastructure_for_real_simulation(copy.deepcopy(cfg), None, sim_input)


def det_infra(cfg: Dict[str, Any], sim_input: Path, seed: int) -> Dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmp:
        space = Path(tmp) / "space_with_network.json"
        space.write_text(json.dumps(cfg))
        with contextlib.redirect_stdout(io.StringIO()):
            return generate_deterministic_infrastructure(str(space), sim_input, str(Path(tmp) / "infrastructure.json"), seed)


def jsonable(x: Any) -> Any:
    return json.loads(json.dumps(x, default=str))


def _node_names(infra: Dict[str, Any]) -> List[str]:
    if "nodes" in infra:
        return [n["node_name"] for n in infra["nodes"]]
    return sorted(infra["network_maps"])


def compare_topology(cfg2: Dict[str, Any], cfg3: Dict[str, Any], sim_input: Path) -> Dict[str, Any]:
    seed = int(cfg2["network"]["topology"]["seed"])
    res: Dict[str, Any] = {"seed": seed}
    for name, fn in (("live", lambda c: live_infra(c, sim_input)), ("generator", lambda c: det_infra(c, sim_input, seed))):
        a, b = jsonable(fn(cfg2)), jsonable(fn(cfg3))
        entries: List[Dict[str, Any]] = []
        diff(a, b, name, entries)
        bad = [e for e in entries if not allowed(e)]
        links3 = (b.get("link_topology") or {}).get("links") or {}
        links2 = (a.get("link_topology") or {}).get("links") or {}
        res[name] = {
            "n_diff_fields": len(entries),
            "n_unexpected": len(bad),
            "unexpected": bad[:20],
            "links_same_keys": sorted(links2) == sorted(links3),
            "routes_equal": (a.get("link_topology") or {}).get("routes") == (b.get("link_topology") or {}).get("routes"),
            "node_names_equal": _node_names(a) == _node_names(b),
            "n_links": len(links3),
            "n_links_changed": len({e["path"].split(".links.")[1].split(".")[0] for e in entries if ".links." in e["path"]}),
            "fields_changed": sorted({e["path"].rsplit(".", 1)[1] for e in entries if ".links." in e["path"]}),
            "other_fields_changed": sorted({e["path"] for e in entries if ".links." not in e["path"] and "access_classes" not in e["path"]}),
        }
        if name == "generator":
            res[name]["replica_placements_equal"] = a.get("replica_placements") == b.get("replica_placements")
            res[name]["queues_equal"] = a.get("queue_distributions") == b.get("queue_distributions") if "queue_distributions" in a else None
        res[name]["classes"] = (b.get("link_topology") or {}).get("access_classes")
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--w2-inputs", type=Path, required=True)
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--diff-out", type=Path, required=True)
    ap.add_argument("--sim-input", type=Path, default=ROOT / "data" / "nofs-ids")
    a = ap.parse_args()

    mix = dict(DEFAULT_ACCESS_MIX)
    report: Dict[str, Any] = {"mix": mix, "rungs": {}}
    a.out.mkdir(parents=True, exist_ok=True)
    for tag in a.tags:
        src, dst = a.w2_inputs / f"wf1_{tag}", a.out / f"wf1_{tag}"
        if dst.exists():
            raise SystemExit(f"FAIL LOUD: {dst} exists; a built rung is frozen")
        manifest = json.loads((a.w2_inputs / f"manifest_{tag}.json").read_text())
        shas = manifest["rungs"][tag]["wl_sha256"]
        (dst / "wl").mkdir(parents=True)
        (dst / "cfg").mkdir()
        for name, want in shas.items():
            if sha256(src / "wl" / name) != want:
                raise SystemExit(f"FAIL LOUD: W2 {tag}/{name} does not match its manifest sha256")
            shutil.copyfile(src / "wl" / name, dst / "wl" / name)
            if sha256(dst / "wl" / name) != want:
                raise SystemExit(f"FAIL LOUD: copy of {tag}/{name} differs from the W2 file")
        rows = {}
        shares = {"wired": 0, "wifi": 0, "cellular": 0}
        for cfg_path in sorted((src / "cfg").glob("*.json")):
            cfg2 = json.loads(cfg_path.read_text())
            cfg3 = copy.deepcopy(cfg2)
            cfg3["network"]["backbone"]["access_classes"] = {"mix": mix}
            diff_cfg: List[Dict[str, Any]] = []
            diff(cfg2, cfg3, "cfg", diff_cfg)
            if [e["path"] for e in diff_cfg] != ["cfg.network.backbone.access_classes"]:
                raise SystemExit(f"FAIL LOUD: {cfg_path.name} W3 config differs from W2 beyond access_classes: {diff_cfg}")
            (dst / "cfg" / cfg_path.name).write_text(json.dumps(cfg3))
            row = compare_topology(cfg2, cfg3, a.sim_input)
            row["cfg_diff"] = [e["path"] for e in diff_cfg]
            rows[cfg_path.stem] = row
            for spec in (row["live"]["classes"] or {}).values():
                shares[spec["class"]] += 1
        n_nodes = sum(shares.values())
        report["rungs"][tag] = {
            "topologies": rows,
            "class_counts": shares,
            "class_shares": {k: v / n_nodes for k, v in shares.items()},
            "wl_sha256": shas,
        }
        shutil.copyfile(a.w2_inputs / f"manifest_{tag}.json", a.out / f"manifest_{tag}.json")
    a.diff_out.write_text(json.dumps(report, indent=1))

    bad = 0
    for tag, r in report["rungs"].items():
        for name, row in r["topologies"].items():
            ok = all(row[p]["n_unexpected"] == 0 and row[p]["links_same_keys"] and row[p]["routes_equal"]
                     and row[p]["node_names_equal"] and not row[p]["other_fields_changed"] for p in ("live", "generator")) \
                and row["generator"]["replica_placements_equal"]
            bad += not ok
            g, lv = row["generator"], row["live"]
            print(f"{'ok ' if ok else 'BAD'} {tag} {name} seed={row['seed']} links={lv['n_links']} changed={lv['n_links_changed']} "
                  f"fields={lv['fields_changed']} routes_eq={lv['routes_equal']} replicas_eq={g['replica_placements_equal']} "
                  f"other={lv['other_fields_changed'] + g['other_fields_changed']}")
        print(f"[{tag}] class counts {r['class_counts']} shares " + " ".join(f"{k}={v:.3f}" for k, v in r['class_shares'].items()))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
