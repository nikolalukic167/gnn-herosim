#!/usr/bin/env python3
"""Reproducibility audit: regenerate infrastructure from recorded seeds and diff.

WS5 of the co-sim deep-dive campaign. For N sampled datasets per collection, re-run
`generate_deterministic_infrastructure` from the dataset's own `space_with_network.json`
and recorded seed, and diff the result against the stored `infrastructure.json`.

Scope is deliberately infrastructure-only: the workload draw depends on a template index
that was never recorded per dataset (the generator's `template_idx` advances globally),
so workload reproducibility is not decidable from the artifacts — that fact is itself a
finding, recorded here as `workload_check: "undecidable_no_template_idx"`.

Usage:
  PYTHONPATH=. python3 scripts_cosim/audit_regen_reproducibility.py \
      --collection gnn_datasets_4tasks_contention_v2 ... [--per-collection 5] [--json OUT]
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.generate_infrastructure import generate_deterministic_infrastructure  # noqa: E402


def canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def diff_infra(stored: Dict[str, Any], regen: Dict[str, Any]) -> Dict[str, Any]:
    """Section-level diff: which top-level keys differ, with a coarse magnitude."""
    out: Dict[str, Any] = {"identical": True, "differing_sections": []}
    keys = set(stored) | set(regen)
    keys.discard("metadata")  # carries generation_time; never comparable
    for key in sorted(keys):
        a, b = stored.get(key), regen.get(key)
        if canonical(a) == canonical(b):
            continue
        out["identical"] = False
        detail: Dict[str, Any] = {"section": key}
        if key == "network_maps" and isinstance(a, dict) and isinstance(b, dict):
            edges_a = {(s, t) for s, m in a.items() for t in m}
            edges_b = {(s, t) for s, m in b.items() for t in m}
            detail["edges_stored"] = len(edges_a)
            detail["edges_regen"] = len(edges_b)
            detail["edges_only_stored"] = len(edges_a - edges_b)
            detail["edges_only_regen"] = len(edges_b - edges_a)
            shared = edges_a & edges_b
            lat_diff = sum(
                1 for (s, t) in shared if canonical(a[s][t]) != canonical(b[s][t])
            )
            detail["shared_edges_latency_diff"] = lat_diff
        elif key == "replica_placements" and isinstance(a, dict) and isinstance(b, dict):
            pa = {(tt, canonical(r)) for tt, rs in a.items() for r in rs}
            pb = {(tt, canonical(r)) for tt, rs in b.items() for r in rs}
            detail["replicas_stored"] = len(pa)
            detail["replicas_regen"] = len(pb)
            detail["replicas_only_stored"] = len(pa - pb)
        elif key == "queue_distributions" and isinstance(a, dict) and isinstance(b, dict):
            flat_a = {
                (tt, k): v for tt, m in a.items() for k, v in (m or {}).items()
            }
            flat_b = {
                (tt, k): v for tt, m in b.items() for k, v in (m or {}).items()
            }
            common = set(flat_a) & set(flat_b)
            detail["queues_diff"] = sum(1 for k in common if flat_a[k] != flat_b[k])
            detail["queues_total"] = len(common)
        out["differing_sections"].append(detail)
    return out


def audit_dataset(ds_dir: Path, sim_input: Path) -> Dict[str, Any]:
    row: Dict[str, Any] = {"dataset": ds_dir.name}
    space_path = ds_dir / "space_with_network.json"
    infra_path = ds_dir / "infrastructure.json"
    if not space_path.exists() or not infra_path.exists():
        row["status"] = "missing_inputs"
        return row
    config = json.loads(space_path.read_text())
    stored = json.loads(infra_path.read_text())
    seed = (
        (stored.get("metadata") or {}).get("seed")
        or config.get("network", {}).get("topology", {}).get("seed")
    )
    row["seed"] = seed
    if seed is None:
        row["status"] = "no_recorded_seed"
        return row
    with tempfile.TemporaryDirectory() as td:
        out_file = Path(td) / "infrastructure.json"
        try:
            regen = generate_deterministic_infrastructure(
                str(space_path), sim_input, str(out_file), int(seed)
            )
        except Exception as exc:
            row["status"] = "regen_error"
            row["error"] = f"{type(exc).__name__}: {exc}"
            return row
    row["diff"] = diff_infra(stored, regen)
    row["status"] = "reproducible" if row["diff"]["identical"] else "diverged"
    row["workload_check"] = "undecidable_no_template_idx"
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sim-root", default="simulation_data", type=Path)
    ap.add_argument("--sim-input", default="data/nofs-ids", type=Path)
    ap.add_argument("--collection", action="append", required=True)
    ap.add_argument("--per-collection", type=int, default=5)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    report: Dict[str, Any] = {}
    for name in args.collection:
        base = args.sim_root / name
        ds_dirs = sorted(d for d in base.glob("ds_*") if d.is_dir())
        if not ds_dirs:
            print(f"[{name}] no datasets", flush=True)
            continue
        # Deterministic spread: first, last, and evenly spaced between.
        n = min(args.per_collection, len(ds_dirs))
        idx = sorted({round(i * (len(ds_dirs) - 1) / max(n - 1, 1)) for i in range(n)})
        rows: List[Dict[str, Any]] = []
        for i in idx:
            import contextlib, io
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                row = audit_dataset(ds_dirs[i], args.sim_input)
            rows.append(row)
        statuses = [r["status"] for r in rows]
        print(f"[{name}] {dict((s, statuses.count(s)) for s in set(statuses))}", flush=True)
        for r in rows:
            if r["status"] == "diverged":
                secs = [d["section"] for d in r["diff"]["differing_sections"]]
                print(f"    {r['dataset']} (seed={r['seed']}): differs in {secs}", flush=True)
        report[name] = rows

    if args.json:
        args.json.write_text(json.dumps(report, indent=1))
        print(f"Wrote {args.json}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
