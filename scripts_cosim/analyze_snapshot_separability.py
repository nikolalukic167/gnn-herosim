#!/usr/bin/env python3
"""WS3c: aggregate live-regime separability over snapshot pseudo-dataset sweeps.

Consumes the output of `snapshot_separability_sweep.py` (dirs of ds_snap_* pseudo-
datasets under one or more roots), computes the M4 statistics per snapshot via
`separability_diagnostic`'s own functions (so the metric code is identical to the
co-sim corpus baseline), and reports distributions stratified by:

  * gate condition (dir name: <gate>_<cond>_<cell>[ _<arm> ])
  * snapshot time tercile within its cell (early / mid / late trace)
  * collapse cells (cell03/cell05) vs healthy cells

Also refits at K=4 (per-task top-4 by the same schedule-time ECT, recomputed from the
source snapshot JSONL) as the pre-registered pruning-sensitivity check — no new
simulations needed because the K=4 candidate lists are prefixes of the K=6 ones.

Pre-registered verdict (from the campaign plan): "live regime is non-additive" iff
median additive_choice_regret_rel > 0.02 AND median one-integer repair < 0.8 over
>=100 fitted snapshots.

Usage:
  PYTHONPATH=. python3 scripts_cosim/analyze_snapshot_separability.py \
      --sweep-root simulation_data/snapshot_sweeps \
      [--sweep-root simulation_data/snapshot_sweeps_mlpcollapse] \
      [--snapshots-root simulation_data/normal_sim_sweeps] \
      [--json OUT]
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts_cosim.separability_diagnostic import (  # noqa: E402
    load_combos,
    load_link_context,
    variance_decomposition,
)
from src.placement.scheduling_cost import (  # noqa: E402
    expected_completion_from_snapshot_candidate,
)

COLLAPSE_CELLS = {"cell03_p15_s9003", "cell05_p20_s9005"}


def topk_platform_sets(snapshot: Dict[str, Any], k: int) -> List[set]:
    """Per task, the (node_id, platform_id) set of its top-k candidates by ECT."""
    out: List[set] = []
    for task in snapshot.get("tasks", []):
        scored = sorted(
            task.get("candidates", []),
            key=lambda c: expected_completion_from_snapshot_candidate(
                c, int(c.get("queue_length", 0)), 0
            ),
        )
        out.append({(int(c["node_id"]), int(c["platform_id"])) for c in scored[:k]})
    return out


def analyze_pseudo_dataset(
    ds_dir: Path,
    snapshot: Optional[Dict[str, Any]],
    k4: bool,
) -> Optional[Dict[str, Any]]:
    combos = load_combos(ds_dir)
    if not combos:
        return None
    task_ids = sorted(combos[0][0].keys())
    link_ctx = load_link_context(ds_dir)
    m4 = variance_decomposition(combos, task_ids, link_ctx)
    if m4 is None:
        return None
    meta = json.loads((ds_dir / "snapshot_meta.json").read_text())
    row: Dict[str, Any] = {
        "dataset": ds_dir.name,
        "time": float(meta.get("time", 0.0)),
        "n_combos": len(combos),
        "m4": m4,
    }
    if k4 and snapshot is not None:
        keep = topk_platform_sets(snapshot, 4)
        filtered = [
            (pp, rtt)
            for pp, rtt in combos
            if all(pp[t] in keep[i] for i, t in enumerate(task_ids))
        ]
        if len(filtered) >= 32:
            m4_k4 = variance_decomposition(filtered, task_ids, link_ctx)
            if m4_k4 is not None:
                row["m4_k4"] = m4_k4
                row["n_combos_k4"] = len(filtered)
    return row


def dist(vals: List[float]) -> Dict[str, Any]:
    if not vals:
        return {"n": 0}
    s = sorted(vals)
    return {
        "n": len(vals),
        "mean": st.mean(vals),
        "median": st.median(vals),
        "p90": s[min(len(s) - 1, int(0.9 * len(s)))],
        "max": max(vals),
    }


def m4_field(rows: List[Dict[str, Any]], key: str, sub: str = "m4") -> List[float]:
    out = []
    for r in rows:
        v = (r.get(sub) or {}).get(key)
        if v is not None:
            out.append(float(v))
    return out


def summarize(rows: List[Dict[str, Any]], label: str) -> Dict[str, Any]:
    r2 = m4_field(rows, "additive_r2")
    regret = m4_field(rows, "additive_choice_regret_rel")
    coll_regret = m4_field(rows, "additive_plus_collision_choice_regret_rel")
    # one-integer repair fraction per snapshot: 1 if the collision column fixes the
    # additive argmin's regret, else the residual share.
    repair = []
    for a, b in zip(regret, coll_regret):
        if a > 0:
            repair.append(max(0.0, 1.0 - b / a))
    out = {
        "label": label,
        "additive_r2": dist(r2),
        "additive_choice_regret_rel": dist(regret),
        "one_integer_repair_frac_when_regret": dist(repair),
        "regret_gt_2pct_fraction": (
            sum(1 for v in regret if v > 0.02) / len(regret) if regret else None
        ),
    }
    k4regret = m4_field(rows, "additive_choice_regret_rel", "m4_k4")
    if k4regret:
        out["k4_additive_choice_regret_rel"] = dist(k4regret)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sweep-root", action="append", type=Path, required=True)
    ap.add_argument(
        "--snapshots-root",
        type=Path,
        default=Path("simulation_data/normal_sim_sweeps"),
        help="Where the capture sweep dirs live, for the K=4 refit",
    )
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--no-k4", action="store_true")
    args = ap.parse_args()

    all_rows: List[Dict[str, Any]] = []
    for root in args.sweep_root:
        for cell_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            # cell_dir name: <gate>_<cond>_<cellname>[ _<arm> ]
            name = cell_dir.name
            # Load source snapshots for the K=4 refit, indexed by snapshot_id.
            snap_index: Dict[int, Dict[str, Any]] = {}
            if not args.no_k4:
                candidates = list(args.snapshots_root.glob(f"*/snapshots/{name}.jsonl"))
                if candidates:
                    with open(candidates[0]) as f:
                        for line in f:
                            if line.strip():
                                s = json.loads(line)
                                snap_index[int(s.get("snapshot_id", -1))] = s
            for ds_dir in sorted(cell_dir.glob("ds_snap_*")):
                sid = int(ds_dir.name.rsplit("_", 1)[-1])
                row = analyze_pseudo_dataset(ds_dir, snap_index.get(sid), not args.no_k4)
                if row is None:
                    print(f"  SKIPPED (no fit): {cell_dir.name}/{ds_dir.name}")
                    continue
                row["cell_dir"] = name
                parts = name.split("_")
                row["gate"] = parts[0]
                row["cell"] = next((p for p in parts if p.startswith("cell")), "?")
                # everything between gate and cellXX is the condition; a trailing
                # mlp/mlptempfix arm may follow the cell in mlpcollapse sweeps.
                cell_token = row["cell"].split("_")[0]
                cond = name.split(parts[0] + "_", 1)[1].split("_cell", 1)[0]
                row["cond"] = cond
                row["arm"] = parts[-1] if parts[-1] in ("mlp", "mlptempfix") else "knative"
                # Reconstruct the full cell name (cellNN_pXX_sYYYY spans 3 tokens).
                rest = name[name.index(cell_token):]
                if row["arm"] != "knative":
                    rest = rest[: -(len(row["arm"]) + 1)]
                row["cell"] = rest
                all_rows.append(row)

    if not all_rows:
        raise SystemExit("FAIL LOUD: no analyzable pseudo-datasets found")

    # Time terciles are computed within each cell_dir so trace-length differences
    # across gates don't leak into the strata.
    by_dir: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in all_rows:
        by_dir[r["cell_dir"]].append(r)
    for rows in by_dir.values():
        times = sorted(r["time"] for r in rows)
        t1 = times[len(times) // 3]
        t2 = times[2 * len(times) // 3]
        for r in rows:
            r["tercile"] = "early" if r["time"] <= t1 else ("mid" if r["time"] <= t2 else "late")

    report: Dict[str, Any] = {"n_snapshots": len(all_rows), "strata": []}
    report["strata"].append(summarize(all_rows, "ALL"))
    for arm in sorted({r["arm"] for r in all_rows}):
        report["strata"].append(
            summarize([r for r in all_rows if r["arm"] == arm], f"arm={arm}")
        )
    for gate in sorted({(r["gate"], r["cond"]) for r in all_rows}):
        rows = [r for r in all_rows if (r["gate"], r["cond"]) == gate]
        report["strata"].append(summarize(rows, f"gate={gate[0]}/{gate[1]}"))
    for terc in ("early", "mid", "late"):
        report["strata"].append(
            summarize([r for r in all_rows if r["tercile"] == terc], f"tercile={terc}")
        )
    collapse_rows = [r for r in all_rows if any(c in r["cell_dir"] for c in COLLAPSE_CELLS)]
    healthy_rows = [r for r in all_rows if r not in collapse_rows]
    report["strata"].append(summarize(collapse_rows, "cells=collapse(cell03+cell05)"))
    report["strata"].append(summarize(healthy_rows, "cells=healthy"))

    # Pre-registered verdict on the primary (knative-proxy) capture population.
    primary = [r for r in all_rows if r["arm"] == "knative"]
    regret = m4_field(primary, "additive_choice_regret_rel")
    coll = m4_field(primary, "additive_plus_collision_choice_regret_rel")
    repair = [max(0.0, 1.0 - b / a) for a, b in zip(regret, coll) if a > 0]
    verdict = {
        "n_fitted": len(regret),
        "median_regret": st.median(regret) if regret else None,
        "median_one_integer_repair": st.median(repair) if repair else None,
        "non_additive": bool(
            len(regret) >= 100
            and st.median(regret) > 0.02
            and (st.median(repair) if repair else 1.0) < 0.8
        ),
    }
    report["preregistered_verdict"] = verdict

    for s in report["strata"]:
        rr = s["additive_choice_regret_rel"]
        r2 = s["additive_r2"]
        k4 = s.get("k4_additive_choice_regret_rel")
        print(
            f"{s['label']:38s} n={rr.get('n',0):4d} "
            f"R2 med={r2.get('median',float('nan')):.5f} "
            f"regret med={rr.get('median',float('nan')):.5f} "
            f"p90={rr.get('p90',float('nan')):.5f} "
            f">2%={100*(s['regret_gt_2pct_fraction'] or 0):.1f}% "
            + (f"| K4 med={k4['median']:.5f}" if k4 else "")
        )
    print(f"\nPre-registered verdict: {json.dumps(verdict)}")

    if args.json:
        args.json.write_text(json.dumps(report, indent=1, default=float))
        print(f"Wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
