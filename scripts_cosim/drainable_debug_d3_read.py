#!/usr/bin/env python3
"""drainable_debug_v1 D3 -- is the drainable regime an artifact of the target concurrency?

Registered in `docs/lineages/drainable_debug_v1.md`; the bar is the module constant below and
was committed before any probe ran.

The Knative autoscaler adds a replica when `ceil(total_queued / Q) > replicas`, with
`Q = HEROSIM_QUEUE_LENGTH` and a default of 100 (`src/placement/constants.py:11`). Platforms
serve one task at a time, so `Q = 100` means a replica must accumulate a hundred queued tasks
before a second one is created. On a cluster that is ~98.5 % idle at the drainable rung, that
is not a throughput setting, it is a packing setting -- and it is the setting under which every
comparison in this program was made, including every co-sim corpus state
(`src/executecosimulation.py:1348,1601,2532,2667`).

If Knative's own latency collapses when Q is lowered, then the "drainable regime" the parents
gated at is partly an artifact of that constant, and every learned-vs-reactive contrast has to
be read at a sensible Q as well as at 100. If it does not, the rung is real as measured.

Only `knative_network` is run: the question is about the cluster, not about any learned arm.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# --- bar (drainable_debug_v1, signed 2026-09-14, before any probe ran) ----------------------
# If the best latency at Q <= 4 is below this fraction of the Q = 100 latency, the rung's
# regime is a target-concurrency artifact and Phase 2 runs at both values.
D3_ARTIFACT_MAX_FRACTION = 0.50
D3_LOW_Q = (1, 2, 4)
D3_REFERENCE_Q = 100
# -------------------------------------------------------------------------------------------


def read_rung(results_dir: Path, rung: int, qs: List[int]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"rung": rung, "by_q": {}}
    for q in qs:
        path = results_dir / f"dbg_D3_f{rung}_q{q}" / "knative_network.summary.json"
        if not path.exists():
            out["by_q"][q] = {"missing": str(path)}
            continue
        d = json.loads(path.read_text())
        recorded = d.get("queue_length")
        if recorded is not None and int(recorded) != q:
            # The summary carries the value the run actually resolved. A mismatch means the
            # env var did not reach the simulator and the arm is a different experiment than
            # its directory name claims -- exactly the confound class that cost this lineage's
            # parent three reads.
            raise SystemExit(
                f"FAIL LOUD: {path} records queue_length={recorded} but its directory says "
                f"Q={q}; the probe did not run at the intended target concurrency"
            )
        out["by_q"][q] = {
            "latency": d["averageElapsedTime"],
            "queue": d["averageQueueTime"],
            "scale_events": d["scaleEventCount"],
            "occupation": d["averageOccupation"],
            "unused_platforms_pct": d["unusedPlatforms"],
            "end_time": d["endTime"],
            "queue_length": recorded,
        }
    return out


def verdict(rung: Dict[str, Any]) -> Dict[str, Any]:
    by_q = rung["by_q"]
    ref = by_q.get(D3_REFERENCE_Q)
    if not ref or "missing" in ref:
        return {"verdict": "VOID", "why": f"no Q={D3_REFERENCE_Q} reference run"}
    lows = {q: by_q[q]["latency"] for q in D3_LOW_Q if q in by_q and "missing" not in by_q[q]}
    if not lows:
        return {"verdict": "VOID", "why": "no low-Q run"}
    best_q = min(lows, key=lambda q: lows[q])
    fraction = lows[best_q] / ref["latency"]
    return {
        "reference_latency": ref["latency"],
        "best_low_q": best_q,
        "best_low_q_latency": lows[best_q],
        "fraction_of_reference": fraction,
        "bar": D3_ARTIFACT_MAX_FRACTION,
        "verdict": "TARGET-CONCURRENCY-ARTIFACT"
        if fraction < D3_ARTIFACT_MAX_FRACTION
        else "REGIME-IS-REAL",
    }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, required=True, help="the gate's results/ directory")
    ap.add_argument("--rungs", type=int, nargs="+", default=[4000, 2000])
    ap.add_argument("--qs", type=int, nargs="+", default=[1, 2, 4, 10, 100])
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)

    reading: Dict[str, Any] = {
        "lineage": "drainable_debug_v1",
        "read": "D3",
        "bar": {
            "d3_artifact_max_fraction": D3_ARTIFACT_MAX_FRACTION,
            "low_q": list(D3_LOW_Q),
            "reference_q": D3_REFERENCE_Q,
        },
        "rungs": {},
    }
    print("%6s %5s %10s %9s %9s %8s %11s" % ("rung", "Q", "latency", "queue", "scale", "occ", "unusedPlat"))
    for rung in args.rungs:
        row = read_rung(args.results, rung, args.qs)
        row["verdict"] = verdict(row)
        reading["rungs"][str(rung)] = row
        for q in args.qs:
            cell = row["by_q"].get(q, {})
            if "missing" in cell:
                print("x%-5d %5d  MISSING" % (rung, q))
                continue
            print("x%-5d %5d %10.2f %9.2f %9d %8.4f %11.1f" % (
                rung, q, cell["latency"], cell["queue"], cell["scale_events"],
                cell["occupation"], cell["unused_platforms_pct"]))
        v = row["verdict"]
        if v["verdict"] == "VOID":
            print(f"[D3 x{rung}] VOID -- {v['why']}")
        else:
            print(f"[D3 x{rung}] {v['verdict']}: best low Q={v['best_low_q']} at "
                  f"{v['best_low_q_latency']:.2f} s = {100 * v['fraction_of_reference']:.1f}% of "
                  f"Q={D3_REFERENCE_Q}'s {v['reference_latency']:.2f} s "
                  f"(bar {100 * D3_ARTIFACT_MAX_FRACTION:.0f}%)")

    if args.out:
        args.out.write_text(json.dumps(reading, indent=1))
        print(f"[wrote] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
