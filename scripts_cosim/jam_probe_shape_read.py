#!/usr/bin/env python3
"""JAM PROBE -- is the learned arms' loss to reactive Knative a jam, or is it uniform?

SCOPING READ. No pre-registered bars, and it therefore CANNOT close anything. It exists to
order work: `offline_live_transfer_v1` R3 found that 50.5 % of the 1.7x gap between two
training seeds of one recipe lives in ONE decile of one trace and then recovers, and the
question that decides whether that is a lever is whether the HEADLINE loss -- learned arm
against reactive Knative -- has the same shape.

  concentrated in a few deciles, with recovery
      the graph arm is not steadily worse at placement; it is jam-prone, and the fix is a
      different problem from better placement scoring.

  spread evenly across the trace
      the graph arm is simply worse. The excursion is a seed-level curiosity and this line
      of work stops.

The statistic is deliberately plain, because a scoping read that needs a subtle statistic is
not a scoping read: per decile of arrival order, the mean queue-time excess of each learned
arm over reactive, and what share of the total excess the worst decile carries. A uniform
loss puts ~1/10 of the excess in its worst decile; R3's seed excursion put 0.505 in one.

Whatever this shows, the follow-ups get their own registration with bars signed before their
data, and this read is named there as the motivating measurement -- the same handling
offline_live_transfer_v1 gave its own scoping reads.

Usage:
    python3 scripts_cosim/jam_probe_shape_read.py \
        --reference <reactive capture>.json \
        --arm best=<capture>.json --arm worst=<capture>.json \
        --out simulation_data/jam_probe/shape.json
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts_cosim.offline_live_transfer_v1_decile_read import (  # noqa: E402
    DecileReadError,
    R3_DECILES,
    decile_bounds,
    decile_of,
    load_arm,
)

# A uniform loss puts this share of its excess in any one decile. Reported for scale only --
# this is NOT a bar, and no verdict below is a registered one.
UNIFORM_SHARE = 1.0 / R3_DECILES


def shape(
    reference: Dict[int, Tuple[float, float, float]],
    arm: Dict[int, Tuple[float, float, float]],
    bounds: List[float],
) -> Dict[str, Any]:
    shared = sorted(set(reference) & set(arm))
    if len(shared) != len(reference) or len(shared) != len(arm):
        raise DecileReadError(
            f"captures cover different tasks ({len(reference)} vs {len(arm)}, "
            f"{len(shared)} shared) -- the same trace must produce the same task set"
        )
    rows: List[Dict[str, Any]] = []
    for d in range(R3_DECILES):
        ids = [t for t in shared if decile_of(reference[t][0], bounds) == d]
        ref_q = st.fmean(reference[t][1] for t in ids)
        arm_q = st.fmean(arm[t][1] for t in ids)
        rows.append({"decile": d + 1, "n": len(ids), "reference_queue_s": ref_q,
                     "arm_queue_s": arm_q, "excess_s": arm_q - ref_q})
    mass = sum(r["excess_s"] for r in rows)
    if mass <= 0:
        return {"rows": rows, "excess_mass_s": mass,
                "note": "the arm carries no net excess over the reference"}
    for r in rows:
        r["share"] = r["excess_s"] / mass
    shares = sorted((r["share"] for r in rows), reverse=True)
    return {
        "rows": rows,
        "excess_mass_s": mass,
        "worst_decile_share": shares[0],
        "top3_share": sum(shares[:3]),
        "uniform_share_for_scale": UNIFORM_SHARE,
        "full_reference_queue_s": st.fmean(reference[t][1] for t in shared),
        "full_arm_queue_s": st.fmean(arm[t][1] for t in shared),
        "full_reference_elapsed_s": st.fmean(reference[t][2] for t in shared),
        "full_arm_elapsed_s": st.fmean(arm[t][2] for t in shared),
    }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--arm", action="append", required=True, metavar="NAME=PATH")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    print(f"[jam] loading reference {args.reference}", flush=True)
    reference = load_arm(args.reference)
    bounds = decile_bounds([v[0] for v in reference.values()])

    out: Dict[str, Any] = {"read": "jam_probe_shape", "status": "SCOPING-NO-BARS",
                           "reference": str(args.reference), "arms": {}}
    for spec in args.arm:
        if "=" not in spec:
            raise SystemExit(f"FAIL LOUD: --arm takes NAME=PATH, got {spec!r}")
        name, _, path = spec.partition("=")
        print(f"[jam] loading arm {name} {path}", flush=True)
        out["arms"][name] = shape(reference, load_arm(Path(path)), bounds)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))

    for name, res in out["arms"].items():
        print(f"\n=== {name} vs reactive ===")
        print(f"   mean elapsed {res['full_arm_elapsed_s']:.2f} s vs reactive "
              f"{res['full_reference_elapsed_s']:.2f} s   |   mean queue "
              f"{res['full_arm_queue_s']:.2f} s vs {res['full_reference_queue_s']:.2f} s")
        if "share" not in res["rows"][0]:
            print(f"   {res.get('note')}")
            continue
        print(f"   {'decile':>7} {'reactive q':>11} {'arm q':>11} {'excess':>10} {'share':>8}")
        for r in res["rows"]:
            print(f"   {r['decile']:>7} {r['reference_queue_s']:>11.3f}"
                  f" {r['arm_queue_s']:>11.3f} {r['excess_s']:>10.3f} {r['share']:>8.3f}")
        print(f"   worst decile carries {res['worst_decile_share']:.3f} of the excess, "
              f"top 3 carry {res['top3_share']:.3f}  (uniform would be "
              f"{UNIFORM_SHARE:.3f} and {3 * UNIFORM_SHARE:.3f})")
    print(f"\n[jam] SCOPING READ -- no bars, orders work, closes nothing")
    print(f"[wrote] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
