#!/usr/bin/env python3
"""accel_replica_v1 W2 read: per probe cell (seed), the share of scored batches whose best POINTWISE plan loses more than 1 % to the joint optimum
(joint_headroom_screen.py: regret_rel > 0.01; status ok; underdetermined datasets are not scored), per rung and pooled over both rungs.
Gate: at least 3 of the 4 seeds above 50 % (pooled over the two rungs). The pooled share over all batches is reported next to scale_160_v1's 69 %.

  accel_w2_read.py <probe root> [<second root for a descriptive comparison>]
"""
import json
import statistics as st
import sys
from pathlib import Path


def read(root: Path):
    out = {}
    for cell in sorted(root.glob("c160s24p0.6s99*")):
        f = cell / "headroom_k5.jsonl"
        if not f.is_file():
            continue
        rows = [json.loads(l) for l in f.open() if l.strip()]
        ok = [r for r in rows if r.get("status") == "ok" and r.get("regret_rel") is not None]
        by = {}
        for rung in ("moderate", "heavy"):
            rr = [r for r in ok if r["rung"] == rung]
            by[rung] = {"scored": len(rr), "above_1pct": sum(r["regret_rel"] > 0.01 for r in rr),
                        "share": (sum(r["regret_rel"] > 0.01 for r in rr) / len(rr)) if rr else None,
                        "median_regret": st.median(r["regret_rel"] for r in rr) if rr else None}
        out[cell.name[-4:]] = {"rows": len(rows), "scored": len(ok), "above_1pct": sum(r["regret_rel"] > 0.01 for r in ok),
                               "share": (sum(r["regret_rel"] > 0.01 for r in ok) / len(ok)) if ok else None,
                               "status": {s: sum(r.get("status") == s for r in rows) for s in sorted({r.get("status") for r in rows})}, **by}
    return out


def show(name, d):
    print(f"== {name}")
    for seed, v in d.items():
        print(f"  seed {seed}: scored {v['scored']}/{v['rows']}  share>1% {v['share']:.3f}   moderate {v['moderate']['above_1pct']}/{v['moderate']['scored']}"
              f" ({v['moderate']['share']:.3f}, median regret {v['moderate']['median_regret']:.3f})   heavy {v['heavy']['above_1pct']}/{v['heavy']['scored']}"
              f" ({v['heavy']['share']:.3f}, median regret {v['heavy']['median_regret']:.3f})   status {v['status']}")
    tot = sum(v["scored"] for v in d.values()); hit = sum(v["above_1pct"] for v in d.values())
    n_pass = sum(1 for v in d.values() if v["share"] is not None and v["share"] > 0.5)
    print(f"  seeds above 50 %: {n_pass} of {len(d)} (gate: >= 3 of 4);  pooled share {hit}/{tot} = {hit / tot:.3f}  (scale_160_v1: 0.69)")


if __name__ == "__main__":
    for r in sys.argv[1:]:
        show(r, read(Path(r)))
