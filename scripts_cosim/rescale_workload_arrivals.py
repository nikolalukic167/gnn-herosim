#!/usr/bin/env python3
"""Play a production trace at a lower arrival rate by stretching its timestamps.

The peer_affinity live gate runs a 2,650 arrivals/s trace onto 6 servers, which is deep
overload: queueing is ~100% of RTT and the peer-exchange term the checkpoints optimise is
~0.01% of it, so no placement policy can show its mechanism there. This scales every
event timestamp by `--factor` (and nothing else -- same events, same types, same peer
groups, same payloads, same demand scales), giving a load ladder on one trace whose only
moving part is the arrival rate.

Usage:
  python3 scripts_cosim/rescale_workload_arrivals.py --workload <in.json> --factor 10 \
      --output <out.json>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workload", type=Path, required=True)
    ap.add_argument("--factor", type=float, required=True, help="multiply every timestamp by this (>1 = slower)")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.factor <= 0:
        raise SystemExit("FAIL LOUD: --factor must be positive")
    d = json.loads(args.workload.read_text())
    events = d["events"]
    if not events:
        raise SystemExit(f"FAIL LOUD: {args.workload} has no events")
    span_before = events[-1]["timestamp"] - events[0]["timestamp"]
    for ev in events:
        ev["timestamp"] = ev["timestamp"] * args.factor
    span_after = events[-1]["timestamp"] - events[0]["timestamp"]
    d["duration"] = float(d.get("duration") or 0) * args.factor
    d["arrival_rescale"] = {
        "source": str(args.workload), "factor": args.factor,
        "events": len(events), "span_before_s": span_before, "span_after_s": span_after,
        "rate_before_per_s": len(events) / span_before if span_before else None,
        "rate_after_per_s": len(events) / span_after if span_after else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(d))
    print(f"[rescale] {args.output}: {len(events)} events, "
          f"{d['arrival_rescale']['rate_before_per_s']:.1f} -> "
          f"{d['arrival_rescale']['rate_after_per_s']:.1f} arrivals/s (x{args.factor})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
