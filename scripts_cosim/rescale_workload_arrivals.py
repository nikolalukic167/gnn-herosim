#!/usr/bin/env python3
"""Play a production trace at a lower arrival rate by stretching its timestamps.

The peer_affinity live gate runs a 2,650 arrivals/s trace onto 6 servers, which is deep
overload: queueing is ~100% of RTT and the peer-exchange term the checkpoints optimise is
~0.01% of it, so no placement policy can show its mechanism there. This scales every
event timestamp by `--factor` (and nothing else -- same events, same types, same peer
groups, same payloads, same demand scales), giving a load ladder on one trace whose only
moving part is the arrival rate.

`--max-events` additionally keeps only the first N events. Stretching a 450,729-event trace
by 4,000x gives a 678,000 s arrival span, and the autoscaler reconciles once per simulated
second, so the run accumulates ~700k x |task types| `systemEvents` rows and a multi-GB raw
result. Truncation is the cheap half of the same knob: the arrival process is unchanged, so
the steady state a load screen reads is identical, and peer pairs that reach past the cut are
dropped with the tasks they reference rather than left dangling. N must be a multiple of the
peer group size, or the last group is cut in half and no batch over it can align.

`--skip-events` takes the window from K events in instead of from the start (2026-09-19,
`unsaturated_scale_v2`). Every gate in this programme has read the SAME first 50,000 events of
a 450,729-event production trace, so ~89 % of real arrival data has never been used and every
result rests on one arrival window. `unsaturated_scale_v1` measured what that costs: between
the first and second halves of this one window the arms' median result against reactive swings
**19 pp**, against an effect of 5-8 pp, and flips sign. Disjoint windows are the honest
replication unit for that, and they are real trace data rather than a resampling of it.

Timestamps are REBASED so the window's first event lands where the full trace's first event
does; without that a window starting 1,000 s in makes every policy idle through 1,000 s of
simulated time before the first arrival. `peer_exchange` pairs are re-indexed to the window and
kept only when BOTH endpoints are inside it. K must be a multiple of the peer group size for
the same reason N is. `--skip-events 0` is byte-identical to omitting it, which is pinned by
`tests/test_rescale_window_offset.py` -- every trace already on disk must stay reproducible.

Usage:
  python3 scripts_cosim/rescale_workload_arrivals.py --workload <in.json> --factor 10 \
      --output <out.json> [--max-events 50000] [--skip-events 100000] [--group-size 10]
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
    ap.add_argument("--max-events", type=int, default=None,
                    help="keep only the first N events (and the peer pairs wholly inside them)")
    ap.add_argument("--skip-events", type=int, default=0,
                    help="take the window from K events in instead of from the start; "
                         "timestamps are rebased and peer pairs re-indexed (default 0)")
    ap.add_argument("--group-size", type=int, default=10,
                    help="peer group size; --max-events must be a multiple of it")
    args = ap.parse_args()
    if args.factor <= 0:
        raise SystemExit("FAIL LOUD: --factor must be positive")
    d = json.loads(args.workload.read_text())
    events = d["events"]
    if not events:
        raise SystemExit(f"FAIL LOUD: {args.workload} has no events")
    skip = args.skip_events
    if skip < 0:
        raise SystemExit("FAIL LOUD: --skip-events must be non-negative")
    if skip % args.group_size != 0:
        raise SystemExit(
            f"FAIL LOUD: --skip-events {skip} is not a multiple of --group-size "
            f"{args.group_size}; the window would start mid peer group")
    if skip and args.max_events is None:
        raise SystemExit("FAIL LOUD: --skip-events needs --max-events; a window has two ends")
    if skip >= len(events):
        raise SystemExit(f"FAIL LOUD: --skip-events {skip} is past the trace ({len(events)} events)")
    truncation = None
    if args.max_events is not None:
        n = args.max_events
        if n <= 0:
            raise SystemExit("FAIL LOUD: --max-events must be positive")
        if n % args.group_size != 0:
            raise SystemExit(
                f"FAIL LOUD: --max-events {n} is not a multiple of --group-size "
                f"{args.group_size}; the last peer group would be cut in half")
        if skip + n > len(events):
            raise SystemExit(f"FAIL LOUD: window [{skip}, {skip + n}) exceeds the trace "
                             f"({len(events)} events)")
        pairs_before = len(d.get("peer_exchange") or [])
        events_before = len(events)
        # Rebase to the FULL trace's first timestamp, so window K starts where window 0 does and
        # no policy idles through the skipped span. At skip=0 this subtracts exactly 0.0.
        t_origin = events[0]["timestamp"]
        t_window = events[skip]["timestamp"]
        events = events[skip:skip + n]
        if skip:
            shift = t_window - t_origin
            for ev in events:
                ev["timestamp"] = ev["timestamp"] - shift
        d["events"] = events
        # Pair indices are into the ORIGINAL event list, so they must be re-indexed to the
        # window and kept only when BOTH endpoints are inside it -- a pair reaching out of the
        # window would otherwise silently point at whatever task now sits at that index.
        d["peer_exchange"] = [[p[0] - skip, p[1] - skip] + list(p[2:])
                              for p in (d.get("peer_exchange") or [])
                              if skip <= p[0] < skip + n and skip <= p[1] < skip + n]
        truncation = {"max_events": n, "group_size": args.group_size,
                      "events_before": events_before, "events_after": len(events),
                      "peer_pairs_before": pairs_before,
                      "peer_pairs_after": len(d["peer_exchange"])}
        if skip:
            truncation.update(skip_events=skip, window=[skip, skip + n],
                              rebased_by_s=t_window - t_origin)
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
        "truncation": truncation,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(d))
    if truncation:
        print(f"[rescale] truncated to {truncation['max_events']} events, peer pairs "
              f"{truncation['peer_pairs_before']} -> {truncation['peer_pairs_after']}")
    print(f"[rescale] {args.output}: {len(events)} events, "
          f"{d['arrival_rescale']['rate_before_per_s']:.1f} -> "
          f"{d['arrival_rescale']['rate_after_per_s']:.1f} arrivals/s (x{args.factor})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
