"""r1_attribution_v1 production build: the snapshots one cell offers to make_warm_corpus, in an order whose every prefix is spread
evenly over the whole capture.

Drops are counted by reason, not silently: a snapshot the build would reject anyway (not a consecutive batch of >= 2 tasks, captured
before --min-time) never enters the shard. The kept snapshots are written in bit-reversal order, so make_warm_corpus --limit-batches N
takes N snapshots spread over the entire trace, and a rejection in that prefix is replaced by the next snapshot in the same order
(the top-up) rather than by a truncated stride (`ok[::step][:want]` dropped the trace's last ~10 %).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def spread_order(n: int) -> list:
    """0..n-1 such that every prefix is evenly spread: bit-reversal permutation of the next power of two, filtered to < n."""
    if n <= 0:
        return []
    bits = max(1, (n - 1).bit_length())
    out = []
    for i in range(1 << bits):
        r = int(format(i, f"0{bits}b")[::-1], 2)
        if r < n:
            out.append(r)
    return out


def main() -> int:
    from scripts_cosim.make_warm_corpus import SnapshotRejected, check_consecutive_batch

    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshots", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--min-time", type=float, default=0.0)
    ap.add_argument("--min-batch", type=int, default=2)
    a = ap.parse_args()
    keep, dropped, total = [], {}, 0
    for line in a.snapshots.read_text().splitlines():
        if not line.strip():
            continue
        total += 1
        snap = json.loads(line)
        try:
            if float(snap.get("time", 0.0)) < a.min_time:
                raise SnapshotRejected("before_min_time")
            check_consecutive_batch(snap, a.min_batch)
        except SnapshotRejected as exc:
            key = str(exc) if str(exc) == "before_min_time" else "not_a_consecutive_batch"
            dropped[key] = dropped.get(key, 0) + 1
            continue
        keep.append(line)
    order = spread_order(len(keep))
    a.out.write_text("".join(keep[i].rstrip("\n") + "\n" for i in order))
    stats = {"snapshots": str(a.snapshots), "read": total, "eligible": len(keep), "dropped_by_reason": dropped,
             "dropped": total - len(keep)}
    a.out.with_suffix(".stats.json").write_text(json.dumps(stats))
    print(f"[shard] {stats}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
