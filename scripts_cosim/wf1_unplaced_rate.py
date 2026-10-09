"""r1_attribution_v1: the share of offered batches the `unplaced_partner` rule rejects, per rung (the 10 % pause line).

Same predicate as make_warm_corpus: a partner outside the batch with no node at the snapshot instant and no `future` entry. Offered =
snapshots that are a consecutive batch of >= 2 tasks captured at or after --min-time (the shard's eligible set)."""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> int:
    from scripts_cosim.make_warm_corpus import SnapshotRejected, check_consecutive_batch

    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshots-dir", type=Path, required=True)
    ap.add_argument("--min-time", type=float, default=360.0)
    a = ap.parse_args()
    per = defaultdict(lambda: [0, 0])
    for f in sorted(a.snapshots_dir.glob("cc40s*_*_g*.jsonl")):
        if f.name.startswith("shard_"):
            continue
        rung = re.match(r"cc40s\d+_(.+)_g\d", f.stem).group(1)
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            snap = json.loads(line)
            try:
                if float(snap.get("time", 0.0)) < a.min_time:
                    continue
                check_consecutive_batch(snap, 2)
            except SnapshotRejected:
                continue
            fid = snap.get("fidelity") or {}
            future = fid.get("future") or {}
            waiting = any(node is None and str(other) not in future
                          for row in (fid.get("peers") or {}).values() for other, node, _p in row)
            per[rung][0] += 1
            per[rung][1] += bool(waiting)
    out = {r: {"offered": n, "unplaced_partner": k, "rate": k / n if n else None} for r, (n, k) in sorted(per.items())}
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
