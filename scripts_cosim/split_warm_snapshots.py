#!/usr/bin/env python3
"""peer_affinity_warm_v1: pick the aligned snapshots of a capture run and shard them.

A capture run (LIVE_AUDIT_SNAPSHOT_PATH, stride 2000 on the task id) yields one snapshot
per 2,000 arrivals; this keeps the ones that are a whole peer group (make_warm_corpus's
alignment rule, so the shard count is the dataset count), takes every `--every`-th of
them so the selection is spread over the run rather than front-loaded, and writes
`--shards` JSONL files of `--per` snapshots for the array job. Selection is
deterministic (no RNG) and printed, so the manifest of what went in is the command line.

  python3 scripts_cosim/split_warm_snapshots.py --snapshots <run.jsonl> \\
      --out-dir simulation_data/peer_affinity_warm_v1/w0 --prefix knb --shards 10 --per 5 --every 3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts_cosim.make_warm_corpus import SnapshotRejected, check_aligned_peer_group  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--snapshots", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--shards", type=int, required=True)
    ap.add_argument("--per", type=int, required=True)
    ap.add_argument("--every", type=int, default=1, help="keep every n-th aligned snapshot")
    ap.add_argument("--offset", type=int, default=0, help="skip this many aligned snapshots first")
    ap.add_argument("--group-size", type=int, default=10)
    ap.add_argument("--min-time", type=float, default=0.0)
    args = ap.parse_args()

    aligned = []
    rejected = 0
    with open(args.snapshots) as fh:
        for line in fh:
            if not line.strip():
                continue
            snap = json.loads(line)
            try:
                check_aligned_peer_group(snap, args.group_size)
            except SnapshotRejected:
                rejected += 1
                continue
            if float(snap.get("time", 0.0)) < args.min_time:
                rejected += 1
                continue
            aligned.append(snap)
    chosen = aligned[args.offset::args.every]
    need = args.shards * args.per
    if len(chosen) < need:
        raise SystemExit(
            f"FAIL LOUD: {len(chosen)} aligned snapshots after --every {args.every} / --offset "
            f"{args.offset}, need {need} ({args.shards} x {args.per}); {rejected} rejected"
        )
    chosen = chosen[:need]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for k in range(args.shards):
        path = args.out_dir / f"{args.prefix}_shard{k}.jsonl"
        with open(path, "w") as fh:
            for snap in chosen[k * args.per:(k + 1) * args.per]:
                fh.write(json.dumps(snap, separators=(",", ":")) + "\n")
    times = [round(float(s.get("time", 0.0)), 1) for s in chosen]
    ids = [int(s.get("snapshot_id", -1)) for s in chosen]
    print(f"[split] {args.snapshots}: {len(aligned)} aligned ({rejected} rejected) -> {need} chosen "
          f"in {args.shards} shards of {args.per}; times {times[0]}..{times[-1]} s; snapshot ids {ids}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
