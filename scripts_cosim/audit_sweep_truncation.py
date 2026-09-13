#!/usr/bin/env python3
"""Sweep-truncation census: expected placement count vs rows in placements.jsonl.

The brute-force sweep loop can lose placements silently (worker exception or timeout →
row omitted from placements.jsonl while best.json is still written; instrumented
fail-loud in executecosimulation.py since 2026-08-23). Every dataset generated before
that carries no loss counters — but `placement_metadata.json` has always recorded
`num_placements` (the enumerated combination count), so truncation is detectable as
`rows(placements.jsonl) != num_placements`.

A truncated sweep can silently change the "optimum" the training label is taken from,
so this census is gate D1 of the co-sim deep-dive campaign.

Usage:
  python3 scripts_cosim/audit_sweep_truncation.py \
      --collection gnn_datasets_4tasks_contention_v2 ... [--json OUT] [--max-list 10]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


def count_lines(path: Path) -> int:
    n = 0
    with open(path, "rb") as f:
        for _ in f:
            n += 1
    return n


def audit_dataset(ds_dir: Path) -> Dict[str, Any]:
    meta_path = ds_dir / "placement_metadata.json"
    jsonl_path = ds_dir / "placements" / "placements.jsonl"
    row: Dict[str, Any] = {"dataset": ds_dir.name}

    if not jsonl_path.exists():
        row["status"] = "missing_jsonl"
        return row
    if not meta_path.exists():
        row["status"] = "missing_metadata"
        row["rows"] = count_lines(jsonl_path)
        return row

    try:
        meta = json.loads(meta_path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        row["status"] = "corrupt_metadata"
        row["error"] = str(exc)
        return row

    expected = meta.get("num_placements")
    rows = count_lines(jsonl_path)
    row["expected"] = expected
    row["rows"] = rows
    if expected is None:
        row["status"] = "metadata_without_count"
    elif rows == expected:
        row["status"] = "complete"
    else:
        row["status"] = "truncated" if rows < expected else "overfull"
        row["missing"] = int(expected) - rows
    # Post-2026-08-23 sweeps carry the loss counters directly.
    for key in ("timed_out", "worker_failed", "worker_exception", "early_terminated"):
        if key in meta:
            row[key] = meta[key]
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sim-root", default="simulation_data", type=Path)
    ap.add_argument(
        "--collection", action="append", required=True, help="Collection dir name; repeatable"
    )
    ap.add_argument("--json", type=Path, default=None, help="Write full JSON report here")
    ap.add_argument("--max-list", type=int, default=10, help="Worst offenders listed per collection")
    args = ap.parse_args()

    report: Dict[str, Any] = {}
    for name in args.collection:
        base = args.sim_root / name
        ds_dirs = sorted(d for d in base.glob("ds_*") if d.is_dir())
        if not ds_dirs:
            print(f"[{name}] NO ds_* directories under {base}", flush=True)
            report[name] = {"error": "no_datasets"}
            continue

        rows: List[Dict[str, Any]] = [audit_dataset(d) for d in ds_dirs]
        by_status: Dict[str, int] = {}
        missing_total = 0
        for r in rows:
            by_status[r["status"]] = by_status.get(r["status"], 0) + 1
            missing_total += int(r.get("missing", 0) or 0)
        truncated = [r for r in rows if r["status"] in ("truncated", "overfull")]
        truncated.sort(key=lambda r: -abs(r.get("missing", 0)))

        summary = {
            "datasets": len(rows),
            "by_status": by_status,
            "missing_rows_total": missing_total,
            "worst": truncated[: args.max_list],
        }
        report[name] = {"summary": summary, "datasets": rows}
        print(f"[{name}] {len(rows)} datasets: {by_status} missing_rows_total={missing_total}", flush=True)
        for r in truncated[: args.max_list]:
            print(
                f"    {r['dataset']}: expected={r.get('expected')} rows={r.get('rows')} "
                f"({r['status']})",
                flush=True,
            )

    if args.json:
        slim = {
            name: (payload if "error" in payload else {"summary": payload["summary"],
                                                        "datasets": payload["datasets"]})
            for name, payload in report.items()
        }
        args.json.write_text(json.dumps(slim, indent=2))
        print(f"Wrote {args.json}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
