#!/usr/bin/env python3
"""route_c_link_transfer_v1 SCREEN — the registered manipulation check (LINEAGES.md,
registration 2026-08-26).

A rung is only VALID if link waiting is a material share of RTT; otherwise no verdict may
be read from it. The registered statistic, per rung corpus:

    per dataset:  share = sum over sweep rows of link_wait_total
                          / sum over sweep rows of rtt
    rung:         median share over datasets;  VALID iff median >= 0.10

Reads the per-plan link fields that HEROSIM_RETAIN_LINK_STATS=1 writes into
placements.jsonl. Fail-loud contract: a corpus with any row missing link_wait_total
raises (a sweep mixing rows with and without link fields is undecomposable — same rule
as task_times); an incomplete sweep raises via placement_metadata.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

VALID_THRESHOLD = 0.10  # registered; see LINEAGES route_c_link_transfer_v1 SCREEN


def dataset_share(ds_dir: Path) -> dict:
    meta_path = ds_dir / "placement_metadata.json"
    with open(meta_path) as fh:
        meta = json.load(fh)
    if not meta.get("sweep_complete"):
        raise RuntimeError(f"{ds_dir}: sweep_complete is not true — refusing to read a "
                           "truncated sweep")
    wait = rtt = transfer_avg_sum = 0.0
    n = 0
    with open(ds_dir / "placements" / "placements.jsonl") as fh:
        for line in fh:
            row = json.loads(line)
            if "link_wait_total" not in row:
                raise RuntimeError(
                    f"{ds_dir}: row without link_wait_total — corpus was not generated "
                    "with HEROSIM_RETAIN_LINK_STATS=1; the manipulation check cannot run")
            wait += float(row["link_wait_total"])
            rtt += float(row["rtt"])
            transfer_avg_sum += float(row["link_transfer_avg"])
            n += 1
    if n == 0 or rtt <= 0:
        raise RuntimeError(f"{ds_dir}: empty sweep or non-positive summed rtt")
    return {"dataset": ds_dir.name, "n_rows": n, "share": wait / rtt,
            "mean_link_transfer_avg": transfer_avg_sum / n}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", action="append", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    reports = []
    for corpus in args.corpus:
        ds_dirs = sorted(d for d in corpus.glob("ds_*") if d.is_dir())
        if not ds_dirs:
            raise RuntimeError(f"{corpus}: no ds_* directories")
        rows = [dataset_share(d) for d in ds_dirs]
        shares = sorted(r["share"] for r in rows)
        median = shares[len(shares) // 2]
        report = {
            "corpus": str(corpus),
            "n_datasets": len(rows),
            "median_share": median,
            "min_share": shares[0],
            "max_share": shares[-1],
            "valid_threshold": VALID_THRESHOLD,
            "rung_valid": median >= VALID_THRESHOLD,
            "per_dataset": rows,
        }
        reports.append(report)
        print(f"{corpus}: n={len(rows)}  link-wait share of rtt "
              f"median={median:.4f}  min={shares[0]:.4f}  max={shares[-1]:.4f}  "
              f"-> rung {'VALID' if report['rung_valid'] else 'INVALID'} "
              f"(threshold {VALID_THRESHOLD})")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(reports, fh, indent=1, sort_keys=True)
        print(f"written: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
