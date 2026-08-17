#!/usr/bin/env python3
"""Build total_rtt table for Regime B transfer gate3 sweep. Fail loud on missing/zero."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict

POLICIES = (
    "knative",
    "ect_pull",
    "gnn873_argmax",
    "distill_seq_reforward_pull",
)
CONFIGS = (
    "default_20_20_p50",
    "01_balanced_40_40_p50",
    "05_sparse_40_40_p25",
)


def _rtt(path: Path) -> float:
    if not path.is_file():
        raise FileNotFoundError(f"FAIL LOUD: missing result {path}")
    data = json.loads(path.read_text())
    rtt = data.get("total_rtt")
    if rtt is None:
        raise KeyError(f"FAIL LOUD: no total_rtt in {path}")
    val = float(rtt)
    if not math.isfinite(val) or val <= 0:
        raise ValueError(f"FAIL LOUD: bad total_rtt={val} in {path}")
    return val


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sweep-dir", type=Path, required=True)
    args = ap.parse_args()
    results = args.sweep_dir / "results"
    table: Dict[str, Dict[str, Any]] = {}
    missing = []
    for cfg in CONFIGS:
        table[cfg] = {}
        for pol in POLICIES:
            path = results / f"{cfg}_{pol}.json"
            try:
                table[cfg][pol] = _rtt(path)
            except (FileNotFoundError, KeyError, ValueError) as exc:
                missing.append(str(exc))
                table[cfg][pol] = None
    if missing:
        raise SystemExit("FAIL LOUD:\n  " + "\n  ".join(missing))

    sums = {pol: sum(float(table[c][pol]) for c in CONFIGS) for pol in POLICIES}
    kn = sums["knative"]
    report = {
        "sweep_dir": str(args.sweep_dir),
        "metric": "total_rtt",
        "workload": "workload-100-100",
        "configs": table,
        "sum": sums,
        "vs_knative": {
            pol: (sums[pol] / kn if kn > 0 else None) for pol in POLICIES
        },
        "interpretation": (
            "Transfer of FilterStore-distilled GNN onto real 100-100 gate3. "
            "A loss vs Kn/873 is the expected honest result — do not ship."
        ),
    }
    out = args.sweep_dir / "compare.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {out}")
    hdr = f"{'config':<28}" + "".join(f"{p:>16}" for p in POLICIES)
    print(hdr)
    for cfg in CONFIGS:
        line = f"{cfg:<28}"
        for pol in POLICIES:
            line += f"{table[cfg][pol]:16.0f}"
        print(line)
    line = f"{'SUM':<28}"
    for pol in POLICIES:
        line += f"{sums[pol]:16.0f}"
    print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
