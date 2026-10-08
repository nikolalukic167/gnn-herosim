#!/usr/bin/env python3
"""workload_fix_v1 provisional load: arrival multipliers at which CD's median queue share is ~0.1 and ~0.3.

Protocol of load_recalibration_v1 (docs/lineages/workload_fix_v1.md, "Provisional load"): the calibration
topologies x two windows, the CURRENT workload (legacy payloads, single origin), R1 physics, at most 8 evaluated
multipliers per target. A multiplier m scales the x1 arrival rate (timestamps and batch_timeout by 1/m, the ladder
protocol). Bisection is in log m. The two bracket ends are evaluated first (steps 1-2, shared by the targets), then
midpoints. The statistic is the median over (topology, window) cells of CD's queue_share (mean queue time / mean
latency).

Rules fixed before the first step:
  * a step needs >= `--min-cells` of its cells to finish; fewer counts as overload (share above target) and is
    marked FAILED-CELLS, because a rung where CD does not finish is not a usable rung;
  * the answer is the evaluated multiplier with share closest to the target; the bracketing pair is reported too;
  * if the ends do not bracket a target, the search stops for it and says so.

  workload_fix_v1_bisect.py --work <dir> --grounded-wl <x1 wl> --cfg-dir <cfg> --topologies 9601 9602 9603 9604 \\
      --windows g0 g1 --log <steps.json>        (run on datalab, R1 environment exported)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics as st
import subprocess
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
TARGETS = (0.1, 0.3)


def tag_for(m: float) -> str:
    return "m" + f"{m:.4f}".replace(".", "p")


def next_multiplier(lo: float, hi: float) -> float:
    return round(math.exp(0.5 * (math.log(lo) + math.log(hi))), 4)


def search(target: float, evaluate: Callable[[float], Optional[float]], m_lo: float, m_hi: float, max_steps: int) -> dict:
    """Log-bisection for share(m) = target; ``evaluate`` returns the median share, or ``inf`` for FAILED-CELLS.
    Evaluations are cached by the caller, so a repeated multiplier is not a repeated step. Returns the step list."""
    steps = []

    def step(m: float) -> float:
        share = evaluate(m)
        steps.append({"step": len(steps) + 1, "m": m, "share": share})
        return share

    s_lo, s_hi = step(m_lo), step(m_hi)
    out = {"target": target, "steps": steps}
    if not s_lo < target:
        out["status"] = "UNBRACKETED-LOW"
    elif not s_hi >= target:
        out["status"] = "UNBRACKETED-HIGH"
    else:
        lo, hi = m_lo, m_hi
        while len(steps) < max_steps:
            mid = next_multiplier(lo, hi)
            if mid <= lo or mid >= hi:
                break
            s = step(mid)
            lo, hi = (mid, hi) if s < target else (lo, mid)
        out["status"] = "BRACKETED"
        out["bracket"] = [lo, hi]
    finite = [s for s in steps if math.isfinite(s["share"])]
    best = min(finite, key=lambda s: abs(s["share"] - target)) if finite else None
    out["closest"] = best
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--grounded-wl", type=Path, required=True)
    ap.add_argument("--cfg-dir", type=Path, required=True)
    ap.add_argument("--topologies", type=int, nargs="+", required=True)
    ap.add_argument("--windows", nargs="+", default=["g0", "g1"])
    ap.add_argument("--m-lo", type=float, default=0.5)
    ap.add_argument("--m-hi", type=float, default=24.0)
    ap.add_argument("--max-steps", type=int, default=8)
    ap.add_argument("--min-cells", type=int, default=6)
    ap.add_argument("--parallel", type=int, default=16)
    ap.add_argument("--timeout", type=int, default=5400)
    ap.add_argument("--log", type=Path, required=True)
    a = ap.parse_args()
    for k, v in (("HEROSIM_TRANSFER_MODEL", "pipelined"), ("HEROSIM_REPLICA_RELEASE", "1"), ("HEROSIM_SCALEOUT", "kpa"),
                 ("GATE_FIXED_POLICY_TIME_SCALE", "1.0")):
        if os.environ.get(k) != v:
            raise SystemExit(f"FAIL LOUD: {k}={os.environ.get(k)!r}; workload_fix_v1 runs on R1 ({k}={v})")
    a.work.mkdir(parents=True, exist_ok=True)
    cache: Dict[float, float] = {}
    cells: Dict[str, dict] = {}

    def evaluate(m: float) -> float:
        if m in cache:
            return cache[m]
        tag = tag_for(m)
        subprocess.run([sys.executable, str(ROOT / "scripts_cosim/workload_fix_v1_build.py"), "--grounded-wl", str(a.grounded_wl),
                        "--cfg-dir", str(a.cfg_dir), "--topologies", *map(str, a.topologies), "--rung", f"{tag}={1.0 / m!r}",
                        "--out", str(a.work / tag)], check=True, cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT)})
        out = a.work / tag / "gate"
        env = {**os.environ, "WF1_RUNGS": tag, "WF1_TOPOS": ",".join(map(str, a.topologies)),
               "WF1_CAL_WINDOWS": ",".join(a.windows)}
        subprocess.run([sys.executable, str(ROOT / "scripts_cosim/fresh_topo_burst_v1_gate.py"), "wf1cal", "--inputs",
                        str(a.work / tag), "--out", str(out), "--parallel", str(a.parallel), "--mem", "4G",
                        "--timeout", str(a.timeout), "--no-scope"], check=False, cwd=ROOT, env=env)
        shares, lat = [], []
        for t in a.topologies:
            for w in a.windows:
                f = out / f"cc40s{t}__{w}{tag}__cd_s0.summary.json"
                if f.exists():
                    r = json.loads(f.read_text())
                    shares.append(float(r["queue_share"]))
                    lat.append(float(r["averageElapsedTime"]))
        cells[tag] = {"m": m, "n_cells": len(shares), "of": len(a.topologies) * len(a.windows), "shares": shares,
                      "median_latency_s": st.median(lat) if lat else None}
        cache[m] = st.median(shares) if len(shares) >= a.min_cells else math.inf
        print(f"[eval] m={m} {tag}: {len(shares)}/{cells[tag]['of']} cells, median queue share "
              f"{cache[m] if math.isfinite(cache[m]) else 'FAILED-CELLS'}", flush=True)
        return cache[m]

    # The targets are searched one after the other; an evaluation both need (the bracket ends) runs once.
    results = [search(t, evaluate, a.m_lo, a.m_hi, a.max_steps) for t in TARGETS]
    doc = {"targets": list(TARGETS), "topologies": a.topologies, "windows": a.windows, "m_range": [a.m_lo, a.m_hi],
           "max_steps": a.max_steps, "min_cells": a.min_cells, "results": results, "evaluations": cells}
    a.log.write_text(json.dumps(doc, indent=1, default=str))
    for r in results:
        print(json.dumps({k: r[k] for k in ("target", "status", "closest")}, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
