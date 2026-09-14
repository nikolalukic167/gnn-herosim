#!/usr/bin/env python3
"""drainable_regime_v1 -- the S0 screen read (docs/lineages/drainable_regime_v1.md, stage 1).

Everything here is fixed by the registration; this script only executes it. Per rung it reads
the arm summaries written by drainable_regime_v1_s0_screen.sbatch and the LIVE_AUDIT snapshots
the batch arm wrote, and applies the registered bars:

  S1 mechanism        totalPeerExchangeTime / total_rtt      >= 20 %
  S2 trained range    live dim-7 queue column p90            <= 42     (cold-corpus maximum)
  S3 placement        totalPeerRendezvousWait / total_rtt    <  50 %
  S4 corpus           peer-group-aligned audit snapshots     >= 12
  S5 control          R0 (factor 300) reproduces overload: peer share < 1 %, queue share > 99 %

A rung PASSES on S1 and S2 and S3 and S4. The screen reads GO (naming the rungs that pass) or
NO-GO, and VOID if S5 fails -- a VOID means the rescale moved something other than the arrival
rate, and no rung may be read at all.

dim-7 under `legacy_v0` is raw queue depth (its divisor is 1.0 in 516/516 training datasets),
so the column is read straight off each candidate's `queue_length` in the snapshot; "busy"
means queue_length > 0, as in the W0 screen.

S4 does not re-implement "aligned": it calls `make_warm_corpus.check_aligned_peer_group`, the
bridge's own test (exactly `group_size` tasks, consecutive ids, starting at a multiple of the
group size). S4 asks whether the bridge could cut datasets at this rung, so the bridge's
definition is the only one that can answer it; a local copy would drift and S4 would lie.

Usage:
  drainable_regime_s0_read.py --results-dir <dir> --snapshots-dir <dir> --output <json>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_warm_corpus import SnapshotRejected, check_aligned_peer_group  # noqa: E402

BARS = {
    "s1_peer_share_min_pct": 20.0,
    "s2_dim7_busy_p90_max": 42.0,
    "s3_rendezvous_share_max_pct": 50.0,
    "s4_aligned_snapshots_min": 12,
    "s5_control_peer_share_max_pct": 1.0,
    "s5_control_queue_share_min_pct": 99.0,
}
CONTROL_FACTOR = 300.0
GROUP_SIZE = 10


def percentile(values: List[float], q: float) -> float:
    """Linear-interpolation percentile; avoids a numpy import in the read path."""
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return float(s[0])
    pos = (len(s) - 1) * (q / 100.0)
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    return float(s[lo] + (s[hi] - s[lo]) * (pos - lo))


def load_summary(path: Path) -> dict:
    if not path.exists():
        raise RuntimeError(f"missing summary {path} -- the screen is not complete, refusing to read")
    s = json.loads(path.read_text())
    for key in ("total_rtt", "totalPeerExchangeTime", "averageQueueTime", "averageElapsedTime"):
        if s.get(key) is None:
            raise RuntimeError(f"{path}: summary has no {key!r}")
    if not s.get("num_tasks"):
        raise RuntimeError(f"{path}: summary has no num_tasks")
    return s


def shares(summary: dict) -> Dict[str, float]:
    total = float(summary["total_rtt"])
    if total <= 0:
        raise RuntimeError("total_rtt must be positive")
    elapsed = float(summary["averageElapsedTime"])
    return {
        "peer_share_pct": 100.0 * float(summary["totalPeerExchangeTime"]) / total,
        "rendezvous_share_pct": 100.0 * float(summary.get("totalPeerRendezvousWait") or 0.0) / total,
        "queue_share_pct": (100.0 * float(summary["averageQueueTime"]) / elapsed) if elapsed > 0 else 0.0,
    }


def queue_column(snapshot_path: Optional[Path]) -> Dict[str, float]:
    """dim-7 over busy candidate platforms, plus the aligned-snapshot count (S2 and S4)."""
    if snapshot_path is None or not snapshot_path.exists():
        return {"snapshots": 0, "aligned_snapshots": 0, "busy_slots": 0,
                "dim7_busy_p50": 0.0, "dim7_busy_p90": 0.0, "dim7_busy_max": 0.0,
                "batch_size_mean": 0.0}
    busy: List[float] = []
    snapshots = aligned = 0
    batch_sizes: List[int] = []
    for line in snapshot_path.read_text().splitlines():
        if not line.strip():
            continue
        snap = json.loads(line)
        snapshots += 1
        tasks = snap.get("tasks") or []
        batch_sizes.append(len(tasks))
        try:
            check_aligned_peer_group(snap, GROUP_SIZE)
            aligned += 1
        except SnapshotRejected:
            pass
        for task in tasks:
            for cand in task.get("candidates") or []:
                q = float(cand.get("queue_length") or 0.0)
                if q > 0:
                    busy.append(q)
    return {
        "snapshots": snapshots,
        "aligned_snapshots": aligned,
        "busy_slots": len(busy),
        "dim7_busy_p50": percentile(busy, 50),
        "dim7_busy_p90": percentile(busy, 90),
        "dim7_busy_max": max(busy) if busy else 0.0,
        "batch_size_mean": (sum(batch_sizes) / len(batch_sizes)) if batch_sizes else 0.0,
    }


def read_rung(factor: float, results_dir: Path, snapshots_dir: Path) -> dict:
    tag = f"f{int(factor)}"
    batch = load_summary(results_dir / f"knative_network_batch_{tag}.summary.json")
    plain_path = results_dir / f"knative_network_{tag}.summary.json"
    plain = load_summary(plain_path) if plain_path.exists() else None
    sh = shares(batch)
    qc = queue_column(snapshots_dir / f"{tag}.jsonl")
    s1 = sh["peer_share_pct"] >= BARS["s1_peer_share_min_pct"]
    s2 = qc["dim7_busy_p90"] <= BARS["s2_dim7_busy_p90_max"]
    s3 = sh["rendezvous_share_pct"] < BARS["s3_rendezvous_share_max_pct"]
    s4 = qc["aligned_snapshots"] >= BARS["s4_aligned_snapshots_min"]
    out = {
        "factor": factor,
        "arrivals_per_s": (float(batch["num_tasks"]) / float(batch["arrival_span_s"]))
        if batch.get("arrival_span_s") else None,
        "batch": {k: batch.get(k) for k in (
            "num_tasks", "total_rtt", "endTime", "averageElapsedTime", "averageQueueTime",
            "averageCommunicationsTime", "averageExecutionTime", "totalPeerExchangeTime",
            "totalPeerRendezvousWait", "averageOccupation", "scaleEventCount")},
        "shares": sh,
        "queue_column": qc,
        "S1_mechanism": s1,
        "S2_trained_range": s2,
        "S3_placement_sensitivity": s3,
        "S4_corpus_feasibility": s4,
        "passes": bool(s1 and s2 and s3 and s4),
    }
    if plain is not None:
        out["plain"] = {k: plain.get(k) for k in ("total_rtt", "endTime", "averageElapsedTime")}
        out["policies_separate_pct"] = 100.0 * (
            float(plain["total_rtt"]) - float(batch["total_rtt"])) / float(plain["total_rtt"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results-dir", type=Path, required=True)
    ap.add_argument("--snapshots-dir", type=Path, required=True)
    ap.add_argument("--factors", type=float, nargs="+", default=[300.0, 1000.0, 2000.0, 4000.0])
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    rungs = [read_rung(f, args.results_dir, args.snapshots_dir) for f in args.factors]
    control = next((r for r in rungs if r["factor"] == CONTROL_FACTOR), None)
    if control is None:
        raise RuntimeError(f"no control rung at factor {CONTROL_FACTOR} -- S5 cannot be evaluated")
    s5 = (control["shares"]["peer_share_pct"] < BARS["s5_control_peer_share_max_pct"]
          and control["shares"]["queue_share_pct"] > BARS["s5_control_queue_share_min_pct"])
    passing = [r["factor"] for r in rungs if r["passes"]]
    if not s5:
        verdict = "VOID"
    elif passing:
        verdict = "GO"
    else:
        verdict = "NO-GO"
    out = {
        "lineage": "drainable_regime_v1",
        "stage": "S0",
        "registration": "docs/lineages/drainable_regime_v1.md",
        "bars": BARS,
        "rungs": rungs,
        "S5_control_holds": s5,
        "passing_factors": passing,
        "verdict": verdict,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))
    for r in rungs:
        print(f"[rung x{int(r['factor']):5d}] peer {r['shares']['peer_share_pct']:6.2f}% "
              f"queue {r['shares']['queue_share_pct']:7.3f}% rendezvous {r['shares']['rendezvous_share_pct']:6.2f}% "
              f"dim7_p90 {r['queue_column']['dim7_busy_p90']:8.1f} aligned {r['queue_column']['aligned_snapshots']:3d} "
              f"| S1 {int(r['S1_mechanism'])} S2 {int(r['S2_trained_range'])} S3 {int(r['S3_placement_sensitivity'])} "
              f"S4 {int(r['S4_corpus_feasibility'])} -> {'PASS' if r['passes'] else 'fail'}")
    print(f"[S5 control] holds={s5}")
    print(f"[verdict] {verdict}" + (f" at factors {passing}" if passing else ""))
    print(f"[read] wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
