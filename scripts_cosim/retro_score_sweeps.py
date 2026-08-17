#!/usr/bin/env python3
"""
Retro-score every live-sim result JSON on disk with the tail metrics.

The 99-quantile response-time distribution has always been written into result
JSONs but no compare script read it, so every historical sweep can be scored on
p50/p90/p99 without re-running a single simulation. Also records the declared
warmth physics per sweep, since undeclared runs are not cross-comparable.

Writes `<sweep>/tail_metrics.json` per sweep plus a combined index.

Usage:
  pipenv run python3 scripts_cosim/retro_score_sweeps.py
  pipenv run python3 scripts_cosim/retro_score_sweeps.py --sweeps-root <dir> --sweep <name>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts_cosim.sweep_metrics import MetricExtractionError, load_metrics  # noqa: E402

SKIP_NAMES = {
    "compare.json",
    "manifest.json",
    "summary.json",
    "tail_metrics.json",
    "infrastructure.json",
    "workload.json",
}
# Sweep dirs also hold simulation inputs; those are skipped, but anything that
# looks like a result and cannot be scored is a hard failure.
INPUT_MARKERS = ('"pci"', '"platform_types"', '"events"', '"brute_force"')
DEFAULT_ROOT = Path("simulation_data/normal_sim_sweeps")


def result_files(sweep_dir: Path) -> List[Path]:
    candidates: List[Path] = []
    for path in sorted(sweep_dir.rglob("*.json")):
        if path.name in SKIP_NAMES or path.name.endswith(".decode_stats.json"):
            continue
        if "configs" in path.parts:
            continue
        candidates.append(path)
    return candidates


def is_simulation_input(path: Path) -> bool:
    with open(path, "rb") as fh:
        head = fh.read(65536).decode("utf-8", "ignore")
    if '"total_rtt"' in head or '"status"' in head:
        return False
    return any(marker in head for marker in INPUT_MARKERS)


def score_sweep(sweep_dir: Path) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, str]] = []
    physics: Dict[str, int] = {}
    n_inputs = 0

    for path in result_files(sweep_dir):
        if is_simulation_input(path):
            n_inputs += 1
            continue
        try:
            metrics = load_metrics(path, require_tail=False)
        except MetricExtractionError as exc:
            failures.append({"path": str(path), "error": str(exc)})
            continue
        key = str(metrics["warmth_physics"])
        physics[key] = physics.get(key, 0) + 1
        rows.append(
            {
                "name": metrics["name"],
                "total_rtt": metrics["total_rtt"],
                "p50": metrics["p50"],
                "p90": metrics["p90"],
                "p99": metrics["p99"],
                "averageQueueTime": metrics["averageQueueTime"],
                "coldStartProportion": metrics["coldStartProportion"],
                "warmth_physics": metrics["warmth_physics"],
                "warmth_physics_source": metrics["warmth_physics_source"],
            }
        )

    return {
        "sweep": sweep_dir.name,
        "sweep_dir": str(sweep_dir),
        "n_scored": len(rows),
        "n_unscorable": len(failures),
        "n_simulation_inputs_skipped": n_inputs,
        "warmth_physics_counts": physics,
        "results": rows,
        "unscorable": failures,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweeps-root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--sweep", action="append", default=None, help="sweep name (repeatable)")
    ap.add_argument("--index", type=Path, default=None)
    ap.add_argument(
        "--no-write",
        action="store_true",
        help="print only; do not write tail_metrics.json into sweep dirs",
    )
    args = ap.parse_args()

    root = args.sweeps_root
    if not root.is_dir():
        raise FileNotFoundError(f"No sweeps root: {root}")

    names = args.sweep or sorted(p.name for p in root.iterdir() if p.is_dir())
    index: List[Dict[str, Any]] = []
    total_scored = 0
    total_failed = 0

    print(f"{'sweep':<56} {'scored':>6} {'no_tail':>7}  physics")
    print("-" * 100)
    for name in names:
        sweep_dir = root / name
        if not sweep_dir.is_dir():
            raise FileNotFoundError(f"No such sweep: {sweep_dir}")
        report = score_sweep(sweep_dir)
        if report["n_scored"] == 0 and report["n_unscorable"] == 0:
            continue
        total_scored += report["n_scored"]
        total_failed += report["n_unscorable"]
        physics_str = ", ".join(f"{k}×{v}" for k, v in sorted(report["warmth_physics_counts"].items()))
        print(f"{name:<56} {report['n_scored']:>6} {report['n_unscorable']:>7}  {physics_str}")
        if not args.no_write:
            (sweep_dir / "tail_metrics.json").write_text(json.dumps(report, indent=2) + "\n")
        index.append(
            {
                "sweep": name,
                "n_scored": report["n_scored"],
                "n_unscorable": report["n_unscorable"],
                "warmth_physics_counts": report["warmth_physics_counts"],
            }
        )

    print("-" * 100)
    print(f"{'TOTAL':<56} {total_scored:>6} {total_failed:>7}")

    if not args.no_write:
        out = args.index or (root / "tail_metrics_index.json")
        out.write_text(json.dumps({"sweeps": index}, indent=2) + "\n")
        print(f"Wrote {out}")

    if total_failed:
        print(
            f"\nERROR: {total_failed} result-shaped JSON files could not be scored",
            file=sys.stderr,
        )
        for entry in index:
            if entry["n_unscorable"]:
                print(f"  {entry['sweep']}: {entry['n_unscorable']}", file=sys.stderr)
        return 1

    if total_scored == 0:
        print("ERROR: nothing scored", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
